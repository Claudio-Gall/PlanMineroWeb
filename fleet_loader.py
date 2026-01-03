"""
Fleet Loader - SIMPLE AND CORRECT
Primary source: Pala-Fase (has equipment, phase, and tonnage)
Secondary: KPI-Palas only for Cargador Frontal (not in Pala-Fase)
"""
import pandas as pd
import numpy as np

def load_fleet_clean(file_path):
    """
    Smart crosscheck loader:
    - Primary: Pala-Fase for equipment with individual rows (P05, P06, Bulldozer)
    - Secondary: KPI-Palas for Pala 03, Pala 04, Cargador Frontal
    - Crosscheck: Detect Remanejo periods from Pala-Fase to assign correct phase
    """
    
    print(f"--- LOADING FLEET (CROSSCHECK) FROM: {file_path} ---")
    
    result_data = []
    
    # STEP 1: Load all data from Pala-Fase
    df_pf = pd.read_excel(file_path, sheet_name="Pala-Fase", header=None)
    pf_data, remanejo_periods = _load_pala_fase_with_remanejo(df_pf)
    print(f"  Pala-Fase: {len(pf_data)} records")
    print(f"  Remanejo periods detected: {len(remanejo_periods)}")
    result_data.extend(pf_data)
    
    # STEP 2: Load Pala 03, Pala 04, Cargador Frontal from KPI-Palas
    df_kpi = pd.read_excel(file_path, sheet_name="KPI-Palas", header=None)
    kpi_data = _load_from_kpi_palas(df_kpi, remanejo_periods)
    print(f"  KPI-Palas (P03, P04, CF): {len(kpi_data)} records")
    result_data.extend(kpi_data)
    
    # STEP 3: Load Camiones (Trucks) from KPI-Camiones
    try:
        df_cam = pd.read_excel(file_path, sheet_name="KPI-Camiones", header=0)
        cam_data = _load_camiones(df_cam)
        print(f"  KPI-Camiones: {len(cam_data)} records")
        result_data.extend(cam_data)
    except Exception as e:
        print(f"  KPI-Camiones: Could not load ({str(e)})")
    
    # Create DataFrame
    df_result = pd.DataFrame(result_data)
    
    print(f"Total rows: {len(df_result)}")
    if not df_result.empty:
        print(f"Equipos: {sorted(df_result['Equipo'].unique())}")
        print(f"Fases: {sorted(df_result['Fase'].unique())}")
    
    return df_result


def _load_pala_fase_with_remanejo(df):
    """
    Load data from Pala-Fase and also extract Remanejo periods.
    Returns: (data_list, remanejo_periods_set)
    """
    
    month_map = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    
    # Build year mapping
    years_row = df.iloc[0].tolist()
    filled_years = []
    current_year = 2026
    for val in years_row:
        if pd.notna(val):
            val_str = str(val).strip().replace('.0', '')
            if val_str.isdigit():
                current_year = int(val_str)
        filled_years.append(current_year)
    
    # Build column time mapping
    months_row = df.iloc[1].tolist()
    col_time_map = {}
    
    for col_idx in range(3, len(months_row)):  # Start from col 3 to include Enero 2026
        year = filled_years[col_idx] if col_idx < len(filled_years) else 2026
        period_str = str(months_row[col_idx]).upper().strip()
        
        # Check for individual months
        if period_str in month_map:
            col_time_map[col_idx] = (year, month_map[period_str])
        # Check for trimester format: "1ER TRIMESTRE_2028", etc
        elif 'TRIMESTRE' in period_str:
            if '1ER' in period_str:
                col_time_map[col_idx] = (year, 1, 'Q1')  # (year, month, quarter_marker)
            elif '2DO' in period_str:
                col_time_map[col_idx] = (year, 4, 'Q2')
            elif '3ER' in period_str:
                col_time_map[col_idx] = (year, 7, 'Q3')
            elif '4TO' in period_str:
                col_time_map[col_idx] = (year, 10, 'Q4')
    
    # Extract data
    data = []
    remanejo_periods = set()  # Track (year, month) tuples where Remanejo is active
    
    for row_idx in range(3, len(df)):
        equipo_raw = str(df.iloc[row_idx, 1]).strip().upper()
        fase_raw = str(df.iloc[row_idx, 2]).strip().upper()
        
        if pd.isna(df.iloc[row_idx, 1]) or equipo_raw == '' or 'TOTAL' in equipo_raw:
            continue
        
        # Check if this is the Remanejo row
        is_remanejo_row = 'REMANEJO' in equipo_raw and 'REMANEJO' in fase_raw
        
        if is_remanejo_row:
            # Track which periods have Remanejo activity (don't add as equipment data)
            for col_idx, period_info in col_time_map.items():
                year, month = period_info[0], period_info[1]  # Works for both 2 and 3-tuples
                ton_val = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
                if pd.notna(ton_val) and ton_val > 0:
                    remanejo_periods.add((year, month))
            continue  # Don't add "Remanejo" as an equipment
        
        # Map equipment and phase
        equipo = _map_equipo(equipo_raw)
        fase = _map_fase(fase_raw)
        
        if not equipo or not fase:
            continue
        
        # Extract tonnage for each month/quarter
        for col_idx, period_info in col_time_map.items():
            year, month = period_info[0], period_info[1]
            is_quarter = len(period_info) == 3  # Has quarter marker
            
            ton_val = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
            
            if pd.notna(ton_val) and ton_val > 0:
                month_names = {
                    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
                }
                
                if is_quarter:
                    quarter = period_info[2]  # Get quarter marker (Q1, Q2, etc)
                    periodo = f"{quarter} {year}"
                else:
                    periodo = f"{month_names[month]} {year}"
                    quarter = f"Q{(month - 1) // 3 + 1}"
                
                data.append({
                    'Year': year,
                    'Month': month,
                    'Quarter': quarter,
                    'Periodo': periodo,
                    'Equipo': equipo,
                    'Fase': fase,
                    'Ton': ton_val,
                    'Source': 'Pala-Fase'
                })
    
    return data, remanejo_periods


def _load_from_kpi_palas(df, remanejo_periods):
    """
    Load Pala 03, Pala 04, and Cargador Frontal from KPI-Palas.
    Use remanejo_periods to assign correct phase to P03 and P04.
    """
    
    month_map = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    
    
    # Find equipment columns in KPI-Palas - Row 1 has IDs (CF, P03, P04, etc)
    equip_cols = {}
    for col_idx in range(3, df.shape[1]):
        header = str(df.iloc[2, col_idx]).upper().strip()
        if "K TON" in header:
            equip_id = str(df.iloc[1, col_idx]).strip().upper()
            # Only load CF, P03, P04 (others are in Pala-Fase)
            if equip_id in ['CF', 'P03', 'P04']:
                mapped = _map_equipo(equip_id)
                if mapped:
                    equip_cols[col_idx] = mapped
    
    # Extract data
    data = []
    current_year = 2026
    
    for row_idx in range(3, df.shape[0]):
        # Get year
        year_val = df.iloc[row_idx, 0]
        if pd.notna(year_val):
            year_str = str(year_val).strip().replace('.0', '')
            if year_str.isdigit():
                current_year = int(year_str)
        
        # Get period (month or trimester)
        period_str = str(df.iloc[row_idx, 1]).upper().strip()
        month_num = None
        is_quarter = False
        quarter_marker = None
        
        # Check for individual months
        if period_str in month_map:
            month_num = month_map[period_str]
        # Check for trimester format
        elif 'TRIMESTRE' in period_str:
            is_quarter = True
            if '1ER' in period_str:
                month_num = 1
                quarter_marker = 'Q1'
            elif '2DO' in period_str:
                month_num = 4
                quarter_marker = 'Q2'
            elif '3ER' in period_str:
                month_num = 7
                quarter_marker = 'Q3'
            elif '4TO' in period_str:
                month_num = 10
                quarter_marker = 'Q4'
        
        if month_num is not None:
            
            # Extract values for each equipment
            for col_idx, equipo in equip_cols.items():
                ton_val = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
                
                if pd.notna(ton_val) and ton_val > 0:
                    # Year-specific filtering: P04 appears in Pala-Fase from 2027+
                    # Don't load P04 from KPI-Palas in 2027+ to avoid duplication
                    if equipo == 'Pala 04' and current_year >= 2027:
                        continue
                    
                    # Determine phase - year-specific Remanejo composition
                    if (current_year, month_num) in remanejo_periods:
                        # Remanejo composition changes by year:
                        #  2026: Pala 03 + Pala 04
                        #  2027+: Pala 03 + Cargador Frontal
                        if current_year == 2026:
                            if equipo in ['Pala 03', 'Pala 04']:
                                fase = 'Remanejo'
                            else:
                                fase = 'Fase 5'  # CF in 2026 has Fase 5
                        else:  # 2027 onwards
                            if equipo in ['Pala 03', 'Cargador Frontal']:
                                fase = 'Remanejo'
                            else:
                                fase = 'Fase 5'  # P04 in 2027+ has Fase 5
                    else:
                        fase = 'Fase 5'  # Default when no Remanejo
                    
                    month_names = {
                        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
                    }
                    
                    if is_quarter:
                        periodo = f"{quarter_marker} {current_year}"
                        quarter = quarter_marker
                    else:
                        periodo = f"{month_names[month_num]} {current_year}"
                        quarter = f"Q{(month_num - 1) // 3 + 1}"
                    
                    data.append({
                        'Year': current_year,
                        'Month': month_num,
                        'Quarter': quarter,
                        'Periodo': periodo,
                        'Equipo': equipo,
                        'Fase': fase,
                        'Ton': ton_val,
                        'Source': 'KPI-Palas'
                    })
    
    return data


def _map_equipo(raw):
    """Map equipment name"""
    raw = raw.upper()
    if 'P06' in raw or 'PALA 6' in raw:
        return 'Pala 06'
    elif 'P05' in raw or 'PALA 5' in raw:
        return 'Pala 05'
    elif 'P04' in raw or 'PALA 4' in raw:
        return 'Pala 04'
    elif 'P03' in raw or 'PALA 3' in raw:
        return 'Pala 03'
    elif 'BULL' in raw:
        return 'Bulldozer'
    elif 'CF' == raw or 'FRONTAL' in raw or 'CARGADOR' in raw:
        return 'Cargador Frontal'
    elif 'REMANEJO' in raw:
        return 'Remanejo'
    return None


def _map_fase(raw):
    """Map phase name"""
    raw = raw.upper()
    if 'F05' in raw:  # Matches F05 and F05C
        return 'Fase 5'
    elif 'F04' in raw:
        return 'Fase 4'
    elif 'F03' in raw:
        return 'Fase 3'
    elif 'REMANEJO' in raw:
        return 'Remanejo'
    return None


def _load_camiones(df):
    """
    Load Camiones (Trucks) data from KPI-Camiones sheet
    Returns list of dicts with: Year, Month, Quarter, Periodo, Equipo='Camiones', Ton, Rend
    """
    month_map = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    
    data = []
    
    for _, row in df.iterrows():
        # Get year
        year_val = row.get('Año', None)
        if pd.isna(year_val):
            continue
        
        year = int(year_val) if isinstance(year_val, (int, float)) else int(str(year_val).strip().replace('.0', ''))
        
        # Get month or trimester
        period_str = str(row.get('Periodo', '')).upper().strip()
        month_num = None
        is_quarter = False
        quarter_marker = None
        
        # Check for individual months
        if period_str in month_map:
            month_num = month_map[period_str]
        # Check for trimester format
        elif 'TRIMESTRE' in period_str:
            is_quarter = True
            if '1ER' in period_str:
                month_num = 1
                quarter_marker = 'Q1'
            elif '2DO' in period_str:
                month_num = 4
                quarter_marker = 'Q2'
            elif '3ER' in period_str:
                month_num = 7
                quarter_marker = 'Q3'
            elif '4TO' in period_str:
                month_num = 10
                quarter_marker = 'Q4'
        
        if month_num is None:
            continue
        
        quarter = quarter_marker if is_quarter else f"Q{(month_num - 1) // 3 + 1}"
        
        month_names = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        
        if is_quarter:
            periodo = f"{quarter_marker} {year}"
        else:
            periodo = f"{month_names[month_num]} {year}"
        
        # Get N° Camiones from column H
        n_camiones = None
        for col in df.columns:
            if 'CAMIONES' in str(col).upper() and 'N°' in str(col).upper():
                n_camiones = row.get(col, None)
                break
        
        # Fallback: try just "N° Camiones" or similar variations
        if n_camiones is None:
            n_camiones = row.get('N° Camiones', None)
        
        # Get performance (RenDCamion)
        rend_val = None
        for col in df.columns:
            if 'REND' in str(col).upper() and 'CAM' in str(col).upper():
                rend_val = row.get(col, None)
                if pd.notna(rend_val):
                    break
        
        # Only add if we have valid N° Camiones data
        if pd.notna(n_camiones) and n_camiones > 0:
            data_entry = {
                'Year': year,
                'Month': month_num,
                'Quarter': quarter,
                'Periodo': periodo,
                'Equipo': 'Camiones',
                'Fase': 'Transporte',  # Camiones don't have phases, use generic
                'Ton': float(n_camiones),  # Store N° Camiones in Ton column
                'Source': 'KPI-Camiones'
            }
            
            if pd.notna(rend_val):
                data_entry['Rend'] = float(rend_val)
            
            data.append(data_entry)
    
    return data


if __name__ == "__main__":
    df = load_fleet_clean("plan_budget_real.xlsx")
    
    if not df.empty:
        print("\n" + "="*70)
        print("ENERO 2026 - VERIFICACIÓN POR EQUIPO Y FASE")
        print("="*70)
        
        df_ene2026 = df[(df['Year'] == 2026) & (df['Month'] == 1)]
        
        for equipo in sorted(df_ene2026['Equipo'].unique()):
            eq_data = df_ene2026[df_ene2026['Equipo'] == equipo]
            print(f"\n{equipo}:")
            for _, row in eq_data.iterrows():
                print(f"  {row['Fase']}: {row['Ton']:.2f} kTon")
