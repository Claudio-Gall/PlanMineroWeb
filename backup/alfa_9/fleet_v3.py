import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def load_fleet_data_v3_hybrid(file_path): 
    """
    MASTER STRATEGY V13: CORRECT MULTI-YEAR STRUCTURE
    Based on user confirmation:
    - 2026: Rows 4-15 (monthly, 12 rows)
    - 2027: Rows 16-27 (monthly, 12 rows)
    - 2028: Rows 28-31 (quarterly, 4 rows)
    - 2029: Row 32+ (quarterly)
    
    Year headers are 1 row ABOVE data blocks (rows 3, 15, 27, 31).
    """
    print(f"--- LOADING FLEET V13 (CORRECT STRUCTURE) FROM: {file_path} ---")
    data_rows = []
    
    try:
        # --- PART A: PHASE MAP FROM PALA-FASE ---
        df_pf = pd.read_excel(file_path, sheet_name="Pala-Fase", header=None)
        
        time_map_pf = {} 
        current_year = 2026
        months_dict = {
            'ENE':1,'FEB':2,'MAR':3,'ABR':4,'MAY':5,'JUN':6,'JUL':7,'AGO':8,'SEP':9,'OCT':10,'NOV':11,'DIC':12,
            'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,'SEPTIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12
        }
        
        for c in range(2, df_pf.shape[1]):
            val_y = df_pf.iloc[0, c]
            if pd.notna(val_y): 
                try:
                    current_year = int(val_y)
                except:
                    pass
            val_m = str(df_pf.iloc[1, c]).strip().upper()[:3]
            if val_m in months_dict:
                time_map_pf[c] = (current_year, months_dict[val_m])

        # Apply FFILL to Col 0 and 1 for merged cells
        df_pf[0] = df_pf[0].fillna(method='ffill')
        df_pf[1] = df_pf[1].fillna(method='ffill')

        # Build Phase Proportions Map
        # keyed by (month, equipment) -> List of {"phase": phase, "ton": value}
        phase_lookup = {}
        
        for i in range(2, len(df_pf)):
            mega_grupo = str(df_pf.iloc[i, 0]).strip().upper()
            equipo_label = str(df_pf.iloc[i, 1]).strip().upper()
            fase_label = str(df_pf.iloc[i, 2]).strip().upper()
            
            equip = None
            if "BULL" in equipo_label or "BULL" in mega_grupo: equip = "Bulldozer"
            elif "P04" in equipo_label: equip = "Pala 04"
            elif "P05" in equipo_label: equip = "Pala 05"
            elif "P06" in equipo_label: equip = "Pala 06"
            
            phase = None
            if "F05" in fase_label: phase = "Fase 5"
            elif "F04" in fase_label: phase = "Fase 4" 
            elif "F03" in fase_label: phase = "Fase 3"
            elif "REMANEJO" in mega_grupo: phase = "Remanejo"
            
            if equip and phase and "TOTAL" not in equipo_label:
                for c, (yr, mn) in time_map_pf.items():
                    val = pd.to_numeric(df_pf.iloc[i, c], errors='coerce')
                    if pd.notna(val) and val > 0:
                        if (mn, equip) not in phase_lookup:
                            phase_lookup[(mn, equip)] = []
                        phase_lookup[(mn, equip)].append({"phase": phase, "ton": val})
                        
                        # Bulldozers are ONLY in this sheet
                        if equip == "Bulldozer":
                            data_rows.append({
                                'Year': yr, 'Month': mn, 
                                'Quarter': f"Q{(mn-1)//3 + 1}",
                                'Equipo': equip, 'Fase': phase,
                                'Ton': val, 'Source': 'Pala-Fase'
                            })

        # --- PART B: KPI-PALAS (MULTI-YEAR BLOCKS) ---
        df_kpi = pd.read_excel(file_path, sheet_name="KPI-Palas", header=None)
        
        # Identify equipment columns
        equip_cols = []
        for c in range(df_kpi.shape[1]):
            h0 = str(df_kpi.iloc[0, c]).strip().upper()
            h1 = str(df_kpi.iloc[1, c]).strip().upper()
            h2 = str(df_kpi.iloc[2, c]).strip().upper()
            
            combined = f"{h0} {h1}"
            eq = None
            if "P03" in combined: eq = "Pala 03"
            elif "P04" in combined: eq = "Pala 04"
            elif "P05" in combined: eq = "Pala 05"
            elif "P06" in combined: eq = "Pala 06"
            elif "CF" in combined or "CARGADOR" in combined: eq = "Cargador Frontal"
            
            if eq and ("TON" in h2 or "TON" in h1):
                equip_cols.append((c, eq))
        
        # Define year blocks based on ACTUAL diagnostic
        year_blocks = [
            (3, 2026, 3, 14, False),    # Monthly
            (15, 2027, 15, 26, False),  # Monthly
            (27, 2028, 27, 30, True),   # Quarterly
            (31, 2029, 31, 31, True)    # Quarterly
        ]
        
        for header_row, year, start_row, end_row, is_quarterly in year_blocks:
            if not is_quarterly:
                # MONTHLY PARSING (2026, 2027)
                row_month_map = {}
                for i in range(start_row, min(end_row + 1, len(df_kpi))):
                    val_m = str(df_kpi.iloc[i, 1]).strip().upper()
                    if val_m in months_dict:
                        row_month_map[i] = months_dict[val_m]
                
                for row_idx, month_num in row_month_map.items():
                    q = f"Q{(month_num-1)//3 + 1}"
                    for col_idx, eq in equip_cols:
                        raw_val = str(df_kpi.iloc[row_idx, col_idx]).replace(',', '')
                        val_total = pd.to_numeric(raw_val, errors='coerce')
                        
                        if pd.notna(val_total) and val_total > 0:
                            if eq in ["Pala 03", "Cargador Frontal"]:
                                data_rows.append({
                                    'Year': year, 'Month': month_num, 'Quarter': q,
                                    'Equipo': eq, 'Fase': 'Remanejo',
                                    'Ton': val_total, 'Source': 'KPI-Palas'
                                })
                            else:
                                # Apply proportions for P04, P05, P06
                                phases_data = phase_lookup.get((month_num, eq), [])
                                if not phases_data:
                                    # Fallback if not in Pala-Fase map for that month
                                    data_rows.append({
                                        'Year': year, 'Month': month_num, 'Quarter': q,
                                        'Equipo': eq, 'Fase': 'Fase 3', # Fallback phase
                                        'Ton': val_total, 'Source': 'KPI-Palas'
                                    })
                                else:
                                    sum_pf_ton = sum([p['ton'] for p in phases_data])
                                    for p in phases_data:
                                        prop = p['ton'] / sum_pf_ton
                                        data_rows.append({
                                            'Year': year, 'Month': month_num, 'Quarter': q,
                                            'Equipo': eq, 'Fase': p['phase'],
                                            'Ton': val_total * prop, 'Source': 'KPI-Palas'
                                        })
            
            else:
                # QUARTERLY PARSING (2028, 2029)
                row_quarter_map = {}
                for i in range(start_row, min(end_row + 1, len(df_kpi))):
                    val_1 = str(df_kpi.iloc[i, 1]).strip()
                    if "TRIMESTRE" in val_1.upper() or "Trimestre" in val_1:
                        if "1" in val_1 or "PRIMER" in val_1.upper() or "1ER" in val_1.upper():
                            row_quarter_map[i] = 1
                        elif "2" in val_1 or "SEGUNDO" in val_1.upper() or "2DO" in val_1.upper():
                            row_quarter_map[i] = 2
                        elif "3" in val_1 or "TERCER" in val_1.upper() or "3ER" in val_1.upper():
                            row_quarter_map[i] = 3
                        elif "4" in val_1 or "CUARTO" in val_1.upper() or "4TO" in val_1.upper():
                            row_quarter_map[i] = 4
                
                for row_idx, quarter_num in row_quarter_map.items():
                    q = f"Q{quarter_num}"
                    # For quarterly, we use "Average Phase" or just match any month in that quarter
                    # Let's check month 1 of the quarter for proportions
                    m_guide = quarter_num * 3 - 1
                    
                    for col_idx, eq in equip_cols:
                        raw_val = str(df_kpi.iloc[row_idx, col_idx]).replace(',', '')
                        val_total = pd.to_numeric(raw_val, errors='coerce')
                        
                        if pd.notna(val_total) and val_total > 0:
                            if eq in ["Pala 03", "Cargador Frontal"]:
                                data_rows.append({
                                    'Year': year, 'Month': m_guide, 'Quarter': q,
                                    'Equipo': eq, 'Fase': 'Remanejo',
                                    'Ton': val_total, 'Source': 'KPI-Palas'
                                })
                            else:
                                # Try to find proportions for ANY month in the quarter
                                phases_data = []
                                for m_in_q in [quarter_num*3-2, quarter_num*3-1, quarter_num*3]:
                                    phases_data = phase_lookup.get((m_in_q, eq), [])
                                    if phases_data: break
                                    
                                if not phases_data:
                                    data_rows.append({
                                        'Year': year, 'Month': m_guide, 'Quarter': q,
                                        'Equipo': eq, 'Fase': 'Fase 3',
                                        'Ton': val_total, 'Source': 'KPI-Palas'
                                    })
                                else:
                                    sum_pf_ton = sum([p['ton'] for p in phases_data])
                                    for p in phases_data:
                                        prop = p['ton'] / sum_pf_ton
                                        data_rows.append({
                                            'Year': year, 'Month': m_guide, 'Quarter': q,
                                            'Equipo': eq, 'Fase': p['phase'],
                                            'Ton': val_total * prop, 'Source': 'KPI-Palas'
                                        })

        print(f"DEBUG: Total rows extracted: {len(data_rows)}")
        return pd.DataFrame(data_rows)

    except Exception as e:
        print(f"Error V13: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
