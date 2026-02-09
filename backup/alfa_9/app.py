import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import fleet_v3 # NEW MODULE
import google.generativeai as genai
import os
import random
import base64

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Anglo American - Plan Minero", page_icon="💎")

# --- 2. FUNCIONES UTILITARIAS ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Cargar imágenes visuales clave
img_bg_b64 = get_base64_of_bin_file("static/fondo_mina2.png")
img_logo_b64 = get_base64_of_bin_file("static/logo.png")

# CSS INYECTADO CON AJUSTES PARA STREAMLIT
def load_css():
    try:
        with open("mockup/style.css", "r") as f:
            css = f.read()
            
            # 1. FIX FONDO
            if img_bg_b64:
                bg_fix = f"""
                <style>
                .bg-image {{
                    background-image: url("data:image/png;base64,{img_bg_b64}") !important;
                    opacity: 0.35; 
                    mix-blend-mode: luminosity; 
                    z-index: 0;
                }}
                </style>
                """
                st.markdown(bg_fix, unsafe_allow_html=True)

            # 2. FIX STREAMLIT UI
            st_fixes = """
            <style>
            [data-testid="stAppViewContainer"] {background-color: #050910;}
            [data-testid="stHeader"] {background: transparent;}
            [data-testid="stToolbar"] {visibility: hidden;}
            .block-container {
                padding-top: 1rem; padding-bottom: 0rem;
                padding-left: 1rem; padding-right: 1rem;
                max-width: 100% !important;
            }
            footer {visibility: hidden;}
            </style>
            """
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
            st.markdown(st_fixes, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("⚠️ CSS File not found.")

load_css()

# --- 2.1 HELPER DE CARGA FLOTAS (Long Format) ---
def load_long_format_data(file_path, sheet_name, header_row=0):
    """
    Carga hojas en formato Vertical (Tiempo en filas).
    Retorna DF indexado por (Year, Periodo_Clean) para lookup rápido.
    """
    try:
        # Leer header específico
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, engine='openpyxl')
        
        # Estandarizar nombres de columnas de tiempo (Col 0 y 1 usualmente)
        # Asumimos que la Col 0 es Año y Col 1 es Mes/Periodo, pero nombres pueden variar
        # Renombramos forzadamente las primeras columnas para estandarizar
        cols = df.columns.tolist()
        if len(cols) > 2:
            df.columns = ['Year_Raw', 'Period_Raw'] + cols[2:]
        
        # Limpiar datos
        df['Year_Raw'] = pd.to_numeric(df['Year_Raw'], errors='coerce')
        df = df.dropna(subset=['Year_Raw']) # Eliminar filas sin año
        df['Year_Int'] = df['Year_Raw'].astype(int)
        
        # Limpiar Periodo (Strip espacios)
        df['Period_Clean'] = df['Period_Raw'].astype(str).str.strip()
        
        # Crear indice compuesto
        df.set_index(['Year_Int', 'Period_Clean'], inplace=True)
        return df
        
    except Exception as e:
        print(f"Error loading {sheet_name}: {e}")
        return pd.DataFrame()

# --- 3. DATOS (CARGA REAL MULTI-ANUAL) ---
# --- 2.2 ROBUST EXCEL LOADER (Universal "Dry & Perfect" Logic) ---
def load_and_clean_excel(file_path, sheet_name, ffill_cols=None, ffill_rows=None):
    """
    Universal loader that fixes 'Merged Cells' (NaN blocks) by Forward Filling.
    - ffill_cols: List of column indices to ffill (Vertical Merges).
    - ffill_rows: List of row indices to ffill (Horizontal Merges).
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 1. Fix Horizontal Merges (Headers)
        if ffill_rows:
            for r in ffill_rows:
                if r < df.shape[0]:
                    df.iloc[r, :] = df.iloc[r, :].ffill()
        
        # 2. Fix Vertical Merges (Index/Metadata)
        if ffill_cols:
            for c in ffill_cols:
                if c < df.shape[1]:
                    df.iloc[:, c] = df.iloc[:, c].ffill()
                    
        return df
    except Exception as e:
        st.error(f"Error loading {sheet_name}: {e}")
        return pd.DataFrame()

# --- AUX HELPER: STACKED LOADER (PERFOS/SERVICIOS) ---
def clean_stacked_headers(df):
    df.iloc[1, :] = df.iloc[1, :].ffill() # Year
    df.iloc[2, :] = df.iloc[2, :].ffill() # Month
    return df

def parse_row_label_stateful(full_label, sheet_type):
    full_label = str(full_label).upper().replace("NAN", " ").strip()
    detected_metric = None
    cleanup_str = ""
    
    if sheet_type == 'Perfos':
        if 'MTS' in full_label and 'PROD' in full_label:
            detected_metric = "Mts Producción"
            cleanup_str = "MTS PRODUCCION"
            full_label = full_label.replace("PRODUCCIÓN", "PRODUCCION")
        elif 'MTS' in full_label and ('PREC' in full_label or 'PRE-CORTE' in full_label):
            detected_metric = "Mts Precorte"
            cleanup_str = "MTS PRECORTE"
        elif ('HORA' in full_label or 'H.OP' in full_label) and 'OPER' in full_label:
            detected_metric = "Horas Operativas"
            cleanup_str = "HORAS OPERATIVAS"
    elif sheet_type == 'Servicios':
        if ('HORA' in full_label or 'H.OP' in full_label) and 'OPER' in full_label:
            detected_metric = "Horas Operativas"
            cleanup_str = "HORAS OPERATIVAS"

    cleaned_equip = full_label
    if detected_metric:
        for k in cleanup_str.split():
            cleaned_equip = cleaned_equip.replace(k, "")
    
    cleaned_equip = cleaned_equip.strip().strip(".-: ")
    if "TOTAL" in cleaned_equip: return detected_metric, None
    return detected_metric, cleaned_equip

def extract_stacked_sheet(df, sheet_type):
    records = []
    current_metric = None
    for r in range(3, df.shape[0]):
        c1 = df.iloc[r, 1]
        c2 = df.iloc[r, 2]
        full_label = (str(c1) + " " + str(c2)).upper().replace("NAN", " ").strip()
        if len(full_label) < 2: continue
        
        new_metric, equip_name = parse_row_label_stateful(full_label, sheet_type)
        if new_metric: current_metric = new_metric
        
        eff_equip = None
        if new_metric:
            if equip_name and len(equip_name) > 1: eff_equip = equip_name
        else:
            if current_metric and equip_name and len(equip_name) > 1: eff_equip = equip_name
        
        if eff_equip:
            if any(k in eff_equip for k in ['DIAS', 'UTIL', 'DISP', 'REND', 'USO', 'MAX']): eff_equip = None
        
        if current_metric and eff_equip:
            for c in range(3, df.shape[1]):
                try:
                    year_val = str(df.iloc[1, c]).replace(".0","").strip()
                    month_val = str(df.iloc[2, c]).strip()
                    if int(year_val) < 2026: continue
                except: continue
                
                val = pd.to_numeric(df.iloc[r, c], errors='coerce')
                if pd.notna(val) and val != 0:
                    records.append({
                        'Year': int(year_val),
                        'Month': month_val,
                        'Periodo': f"{month_val} {year_val}",
                        'Equipo': eff_equip,
                        'Metric': current_metric,
                        'Value': val,
                        'Source': f'KPI-{sheet_type}'
                    })
    return pd.DataFrame(records)

def load_new_kpis_integration(f_path):
    try:
        df_p = pd.read_excel(f_path, sheet_name='KPI-Perfos', header=None, engine='openpyxl')
        df_p = clean_stacked_headers(df_p)
        df_perfos = extract_stacked_sheet(df_p, 'Perfos')
    except: df_perfos = pd.DataFrame()

    try:
        df_s = pd.read_excel(f_path, sheet_name='KPI-Servicios', header=None, engine='openpyxl')
        df_s = clean_stacked_headers(df_s)
        df_serv = extract_stacked_sheet(df_s, 'Servicios')
    except: df_serv = pd.DataFrame()
    
    return df_perfos, df_serv

@st.cache_data
def load_data_v2():
    file_path = 'plan_budget_real.xlsx'
    
    try:
        # --- AUX HELPER: STACKED LOADER (PERFOS/SERVICIOS) ---
        def clean_stacked_headers(df):
            df.iloc[1, :] = df.iloc[1, :].ffill() # Year
            df.iloc[2, :] = df.iloc[2, :].ffill() # Month
            return df

        def parse_row_label_stateful(full_label, sheet_type):
            full_label = str(full_label).upper().replace("NAN", " ").strip()
            detected_metric = None
            cleanup_str = ""
            
            if sheet_type == 'Perfos':
                if 'MTS' in full_label and 'PROD' in full_label:
                    detected_metric = "Mts Producción"
                    cleanup_str = "MTS PRODUCCION"
                    full_label = full_label.replace("PRODUCCIÓN", "PRODUCCION")
                elif 'MTS' in full_label and ('PREC' in full_label or 'PRE-CORTE' in full_label):
                    detected_metric = "Mts Precorte"
                    cleanup_str = "MTS PRECORTE"
                elif ('HORA' in full_label or 'H.OP' in full_label) and 'OPER' in full_label:
                    detected_metric = "Horas Operativas"
                    cleanup_str = "HORAS OPERATIVAS"
            elif sheet_type == 'Servicios':
                if ('HORA' in full_label or 'H.OP' in full_label) and 'OPER' in full_label:
                    detected_metric = "Horas Operativas"
                    cleanup_str = "HORAS OPERATIVAS"

            cleaned_equip = full_label
            if detected_metric:
                for k in cleanup_str.split():
                    cleaned_equip = cleaned_equip.replace(k, "")
            
            cleaned_equip = cleaned_equip.strip().strip(".-: ")
            if "TOTAL" in cleaned_equip: return detected_metric, None
            return detected_metric, cleaned_equip

        def extract_stacked_sheet(df, sheet_type):
            records = []
            current_metric = None
            for r in range(3, df.shape[0]):
                c1 = df.iloc[r, 1]
                c2 = df.iloc[r, 2]
                full_label = (str(c1) + " " + str(c2)).upper().replace("NAN", " ").strip()
                if len(full_label) < 2: continue
                
                new_metric, equip_name = parse_row_label_stateful(full_label, sheet_type)
                if new_metric: current_metric = new_metric
                
                eff_equip = None
                if new_metric:
                    if equip_name and len(equip_name) > 1: eff_equip = equip_name
                else:
                    if current_metric and equip_name and len(equip_name) > 1: eff_equip = equip_name
                
                if eff_equip:
                    if any(k in eff_equip for k in ['DIAS', 'UTIL', 'DISP', 'REND', 'USO', 'MAX']): eff_equip = None
                
                if current_metric and eff_equip:
                    for c in range(3, df.shape[1]):
                        try:
                            year_val = str(df.iloc[1, c]).replace(".0","").strip()
                            month_val = str(df.iloc[2, c]).strip()
                            if int(year_val) < 2026: continue
                        except: continue
                        
                        val = pd.to_numeric(df.iloc[r, c], errors='coerce')
                        if pd.notna(val) and val != 0:
                            records.append({
                                'Year': int(year_val),
                                'Month': month_val,
                                'Periodo': f"{month_val} {year_val}",
                                'Equipo': eff_equip,
                                'Metric': current_metric,
                                'Value': val,
                                'Source': f'KPI-{sheet_type}'
                            })
            return pd.DataFrame(records)

        def load_new_kpis_integration(f_path):
            try:
                df_p = pd.read_excel(f_path, sheet_name='KPI-Perfos', header=None, engine='openpyxl')
                df_p = clean_stacked_headers(df_p)
                df_perfos = extract_stacked_sheet(df_p, 'Perfos')
            except: df_perfos = pd.DataFrame()

            try:
                df_s = pd.read_excel(f_path, sheet_name='KPI-Servicios', header=None, engine='openpyxl')
                df_s = clean_stacked_headers(df_s)
                df_serv = extract_stacked_sheet(df_s, 'Servicios')
            except: df_serv = pd.DataFrame()
            
            return df_perfos, df_serv

        
        # 1. Planta: Col 0 (Category) matches Merged Cells pattern? 
        # Actually usually Planta is simple, but let's be safe if user flagged "all sheets".
        df_planta = load_and_clean_excel(file_path, 'Planta', ffill_cols=[0]) 
        
        # 2. Data Tecnica: Cols 0, 1 (Origen, Fase) are heavily merged.
        df_dt = load_and_clean_excel(file_path, 'Data Tecnica', ffill_cols=[0, 1, 2])
        
        # 3. KPI-Palas: Row 0 (Year) is merged horizontally.
        df_palas_wide = load_and_clean_excel(file_path, 'KPI-Palas', ffill_rows=[0])
        
        # 4. Pala-Fase (Bull): Rows 0,1 (Time) and Cols 0,1 (Metadata) are merged.
        df_bull = load_and_clean_excel(file_path, 'Pala-Fase', ffill_rows=[0, 1], ffill_cols=[0, 1])
        
        # 4. Pala-Fase: Rows 0,1 (Time) and Cols 0,1 (Metadata) are merged.
        # Note: We use the helper, but `extract_pala_fase` does its own logic too.
        # We can pass the raw DF to it, but it expects `xls_file` path. 
        # Refactor `extract_pala_fase` to accept DF? 
        # For now, let's keep `extract_pala_fase` reading its own file but using robust logic inside.
        # Or better: Read here and pass DF. But `extract_pala_fase` logic is complex.
        # Let's LEAVE extract_pala_fase as is (it works) but ensure the filename is correct inside it?
        # Actually `extract_pala_fase` takes `file_path`.
        
        # Update: We need to pass the CORRECT file_path to extract_pala_fase.
        
        # ... (rest of logic)

        df_perfos = pd.read_excel(file_path, sheet_name='KPI-Perfos', header=None, engine='openpyxl')
        df_servicios = pd.read_excel(file_path, sheet_name='KPI-Servicios', header=None, engine='openpyxl')
        df_envios = pd.read_excel(file_path, sheet_name='Envios Desglosados por Fases', header=None, engine='openpyxl')
        df_camiones_wide = pd.read_excel(file_path, sheet_name='KPI-Camiones', header=None, engine='openpyxl')

        # A.2 LEER HOJAS DE FLOTA (Long - Vertical)
        # df_palas_long = load_long_format_data(file_path, 'KPI-Palas', header_row=1) # Replaced by df_palas_wide
        df_cam_long = load_long_format_data(file_path, 'KPI-Camiones', header_row=0)
        
        # A.3 LEER NUEVOS KPIS (Perfos/Servicios - Stacked)
        df_perfos, df_servicios_clean = load_new_kpis_integration(file_path)
        
        # --- HELPERS SEMÁNTICOS (BÚSQUEDA INTELIGENTE) ---
        def find_header_row(df, keywords):
            for idx, row in df.iterrows():
                if idx > 20: break # Limit search to first 20 rows
                s = " ".join([str(x) for x in row if pd.notna(x)])
                if any(k.lower() in s.lower() for k in keywords): return idx
            return 0 # Fallback to row 0 if no header found

        def get_col_map(df, sheet_name, validate_sub_header=False):
            # Find the row that contains month/quarter names
            h_row = find_header_row(df, ['Enero', 'Jan', 'Q1', 'Trimestre'])
            # Year is usually one row above the month/quarter header
            y_row = h_row - 1 if h_row > 0 else 0
            
            mapping = {}
            curr_year = None
            for c in range(df.shape[1]):
                # Detect Year
                y_val_raw = str(df.iloc[y_row, c]).replace('.0','')
                if y_val_raw.isdigit() and int(y_val_raw) > 2020 and int(y_val_raw) < 2050: 
                    curr_year = int(y_val_raw)
                
                # Detect Month/Quarter
                m_val_raw = str(df.iloc[h_row, c]).strip()
                if curr_year:
                    # Normalize quarter labels
                    m_val = m_val_raw
                    if '1er' in m_val_raw: m_val = 'Q1'
                    elif '2do' in m_val_raw: m_val = 'Q2'
                    elif '3er' in m_val_raw: m_val = 'Q3'
                    elif '4to' in m_val_raw: m_val = 'Q4'

                    valid_labels = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                                    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
                                    'Q1', 'Q2', 'Q3', 'Q4']
                    
                    if m_val in valid_labels:
                          # KEY CHANGE: "Keep First" Logic.
                          # The Excel has multiple columns for "Enero" (e.g. Mineral, Lastre, Total).
                          if (curr_year, m_val) not in mapping:
                              mapping[(curr_year, m_val)] = c
            return mapping

        def find_row(df, keywords, start_row=0, scan_cols=5):
            for r in range(start_row, df.shape[0]):
                row_txt = " ".join([str(df.iloc[r, c]) for c in range(min(scan_cols, df.shape[1])) if pd.notna(df.iloc[r, c])]).lower()
                if all(k.lower() in row_txt for k in keywords):
                    return r
            return None


        # --- MAPEO DINÁMICO ---
        map_planta = get_col_map(df_planta, "Planta")
        map_dt = get_col_map(df_dt, "DT")
        map_dt = get_col_map(df_dt, "DT")
        map_bull = get_col_map(df_bull, "Bull", validate_sub_header=True) # Validate 'K Ton' header
        map_envios = get_col_map(df_envios, "Envios") # Reuse map logic, likely aligns with Data Tecnica
        
        # --- LOCALIZACIÓN DE FILAS (SEMÁNTICA con Fallback) ---
        # Planta
        # Data Tecnica (Movimientos) - INDICES VERIFICADOS (16/12) para "plan_budget_real.xlsx"
        # F03=12, F04=6, F05=18, F05C=24, Remanejo=26
        # To avoid any "search failure" risk, we prioritize the verified indices.
        
        # PLANTA INDICES (Excel 1-indexed to Pandas 0-indexed)
        # Excel 15 (Tratamiento) -> 14
        # Excel 16 (Ley) -> 15
        # Excel 20 (Recuperacion Total) -> 19
        # Excel 21 (Cobre Fino) -> 20
        r_planta_total = 14
        r_trat = 14
        r_ley = 15
        r_recup = 19
        r_cobre = 20
        
        r_f03 = 12 
        r_f04 = 6  
        r_f05 = 18 
        r_f05c = 24 
        r_remanejo = 26 

        # Bull Row
        r_bull = find_row(df_bull, ["Bull"]) # Found at Row 3 in inspection
        if r_bull is None: r_bull = 3 # Fallback

        # StockDR Row (Stock a Stock) in Envios
        r_stock_dr = find_row(df_envios, ["StockDR"]) or find_row(df_envios, ["Doble Remanejo"]) or 36
        
        # --- DEFINICIÓN DE BLOQUES DE TIEMPO ---
        months_labels = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        quarters_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        
        time_blocks = [
            {'year': 2026, 'labels': months_labels, 'total_idx': 15},
            {'year': 2027, 'labels': months_labels, 'total_idx': 28},
            {'year': 2028, 'labels': quarters_labels, 'total_idx': 33},
            {'year': 2029, 'labels': quarters_labels, 'total_idx': None},
        ]
        
        all_dataframes = []
        
        def extract_series(df, row_idx, col_indices):
            data = []
            if row_idx is None: # If row not found, return zeros
                return [0.0] * len(col_indices)

            for c in col_indices:
                if c is not None and c < df.shape[1]:
                    val = pd.to_numeric(df.iloc[row_idx, c], errors='coerce')
                    data.append(val if pd.notna(val) else 0.0)
                else:
                    data.append(0.0) # Column not found for this period
            return data

        def extract_palas_sum(df_wide, year, labels):
            # Return tuple: (Planned_Ton_Sum, Capacity_Ton_Sum)
            # Capacity = Rend * (Days * 24) * Disp * Uso
            
            sums_ton = []
            sums_cap = []
            
            # Start Columns for Shovels (CF, P03, P04, P05, P06)
            # Step is 6 cols: (Ton, Hrs, Rend, Disp, UsoMax, UsoReal)
            # Based on inspection:
            # CF: Col 3 (Ton), 5 (Rend), 6 (Disp), 8 (UsoReal)
            # P03: Col 9...
            shovel_start_cols = [3, 9, 15, 21, 27] 
            
            header_row_palas = find_header_row(df_wide, ['CF', 'P03', 'P04', 'P05', 'P06', 'K Ton'])
            
            for lbl in labels:
                found_ton = 0
                found_cap = 0
                
                for r in range(header_row_palas + 1, df_wide.shape[0]):
                     y_val_raw = str(df_wide.iloc[r, 0]).replace('.0','')
                     m_val_raw = str(df_wide.iloc[r, 1]).strip()
                     
                     # --- LOGIC FOR COMPOSITE LABELS (e.g. '1er Trimestre_2028') ---
                     # Check if row label contains Year and Quarter info
                     row_year = y_val_raw
                     row_lbl = m_val_raw.lower()
                     
                     row_lbl = m_val_raw.lower()
                     
                     if '_' in m_val_raw and any(x in m_val_raw for x in ['2026','2027','2028','2029']):
                         parts = m_val_raw.split('_')
                         # Assume format "1er Trimestre_2028"
                         if len(parts) > 1 and parts[-1].strip().isdigit():
                             row_year = parts[-1].strip()
                             
                         # Map standard quarters
                         if '1er' in row_lbl: row_lbl = 'q1'
                         elif '2do' in row_lbl: row_lbl = 'q2'
                         elif '3er' in row_lbl: row_lbl = 'q3'
                         elif '4to' in row_lbl: row_lbl = 'q4'
                     
                     # -----------------------------------------------------------

                     if (row_year == str(year) or (row_year == 'nan' and r > header_row_palas+1)) and row_lbl == lbl.lower():
                         
                         days = pd.to_numeric(df_wide.iloc[r, 2], errors='coerce') or 30 # Col 2 is Days
                         hours_total = days * 24
                         
                         row_ton = 0
                         row_cap = 0
                         
                         
                         for c_ton in shovel_start_cols:
                             if c_ton < df_wide.shape[1]:
                                 # 1. Planned Ton
                                 val_ton = pd.to_numeric(df_wide.iloc[r, c_ton], errors='coerce')
                                 row_ton += (val_ton if pd.notna(val_ton) else 0.0)
                                 
                                 # 2. Physics Capacity
                                 try:
                                     # Convert to string first to handle '3,500' format
                                     def clean_float(val):
                                         if pd.isna(val): return 0.0
                                         s = str(val).replace(',', '.')
                                         try: return float(s)
                                         except: return 0.0

                                     rend = clean_float(df_wide.iloc[r, c_ton + 2])
                                     disp = clean_float(df_wide.iloc[r, c_ton + 3])
                                     # Switch to UsoMax (Potential) for Capacity Line calculation
                                     # UsoMax is at c_ton + 4 (Col 7, 13...)
                                     uso = clean_float(df_wide.iloc[r, c_ton + 4]) 
                                     
                                     # Automatic percentage detection
                                     if disp > 1.0: disp /= 100.0
                                     if uso > 1.0: uso /= 100.0
                                     
                                     # Capacity = Rend(t/h) * 24 * Days * Disp * UsoMax
                                     if rend > 0:
                                         cap = rend * hours_total * disp * uso
                                         row_cap += (cap / 1000.0)
                                 except:
                                     pass

                         found_ton = row_ton
                         found_cap = row_cap
                         break 
                sums_ton.append(found_ton)
                sums_cap.append(found_cap)
            return sums_ton, sums_cap

            # Ton is C3.
            # Dist is C6.
            # Vel is C7.
            # Assuming Cycle is C5 (guess based on proximity) or calc from Dist/Vel?
            # Cycle (hr) = Dist (km) / Vel (km/h) + Fixed Time (Load/Dump ~ 5min?)
            
            # For simplicity in this phase, we extract TOTAL FLEET rows if available, 
            # OR sum key attributes.
            # Assuming CAEX (Col 3) is the Total Fleet or Main Fleet.
            
            c_ton = 3
            c_dist = 6
            c_vel = 7
            # Assuming Cycle is C5 (guess based on proximity) or calc from Dist/Vel?
            # Cycle (hr) = Dist (km) / Vel (km/h) + Fixed Time (Load/Dump ~ 5min?)
            
            sums_ton = []
            avg_dist = []
            
            header_row_cam = find_header_row(df_wide, ['CAEX', 'Dist', 'Ton', 'K Ton'])
            
            for lbl in labels:
                found_ton = 0
                found_dist = 0
                
                for r in range(header_row_cam + 1, df_wide.shape[0]):
                     if r >= df_wide.shape[0]: break
                     
                     y_val_raw = str(df_wide.iloc[r, 0]).replace('.0','')
                     m_val_raw = str(df_wide.iloc[r, 1]).strip()
                     
                     # Parser Logic (Same as Palas)
                     row_year = y_val_raw
                     row_lbl = m_val_raw.lower()
                     if '_' in m_val_raw and any(x in m_val_raw for x in ['2026','2027','2028','2029']):
                         parts = m_val_raw.split('_')
                         if len(parts) > 1 and parts[-1].strip().isdigit(): row_year = parts[-1].strip()
                         if '1er' in row_lbl: row_lbl = 'q1'
                         elif '2do' in row_lbl: row_lbl = 'q2'
                         elif '3er' in row_lbl: row_lbl = 'q3'
                         elif '4to' in row_lbl: row_lbl = 'q4'
                     
                     if (row_year == str(year) or (row_year == 'nan' and r > header_row_cam+1)) and row_lbl == lbl.lower():
                         
                         def get_val(col):
                             if col < df_wide.shape[1]:
                                 s = str(df_wide.iloc[r, col]).replace(',', '.')
                                 try: return float(s)
                                 except: return 0.0
                             return 0.0

                         val_ton = get_val(c_ton)
                         val_dist = get_val(c_dist)
                         
                         found_ton = val_ton
                         found_dist = val_dist
                         break
                
                sums_ton.append(found_ton)
                avg_dist.append(found_dist) # This is Weighted Avg ideally, but taking value
            
            # Estimate Capacity: If we have Ton, we assume Plan Ton.
            # To get Capacity, we'd need Truck Count * Payload * Cycles.
            # Without Truck Count, we can't calculate Capacity curve purely.
            # So for now, we map Planned Truck Ton.
            return sums_ton, avg_dist

        # --- LÓGICA DE PRIMEROS PRINCIPIOS (FILTRADO DE FLUJOS) ---
        # Normalizar Data Tecnica (FFILL para celdas fusionadas)
        df_dt_filled = df_dt.copy()
        df_dt_filled.iloc[:, 0:4] = df_dt_filled.iloc[:, 0:4].ffill().fillna("")
        
        def extract_flow_sum(source_kws, dest_kws, mat_kws, col_indices):
            indices = []
            for r in range(df_dt_filled.shape[0]):
                row_vals = [str(x).lower() for x in df_dt_filled.iloc[r, 0:4]]
                if len(row_vals) < 4: continue
                
                check_src = (source_kws == ['*']) or any(k in row_vals[0] for k in source_kws)
                check_dest = (dest_kws == ['*']) or any(k in row_vals[3] for k in dest_kws)
                check_mat = (mat_kws == ['*']) or any(k in row_vals[2] for k in mat_kws)
                
                is_total = ('total' in row_vals[1] or 'total' in row_vals[2]) and ('remanejo' not in row_vals[0])
                if check_src and check_dest and check_mat and not is_total:
                    indices.append(r)
            
            total_s = [0.0] * len(col_indices)
            for r_idx in indices:
                s = extract_series(df_dt_filled, r_idx, col_indices)
                total_s = [t + v for t, v in zip(total_s, s)]
            return total_s

        for block in time_blocks:
            yr = block['year']
            labels = block['labels']
            
            # Resolve Columns from Maps for the current year and labels
            cols_p = [map_planta.get((yr, l)) for l in labels]
            cols_d = [map_dt.get((yr, l)) for l in labels]
            cols_b = [map_bull.get((yr, l)) for l in labels] # Bull has same layout as Planta?
            
            # Extract Planta data
            cobre = extract_series(df_planta, r_cobre, cols_p)
            trat = extract_series(df_planta, r_trat, cols_p)
            ley = extract_series(df_planta, r_ley, cols_p)
            recup = extract_series(df_planta, r_recup, cols_p)

            # --- NUEVA LÓGICA DE FLUJOS ---
            # Extract using new helper (Source, Dest, Mat)
            v_mina_planta = extract_flow_sum(['mina'], ['planta'], ['roca'], cols_d)
            v_mina_stock = extract_flow_sum(['mina'], ['stock'], ['roca'], cols_d)
            v_mina_botadero = extract_flow_sum(['mina'], ['botadero'], ['roca'], cols_d)
            v_relleno_botadero = extract_flow_sum(['mina'], ['botadero'], ['relleno'], cols_d)
            v_remanejo = extract_flow_sum(['remanejo'], ['*'], ['*'], cols_d) 
            
            # Extract StockDR
            cols_e = [map_envios.get((yr, l)) for l in labels]
            v_stock_stock = extract_series(df_envios, r_stock_dr, cols_e)
            
            # Recalculate Mov Total with granular components + StockDR
            # Mov Total = (Mina->Planta) + (Mina->Stock) + (Mina->Botadero) + (Stock->Planta/Remanejo) + (Stock->Stock)
            val_mov_calc = [a+b+c+d+e for a,b,c,d,e in zip(v_mina_planta, v_mina_stock, v_mina_botadero, v_relleno_botadero, v_remanejo)]
            val_mov_calc = [v + ss for v, ss in zip(val_mov_calc, v_stock_stock)]
            
            # DERIVE NET STOCK -> PLANTA (Remanejo Total - StockDR)
            # Ensure no negative values if data is dirty
            v_stock_planta = [max(0, r - ss) for r, ss in zip(v_remanejo, v_stock_stock)]
            
            # Extract Data Tecnica data
            mov_f03 = extract_series(df_dt, r_f03, cols_d)
            mov_f04 = extract_series(df_dt, r_f04, cols_d)
            mov_f05_base = extract_series(df_dt, r_f05, cols_d)
            mov_f05c = extract_series(df_dt, r_f05c, cols_d)
            remanejo = extract_series(df_dt, r_remanejo, cols_d)
            
            # Computed movements
            mov_f05 = [a+b for a,b in zip(mov_f05_base, mov_f05c)]
            mov_total = [f3 + f4 + f5 + rem for f3, f4, f5, rem in zip(mov_f03, mov_f04, mov_f05, remanejo)]
            
            # Extract Bull & Palas (Wide) - DISABLED 12/16/2025
            # bull_ton = extract_series(df_bull, r_bull, cols_b)
            # palas_ton_sum, palas_cap_sum = extract_palas_sum(df_palas_wide, yr, labels)
            # cam_ton_sum, cam_dist_sum = extract_camiones_sum(df_camiones_wide, yr, labels)
            
            # Reset to 0
            bull_ton = [0.0] * len(labels)
            palas_ton_sum = [0.0] * len(labels)
            palas_cap_sum = [0.0] * len(labels)
            cam_ton_sum = [0.0] * len(labels)
            cam_dist_sum = [0.0] * len(labels)
            
            # Normalization Cobre (if total_idx is provided)
            if block['total_idx'] is not None:
                try:
                    # Usar r_cobre detectado dinámicamente
                    target_total = pd.to_numeric(df_planta.iloc[r_cobre, block['total_idx']], errors='coerce')
                    current_sum = sum(cobre)
                    if current_sum > 0 and target_total > 0:
                        factor = target_total / current_sum
                        cobre = [x * factor for x in cobre]
                except:
                    pass
            
            # Create DF for the current block
            df_block = pd.DataFrame({
                'Periodo': [f"{l} {yr}" for l in labels],
                'Year': yr,
                'Month': labels if len(labels)==12 else [None]*len(labels), # Assign month if monthly
                'Quarter': labels if len(labels)==4 else [None]*len(labels), # Assign quarter if quarterly
                'Granularity': 'M' if len(labels)==12 else 'Q',
                'Cobre_Fino': cobre,
                'Mov_Total': mov_total,
                'Mov_F03': mov_f03,
                'Mov_F04': mov_f04, # Added Mov_F04
                'Mov_F05': mov_f05,
                'Remanejo': remanejo,
                'Trat_Planta': trat,
                'Ley_CuT': ley,
                'Recup': recup,
                'Costo_Mina': [random.uniform(2.8, 3.2) for _ in labels], # Simulated costs
                'Costo_Planta': [random.uniform(14.5, 15.5) for _ in labels], # Simulated costs
                'Palas_Ton': palas_ton_sum, # Sum of CF..P06
                'Palas_Capacidad': palas_cap_sum, # Calculated Physics
                'Bull_Ton': bull_ton,     # From Pala-Fase
                'Palas_Disp': [0.85]*len(labels), # Placeholder
                'Palas_Util': [0.80]*len(labels), # Placeholder
                'Camiones_Ton': cam_ton_sum,
                'Camiones_Dist': cam_dist_sum,
                'Camiones_Disp': [0.85]*len(labels), # Placeholder
                'Camiones_Util': [0.80]*len(labels), # Placeholder
                'Flow_Mina_Planta': v_mina_planta,
                'Flow_Mina_Stock': v_mina_stock,
                'Flow_Mina_Botadero': v_mina_botadero,
                'Flow_Relleno_Botadero': v_relleno_botadero,
                'Flow_Remanejo': v_remanejo,
                'Flow_Stock_Planta': v_stock_planta,
                'Flow_Stock_Stock': v_stock_stock
            })
            
            all_dataframes.append(df_block)
        
        df_final = pd.concat(all_dataframes, ignore_index=True)

        # Load Detailed Data
        # Fix: Pass df_bull directly AND time_blocks/map_bull for Strict Alignment
        # df_pala_fase = extract_pala_fase(df_bull, df_palas_wide, time_blocks, map_bull) # DEPRECATED
        # df_pala_fase = extract_pala_fase ... (DEPRECATED)

        # --- HYBRID FLEET LOAD V3 ---
        try:
             import fleet_v3
             df_fleet = fleet_v3.load_fleet_data_v3_hybrid(file_path)
             total_fleet_ton = df_fleet['Ton'].sum() if not df_fleet.empty else 0

             if abs(total_fleet_ton - 87768) > 100:
                 st.error(f"⚠️ Alerta de Integridad: Suma Flota {total_fleet_ton:,.0f} vs Target 87,768.")

        except Exception as e_fleet:
             st.error(f"Error loading Fleet V3: {e_fleet}")
             df_fleet = pd.DataFrame()
        
        return {
            'planta': df_final,
            'camiones_long': df_cam_long,
            'fleet': df_fleet,
            'perfos': df_perfos,
            'servicios': df_servicios_clean
        }


    except Exception as e:
        import traceback
        st.error(f"Error load_data: {e}")
        st.code(traceback.format_exc())
        return {
            'planta': pd.DataFrame(),
            'camiones_long': pd.DataFrame(),
            'fleet': pd.DataFrame(),
            'perfos': pd.DataFrame(),
            'servicios': pd.DataFrame()
        }

# --- GLOBAL DATA LOAD (LEGACY SUPPORT) ---
# Some functions might rely on 'df' being available globally.
# We unpack it safely.
try:
    _global_data = load_data_v2()
    df = _global_data.get('planta', pd.DataFrame()) if _global_data else pd.DataFrame()
    df_pala_fase_info = None # Deprecated
except:
    df = pd.DataFrame()
    df_pala_fase_info = None

# --- 4. CONFIG API & CHAT ENGINE (BEFORE UI) ---
api_key = os.environ.get("GEMINI_API_KEY")
try:
    if not api_key:
        api_key = st.secrets["GEMINI_API_KEY"]
except:
    pass

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

def generate_code_response(df_context, user_query):
    if not api_key:
        return "⚠️ API Key no configurada."

    buffer = []
    df_context.info(buf=buffer)
    schema_info = "".join(buffer)
    sample_data = df_context.head(3).to_string()
    
    data_dict = """
    - Cobre_Fino: Producción anual de Cobre Fino (kTon).
    - Mov_F03/F05: Material removido Fases 3 y 5 (kTon).
    - Remanejo: Stock re-procesado (kTon).
    - Trat_Planta: Tonelaje procesado (kTon).
    - Recup: Recuperación Total (%).
    - Costo_Mina/Planta: Costo unitario ($/t).
    """

    prompt = f"""
    Eres un experto analista datos Python.
    DICCIONARIO: {data_dict}
    SCHEMA: {schema_info}
    MUESTRA: {sample_data}
    USUARIO: {user_query}
    Genera código Python para `exec()`.
    1. Usa `df`.
    2. Guarda resultado en `result`.
    3. NO print.
    """
    
    try:
        response = model.generate_content(prompt)
        code = response.text.replace("```python", "").replace("```", "").strip()
        local_vars = {'df': df_context, 'pd': pd}
        exec(code, {}, local_vars)
        return local_vars.get('result', "Sin resultado.")
    except Exception as e:
        return f"Error: {str(e)}"

# --- 5. INTERFAZ HÍBRIDA (HEADER & STYLE) ---
# Inyectar logo dinámicamente
logo_style = "height: 110px; filter: brightness(0) invert(1) drop-shadow(0 0 2px rgba(255,255,255,0.5));"
logo_html = ""
if img_logo_b64:
    logo_html = f'<img src="data:image/png;base64,{img_logo_b64}" class="logo-img" style="{logo_style}">'

bg_style = f"""
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background-image: url('data:image/png;base64,{img_bg_b64}');
    background-size: cover; background-position: center; z-index: 0;
    opacity: 0.35; mix-blend-mode: luminosity; pointer-events: none;
"""

# --- TICKER REGLAS DE ORO (ESPAÑOL) ---
rules_text_es = (
    "🏆 REGLAS DE ORO: "
    "1. Evaluar riesgos y planificar. | "
    "2. Permiso de trabajo y EPP correcto. | "
    "3. Conducción segura. | "
    "4. Zonas restringidas. | "
    "5. Aislación de energías. | "
    "6. Izaje seguro. | "
    "7. Explosivos. | "
    "8. Altura. | "
    "9. Espacios confinados. | "
    "10. Sustancias peligrosas."
)

ticker_html_top = f"""
<div style="position: absolute; top: 20px; right: 20px; width: 45%; z-index:1000;">
    <div style="
        overflow: hidden; white-space: nowrap; box-sizing: border-box;
        background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981;
        border-radius: 20px; padding: 6px 15px; color: #10b981;
        font-family: 'Rajdhani'; font-size: 1.1em; display: flex; align-items: center; font-weight: bold;">
        <div style="display: inline-block; padding-left: 100%; animation: marquee 45s linear infinite;">
            {rules_text_es}
        </div>
    </div>
</div>
<style>@keyframes marquee {{ 0% {{ transform: translate(0, 0); }} 100% {{ transform: translate(-100%, 0); }} }}</style>
"""


html_structure = f"""
<div class="bg-image-box" style="{bg_style}"></div>
<div class="main-container" style="position: relative; z-index: 1;">
<header>
<div class="logo-section">
{logo_html}
<div class="title-group"><h1>Plan Minero Budget</h1><h2 style="font-size: 1.8rem;">2026-2029</h2></div>
</div>
{ticker_html_top}
</header>
<div style="text-align: center; font-size: 0.9em; color: #aaa; margin-top: 5px; margin-bottom: 20px;">
    💎 VALORES ANGLO AMERICAN: Seguridad • Cuidado y Respeto • Integridad • Responsabilidad • Colaboración • Innovación
</div>
<div class="content-area" style="margin-top:10px;"></div>
"""
st.markdown(html_structure, unsafe_allow_html=True)


# --- 6. FUNCIÓN RENDERIZADO PRINCIPAL ---
def render_dashboard(df, df_pala_fase_view, df_fleet=None, key_id="main"):
    df_view = df # Alias for backward compatibility with new chunks
    if df.empty:
        st.warning("No hay datos para esta selección.")
        return
    # Estilos CSS
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
    st.markdown("""
    <style>
        .kpi-card { background-color: #1e1e1e; border-left: 3px solid #00f2ff; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
        .kpi-title { color: #aaa; font-size: 0.7em; text-transform: uppercase; }
        .kpi-value { color: #fff; font-size: 1.4em; font-weight: bold; font-family: 'Arial', sans-serif; }
        .kpi-sub { color: #666; font-size: 0.6em; }
        .unit { font-size: 0.6em; color: #888; }
    </style>
    """, unsafe_allow_html=True)
    
    # --- CÁLCULO DE LOS 10 KPIs ---
    kpi_cu = f"{df['Cobre_Fino'].sum():,.0f}".replace(',', '.')
    kpi_planta = f"{df['Trat_Planta'].sum():,.0f}".replace(',', '.')
    kpi_mov_total = f"{df['Mov_Total'].sum():,.0f}".replace(',', '.')
    
    total_trat = df['Trat_Planta'].sum()
    if total_trat > 0:
        avg_ley = (df['Ley_CuT'] * df['Trat_Planta']).sum() / total_trat
        avg_recup = (df['Recup'] * df['Trat_Planta']).sum() / total_trat
    else:
        avg_ley = 0
        avg_recup = 0
    kpi_ley = f"{avg_ley:.2f}"
    kpi_recup = f"{avg_recup:.1f}"
    
    kpi_palas_cap = f"{df['Palas_Capacidad'].sum():,.0f}".replace(',', '.')

    kpi_f03 = f"{df['Mov_F03'].sum():,.0f}".replace(',', '.')
    kpi_f05 = f"{df['Mov_F05'].sum():,.0f}".replace(',', '.')
    kpi_remanejo = f"{df['Remanejo'].sum():,.0f}".replace(',', '.')

    avg_cost_mina = df['Costo_Mina'].mean() if 'Costo_Mina' in df else 0
    avg_cost_planta = df['Costo_Planta'].mean() if 'Costo_Planta' in df else 0
    kpi_costo_min = f"{avg_cost_mina:.2f}"
    kpi_costo_plant = f"{avg_cost_planta:.1f}"

    # --- RENDERIZADO ROBUSTO (st.columns) ---
    def card(title, value, unit, color="#00f2ff"):
        return f"""
        <div class="kpi-card" style="border-left-color: {color};">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value} <span class="unit">{unit}</span></div>
        </div>
        """

    # Fila 1
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(card("COBRE FINO", kpi_cu, "Ton"), unsafe_allow_html=True)
    with c2: st.markdown(card("TRAT. PLANTA", kpi_planta, "Ton"), unsafe_allow_html=True)
    with c3: st.markdown(card("MOV. TOTAL", kpi_mov_total, "kTon"), unsafe_allow_html=True)
    with c4: st.markdown(card("LEY MEDIA", kpi_ley, "%", "#e1b12c"), unsafe_allow_html=True)
    with c5: st.markdown(card("RECUPERACIÓN TOTAL", kpi_recup, "%"), unsafe_allow_html=True)

    # Fila 2
    c6, c7, c8, c9, c10 = st.columns(5)
    with c6: st.markdown(card("MOV FASE 3", kpi_f03, "kTon", "#ff9f43"), unsafe_allow_html=True)
    with c7: st.markdown(card("MOV FASE 5", kpi_f05, "kTon", "#ff9f43"), unsafe_allow_html=True)
    with c8: st.markdown(card("REMANEJO", kpi_remanejo, "kTon", "#ff9f43"), unsafe_allow_html=True)
    with c9: st.markdown(card("COSTO MINA", kpi_costo_min, "$/t", "#ff4757"), unsafe_allow_html=True)
    with c10: st.markdown(card("COSTO PLANTA", kpi_costo_plant, "$/t", "#ff4757"), unsafe_allow_html=True)

    # --- TABS ---
    tab1, = st.tabs(["🚀 Global"]) # Reset Tabs - Palas and Costos DELETED by User Request

    with tab1:
        # GRÁFICOS ESTRATÉGICOS
        st.markdown("### 📊 Gráficos Estratégicos")
        g1, g2 = st.columns(2)
    
        with g1:
            st.caption("🏭 Producción & Ley")
            base = alt.Chart(df).encode(x=alt.X('Periodo', sort=None))
            bars = base.mark_bar(color='#ff7f0e').encode(
                y=alt.Y('Cobre_Fino', axis=alt.Axis(title='Cobre Fino (Ton)', titleColor='#ff7f0e')),
                tooltip=['Periodo', 'Cobre_Fino', 'Ley_CuT']
            )
            line = base.mark_line(color='#1f77b4', strokeWidth=3).encode(
                y=alt.Y('Ley_CuT', axis=alt.Axis(title='Ley CuT (%)', titleColor='#1f77b4', orient='right')),
            )
            chart1 = alt.layer(bars, line).resolve_scale(y='independent').properties(height=350)
            st.altair_chart(chart1, use_container_width=True)

        with g2:
            st.caption("🏔️ Movimiento Mina (Fases)")
            melt_cols = ['Mov_F03', 'Mov_F04', 'Mov_F05', 'Remanejo']
            valid_melt = [c for c in melt_cols if c in df.columns]
            if valid_melt:
                df_melt = df.melt(id_vars=['Periodo'], value_vars=valid_melt, var_name='Fase', value_name='Kton')
                chart2 = alt.Chart(df_melt).mark_bar().encode(
                    x=alt.X('Periodo', sort=None),
                    y=alt.Y('Kton', stack='zero'),
                    color=alt.Color('Fase', scale=alt.Scale(scheme='category10')),
                    tooltip=['Periodo', 'Fase', 'Kton']
                ).properties(height=350)
                st.altair_chart(chart2, use_container_width=True)

        # --- SANKEY DIAGRAM (FLUJO DE MATERIALES) ---
        st.markdown("### 🌊 Flujo de Materiales (Gráfico Sankey)")
        
        # Aggregation
        s_mina_planta = df['Flow_Mina_Planta'].sum()
        s_mina_stock = df['Flow_Mina_Stock'].sum()
        s_mina_bot = df['Flow_Mina_Botadero'].sum()
        s_relleno_bot = df['Flow_Relleno_Botadero'].sum()
        s_stock_planta = df['Flow_Stock_Planta'].sum() 
        s_stock_stock = df['Flow_Stock_Stock'].sum() 

        # Node Definitions
        # 0: Mina (Roca), 1: Mina (Relleno), 2: Stock, 3: Planta, 4: Botadero
        labels = ["Mina (Roca) 🧨", "Mina (Relleno) 🚜", "Stock 🏔️", "Planta 🏭", "Botadero 🗑️"]
        colors = ["#ff9f43", "#576574", "#5f27cd", "#00d2d3", "#ee5253"]
        
        # Link Definitions
        # M(R)->P, M(R)->S, M(R)->B, M(Re)->B, S->P, S->S
        source = [0, 0, 0, 1, 2, 2] 
        target = [3, 2, 4, 4, 3, 2] 
        value = [s_mina_planta, s_mina_stock, s_mina_bot, s_relleno_bot, s_stock_planta, s_stock_stock]
        link_colors = [
            "rgba(255, 159, 67, 0.4)", "rgba(255, 159, 67, 0.4)", "rgba(255, 159, 67, 0.4)", 
            "rgba(87, 101, 116, 0.4)", "rgba(95, 39, 205, 0.4)", "rgba(95, 39, 205, 0.6)"
        ]

        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15, thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = labels,
              color = colors
            ),
            link = dict(
              source = source, target = target, value = value, color=link_colors
            ))])

        fig_sankey.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white", size=12)
        )
        if 'fig_sankey' in locals():
             st.plotly_chart(fig_sankey, use_container_width=True, key=f"{key_id}_sankey")

        # --- ADVANCED VISUALS (RADAR & WATERFALL) ---
        st.markdown("### 🎯 Estrategia y Balance de Masas")
        v1, v2 = st.columns(2)
        
        with v1:
            st.caption("🕸️ Radar Estratégico (Comparativa Anual)")
            # Prepare Data
            radar_metrics = ['Cobre_Fino', 'Mov_Total', 'Costo_Mina', 'Ley_CuT', 'Recup']
            radar_data = []
            
            # Aggregate by Year (Global View) or just show current Year vs Target?
            # Let's show Evolution of Years present in the view
            years_present = df['Year'].unique()
            
            # Max values for normalization
            max_vals = {m: df.groupby('Year')[m].sum().max() if m not in ['Ley_CuT', 'Recup', 'Costo_Mina'] else df[m].max() for m in radar_metrics}
            
            fig_radar = go.Figure()
            
            pass_colors = ['#00f2ff', '#ff9f43', '#ff4757', '#a29bfe']
            
            for i, y in enumerate(sorted(years_present)):
                df_y = df[df['Year']==y]
                if df_y.empty: continue
                
                # Sum for mass, Mean for grades/costs
                vals = []
                for m in radar_metrics:
                    if m in ['Ley_CuT', 'Recup', 'Costo_Mina']:
                        # Weighted Avg would be better but mean is ok for "Shape"
                        val = df_y[m].mean()
                    else:
                        val = df_y[m].sum()
                    
                    # Normalize 0-1
                    norm = val / max_vals[m] if max_vals[m] > 0 else 0
                    vals.append(norm)
                
                # Close the loop
                vals.append(vals[0])
                theta = radar_metrics + [radar_metrics[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=theta,
                    fill='toself',
                    name=str(y),
                    line_color=pass_colors[i % len(pass_colors)],
                    opacity=0.6
                ))
                
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="white"),
                height=350,
                margin=dict(l=40, r=40, t=20, b=20),
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_id}_radar")

        with v2:
            st.caption("📉 Cascada de Valor (Balance de Masas)")
            # Definitions for Waterfall
            w_mov = df['Mov_Total'].sum()
            w_lastre = df['Flow_Mina_Botadero'].sum() + df['Flow_Relleno_Botadero'].sum()
            w_stock_add = df['Flow_Mina_Stock'].sum()
            w_stock_stock = df['Flow_Stock_Stock'].sum()
            w_remanejo = df['Flow_Remanejo'].sum() 
            w_planta = df['Trat_Planta'].sum() 
            
            # Balance
            calc_planta = w_mov - w_lastre - w_stock_add - w_stock_stock
            
            fig_water = go.Figure(go.Waterfall(
                measure = ["absolute", "relative", "relative", "relative", "total"],
                x = ["Mov Total", "A Botadero", "A Stock", "Stock a Stock", "Alim. Planta"],
                y = [w_mov, -w_lastre, -w_stock_add, -w_stock_stock, None],
                text = [f"{int(w_mov)}", f"-{int(w_lastre)}", f"-{int(w_stock_add)}", f"-{int(w_stock_stock)}", f"{int(calc_planta)}"],
                decreasing = {"marker":{"color":"#ff4757"}},
                increasing = {"marker":{"color":"#2ed573"}},
                totals = {"marker":{"color":"#00f2ff"}}
            ))
            
            fig_water.update_layout(
                title = "Cascada de Tonelajes (kTon)",
                waterfallgap = 0.3,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="white"),
                height=350,
                 margin=dict(l=10, r=10, t=40, b=10),
                 yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_water, use_container_width=True, key=f"{key_id}_waterfall")
        
    # --- DELETED TABS (Palas, Costos) ---
    # User requested to "start from zero" on 12/16/2025 due to data complexity.
    # Code removed to ensure cleanliness.
    # (Legacy Fleet/Costos logic removed)
    st.markdown("---")
    st.markdown("---")
    col_chat_ai = st.container()
    with col_chat_ai:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Analizando vista activa..."}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Consulta...", key=f"chat_{key_id}"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

# --- RENDER FLEET (Modificado Petición Usuario) ---
def render_fleet_dashboard(df_fleet):
    st.markdown("### 🚚 Palas & Camiones")
    
    if df_fleet.empty:
        st.warning("Datos de Flota no disponibles.")
        return
        
    # --- 1. Data Prep ---
    df_fleet['Periodo'] = df_fleet.apply(
        lambda x: f"{int(x['Year'])}-{str(x['Month']).zfill(2)}" if pd.notna(x['Month']) and x['Month']>0 else f"{int(x['Year'])}-XX", 
        axis=1
    )
    df_fleet = df_fleet.sort_values(by=['Year', 'Month', 'Equipo'])
    
    # Filter out empty future periods
    # Only keep periods with actual data
    periods_with_data = df_fleet.groupby('Periodo')['Ton'].sum()
    valid_periods = periods_with_data[periods_with_data > 0].index.tolist()
    df_fleet = df_fleet[df_fleet['Periodo'].isin(valid_periods)]

    # --- 2. Clean Timeline Chart (Faceted by Equipment) ---
    import plotly.express as px
    
    # Phase Colors (Clean, Distinct)
    phase_colors = {
        'Fase 3': '#9D4EDD',      # Purple
        'Fase 4': '#FF6D00',      # Deep Orange  
        'Fase 5': '#06D6A0',      # Teal/Green
        'Remanejo': '#EF476F'     # Pink/Red
    }
    
    # Equipment Order (Most productive first)
    equip_order = ['Pala 06', 'Pala 05', 'Pala 04', 'Pala 03', 'Cargador Frontal', 'Bulldozer']
    
    fig = px.bar(
        df_fleet,
        x="Periodo",
        y="Ton",
        color="Fase",
        facet_col="Equipo",
        facet_col_wrap=3,  # 3 columns = 2 rows
        color_discrete_map=phase_colors,
        category_orders={"Equipo": equip_order},
        labels={"Ton": "kTon", "Periodo": ""},
        template="plotly_dark",
        height=550
    )
    
    # Clean up layout
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(title="", tickfont=dict(size=10))
    
    # Better facet labels (remove "Equipo=")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=13, color='white')))
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=12)
        ),
        margin=dict(l=30, r=30, t=50, b=100),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="fleet_timeline_chart")

    # --- 3. KPI Summary ---
    total_mov = df_fleet['Ton'].sum()
    st.markdown("---")
    
    cols = st.columns(5)
    cols[0].metric("Movimiento Total", f"{total_mov:,.0f} kTon")
    
    for i, eq in enumerate(['Pala 06', 'Pala 05', 'Pala 04', 'Pala 03']):
        val = df_fleet[df_fleet['Equipo'] == eq]['Ton'].sum()
        cols[i+1].metric(f"{eq}", f"{val:,.0f}")


# --- RENDER AUXILIARY (Perfos + Servicios) ---
def render_auxiliary_dashboard(df_perfos, df_serv):
    st.markdown("### 🚜 Perforadoras y Equipos de Servicios")
    
    t1, t2 = st.tabs(["Perforación", "Servicios"])
    
    with t1:
        if df_perfos.empty:
            st.warning("Datos de Perforación no disponibles.")
        else:
            # KPIs Globales
            mts_prod = df_perfos[df_perfos['Metric'] == 'Mts Producción']['Value'].sum()
            mts_prec = df_perfos[df_perfos['Metric'] == 'Mts Precorte']['Value'].sum()
            hrs_op = df_perfos[df_perfos['Metric'] == 'Horas Operativas']['Value'].sum()
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Mts Producción", f"{mts_prod:,.0f} m")
            k2.metric("Mts Precorte", f"{mts_prec:,.0f} m")
            k3.metric("Horas Operativas", f"{hrs_op:,.0f} h")
            
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**📉 Metros (Prod vs Prec)**")
                df_mts = df_perfos[df_perfos['Metric'].isin(['Mts Producción', 'Mts Precorte'])]
                ch1 = alt.Chart(df_mts).mark_bar().encode(
                    x='Periodo:N',
                    y='sum(Value):Q',
                    color='Metric:N',
                    tooltip=['Periodo', 'Metric', 'sum(Value)']
                ).properties(height=300)
                st.altair_chart(ch1, use_container_width=True)
                
            with g2:
                st.markdown("**⏱️ Horas por Equipo**")
                df_hrs = df_perfos[df_perfos['Metric'] == 'Horas Operativas']
                ch2 = alt.Chart(df_hrs).mark_bar().encode(
                    y=alt.Y('Equipo:N', sort='-x'),
                    x='sum(Value):Q',
                    color='Equipo:N',
                    tooltip=['Equipo', 'sum(Value)']
                ).properties(height=300)
                st.altair_chart(ch2, use_container_width=True)

    with t2:
        if df_serv.empty:
            st.warning("Datos de Servicios no disponibles.")
        else:
            hrs_total = df_serv[df_serv['Metric'] == 'Horas Operativas']['Value'].sum()
            st.metric("Total Horas Operativas (Flota Apoyo)", f"{hrs_total:,.0f} h")
            
            # Top Equipos
            top_equipos = df_serv.groupby('Equipo')['Value'].sum().sort_values(ascending=False).head(10).reset_index()
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**🏆 Top 10 Equipos Mayor Uso**")
                ch = alt.Chart(top_equipos).mark_bar().encode(
                    x='Value:Q',
                    y=alt.Y('Equipo:N', sort='-x'),
                    color=alt.value('#2ecc71'),
                    tooltip=['Equipo', 'Value']
                ).properties(height=400)
                st.altair_chart(ch, use_container_width=True)
            
            with c2:
                st.markdown("**📋 Detalle**")
                st.dataframe(top_equipos.style.format({"Value": "{:,.0f}"}), use_container_width=True)

# --- MAIN APP FLOW ---
data_loaded = load_data_v2()

if data_loaded:
    df = data_loaded.get('planta', pd.DataFrame())
    df_fleet = data_loaded.get('fleet', pd.DataFrame())
    df_perfos = data_loaded.get('perfos', pd.DataFrame())
    df_serv = data_loaded.get('servicios', pd.DataFrame())
    
    # Deprecated legacy DF
    df_pala_fase_info = None 

    if df.empty:
        st.error("Data Planta Empty")

    # --- 7. MAIN TABS LOGIC ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- MATRIX NAVIGATION (2 ROWS) ---
    # Row 1: View Context (Panel, Fleet, Aux)
    # Row 2: Time Context (Total, 2026, 2027...)
    
    # 1. Initialize State
    if 'nav_view' not in st.session_state: st.session_state['nav_view'] = "📊 Panel de Control"
    if 'nav_period' not in st.session_state: st.session_state['nav_period'] = "📈 2026-2029"

    # 2. Row 1: View Selection
    c1, c2, c3 = st.columns(3)
    with c1: 
        if st.button("📊 Panel de Control", use_container_width=True): 
            st.session_state['nav_view'] = "📊 Panel de Control"
            st.rerun()
    with c2: 
        if st.button("🚚 Palas & Camiones", use_container_width=True): 
            st.session_state['nav_view'] = "🚚 Palas & Camiones"
            st.rerun()
    with c3: 
        if st.button("🚜 Perforación & Servicios", use_container_width=True): 
            st.session_state['nav_view'] = "🚜 Perforación & Servicios"
            st.rerun()
            
    # 3. Row 2: Period Selection
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1: 
        if st.button("📈 2026-2029", use_container_width=True): 
            st.session_state['nav_period'] = "📈 2026-2029"
            st.rerun()
    with t2: 
        if st.button("📅 2026", use_container_width=True): 
            st.session_state['nav_period'] = "2026"
            st.rerun()
    with t3: 
        if st.button("📅 2027", use_container_width=True): 
            st.session_state['nav_period'] = "2027"
            st.rerun()
    with t4: 
        if st.button("📅 2028", use_container_width=True): 
            st.session_state['nav_period'] = "2028"
            st.rerun()
    with t5: 
        if st.button("📅 2029", use_container_width=True): 
            st.session_state['nav_period'] = "2029"
            st.rerun()
            
    # --- RENDER LOGIC (MATRIX) ---
    st.markdown("---")
    
    current_view = st.session_state['nav_view']
    current_period = st.session_state['nav_period']
    
    # helper to filter data
    def filter_by_period(d, p):
        if p == "📈 2026-2029": return d
        if d.empty: return d
        try:
            yr = int(p)
            return d[d['Year'] == yr]
        except:
            return d

    # Apply Base Filter
    df_f = filter_by_period(df, current_period)
    df_fleet_f = filter_by_period(df_fleet, current_period)
    df_perfos_f = filter_by_period(df_perfos, current_period)
    df_serv_f = filter_by_period(df_serv, current_period)
    
    # ROUTING
    if current_view == "📊 Panel de Control":
        # Panel has special sub-logic for Years (Q/M)
        if current_period == "📈 2026-2029":
             st.caption("Visión Completa del Periodo Presupuestal")
             render_dashboard(df_f, df_pala_fase_info, df_fleet_f, "full_timeline")
        else:
             # Year View with Sub-Navigation
             year_int = int(current_period)
             c1, _ = st.columns([2, 3])
             with c1: 
                 v = st.radio(f"Vista {year_int}:", ["Anual", "Trimestral", "Mensual"], horizontal=True, key=f"v_{year_int}")
             
             if v == "Anual": 
                 render_dashboard(df_f, df_pala_fase_info, df_fleet_f, f"{year_int}_anual")
             elif v == "Trimestral": 
                 q = st.select_slider("Trimestre:", ["Q1","Q2","Q3","Q4"], key=f"q_{year_int}")
                 render_dashboard(df_f[df_f['Quarter']==q], df_pala_fase_info, df_fleet_f, f"{year_int}_{q}")
             else:
                 available_months = df_f[df_f['Month'].notna()]['Month'].unique().tolist()
                 if available_months:
                     m = st.select_slider("Mes:", available_months, key=f"m_{year_int}")
                     render_dashboard(df_f[df_f['Month']==m], df_pala_fase_info, df_fleet_f, f"{year_int}_{m}")
                 else:
                     st.warning("No hay datos mensuales disponibles.")
                     render_dashboard(df_f, df_pala_fase_info, df_fleet_f, f"{year_int}_anual")

    elif current_view == "🚚 Palas & Camiones":
        # Pass filtered Fleet Data
        # Ensure title reflects period
        st.caption(f"Contexto Temporal: {current_period}")
        render_fleet_dashboard(df_fleet_f)
        
    elif current_view == "🚜 Perforación & Servicios":
        st.caption(f"Contexto Temporal: {current_period}")
        render_auxiliary_dashboard(df_perfos_f, df_serv_f)

else:
    st.error("No se pudieron cargar los datos. Verifica el archivo Excel.")

