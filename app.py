import streamlit as st
st.write("🔄 **CARGANDO SISTEMA DE PLANIFICACIÓN... (Si lees esto, el servidor funciona)**")

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import altair as alt
    import fleet_v3  # Auxiliary functions (perfos, servicios)
    import fleet_loader  # Fleet data loader (final clean version)
    import adapter_v4  # Data adapter
    import importlib  # For dynamic reloading
    from chat_code_gen import CodeGenerationChatAgent
    import google.generativeai as genai
    import os
    import random
    import base64
    import numpy as np
except Exception as e:
    st.error(f"❌ CRASH AL INICIAR: {e}")
    st.stop()

# FORCE RELOAD to clear module cache
importlib.reload(fleet_v3)
importlib.reload(fleet_loader)

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Anglo American - Plan Minero", page_icon="💎")
# Updated: Control Absoluto Applied - Cache Invalidated

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
            
            /* SAFE RESTORATION - HIDE ONLY HEADER */
            [data-testid="stHeader"] {background: transparent;}
            
            .block-container {
                padding-top: 1rem; 
                padding-bottom: 0rem;
                padding-left: 1rem; 
                padding-right: 1rem;
                max-width: 100% !important;
            }
            footer {visibility: hidden;}
            </style>
            """
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
            st.markdown(st_fixes, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("⚠️ CSS File not found.")

# load_css() # DEACTIVATED FOR DEBUGGING

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
    # try:
    #     df_p = pd.read_excel(f_path, sheet_name='KPI-Perfos', header=None, engine='openpyxl')
    #     df_p = clean_stacked_headers(df_p)
    #     df_perfos = extract_stacked_sheet(df_p, 'Perfos')
    # except: df_perfos = pd.DataFrame()

    # try:
    #     df_s = pd.read_excel(f_path, sheet_name='KPI-Servicios', header=None, engine='openpyxl')
    #     df_s = clean_stacked_headers(df_s)
    #     df_serv = extract_stacked_sheet(df_s, 'Servicios')
    # except: df_serv = pd.DataFrame()
    
    # NEW LOADER
    df_aux_all = fleet_v3.load_kpi_perfos_servicios(f_path)
    if not df_aux_all.empty:
        df_perfos = df_aux_all[df_aux_all['Category'] == 'Perfos']
        df_serv = df_aux_all[df_aux_all['Category'] == 'Servicios']
    else:
        # Prevent KeyError 'Metric' if empty
        cols = ['Year', 'Month', 'Quarter', 'Category', 'Item', 'Metric', 'Value', 'Sheet']
        df_perfos = pd.DataFrame(columns=cols)
        df_serv = pd.DataFrame(columns=cols)
    return df_perfos, df_serv

# --- 2.3 PERIOD HELPER ---
def generate_period_column(df):
    if df.empty: return df
    month_names = {
        1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
        7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'
    }
    
    def get_period_label(row):
        try:
            y = int(row['Year'])
            if y <= 2027:
                m = row.get('Month', None)
                if pd.isna(m) or m == '': return f"{y}"
                # Handle both string month names and integer IDs
                if isinstance(m, (int, float)):
                    m_label = month_names.get(int(m), str(m))
                else:
                    m_label = str(m)
                return f"{m_label} {y}"
            else:
                q = row.get('Quarter', None)
                if pd.isna(q) or q == '':
                    m = row.get('Month', None)
                    if pd.notna(m) and isinstance(m, (int, float)):
                        q = f"Q{(int(m)-1)//3 + 1}"
                    else:
                        return f"{y}"
                return f"{q} {y}"
        except:
            return str(row.get('Periodo', ''))

    # Sort by Year and Month before creating the label to ensure stability
    if 'Year' in df.columns and 'Month' in df.columns:
        df = df.sort_values(by=['Year', 'Month'])
    
    df['Periodo'] = df.apply(get_period_label, axis=1)
    return df

# --- ADAPTER: CLEAN DATA LOADER ---
@st.cache_data
def load_data_v3():
    file_path = 'plan_budget_real.xlsx'
    
    try:
        # 1. Planta
        df_planta = load_and_clean_excel(file_path, 'Planta', ffill_cols=[0]) 
        
        # 2. Data Tecnica
        df_dt = load_and_clean_excel(file_path, 'Data Tecnica', ffill_cols=[0, 1, 2])
        
        # 2.1 Envios Desglosados (for Doble Remanejo)
        try:
            df_envios = load_and_clean_excel(file_path, 'Envios Desglosados por Fases', ffill_cols=[0, 1, 2])
        except:
            df_envios = pd.DataFrame()
        
        # 3. Auxiliaries (Perfos/Servicios)
        df_perfos, df_servicios_clean = load_new_kpis_integration(file_path)
        
        # 4. Camiones
        df_cam_long = load_long_format_data(file_path, 'KPI-Camiones', header_row=0)

        # 5. Fleet (Pala-Fase / KPI-Palas)
        df_fleet = fleet_v3.load_fleet_data_v3_hybrid(file_path)

        # --- HELPERS SEMÁNTICOS (BÚSQUEDA INTELIGENTE) ---
        def find_header_row(df, keywords):
            for idx, row in df.iterrows():
                if idx > 20: break 
                s = " ".join([str(x) for x in row if pd.notna(x)])
                if any(k.lower() in s.lower() for k in keywords): return idx
            return 0

        def get_col_map(df, sheet_name):
            h_row = find_header_row(df, ['Enero', 'Jan', 'Q1', 'Trimestre'])
            y_row = h_row - 1 if h_row > 0 else 0
            mapping = {}
            curr_year = None
            for c in range(df.shape[1]):
                y_val_raw = str(df.iloc[y_row, c]).replace('.0','')
                if y_val_raw.isdigit() and 2020 < int(y_val_raw) < 2050: 
                    curr_year = int(y_val_raw)
                
                m_val_raw = str(df.iloc[h_row, c]).strip()
                if curr_year:
                    m_val = m_val_raw
                    if '1er' in m_val_raw: m_val = 'Q1'
                    elif '2do' in m_val_raw: m_val = 'Q2'
                    elif '3er' in m_val_raw: m_val = 'Q3'
                    elif '4to' in m_val_raw: m_val = 'Q4'

                    valid_labels = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                                    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
                                    'Q1', 'Q2', 'Q3', 'Q4']
                    if m_val in valid_labels and (curr_year, m_val) not in mapping:
                        mapping[(curr_year, m_val)] = c
            return mapping

        def extract_series(df, row_idx, col_indices, div_by=1.0):
            data = []
            if row_idx is None: return [0.0] * len(col_indices)
            for c in col_indices:
                if c is not None and c < df.shape[1]:
                    val = pd.to_numeric(df.iloc[row_idx, c], errors='coerce')
                    data.append((val if pd.notna(val) else 0.0) / div_by)
                else:
                    data.append(0.0)
            return data

        # Map and Extract Blocks
        map_planta = get_col_map(df_planta, "Planta")
        map_dt = get_col_map(df_dt, "DT")
        map_envios = get_col_map(df_envios, "Envios") if not df_envios.empty else {}
        
        r_trat, r_ley, r_recup, r_cobre = 14, 15, 19, 20
        
        # Robust Keyword Search for Data Tecnica
        df_dt_filled = df_dt.copy()
        df_dt_filled.iloc[:, 0:4] = df_dt_filled.iloc[:, 0:4].ffill().fillna("")

        if not df_envios.empty:
            df_envios_filled = df_envios.copy()
            df_envios_filled.iloc[:, 0:4] = df_envios_filled.iloc[:, 0:4].ffill().fillna("")
        else:
            df_envios_filled = pd.DataFrame()

        def extract_flow_sum(df_target, src_k, phase_k, mat_k, dest_k, col_indices):
            if df_target.empty: return [0.0] * len(col_indices)
            res = [0.0] * len(col_indices)
            for r in range(df_target.shape[0]):
                row_vals = [str(x).lower().strip() for x in df_target.iloc[r, 0:4]]
                if len(row_vals) < 4: continue
                
                check = True
                if src_k != '*': check &= any(k.lower() == row_vals[0] for k in src_k)
                # In DT, Col 1 is Phase. In Envios, Col 1 is also Phase.
                if phase_k != '*': check &= any(k.lower() == row_vals[1] for k in phase_k)
                # In DT/Envios: check Mat and Dest specifically.
                if mat_k != '*': check &= any(k.lower() in row_vals[2] for k in mat_k)
                if dest_k != '*': check &= any(k.lower() in row_vals[3] for k in dest_k)
                
                if check:
                   s = extract_series(df_target, r, col_indices)
                   res = [a + b for a, b in zip(res, s)]
            return res

        months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        
        blocks = [
            {'year': 2026, 'labels': months},
            {'year': 2027, 'labels': months},
            {'year': 2028, 'labels': quarters},
            {'year': 2029, 'labels': quarters},
        ]
        
        all_dfs = []
        for b in blocks:
            y, lbls = b['year'], b['labels']
            cols_p = [map_planta.get((y, l)) for l in lbls]
            cols_d = [map_dt.get((y, l)) for l in lbls]
            cols_e = [map_envios.get((y, l)) for l in lbls] if map_envios else []
            
            # Aggregate Phase Data (Mina Sources)
            v_f03 = extract_flow_sum(df_dt_filled, ['mina'], ['f03'], '*', '*', cols_d)
            v_f04 = extract_flow_sum(df_dt_filled, ['mina'], ['f04'], '*', '*', cols_d)
            v_f05 = extract_flow_sum(df_dt_filled, ['mina'], ['f05'], '*', '*', cols_d)
            v_f05c = extract_flow_sum(df_dt_filled, ['mina'], ['f05c'], '*', '*', cols_d)
            v_rem = extract_flow_sum(df_dt_filled, ['remanejo'], '*', '*', '*', cols_d)
            
            # Flow Data for Sankey (Extracted from ENVIOS as they are WET TONNES)
            # Mina -> Planta
            v_mina_planta = extract_flow_sum(df_envios_filled, ['mina'], '*', ['planta'], ['humedas'], cols_e)
            # Mina -> Stock
            v_mina_stock = extract_flow_sum(df_envios_filled, ['mina'], '*', ['stock'], ['humedas'], cols_e)
            # Mina -> Botadero
            v_mina_bot = extract_flow_sum(df_envios_filled, ['mina'], '*', ['botadero', 'botaderos'], ['humedas'], cols_e)
            # Remanejo -> Planta
            v_rem_planta = extract_flow_sum(df_envios_filled, ['remanejo'], '*', ['planta'], ['humedas'], cols_e)
            # Stock a Stock (Doble Remanejo AK 37)
            v_stock_stock = extract_flow_sum(df_envios_filled, ['remanejo'], '*', ['stockdr', 'doble remanejo'], ['humedas'], cols_e)
            
            # NOTE: v_mina_bot + v_relleno_bot from DT logic. Replaced by Envios.
            # v_relleno_bot = extract_flow_sum(df_dt_filled, ['mina'], '*', ['relleno'], ['botadero', 'botaderos'], cols_d)

            df_b = pd.DataFrame({
                'Periodo': [f"{l} {y}" for l in lbls],
                'Year': y,
                'Month': lbls if len(lbls)==12 else [None]*len(lbls),
                'Quarter': lbls if len(lbls)==4 else [None]*len(lbls),
                'Granularity': 'M' if len(lbls)==12 else 'Q',
                'Cobre_Fino': extract_series(df_planta, r_cobre, cols_p, div_by=1.0), # KEEP AS TONS (T)
                'Trat_Planta': extract_series(df_planta, r_trat, cols_p, div_by=1000.0), # Normalize Ton to kTon
                'Ley_CuT': extract_series(df_planta, r_ley, cols_p),
                'Recup': extract_series(df_planta, r_recup, cols_p),
                'Mov_Total': [a+b+c+d+e for a,b,c,d,e in zip(v_f03, v_f04, v_f05, v_f05c, v_rem)],
                'Mov_F03': v_f03,
                'Mov_F04': v_f04,
                'Mov_F05': [a+b for a,b in zip(v_f05, v_f05c)],
                'Remanejo': v_rem,
                'Palas_Capacidad': [0.0]*len(lbls), # Placeholder for future logic
                'Costo_Mina': [0.0]*len(lbls),
                'Costo_Planta': [0.0]*len(lbls),
                # Sankey Flows
                'Flow_Mina_Planta': v_mina_planta,
                'Flow_Mina_Stock': v_mina_stock,
                'Flow_Mina_Botadero': v_mina_bot,
                'Flow_Relleno_Botadero': [0.0]*len(lbls), # Integrated in mina_bot if any
                'Flow_Remanejo': v_rem, # Total Remanejo Header for KPI
                'Flow_Stock_Planta': v_rem_planta, # Specific flow to Planta
                'Flow_Stock_Stock': v_stock_stock  # Doble Remanejo AK 37
            })
            all_dfs.append(df_b)
            
        df_final = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        # --- AUX HELPER: LOAD PERFOS/SERVICIOS ---
        # Ensure we use the new V3 loader for Perfos/Servicios
        try:
             df_aux = fleet_v3.load_kpi_perfos_servicios(file_path)
             if not df_aux.empty:
                 df_perfos = df_aux[df_aux['Category'] == 'Perfos'].copy()
                 df_serv = df_aux[df_aux['Category'] == 'Servicios'].copy()
             else:
                 df_perfos = pd.DataFrame()
                 df_serv = pd.DataFrame()
        except Exception as e_aux:
             print(f"Error loading Aux V3: {e_aux}")
             df_perfos = pd.DataFrame()
             df_serv = pd.DataFrame()

        # --- POST-PROCESS: ADD PERIODO COLUMN ---
        df_fleet = generate_period_column(df_fleet)
        df_perfos = generate_period_column(df_perfos)
        df_serv = generate_period_column(df_serv)
        
        return {
            'planta': df_final,
            'camiones_long': df_cam_long,
            'fleet': df_fleet,
            'perfos': df_perfos,
            'servicios': df_serv
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
    _global_data = load_data_v3()
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

<!-- MOBILE PWA TAGS -->
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
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
    def safe_sum(col_name):
        if col_name in df.columns:
            return df[col_name].sum()
        return 0.0

    kpi_cu = f"{safe_sum('Cobre_Fino'):,.0f}".replace(',', '.')
    
    # Tratamiento comes in TON from Row 14, user wants kTon
    val_planta = safe_sum('Trat_Planta') / 1000.0
    kpi_planta = f"{val_planta:,.0f}".replace(',', '.')
    
    # Movimiento and Remanejo are in kTon from Data Tecnica
    kpi_mov_total = f"{safe_sum('Mov_Total'):,.0f}".replace(',', '.')
    kpi_remanejo = f"{safe_sum('Remanejo'):,.0f}".replace(',', '.')
    
    total_trat = safe_sum('Trat_Planta')
    if total_trat > 0:
        # Ley (Row 15) and Recup (Row 19) are already percentages/ratios. 
        # Weighted average using Ton (Row 14)
        avg_ley = (df['Ley_CuT'] * df['Trat_Planta']).sum() / total_trat if 'Ley_CuT' in df.columns else 0
        avg_recup = (df['Recup'] * df['Trat_Planta']).sum() / total_trat if 'Recup' in df.columns else 0
    else:
        avg_ley = 0
        avg_recup = 0
    kpi_ley = f"{avg_ley:.3f}" # 3 decimals for precision 0.679
    kpi_recup = f"{avg_recup:.1f}"
    
    kpi_palas_cap = f"{safe_sum('Planta_Capacidad')/1000.0:,.0f}".replace(',', '.') # Convert to kTon

    kpi_f03 = f"{safe_sum('Mov_F03'):,.0f}".replace(',', '.')
    kpi_f05 = f"{safe_sum('Mov_F05'):,.0f}".replace(',', '.')

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
    with c2: st.markdown(card("TRAT. PLANTA", kpi_planta, "kTon"), unsafe_allow_html=True)
    with c3: st.markdown(card("MOV. MINA", kpi_mov_total, "kTon"), unsafe_allow_html=True)
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
            base = alt.Chart(df).encode(
                x=alt.X('Periodo', 
                    sort=None,
                    axis=alt.Axis(
                        labelAngle=-45,
                        labelFontSize=9,
                        titleFontSize=11
                    )
                )
            )
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
                    x=alt.X('Periodo', 
                        sort=None,
                        axis=alt.Axis(
                            labelAngle=-45,
                            labelFontSize=9,
                            titleFontSize=11
                        )
                    ),
                    y=alt.Y('Kton', stack='zero'),
                    color=alt.Color('Fase', scale=alt.Scale(scheme='category10')),
                    tooltip=['Periodo', 'Fase', 'Kton']
                ).properties(height=350)
                st.altair_chart(chart2, use_container_width=True)

        # --- SANKEY DIAGRAM (FLUJO DE MATERIALES) ---
        st.markdown("### 🌊 Flujo de Materiales (Gráfico Sankey)")
        
        # CORRECTED: Use Envios Desglosados data (Waterfall columns) for accurate flows
        has_waterfall_data = all(col in df.columns for col in [
            'Waterfall_Mina_to_Planta',
            'Waterfall_Mina_to_StockMP', 
            'Waterfall_Mina_to_Botaderos',
            'Waterfall_Stock_to_Planta',
            'Waterfall_Doble_Remanejo'
        ])
        
        if has_waterfall_data:
            # ✅ CORRECT VALUES from Envios Desglosados
            s_mina_planta = df['Waterfall_Mina_to_Planta'].sum()
            s_mina_stock = df['Waterfall_Mina_to_StockMP'].sum()
            s_mina_bot = df['Waterfall_Mina_to_Botaderos'].sum()
            s_relleno_bot = df.get('Flow_Relleno_Botadero', pd.Series([0])).sum()  # Keep from DT
            s_stock_planta = df['Waterfall_Stock_to_Planta'].sum()  # ✅ CORRECTED: 13,434 kTon
            s_stock_stock = df['Waterfall_Doble_Remanejo'].sum()     # ✅ CORRECTED: 2,683 kTon
        else:
            # Fallback to old Data Tecnica columns
            s_mina_planta = df.get('Flow_Mina_Planta', pd.Series([0])).sum()
            s_mina_stock = df.get('Flow_Mina_Stock', pd.Series([0])).sum()
            s_mina_bot = df.get('Flow_Mina_Botadero', pd.Series([0])).sum()
            s_relleno_bot = df.get('Flow_Relleno_Botadero', pd.Series([0])).sum()
            s_stock_planta = df.get('Flow_Stock_Planta', pd.Series([0])).sum() 
            s_stock_stock = df.get('Flow_Stock_Stock', pd.Series([0])).sum()

        # Node Definitions
        # 0: Mina (Roca), 1: Mina (Relleno), 2: Stock, 3: Planta, 4: Botadero, 5: Doble Remanejo
        labels_sankey = ["Mina (Roca) 🧨 ", "Mina (Relleno) 🚜 ", "Stock Mineral 🏔️ ", "Planta 🏭 ", "Botadero 🗑️ ", "Doble Remanejo 🔄 "]
        colors_sankey = ["#ff9f43", "#576574", "#5f27cd", "#00d2d3", "#ee5253", "#a29bfe"]
        
        # Link Definitions
        # Self-loops (2->2) make Sankey BLANK in Plotly. Redirect to a sink node.
        source = [0, 0, 0, 1, 2, 2] 
        target = [3, 2, 4, 4, 3, 5] 
        value = [s_mina_planta, s_mina_stock, s_mina_bot, s_relleno_bot, s_stock_planta, s_stock_stock]
        link_colors = [
            "rgba(255, 159, 67, 0.4)", "rgba(255, 159, 67, 0.4)", "rgba(255, 159, 67, 0.4)", 
            "rgba(87, 101, 116, 0.4)", "rgba(95, 39, 205, 0.4)", "rgba(162, 155, 254, 0.4)"
        ]

        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15, thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = labels_sankey,
              color = colors_sankey
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
            
            # NEW: Use Envios Desglosados data for accurate waterfall (source of truth)
            # Check if waterfall columns exist (they should after merge)
            has_waterfall_data = all(col in df.columns for col in [
                'Waterfall_Mina_to_Botaderos',
                'Waterfall_Mina_to_StockMP', 
                'Waterfall_Doble_Remanejo',
                'Waterfall_Mina_to_Planta',
                'Waterfall_Stock_to_Planta'
            ])
            
            if has_waterfall_data:
                # CORRECTED VALUES from Envios Desglosados
                w_bot = df['Waterfall_Mina_to_Botaderos'].sum()
                w_stock_mp = df['Waterfall_Mina_to_StockMP'].sum()
                w_doble_rem = df['Waterfall_Doble_Remanejo'].sum()
                
                # Calculate total movement (Mina + Remanejo)
                mina_movement = df['Waterfall_Mina_to_Planta'].sum() + w_stock_mp + w_bot
                remanejo_movement = df['Waterfall_Stock_to_Planta'].sum() + w_doble_rem
                w_mov = mina_movement + remanejo_movement
                
                # Planta feed (direct from mina + remanejo to planta)
                w_planta_target = df['Waterfall_Mina_to_Planta'].sum() + df['Waterfall_Stock_to_Planta'].sum()
            else:
                # Fallback to old logic if waterfall columns not available
                w_mov = df['Mov_Total'].sum()
                w_bot = df.get('Flow_Mina_Botadero', pd.Series([0])).sum()
                w_stock_mp = df.get('Flow_Mina_Stock', pd.Series([0])).sum()
                w_doble_rem = df.get('Flow_Stock_Stock', pd.Series([0])).sum()
                w_planta_target = (df.get('Trat_Planta', pd.Series([0])).sum() / 1000.0) * 1.025
            
            fig_water = go.Figure(go.Waterfall(
                measure = ["absolute", "relative", "relative", "relative", "total"],
                x = ["Mov. Mina Total", "Hacia Botaderos", "Hacia Stock MP", "Doble Remanejo", "Alimentación Planta"],
                y = [w_mov, -w_bot, -w_stock_mp, -w_doble_rem, None],
                text = [f"{int(w_mov)}", f"-{int(w_bot)}", f"-{int(w_stock_mp)}", f"-{int(w_doble_rem)}", f"{int(w_planta_target)}"],
                textposition = "outside",
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
                 margin=dict(l=10, r=10, t=60, b=10),
                 yaxis=dict(
                     showgrid=False,
                     range=[0, w_mov * 1.15] # Add 15% head space
                 )
            )
            st.plotly_chart(fig_water, use_container_width=True, key=f"{key_id}_waterfall")
        
    # --- DELETED TABS (Palas, Costos) ---
    # User requested to "start from zero" on 12/16/2025 due to data complexity.
    # Code removed to ensure cleanliness.
    # (Legacy Fleet/Costos logic removed)
    # (Legacy Fleet/Costos logic removed)

# --- RENDER FLEET (Modificado Petición Usuario) ---
def render_fleet_dashboard(df_fleet):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    
    st.markdown("### 🚚 Palas & Camiones")
    
    if df_fleet.empty:
        st.warning("Datos de Flota no disponibles.")
        return
        
    # --- 1. Data Prep ---
    
    # Aggregation for stability
    agg_map = {'Ton': 'sum'}
    if 'Rend' in df_fleet.columns: agg_map['Rend'] = 'mean'
    
    df_grouped = df_fleet.groupby(['Year', 'Month', 'Quarter', 'Periodo', 'Equipo', 'Fase'], as_index=False).agg(agg_map)
    
    # Sort by Year and Month to ensure alphabetical order doesn't break timeline
    df_grouped = df_grouped.sort_values(by=['Year', 'Month'])
    sort_order = df_grouped['Periodo'].unique().tolist()

    # Filter out empty future periods
    periods_with_data = df_grouped.groupby('Periodo')['Ton'].sum()
    valid_periods = [p for p in sort_order if p in periods_with_data[periods_with_data > 0].index]
    df_grouped = df_grouped[df_grouped['Periodo'].isin(valid_periods)]

    # Phase Colors
    phase_colors = {
        'Fase 3': '#9D4EDD', 'Fase 4': '#FF6D00', 'Fase 5': '#06D6A0', 'Remanejo': '#EF476F', 'Transporte': '#118AB2'
    }
    
    # --- 2. Clean Timeline Chart (Faceted by Equipment) ---
    st.markdown("#### 🏗️ Movimiento Palas & Apoyo")
    
    # Equipment Order (Most productive first)
    equip_order = ['Pala 06', 'Pala 05', 'Pala 04', 'Pala 03', 'Cargador Frontal', 'Bulldozer']
    df_shovels = df_grouped[df_grouped['Equipo'].isin(equip_order)]
    
    fig = px.bar(
        df_shovels,
        x="Periodo",
        y="Ton",
        color="Fase",
        facet_col="Equipo",
        facet_col_wrap=3,
        facet_row_spacing=0.15, # More space between top and bottom rows
        facet_col_spacing=0.06, # More space between columns
        color_discrete_map=phase_colors,
        category_orders={"Equipo": equip_order, "Periodo": valid_periods},
        labels={"Ton": "kTon", "Periodo": ""},
        template="plotly_dark",
        height=800, # Increased height to accommodate more spacing
        text="Ton"
    )
    
    # Clean up layout
    fig.update_xaxes(
        tickangle=-90, 
        tickfont=dict(size=9, color='white'),
        title_text="", # Hide title "Periodo", keep ticks
        showticklabels=True,
        type='category'
    )
    fig.update_yaxes(title="", tickfont=dict(size=10))
    
    # REMOVE "CUTS" (Borders) and add total labels
    fig.update_traces(
        marker_line_width=0, 
        selector=dict(type='bar'),
        texttemplate='%{text:.0f}', 
        textposition='inside',
        textfont_size=9
    )
    
    # Better facet labels (SIGNIFICANTLY LARGER AND BOLDEST)
    fig.for_each_annotation(lambda a: a.update(
        text=f"<b>{a.text.split('=')[-1]}</b>", 
        font=dict(size=18, color='#00f2ff') # Cyber Cyan for high visibility
    ))
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5, title=None),
        margin=dict(l=40, r=40, t=80, b=120), # More top margin for large titles
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, key="fleet_px_v6")

    # --- 3. Truck Dashboard & Match ---
    m1, m2 = st.columns([2, 1])
    
    with m1:
        st.markdown("#### 🚛 Ciclo de Transporte (Camiones)")
        df_trucks = df_grouped[df_grouped['Equipo'] == 'Camiones']
        
        if not df_trucks.empty:
            fig_t = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Bars: N° Camiones (stored in 'Ton' for trucks)
            fig_t.add_trace(
                go.Bar(
                    x=df_trucks['Periodo'], y=df_trucks['Ton'],
                    name="N° Camiones",
                    marker_color='#118AB2', opacity=0.8,
                    text=df_trucks['Ton'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
                    textposition='inside'
                ),
                secondary_y=False
            )
            
            # Line: Rendimiento
            if 'Rend' in df_trucks.columns and not df_trucks['Rend'].isna().all():
                fig_t.add_trace(
                    go.Scatter(
                        x=df_trucks['Periodo'], y=df_trucks['Rend'],
                        name="RenDCamion (t/h)",
                        line=dict(color='#EF476F', width=3),
                        mode='lines+markers+text',
                        text=df_trucks['Rend'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else ""),
                        textposition='top center',
                        textfont=dict(color='#EF476F', size=11)
                    ),
                    secondary_y=True
                )
            
            fig_t.update_layout(
                template="plotly_dark", height=450,
                margin=dict(l=20, r=20, t=30, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_t.update_xaxes(tickangle=-90, type='category', categoryarray=valid_periods)
            fig_t.update_yaxes(title_text="N° Camiones", secondary_y=False, showgrid=False)
            fig_t.update_yaxes(title_text="Rendimiento (t/h)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_t, use_container_width=True, key="trucks_chart_v5")

    # Match Chart Removed per User Request (Image 1)
    # with m2:
    #    st.markdown("#### 📏 Match Pala-Camión")
    #    ... (Code Removed) ...

    # --- 4. KPI Summary ---
    st.markdown("---")
    cols = st.columns(7)
    
    # Loaders list for correct summation (Including validated Bulldozer Production)
    loaders_list = ['Pala 06', 'Pala 05', 'Pala 04', 'Pala 03', 'Cargador Frontal', 'Bulldozer']
    
    # Calculate total production only from loading equipment
    total_mov_sum = df_grouped[df_grouped['Equipo'].isin(loaders_list)]['Ton'].sum()
    cols[0].metric("Mov. Total Flota", f"{total_mov_sum:,.0f} kTon")
    
    
    kpi_list = ['Pala 06', 'Pala 05', 'Pala 04', 'Pala 03', 'Cargador Frontal', 'Bulldozer']
    for i, eq in enumerate(kpi_list):
        row_eq = df_grouped[df_grouped['Equipo'] == eq]
        val = row_eq['Ton'].sum()
        cols[i+1].metric(eq, f"{val:,.0f}")


# --- RENDER AUXILIARY (Perfos + Servicios) ---
def render_auxiliary_dashboard(df_perfos, df_serv):
    st.markdown("### 🚜 Perforadoras y Equipos de Servicios")
    
    t1, t2 = st.tabs(["Perforación", "Servicios"])
    
    with t1:
        if df_perfos.empty:
            st.warning("Datos de Perforación no disponibles.")
        else:
            st.markdown("#### Detalle Perforación (Mensual/Trimestral)")
            
            # Create periodo labels and sort order
            month_names = {
                1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
            }
            
            def make_period_label(row):
                if row['Year'] >= 2028:
                    return f"{row['Quarter']} {row['Year']}"
                else:
                    return f"{month_names.get(row['Month'], str(row['Month']))} {row['Year']}"
            
            def get_sort_key(row):
                # Create sortable key: year * 100 + month
                return row['Year'] * 100 + row['Month']
            
            df_perfos['PeriodoLabel'] = df_perfos.apply(make_period_label, axis=1)
            df_perfos['SortKey'] = df_perfos.apply(get_sort_key, axis=1)
            
            # === 1. PRODUCCIÓN - METROS ===
            st.markdown("##### 📏 Perforadoras de Producción - Metros")
            
            # KPI for Production Meters
            try:
                prod_val = df_perfos[
                    (df_perfos['Metric'].str.contains('Producc', case=False, na=False)) & 
                    (df_perfos['Item'].str.contains('Total', case=False, na=False))
                ]['Value'].sum()
                st.metric("Total Metros Producción", f"{prod_val:,.0f} m")
            except:
                pass
            
            df_prod_mts = df_perfos[
                (df_perfos['MetricCategory'].str.contains('Mts Producc', case=False, na=False)) &
                (~df_perfos['Item'].str.contains('Total', case=False, na=False))
            ].copy()
            
            if not df_prod_mts.empty:
                # Sort by chronological order
                df_prod_mts = df_prod_mts.sort_values('SortKey')
                period_order = df_prod_mts['PeriodoLabel'].unique()
                
                fig1 = go.Figure()
                for equipo in ['PV-5', 'DMM3-03', 'PV6']:
                    df_eq = df_prod_mts[df_prod_mts['Item'] == equipo]
                    if not df_eq.empty:
                        df_eq_grouped = df_eq.groupby('PeriodoLabel', sort=False)['Value'].sum()
                        # Reindex to maintain period order
                        df_eq_grouped = df_eq_grouped.reindex(period_order, fill_value=0)
                        fig1.add_trace(go.Bar(
                            name=equipo,
                            x=df_eq_grouped.index,
                            y=df_eq_grouped.values,
                            text=df_eq_grouped.apply(lambda x: f"{x:,.0f}" if x > 0 else ""),
                            textposition='inside',
                            textfont=dict(size=10)
                        ))
                
                fig1.update_layout(
                    barmode='stack',
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=80),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(tickangle=-45, title='', categoryorder='array', categoryarray=list(period_order)),
                    yaxis=dict(title='Metros'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig1, use_container_width=True, key="prod_mts_chart")
            
            # === 2. PRODUCCIÓN - HORAS ===
            st.markdown("##### ⏱️ Perforadoras de Producción - Horas Operativas")
            
            # KPI for Production Hours
            try:
                prod_hrs_val = df_perfos[
                    (df_perfos['MetricCategory'].str.contains('Horas Operativas', case=False, na=False)) &
                    (df_perfos['Item'].isin(['PV-5', 'DMM3-03', 'PV6']))
                ]['Value'].sum()
                st.metric("Total Horas Producción", f"{prod_hrs_val:,.0f} h")
            except:
                pass
            
            df_prod_hrs = df_perfos[
                (df_perfos['MetricCategory'].str.contains('Horas Operativas', case=False, na=False)) &
                (df_perfos['Item'].isin(['PV-5', 'DMM3-03', 'PV6']))
            ].copy()
            
            if not df_prod_hrs.empty:
                df_prod_hrs = df_prod_hrs.sort_values('SortKey')
                period_order = df_prod_hrs['PeriodoLabel'].unique()
                
                fig2 = go.Figure()
                for equipo in ['PV-5', 'DMM3-03', 'PV6']:
                    df_eq = df_prod_hrs[df_prod_hrs['Item'] == equipo]
                    if not df_eq.empty:
                        df_eq_grouped = df_eq.groupby('PeriodoLabel', sort=False)['Value'].sum()
                        df_eq_grouped = df_eq_grouped.reindex(period_order, fill_value=0)
                        fig2.add_trace(go.Bar(
                            name=equipo,
                            x=df_eq_grouped.index,
                            y=df_eq_grouped.values,
                            text=df_eq_grouped.apply(lambda x: f"{x:,.0f}" if x > 0 else ""),
                            textposition='inside',
                            textfont=dict(size=10)
                        ))
                
                fig2.update_layout(
                    barmode='stack',
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=80),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(tickangle=-45, title='', categoryorder='array', categoryarray=list(period_order)),
                    yaxis=dict(title='Horas'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig2, use_container_width=True, key="prod_hrs_chart")
            
            # === 3. PRE-CORTE - METROS ===
            st.markdown("##### 📐 Perforadoras Pre-Corte - Metros")
            
            # KPI for Pre-Cut Meters
            try:
                prec_val = df_perfos[
                    (df_perfos['Metric'].str.contains('Precort', case=False, na=False)) &
                    (df_perfos['Item'].str.contains('Total', case=False, na=False))
                ]['Value'].sum()
                st.metric("Total Metros Precorte", f"{prec_val:,.0f} m")
            except:
                pass
            
            df_prec_mts = df_perfos[
                (df_perfos['MetricCategory'].str.contains('Mts Precorte', case=False, na=False)) &
                (~df_perfos['Item'].str.contains('Total', case=False, na=False))
            ].copy()
            
            if not df_prec_mts.empty:
                df_prec_mts = df_prec_mts.sort_values('SortKey')
                period_order = df_prec_mts['PeriodoLabel'].unique()
                
                fig3 = go.Figure()
                for equipo in ['D65 SmartRoc - 15', 'D65 SmartRoc - 14', 'D65 Nueva']:
                    df_eq = df_prec_mts[df_prec_mts['Item'] == equipo]
                    if not df_eq.empty:
                        df_eq_grouped = df_eq.groupby('PeriodoLabel', sort=False)['Value'].sum()
                        df_eq_grouped = df_eq_grouped.reindex(period_order, fill_value=0)
                        fig3.add_trace(go.Bar(
                            name=equipo.replace('D65 SmartRoc - ', 'D65-').replace('D65 Nueva', 'D65-Nueva'),
                            x=df_eq_grouped.index,
                            y=df_eq_grouped.values,
                            text=df_eq_grouped.apply(lambda x: f"{x:,.0f}" if x > 0 else ""),
                            textposition='inside',
                            textfont=dict(size=10)
                        ))
                
                fig3.update_layout(
                    barmode='stack',
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=80),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(tickangle=-45, title='', categoryorder='array', categoryarray=list(period_order)),
                    yaxis=dict(title='Metros'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig3, use_container_width=True, key="prec_mts_chart")
            
            # === 4. PRE-CORTE - HORAS ===
            st.markdown("##### ⏲️ Perforadoras Pre-Corte - Horas Operativas")
            
            # KPI for Pre-Cut Hours
            try:
                prec_hrs_val = df_perfos[
                    (df_perfos['MetricCategory'].str.contains('Horas Operativas', case=False, na=False)) &
                    (df_perfos['Item'].str.contains('D65', case=False, na=False))
                ]['Value'].sum()
                st.metric("Total Horas Pre-Corte", f"{prec_hrs_val:,.0f} h")
            except:
                pass
            
            df_prec_hrs = df_perfos[
                (df_perfos['MetricCategory'].str.contains('Horas Operativas', case=False, na=False)) &
                (df_perfos['Item'].str.contains('D65', case=False, na=False))
            ].copy()
            
            if not df_prec_hrs.empty:
                df_prec_hrs = df_prec_hrs.sort_values('SortKey')
                period_order = df_prec_hrs['PeriodoLabel'].unique()
                
                fig4 = go.Figure()
                for equipo in ['D65 SmartRoc - 15', 'D65 SmartRoc - 14', 'D65 Nueva']:
                    df_eq = df_prec_hrs[df_prec_hrs['Item'] == equipo]
                    if not df_eq.empty:
                        df_eq_grouped = df_eq.groupby('PeriodoLabel', sort=False)['Value'].sum()
                        df_eq_grouped = df_eq_grouped.reindex(period_order, fill_value=0)
                        fig4.add_trace(go.Bar(
                            name=equipo.replace('D65 SmartRoc - ', 'D65-').replace('D65 Nueva', 'D65-Nueva'),
                            x=df_eq_grouped.index,
                            y=df_eq_grouped.values,
                            text=df_eq_grouped.apply(lambda x: f"{x:,.0f}" if x > 0 else ""),
                            textposition='inside',
                            textfont=dict(size=10)
                        ))
                
                fig4.update_layout(
                    barmode='stack',
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=80),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(tickangle=-45, title='', categoryorder='array', categoryarray=list(period_order)),
                    yaxis=dict(title='Horas'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig4, use_container_width=True, key="prec_hrs_chart")

    with t2:
        if df_serv.empty:
            st.warning("Datos de Servicios no disponibles.")
        else:
            st.markdown("#### Equipos de Servicios - Horas Operativas")
            
            # KPI: Total hours
            try:
                total_hrs = df_serv['Value'].sum()
                st.metric("Total Horas Equipos de Servicio", f"{total_hrs:,.0f} h")
            except:
                pass
            
            # Create periodo labels and sort
            month_names = {
                1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
            }
            
            def make_period_label(row):
                if row['Year'] >= 2028:
                    return f"{row['Quarter']} {row['Year']}"
                else:
                    return f"{month_names.get(row['Month'], str(row['Month']))} {row['Year']}"
            
            def get_sort_key(row):
                return row['Year'] * 100 + row['Month']
            
            df_serv['PeriodoLabel'] = df_serv.apply(make_period_label, axis=1)
            df_serv['SortKey'] = df_serv.apply(get_sort_key, axis=1)
            
            # Sort by chronological order
            df_serv_sorted = df_serv.sort_values('SortKey')
            period_order = df_serv_sorted['PeriodoLabel'].unique()
            
            # Equipment order (from image)
            equipment_order = [
                'Motoniveladoras Cat 16-H',
                'Bulldozer Cat D10T',
                'Bulldozer Komatsu D475A',
                'Wheeldozers Cat 834 C (W5)',
                'Wheeldozers Cat 834 H (W6)',
                'Excavadora  PC800                   N°2',
                'Excavadora  PC800 Martillo   N°4',
                'Excavadora  PC450 Martillo   N°3',
                'Retro5',
                'Rodillo',
                'Cargador WA500'
            ]
            
            # Create stacked bar chart
            fig = go.Figure()
            
            for equipo in equipment_order:
                # Find matching equipment (handle spacing differences)
                df_eq = df_serv[df_serv['Item'].str.strip() == equipo.strip()]
                
                if not df_eq.empty:
                    df_eq_grouped = df_eq.groupby('PeriodoLabel', sort=False)['Value'].sum()
                    df_eq_grouped = df_eq_grouped.reindex(period_order, fill_value=0)
                    
                    # Shorten equipment names for legend
                    display_name = equipo.strip()
                    display_name = display_name.replace('Excavadora  PC', 'Exc PC')
                    display_name = display_name.replace('Wheeldozers', 'WD')
                    display_name = display_name.replace('Motoniveladoras', 'Motoniveladora')
                    display_name = display_name.replace('Bulldozer', 'Bulldz')
                    display_name = display_name.replace('Cargador', 'Carg')
                    
                    fig.add_trace(go.Bar(
                        name=display_name,
                        x=df_eq_grouped.index,
                        y=df_eq_grouped.values,
                        text=df_eq_grouped.apply(lambda x: f"{x:,.0f}" if x > 0 else ""),
                        textposition='inside',
                        textfont=dict(size=9)
                    ))
            
            fig.update_layout(
                barmode='stack',
                template='plotly_dark',
                height=500,
                margin=dict(l=20, r=20, t=20, b=80),
                legend=dict(
                    orientation="v", 
                    yanchor="top", 
                    y=0.99, 
                    xanchor="right", 
                    x=0.99,
                    bgcolor='rgba(0,0,0,0.5)',
                    font=dict(size=10)
                ),
                xaxis=dict(tickangle=-45, title='', categoryorder='array', categoryarray=list(period_order)),
                yaxis=dict(title='Horas'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="servicios_hrs_chart")

# --- 3.1 PROCESS PLANTA FOR DASHBOARD (HARDENED) ---
def process_planta_for_dashboard(df_raw):
    """
    Transforms the Wide/Raw 'Planta' sheet into a Long format suitable for the Dashboard.
    Maps Row Labels (e.g., 'Cobre Fino') to Column Keys (e.g., 'Cobre_Fino').
    """
    if df_raw.empty: return pd.DataFrame()
    
    # Define Mapping: { 'Excel Label Keyword': 'Dashboard Key' }
    # Focusing on TOTAL rows for balanced KPIs
    mapping = {
        'planta total planta total cobre fino': 'Cobre_Fino',
        'planta total planta total ton tratamiento (secas)': 'Trat_Planta',
        'planta total planta total ley cut %': 'Ley_CuT',
        'planta total planta total recuperacion total': 'Recup',
        'planta total planta total sag (ton)': 'Planta_Capacidad',
        'costo mina': 'Costo_Mina', # Keep these from elsewhere if not in Total
        'costo planta': 'Costo_Planta',
    }
    
    # Initialize Data Dict
    data = {'Year': [], 'Month': [], 'Periodo': []}
    unique_keys = list(set(mapping.values()))
    for k in unique_keys:
        data[k] = []
        
    try:
        years = df_raw.iloc[0].ffill().fillna(0).tolist()
        months = df_raw.iloc[1].fillna("").tolist()
        
        # Identify Data Rows
        row_map = {} # { 'Dashboard Key': Row Index }
        
        for idx, row in df_raw.iterrows():
            if idx < 3: continue # Skip Year (0), Month (1), and label headers (2).
            
            # Construct Full Label from first few columns (0, 1, 2)
            label_parts = [str(x).lower().strip() for x in row.iloc[:3] if pd.notna(x)]
            full_label = " ".join(label_parts)
            
            for key_word, dash_key in mapping.items():
                if key_word in full_label and dash_key not in row_map:
                    row_map[dash_key] = idx
                    break
        
        num_cols = df_raw.shape[1]  # CRITICAL: Define total columns
                    
        for c in range(0, num_cols):
            y_raw = str(years[c]).split('.')[0] # Clean 2026.0
            m = str(months[c]).strip()
            
            # CRITICAL: Pair Year and Month from same column. Skip non-data columns.
            if not y_raw.isdigit() or int(y_raw) < 2026: continue 
            if "TOTAL" in m.upper() or "YEAR" in m.upper() or m == "" or m == "0": continue
            
            data['Year'].append(int(y_raw))
            data['Month'].append(m)
            data['Periodo'].append(f"{m} {y_raw}")
            
            for k in unique_keys:
                r_idx = row_map.get(k)
                val = 0.0
                if r_idx is not None:
                    try:
                        v_raw = df_raw.iloc[r_idx, c]
                        val = pd.to_numeric(v_raw, errors='coerce')
                        if pd.isna(val) or np.isinf(val): val = 0.0
                    except: val = 0.0
                data[k].append(val)
                
        df_result = pd.DataFrame(data)
        
        # Add Quarter column for trimestral filtering
        if not df_result.empty and 'Month' in df_result.columns:
            month_to_quarter = {
                'Enero': 'Q1', 'Febrero': 'Q1', 'Marzo': 'Q1',
                'Abril': 'Q2', 'Mayo': 'Q2', 'Junio': 'Q2',
                'Julio': 'Q3', 'Agosto': 'Q3', 'Septiembre': 'Q3',
                'Octubre': 'Q4', 'Noviembre': 'Q4', 'Diciembre': 'Q4'
            }
            df_result['Quarter'] = df_result['Month'].map(month_to_quarter)
        
        return df_result
        
    except Exception as e:
        print(f"Error processing Planta for Dashboard: {e}")
        return pd.DataFrame()

# --- 3.2 PROCESS DATA TECNICA (PHASES & FLOWS) ---
def process_data_tecnica_full(df_dt_raw):
    """
    Extracts Phase info (F03, F05) AND Material Flows (Sankey keys) from Data Tecnica.
    """
    if df_dt_raw.empty: return pd.DataFrame()
    
    try:
        years = df_dt_raw.iloc[0].ffill().fillna(0).tolist()
        months = df_dt_raw.iloc[1].fillna("").tolist()
        
        start_col = 4 # Validated via inspection
        num_cols = df_dt_raw.shape[1]
        
        # REQUIRED KEYS for Dashboard (Phases + Sankey + Total)
        keys = [
            'Mov_F03', 'Mov_F04', 'Mov_F05', 'Mov_Total', 'Remanejo',
            'Flow_Mina_Planta', 'Flow_Mina_Stock', 'Flow_Mina_Botadero',
            'Flow_Relleno_Botadero', 'Flow_Stock_Planta', 'Flow_Stock_Stock'
        ]
        
        data = {'Year': [], 'Month': []}
        for k in keys: data[k] = []
        
        # Pre-calculate masks for speed
        # Col 0: Origen, Col 1: Fase, Col 2: Tipo Material, Col 3: Destino
        c0 = df_dt_raw.iloc[:, 0].astype(str).str.strip()
        c1 = df_dt_raw.iloc[:, 1].astype(str).str.strip()
        c2 = df_dt_raw.iloc[:, 2].astype(str).str.strip()
        c3 = df_dt_raw.iloc[:, 3].astype(str).str.strip()
        
        num_cols = df_dt_raw.shape[1]  # CRITICAL: Define total columns
        
        for c in range(0, num_cols):
            y_raw = str(years[c]).split('.')[0] # Clean 2026.0
            m = str(months[c]).strip()
            
            # CRITICAL: Aligned pairing - Same column for Year and Month
            if not y_raw.isdigit() or int(y_raw) < 2026: continue 
            if "TOTAL" in m.upper() or "YEAR" in m.upper() or m == "" or m == "0": continue
            
            # --- EXTRACTOR ---
            def get_sum(mask):
                return pd.to_numeric(df_dt_raw.loc[mask, c], errors='coerce').fillna(0).sum()
                
            # 1. PHASES (Case-insensitive check)
            data['Mov_F03'].append(get_sum(c1.str.contains("F03", case=False, na=False)))
            data['Mov_F04'].append(get_sum(c1.str.contains("F04", case=False, na=False)))
            data['Mov_F05'].append(get_sum(c1.str.contains("F05", case=False, na=False)))
            
            # 2. FLOWS (SANKEY) - Support variations in plural/singular
            # Mina -> Planta
            data['Flow_Mina_Planta'].append(get_sum((c0.str.contains("Mina", case=False)) & (c3.str.contains("Planta", case=False))))
            
            # Mina -> Stock
            data['Flow_Mina_Stock'].append(get_sum((c0.str.contains("Mina", case=False)) & (c3.str.contains("Stock", case=False))))
            
            # Mina -> Botadero
            data['Flow_Mina_Botadero'].append(get_sum((c0.str.contains("Mina", case=False)) & (c2.str.contains("Roca", case=False)) & (c3.str.contains("Botadero", case=False))))
            
            # Relleno -> Botadero
            data['Flow_Relleno_Botadero'].append(get_sum((c2.str.contains("Relleno", case=False)) & (c3.str.contains("Botadero", case=False))))
            
            # Stock -> Planta (Rehandling)
            data['Flow_Stock_Planta'].append(get_sum((c0.str.contains("Stock|Remanejo", case=False, regex=True)) & (c3.str.contains("Planta", case=False))))
            
            # Stock -> Stock (Internal)
            data['Flow_Stock_Stock'].append(get_sum((c0.str.contains("Stock|Remanejo", case=False, regex=True)) & (c3.str.contains("Stock|Remanejo", case=False, regex=True))))
            
            # 3. MOVIMIENTO TOTAL (Target 87,768: Mina + Stock Sources)
            # This matches user's ground truth for 'Mov Mina' card.
            mov_total_mask = c0.str.contains("Mina|Stock|Remanejo", case=False, regex=True)
            data['Mov_Total'].append(get_sum(mov_total_mask))
            
            # 4. REMANEJO (Target 16,117: Stock Sources)
            remanejo_mask = c0.str.contains("Stock|Remanejo", case=False, regex=True)
            data['Remanejo'].append(get_sum(remanejo_mask))
            
            data['Year'].append(int(y_raw))
            data['Month'].append(m)
            
        return pd.DataFrame(data)

    except Exception as e:
        print(f"Error Processing Data Tecnica: {e}")
        return pd.DataFrame()

# --- 3.3 PROCESS ENVIOS DESGLOSADOS (FOR WATERFALL CHART) ---
def process_envios_desglosados(df_envios_raw):
    """
    Extracts detailed flow data from 'Envios Desglosados por Fases' for waterfall chart.
    This sheet has the correct breakdown of flows to Botaderos, Stock, etc.
    """
    if df_envios_raw.empty: return pd.DataFrame()
    
    try:
        years = df_envios_raw.iloc[0].ffill().fillna(0).tolist()
        months = df_envios_raw.iloc[1].fillna("").tolist()
        
        keys = [
            'Waterfall_Mina_to_Planta',
            'Waterfall_Mina_to_StockMP', 
            'Waterfall_Mina_to_Botaderos',
            'Waterfall_Stock_to_Planta',
            'Waterfall_Doble_Remanejo'
        ]
        
        data = {'Year': [], 'Month': []}
        for k in keys: data[k] = []
        
        num_cols = df_envios_raw.shape[1]
        
        for c in range(4, num_cols):  # Data starts at column 4
            y_raw = str(years[c]).split('.')[0]
            m = str(months[c]).strip()
            
            if not y_raw.isdigit() or int(y_raw) < 2026: continue
            if "TOTAL" in m.upper() or "YEAR" in m.upper() or m == "" or m == "0": continue
            
            # Accumulators for this month
            flows_month = {
                'Mina_to_Planta': 0,
                'Mina_to_StockMP': 0,
                'Mina_to_Botaderos': 0,
                'StockMP_to_Planta': 0,
                'Remanejo_to_Planta': 0,
                'Remanejo_to_StockDR': 0,
                'StockDR_to_Stock': 0
            }
            
            for idx, row in df_envios_raw.iterrows():
                if idx < 2: continue
                
                origen = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                destino = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
                metrica = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
                
                # Only process tonnage rows (not grades)
                if "K Ton" not in metrica:
                    continue
                
                valor = pd.to_numeric(row.iloc[c], errors='coerce')
                if pd.isna(valor):
                    valor = 0
                
                # Classify flows - CRITICAL: Capture ALL Stock types
                if "Mina" in origen and "Planta" in destino:
                    flows_month['Mina_to_Planta'] += valor
                elif "Mina" in origen and ("StockMP" in destino or "Stock" in destino):
                    # FIXED: Sum both "Stock" and "StockMP (mismo periodo)"
                    flows_month['Mina_to_StockMP'] += valor
                elif "Mina" in origen and "Botaderos" in destino:
                    flows_month['Mina_to_Botaderos'] += valor
                elif "StockMP" in origen and "Planta" in destino:
                    flows_month['StockMP_to_Planta'] += valor
                elif "Remanejo" in origen and "Planta" in destino:
                    flows_month['Remanejo_to_Planta'] += valor
                elif "Remanejo" in origen and "StockDR" in destino:
                    flows_month['Remanejo_to_StockDR'] += valor
                elif "StockDR" in origen and "Stock" in destino:
                    flows_month['StockDR_to_Stock'] += valor
            
            # Aggregate for waterfall
            data['Waterfall_Mina_to_Planta'].append(flows_month['Mina_to_Planta'])
            data['Waterfall_Mina_to_StockMP'].append(flows_month['Mina_to_StockMP'])
            data['Waterfall_Mina_to_Botaderos'].append(flows_month['Mina_to_Botaderos'])
            
            # Stock to Planta (combines StockMP and Remanejo sources)
            stock_planta = flows_month['StockMP_to_Planta'] + flows_month['Remanejo_to_Planta']
            data['Waterfall_Stock_to_Planta'].append(stock_planta)
            
            # Doble Remanejo (Stock->Stock movements)
            doble_rem = flows_month['StockDR_to_Stock'] + flows_month['Remanejo_to_StockDR']
            data['Waterfall_Doble_Remanejo'].append(doble_rem)
            
            data['Year'].append(int(y_raw))
            data['Month'].append(m)
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error Processing Envios Desglosados: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# --- 2. CARGA DE DATOS ---
@st.cache_data(show_spinner=True, ttl=60)  # Reduced TTL to force reload
def load_data_v5():
    """Carga de datos centralizada y optimizada v5 (AI RAW MODE)."""
    file_path = "plan_budget_real.xlsx"
    
    if not os.path.exists(file_path):
        return None
        
    try:
        # Load Main Data (Planta) - RAW
        df_planta_raw = load_and_clean_excel(file_path, 'Planta', ffill_cols=[0])
        
        # TRANSFORM FOR DASHBOARD (PLANTA)
        df_planta = process_planta_for_dashboard(df_planta_raw)
        
        # NEW: LOAD & MERGE PHASES & FLOWS (DATA TECNICA)
        df_dt_raw = load_and_clean_excel(file_path, 'Data Tecnica', ffill_cols=[0])
        df_dt_full = process_data_tecnica_full(df_dt_raw)
        
        # DEBUG: Check dataframe sizes
        print(f"DEBUG load_data_v5: df_planta rows = {len(df_planta)}")
        print(f"DEBUG load_data_v5: df_dt_full rows = {len(df_dt_full)}")
        
        if not df_dt_full.empty and not df_planta.empty:
            # Merge on Year, Month to Inject F03/F05/Flows
            df_planta = pd.merge(df_planta, df_dt_full, on=['Year', 'Month'], how='left')
            # Fill NaNs with 0 for all injected columns
            cols_to_fill = [c for c in df_dt_full.columns if c not in ['Year', 'Month']]
            for c in cols_to_fill:
                 if c in df_planta.columns:
                     df_planta[c] = df_planta[c].fillna(0)
        
        # NEW: LOAD & MERGE ENVIOS DESGLOSADOS (FOR WATERFALL)
        df_envios_raw = load_and_clean_excel(file_path, 'Envios Desglosados por Fases', ffill_cols=[0])
        df_envios = process_envios_desglosados(df_envios_raw)
        
        print(f"DEBUG load_data_v5: df_envios rows = {len(df_envios)}")
        
        if not df_envios.empty and not df_planta.empty:
            # Merge waterfall data
            df_planta = pd.merge(df_planta, df_envios, on=['Year', 'Month'], how='left')
            # Fill NaNs with 0
            cols_to_fill_envios = [c for c in df_envios.columns if c not in ['Year', 'Month']]
            for c in cols_to_fill_envios:
                if c in df_planta.columns:
                    df_planta[c] = df_planta[c].fillna(0)
        
        # Load Fleet Data (Clean Loader V2 - Fixed Period Mapping)
        df_fleet = fleet_loader.load_fleet_clean(file_path)
        
        # Fleet loader generates correct Periodo labels - no synchronization needed
        
        
        # DEBUG: Verify Periodo column
        print(f"DEBUG: df_fleet columns = {df_fleet.columns.tolist() if not df_fleet.empty else 'EMPTY'}")
        print(f"DEBUG: df_fleet shape = {df_fleet.shape}")
        if not df_fleet.empty and 'Periodo' in df_fleet.columns:
            print(f"DEBUG: Fleet Periodo samples = {df_fleet['Periodo'].unique()[:5].tolist()}") 
        
        # Load Auxiliary Data (Perfos & Servicios)
        df_aux = fleet_v3.load_kpi_perfos_servicios(file_path)
        df_perfos = df_aux[df_aux['Category']=='Perfos'] if not df_aux.empty else pd.DataFrame()
        df_serv = df_aux[df_aux['Category']=='Servicios'] if not df_aux.empty else pd.DataFrame()
        
        # NEW: Load AI Context (Raw Smart Sheets)
        import ai_loader
        importlib.reload(ai_loader) # Force reload
        ai_raw = ai_loader.get_ai_context(file_path)

        return {
            'planta': df_planta,
            'fleet': df_fleet,
            'perfos': df_perfos, 
            'servicios': df_serv,
            'ai_raw': ai_raw # Complete raw data for AI
        }
    except Exception as e:
        print(f"Error loading V5: {e}")
        st.error(f"CRITICAL ERROR LOADING DATA V5: {e}") 
        import traceback
        st.code(traceback.format_exc())
        return None

# --- MAIN APP FLOW ---
data_loaded = load_data_v5()

if data_loaded:
    df = data_loaded.get('planta', pd.DataFrame())
    df_fleet = data_loaded.get('fleet', pd.DataFrame())
    df_perfos = data_loaded.get('perfos', pd.DataFrame())
    df_serv = data_loaded.get('servicios', pd.DataFrame())
    ai_raw_context = data_loaded.get('ai_raw', {}) # Raw Dict
    
    
    # Deprecated legacy DF
    df_pala_fase_info = None 

    if df.empty:
        # Fallback if Planta fails but others succeed
        if not df_fleet.empty: pass 
        else: st.error("Data Planta Empty")

    # --- 7. MAIN TABS LOGIC ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- MATRIX NAVIGATION (2 ROWS) ---
    # Row 1: View Context (Panel, Fleet, Aux)
    # Row 2: Time Context (Total, 2026, 2027...)
    
    # 1. Initialize State
    # 1. Initialize State
    if 'nav_view' not in st.session_state: st.session_state['nav_view'] = "📊 Panel de Control"
    if 'nav_period' not in st.session_state: st.session_state['nav_period'] = "📈 2026-2029"

    # 2. Row 1: View Selection
    c1, c2, c3, c4 = st.columns(4)
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
    with c4:
        if st.button("🤖 Chat IA", use_container_width=True):
            st.session_state['nav_view'] = "🤖 Chat IA"
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
        
    elif current_view == "🤖 Chat IA":
        # CSS FOR CHAT IA
        st.markdown("""
            <style>
                .chat-welcome { text-align: center; margin-top: 2rem; margin-bottom: 2rem; }
                .chat-welcome h1 { font-size: 3.5rem; font-weight: 400; color: #fff; margin-bottom: 0.5rem; letter-spacing: -1px; }
                .chat-welcome p { font-size: 1.2rem; color: #9aa0a6; margin-bottom: 3rem; }
                .suggestion-list { max-width: 600px; margin: 0 auto; text-align: left; }
                .suggestion-item { 
                    display: flex; align-items: center; padding: 12px; 
                    color: #e8eaed; font-size: 0.95rem; cursor: pointer;
                    transition: background 0.2s; border-radius: 8px;
                }
                .suggestion-item:hover { background: rgba(255,255,255,0.05); color: #00f2ff; }
                .suggestion-icon { margin-right: 15px; color: #9aa0a6; font-size: 1.2rem; }
            </style>
        """, unsafe_allow_html=True)

        col_header, col_clear = st.columns([10, 2])
        with col_header: st.title("Asistente Virtual Plan Minero")
        with col_clear: 
            if st.button("🗑️ Limpiar Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # --- CLOUD STATUS DIAGNOSTIC (MAIN VIEW) ---
        try:
            import cloud_manager
            is_cloud_ok = cloud_manager.check_cloud_status()
            if is_cloud_ok:
                st.success("🟢 Memoria Cloud: Conectada (Online)")
            else:
                st.error("🔴 Memoria Cloud: Desconectada (Verifica Secrets)")
        except Exception as e:
            st.error(f"Cloud Error: {e}")
        # -------------------------------
                
        # Initialize Agent
        if 'chat_agent_inst' not in st.session_state:
            try:
                # NEW MULTI-CONTEXT INIT V2 (RAW AI)
                data_context = {
                    'planta': df,
                    'fleet': df_fleet, 
                    'perfos': df_perfos,
                    'servicios': df_serv,
                    'ai_raw': ai_raw_context # PROCESSED RAW DATA
                }
                agent = CodeGenerationChatAgent(data_context)
                
                # LOAD KEY SAFELY
                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                except:
                    api_key = os.environ.get("GEMINI_API_KEY")
                
                agent.initialize(api_key=api_key)
                st.session_state['chat_agent_inst'] = agent
            except Exception as e:
                st.error(f"Error iniciando IA: {e}")
                
        # Initialize History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display History or Welcome Screen
        if not st.session_state.messages:

            st.markdown(f"""
                <style>
                    .chat-welcome {{
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 60vh;
                        text-align: center;
                    }}
                    .chat-welcome h1 {{
                        font-family: 'Google Sans', sans-serif;
                        font-size: 4rem;
                        font-weight: 500;
                        color: #ffffff;
                        margin-bottom: 1rem;
                        background: linear-gradient(90deg, #4285F4, #9B72CB, #D96570);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                    }}
                    .chat-welcome p {{
                        font-size: 1.5rem;
                        color: #bdc1c6;
                        max-width: 600px;
                    }}
                </style>
                <div class="chat-welcome">
                    <h1>Descubre el Modo IA</h1>
                    <p>Haz preguntas detalladas para obtener mejores respuestas</p>
                </div>
            """, unsafe_allow_html=True)
            # SUGGESTIONS REMOVED AS REQUESTED
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message.get("type") == "text":
                        st.markdown(message["content"])
                    elif message.get("type") == "dataframe":
                        st.dataframe(message["content"], hide_index=True, use_container_width=True)
                    elif message.get("type") == "plot":
                        st.plotly_chart(message["content"])

        # Input
        if prompt := st.chat_input("Escribe tu pregunta minera aquí..."):
            # Display User Message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
            
            # Generate Answer
            with st.spinner("Analizando datos..."):
                agent = st.session_state.get('chat_agent_inst')
                if agent:
                    response = agent.ask(prompt, history=st.session_state.messages)
                    
                    # Display AI Message
                    with st.chat_message("assistant"):
                        if response['type'] == 'text':
                            st.markdown(response['content'])
                        elif response['type'] == 'error':
                            st.error(response['content'])
                            with st.expander("Ver Código Generado"):
                                st.code(response.get('code', ''), language='python')
                        elif response['type'] == 'dataframe':
                            st.dataframe(response['content'], hide_index=True, use_container_width=True)
                        elif response['type'] == 'plot':
                            st.plotly_chart(response['content'])
                            if 'data' in response:
                                with st.expander("Ver Datos"):
                                    st.dataframe(response['data'])
                    
                    # Store in History
                    # Note: We can't store Plotly objects easily in session state persistence if simple pickle? 
                    # Streamlit handles it fine usually.
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response['content'], 
                        "type": response['type']
                    })
                else:
                    st.error("Agente no inicializado.")

else:
    st.error("No se pudieron cargar los datos. Verifica el archivo Excel.")

