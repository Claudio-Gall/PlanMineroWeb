import pandas as pd
import numpy as np

file_path = "plan_budget_real.xlsx"

def load_and_clean_excel(file_path, sheet_name, ffill_cols=None, ffill_rows=None):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        if ffill_rows:
            for r in ffill_rows:
                if r < df.shape[0]: df.iloc[r, :] = df.iloc[r, :].ffill()
        if ffill_cols:
            for c in ffill_cols:
                if c < df.shape[1]: df.iloc[:, c] = df.iloc[:, c].ffill()
        return df
    except Exception as e:
        print(f"Error {sheet_name}: {e}")
        return pd.DataFrame()

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
        y_val_raw = str(df.iloc[y_row, c]).replace('.0','').strip() # Added strip
        if y_val_raw.isdigit() and 2020 < int(y_val_raw) < 2050: 
            curr_year = int(y_val_raw)
        
        m_val_raw = str(df.iloc[h_row, c]).strip()
        if curr_year:
            m_val_raw_lower = m_val_raw.lower()
            m_val = m_val_raw # Default keep original
            
            # Robust Mapping
            if '1er' in m_val_raw_lower or 'q1' in m_val_raw_lower: m_val = 'Q1'
            elif '2do' in m_val_raw_lower or 'q2' in m_val_raw_lower: m_val = 'Q2'
            elif '3er' in m_val_raw_lower or 'q3' in m_val_raw_lower: m_val = 'Q3'
            elif '4to' in m_val_raw_lower or 'q4' in m_val_raw_lower: m_val = 'Q4'

            valid_labels = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
                            'Q1', 'Q2', 'Q3', 'Q4']
                            
            if m_val in valid_labels: # removed uniqueness check for debug
                 # print(f"Found {sheet_name}: {curr_year} {m_val} at Col {c}")
                 mapping[(curr_year, m_val)] = c
    return mapping

print("--- DEBUGGING 2028 MAPPING ---")
try:
    print("Loading Data Tecnica...")
    df_dt = load_and_clean_excel(file_path, 'Data Tecnica', ffill_cols=[0, 1, 2])
    map_dt = get_col_map(df_dt, "DT")
    print(f"DT Matrix Keys 2028: {[k for k in map_dt.keys() if k[0]==2028]}")

    print("\nLoading Planta...")
    df_planta = load_and_clean_excel(file_path, 'Planta', ffill_cols=[0])
    map_planta = get_col_map(df_planta, "Planta")
    print(f"Planta Matrix Keys 2028: {[k for k in map_planta.keys() if k[0]==2028]}")
    
except Exception as e:
    print(f"Fatal Error: {e}")
