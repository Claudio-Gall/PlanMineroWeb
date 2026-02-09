import pandas as pd

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

def find_header_row(df, keywords):
    for idx, row in df.iterrows():
        if idx > 20: break 
        s = " ".join([str(x) for x in row if pd.notna(x)])
        if any(k.lower() in s.lower() for k in keywords): return idx
    return 0

try:
    print("LOADING DATA TECNICA...")
    # Read without header initially to find it
    df_dt = pd.read_excel(file_path, sheet_name='Data Tecnica', header=None, engine='openpyxl')
    
    h_row = find_header_row(df_dt, ['Enero', 'Jan', 'Q1', 'Trimestre'])
    print(f"Header found at row index: {h_row}")
    
    # Print headers around 2028-2029 columns
    # We assume Year is in row h_row - 1
    y_row = h_row - 1
    
    print("\n--- DETECTED COLUMNS (Year | Period) ---")
    for c in range(df_dt.shape[1]):
        y_val = str(df_dt.iloc[y_row, c]).replace('.0','')
        m_val = str(df_dt.iloc[h_row, c]).strip()
        
        if y_val.isdigit() and int(y_val) >= 2026:
            print(f"Col {c}: {y_val} | {m_val}")

except Exception as e:
    print(f"Error: {e}")
