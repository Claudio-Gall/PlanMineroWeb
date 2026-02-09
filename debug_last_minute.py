import pandas as pd
import numpy as np
import os

# Mock Streamlit to allow function usage
class MockSt:
    def cache_data(self, func): return func
    def error(self, msg): print(f"ST_ERROR: {msg}")
    def warning(self, msg): print(f"ST_WARNING: {msg}")
    def secrets(self): return {}
    
import sys
import types
st = MockSt()
sys.modules['streamlit'] = st

# Define necessary functions from app.py (Simplified for just Data Loading)
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
        print(f"Error loading {sheet_name}: {e}")
        return pd.DataFrame()

def load_data_jan_2026(file_path):
    print(f"Loading from {file_path}...")
    
    # 1. Planta (Tratamiento)
    df_planta = load_and_clean_excel(file_path, 'Planta', ffill_cols=[0])
    # 2. Data Tecnica (Movimiento)
    df_dt = load_and_clean_excel(file_path, 'Data Tecnica', ffill_cols=[0, 1, 2])
    # 3. Costos
    df_costos = pd.read_excel(file_path, sheet_name='Costos', header=2, engine='openpyxl')
    
    # --- HELPER: Find Column for Jan 2026 ---
    # In 'Planta', find 'Enero' under '2026'
    # Row 0: Years, Row 1: Months (indices 0 and 1 in 0-based IF header=None... wait)
    # load_and_clean_excel uses header=None.
    # Row 0 is Year (e.g. 2026), Row 1 is Month (e.g. Enero)
    
    def get_col_idx(df, year_val, month_val):
        # Scan row 0 for Year, row 1 for Month
        # Note: Merged cells might need ffill logic which we did in load_and_clean
        # But we only ffilled first few cols in load_and_clean.
        # We need to scan the header rows properly.
        
        # Heuristic search
        for c in range(df.shape[1]):
            y_cell = str(df.iloc[0, c]).replace('.0','').strip()
            m_cell = str(df.iloc[1, c]).strip()
            if y_cell == str(year_val) and m_cell.lower() == month_val.lower():
                return c
        return -1

    # --- GET INDICES ---
    # Planta
    col_planta = get_col_idx(df_planta, 2026, 'Enero')
    # DT (Uses similar structure usually)
    col_dt = get_col_idx(df_dt, 2026, 'Enero')
    
    print(f"Indices Found -> Planta: {col_planta}, DT: {col_dt}")

    # --- EXTRACT VALUES ---
    # Tratamiento: Row 14 (Index 14) in Planta
    trat_ton = df_planta.iloc[14, col_planta]
    
    # Movimineto: Sum of flows in DT
    # Index 20 is usually Mov Total? Or we calculate it?
    # In app.py we calculated it. Let's look for known rows or sum flows.
    # For now, let's grab specific rows mentioned in app.py logic if possible
    # app.py calculated Mov_Total as sum of F03, F04, F05, Rem.
    # Let's simple check the raw sheet for a "Total" row if it exists, or just use app.py logic implies we sum them.
    # But wait, the user agrees with the Mov Mina (1800 kTon).
    # We trust the Inputs displayed: 1800 kTon, 496 kTon.
    
    # Costos - This is the Tricky Part
    # Costos sheet has different structure?
    # app.py: "df_costos['Año'] = df_costos['Año'].ffill()"
    # It reads with header=2. So columns are actual names.
    # Let's look for a row where Year=2026, Month=Enero
    
    df_costos.columns = [str(c).strip() for c in df_costos.columns]
    # Rename 'Año' if needed, or find it.
    
    # Filter
    # Try to find the row
    row_cost = None
    for idx, r in df_costos.iterrows():
        # Check Year/Month cols. Column names might be 'Año', 'Periodo'
        # Let's inspect columns in the output
        pass

    return trat_ton, df_costos

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"
trat, df_c = load_data_jan_2026(file_path)

print(f"\nRaw Trat (Planta Sheet): {trat}")

# Analyze Costos
print("\n--- Costos Sheet Sample ---")
print(df_c.head())

# Find Jan 2026 Costs
# Assuming columns 'Año', 'Periodo', 'Costo Mina', 'Costo Planta'
try:
    if 'Año' not in df_c.columns and 'Year' not in df_c.columns:
        # Fallback: maybe Col 0 is Year
        df_c['Year'] = df_c.iloc[:,0].ffill()
        df_c['Periodo'] = df_c.iloc[:,1]
    
    # Normalize
    # df_c['Year'] = pd.to_numeric(df_c['Año'], errors='coerce')
    
    jan_2026 = df_c[ (df_c['Año'] == 2026) & (df_c['Periodo'].astype(str).str.contains('Ene')) ]
    if not jan_2026.empty:
        print("\n--- FOUND JAN 2026 COST ---")
        print(jan_2026.iloc[0])
        
        c_mina = jan_2026.iloc[0]['Costo Mina']
        c_planta = jan_2026.iloc[0]['Costo Planta']
        
        # CALCULATION
        # Inputs from User Screenshot (KPIs)
        mov_kton = 1800.0
        trat_kton = 496.0
        rem_kton = 467.0
        
        # User saw: 3.84 and 13.8 in cards.
        print(f"\n--- CALCULATION CHECK ---")
        print(f"Mov: {mov_kton} kTon")
        print(f"Trat: {trat_kton} kTon")
        print(f"Remanejo: {rem_kton} kTon")
        print(f"Costo Mina: {c_mina} $/t")
        print(f"Costo Planta: {c_planta} $/t")
        print(f"Costo Remanejo (Sim): 1.21 $/t")
        
        gasto_mina = mov_kton * c_mina
        gasto_planta = trat_kton * c_planta
        gasto_rem = rem_kton * 1.21
        
        total = gasto_mina + gasto_planta + gasto_rem
        core = gasto_mina + gasto_planta
        
        print(f"\nGasto Mina: {gasto_mina:.3f} M$")
        print(f"Gasto Planta: {gasto_planta:.3f} M$")
        print(f"Gasto Remanejo: {gasto_rem:.3f} M$")
        print(f"---------------------------")
        print(f"TOTAL (Inc. Rem): {total:.3f} M$")
        print(f"CORE (Exc. Rem): {core:.3f} M$")
        
    else:
        print("Could not find Jan 2026 row in Costos")

except Exception as e:
    print(f"Error processing costs: {e}")
