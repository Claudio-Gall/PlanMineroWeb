import pandas as pd
import numpy as np

file_path = "plan_budget_real.xlsx"

try:
    print("LOADING COSTOS...")
    df_costos = pd.read_excel(file_path, sheet_name='Costos', header=2, engine='openpyxl')
    
    # EXACT LOGIC FROM app.py
    df_costos.columns = [str(c).strip() for c in df_costos.columns] # Force column trim
    if 'Año' in df_costos.columns:
        df_costos['Año'] = df_costos['Año'].ffill()
    
    # Normalize Cost Helpers
    def clean_cost_period(p):
        p = str(p).strip().lower()
        # Handle "1er Trimestre_2028" etc.
        if "1er" in p or "q1" in p: return "Q1"
        if "2do" in p or "q2" in p: return "Q2"
        if "3er" in p or "q3" in p: return "Q3"
        if "4to" in p or "q4" in p: return "Q4"
        # Handle months if needed (Capitalize for consistency)
        return p.title()

    df_costos['Year'] = pd.to_numeric(df_costos['Año'], errors='coerce').fillna(0).astype(int)
    df_costos['Period_Clean'] = df_costos['Periodo'].apply(clean_cost_period)
    
    # Create Lookup: (Year, Period) -> {Mina, Planta}
    cost_lookup = {}
    for _, row in df_costos.iterrows():
        key = (row['Year'], row['Period_Clean'])
        cost_lookup[key] = {
            'mina': pd.to_numeric(row['Costo Mina'], errors='coerce') or 0.0,
            'planta': pd.to_numeric(row['Costo Planta'], errors='coerce') or 0.0
        }

    print("\n--- LOOKUP KEYS FOR 2028 ---")
    keys_2028 = [k for k in cost_lookup.keys() if k[0] == 2028]
    print(keys_2028)
    
    if keys_2028:
        print("Sample Value:", cost_lookup[keys_2028[0]])
    else:
        print("NO KEYS FOR 2028 FOUND!")

    print("\n--- LOOKUP KEYS FOR 2029 ---")
    keys_2029 = [k for k in cost_lookup.keys() if k[0] == 2029]
    print(keys_2029)

except Exception as e:
    print(f"Error: {e}")
