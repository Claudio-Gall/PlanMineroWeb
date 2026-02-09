import pandas as pd
import numpy as np

file_path = "plan_budget_real.xlsx"

try:
    print("LOADING COSTOS SIMPLE...")
    df_costos = pd.read_excel(file_path, sheet_name='Costos', header=2, engine='openpyxl')
    
    print("\n--- RAW COLUMNS ---")
    print(df_costos.columns.tolist())
    
    # Check cleaning logic
    df_costos.columns = [str(c).strip() for c in df_costos.columns]
    print("\n--- CLEAN COLUMNS ---")
    print(df_costos.columns.tolist())
    
    if 'Costo Mina' not in df_costos.columns:
        print("CRITICAL: 'Costo Mina' column NOT FOUND (Check spaces inside string?)")
        # fuzzy match check
        for c in df_costos.columns:
            if 'mina' in c.lower():
                print(f"Did you mean '{c}'?")
    else:
        print("Costo Mina found.")
        print("Data Type:", df_costos['Costo Mina'].dtype)
        print("Sample Values:")
        print(df_costos['Costo Mina'].head(10))
        
        # Check specific rows for 2028
        if 'Año' in df_costos.columns:
            df_costos['Año'] = df_costos['Año'].ffill()
            df_costos['Year'] = pd.to_numeric(df_costos['Año'], errors='coerce').fillna(0).astype(int)
            
            df_2028 = df_costos[df_costos['Year'] == 2028]
            print("\n--- 2028 RAW DATA ---")
            print(df_2028[['Periodo', 'Costo Mina', 'Costo Planta']])
            
            # Check Clean Period Logic
            def clean_cost_period(p):
                p = str(p).strip().lower()
                if "1er" in p or "q1" in p: return "Q1"
                if "2do" in p or "q2" in p: return "Q2"
                if "3er" in p or "q3" in p: return "Q3"
                if "4to" in p or "q4" in p: return "Q4"
                return p.title()
                
            df_2028['Period_Clean'] = df_2028['Periodo'].apply(clean_cost_period)
            print("\n--- 2028 CLEAN PERIODS ---")
            print(df_2028[['Periodo', 'Period_Clean']])

except Exception as e:
    print(f"Error: {e}")
