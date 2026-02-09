import pandas as pd
import traceback

file_path = 'plan_budget_real.xlsx'

try:
    print("\n--- LOADING COSTOS ---")
    df = pd.read_excel(file_path, sheet_name='Costos', header=2, engine='openpyxl')
    
    # Simulate App Logic
    if 'Año' in df.columns:
        print("Running ffill on 'Año'...")
        df['Año'] = df['Año'].ffill()
    else:
        print("'Año' column missing!")

    def clean_cost_period(p):
        p = str(p).strip()
        if "1er Trimestre" in p: return "Q1"
        if "2do Trimestre" in p: return "Q2"
        if "3er Trimestre" in p: return "Q3"
        if "4to Trimestre" in p: return "Q4"
        return p

    df['Year'] = pd.to_numeric(df['Año'], errors='coerce').fillna(0).astype(int)
    df['Period_Clean'] = df['Periodo'].apply(clean_cost_period)

    # Check Types
    print("\n--- DTYPES ---")
    print(df.dtypes)

    # Check Sample Keys
    print("\n--- SAMPLE KEYS GENERATED ---")
    for i in range(min(5, len(df))):
        r = df.iloc[i]
        print(f"Row {i}: Year={r['Year']} (Type: {type(r['Year'])}) | Period={repr(r['Period_Clean'])} | CostoMina={r['Costo Mina']}")

    # Check a specific match expected
    target_key = (2026, 'Enero')
    print(f"\n--- LOOKING FOR {target_key} ---")
    
    found = False
    for _, row in df.iterrows():
        k = (row['Year'], row['Period_Clean'])
        if k == target_key:
            print(f"✅ FOUND! {k} -> Mina: {row['Costo Mina']}, Planta: {row['Costo Planta']}")
            found = True
            break
    
    if not found:
        print("❌ NOT FOUND!")

except Exception:
    traceback.print_exc()
