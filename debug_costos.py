import pandas as pd
import os

file_path = 'plan_budget_real.xlsx'

print(f"--- DEBUGGING {file_path} ---")

if not os.path.exists(file_path):
    print("❌ ERROR: File not found!")
    exit()

try:
    # 1. Inspect Raw Structure (First 10 rows)
    print("\n--- RAW CONTENT (Header=None, First 10 rows) ---")
    df_raw = pd.read_excel(file_path, sheet_name='Costos', header=None, nrows=10, engine='openpyxl')
    print(df_raw.to_markdown())

    # 2. Inspect Expected Header (Row 3 -> Index 2)
    print("\n--- ATTEMPTING HEADER=2 ---")
    df_h2 = pd.read_excel(file_path, sheet_name='Costos', header=2, engine='openpyxl')
    print(f"Columns: {df_h2.columns.tolist()}")
    print("First 5 rows:")
    print(df_h2.head().to_markdown())

    # 3. Check for 'Año' behavior
    if 'Año' in df_h2.columns:
        print("\n--- 'Año' Column Sample (Before ffill) ---")
        print(df_h2['Año'].head(15).tolist())
    else:
        print("\n❌ 'Año' column NOT FOUND. Check column names above.")

except Exception as e:
    print(f"\n❌ ERROR READING FILE: {e}")
