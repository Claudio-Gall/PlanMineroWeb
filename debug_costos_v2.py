import pandas as pd
import os
import traceback

file_path = 'plan_budget_real.xlsx'

print(f"Start debugging {file_path}")

try:
    import openpyxl
    print("openpyxl is installed")
except ImportError:
    print("openpyxl is MISSING")

try:
    df = pd.read_excel(file_path, sheet_name='Costos', header=2, engine='openpyxl')
    print("Read success")
    print("Columns:", list(df.columns))
    print(df.head())
except Exception:
    traceback.print_exc()

print("Done")
