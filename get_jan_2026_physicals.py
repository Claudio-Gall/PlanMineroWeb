import pandas as pd

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

# Load Planta with header=13 (Standard from app.py)
df = pd.read_excel(file_path, sheet_name='Planta', header=None)

# Let's find rows for 'Tratamiento' (Row 14 approx), 'Mov Total' (Row 10 approx), 'Recup' (Row 19 approx)
# Indices in pandas are 0-based.
# Excel Row 14 -> Index 13.
# Excel Row 19 -> Index 18.

# Let's print rows 10-25 to identify them and the Jan-26 Column index.
print(df.iloc[10:25, 0:5]) 

# Also check columns to find Jan-26
print("\n--- COLUMN HEADERS (Row 12 presumably) ---")
print(df.iloc[12, 0:15])
