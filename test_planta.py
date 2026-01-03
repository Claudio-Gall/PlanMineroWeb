import pandas as pd
import numpy as np

# Read Planta sheet
df_raw = pd.read_excel('plan_budget_real.xlsx', sheet_name='Planta', header=0)

print("=" * 60)
print("PLANTA SHEET ANALYSIS")
print("=" * 60)
print(f"\nShape: {df_raw.shape}")
print(f"\nFirst row (years): {df_raw.iloc[0, :10].tolist()}")
print(f"Second row (months): {df_raw.iloc[1, :10].tolist()}")

# Show structure
print("\nFirst 3 columns of first 15 rows:")
print(df_raw.iloc[:15, :3].to_string())

# Try to find Cobre Fino row
print("\n" + "=" * 60)
print("SEARCHING FOR KEY ROWS")
print("=" * 60)
for idx in range(min(31, len(df_raw))):
    row_label = " ".join([str(x).lower().strip() for x in df_raw.iloc[idx, :3] if pd.notna(x)])
    if 'cobre' in row_label:
        print(f"Row {idx}: {row_label}")
        print(f"  Sample values: {df_raw.iloc[idx, 3:8].tolist()}")
