import pandas as pd
import ai_loader

file_path = "plan_budget_real.xlsx"

print("--- INSPECTING SHEET: PALA-FASE ---")
try:
    df_raw = ai_loader.load_sheet_smart(file_path, 'Pala-Fase')
    print("COLUMNS FOUND:")
    print(df_raw.columns.tolist())
    
    # Try to find headers based on screenshot: Origen, Pala, Fase
    # The default loader might have named them Col_0, Col_1, Col_2
    print("\nSAMPLE HEAD (First 5 rows):")
    print(df_raw.head().to_dict())
    
    print("\nSEARCHING FOR 'F05' in any column:")
    # Check first few columns
    for col in df_raw.columns[:5]:
        mask = df_raw[col].astype(str).str.contains("F05", case=False, na=False)
        if mask.any():
            print(f"Found F05 in column: {col}")
            print(df_raw[mask].iloc[:5][[col] + list(df_raw.columns[5:10])])

except Exception as e:
    print(f"Error loading Pala-Fase: {e}")
