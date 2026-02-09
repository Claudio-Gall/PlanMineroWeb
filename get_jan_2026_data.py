import pandas as pd
import openpyxl

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

try:
    # 1. Load Planta (Physicals)
    df_planta = pd.read_excel(file_path, sheet_name='Planta', header=13)
    # Filter Jan 2026. Assuming Month columns or row structure.
    # Actually, based on previous interactions, 'Planta' sheet has months in rows or cols?
    # Let's just look at the raw structure first lines to be sure.
    # But usually 'Periodo' or 'Month' column.
    
    # 2. Load Costos
    df_costos = pd.read_excel(file_path, sheet_name='Costos', header=2)
    
    # Print heads to identify Jan 2026
    print("--- PLANTA HEAD ---")
    print(df_planta.head())
    
    print("\n--- COSTOS HEAD ---")
    print(df_costos.head())

except Exception as e:
    print(f"Error: {e}")
