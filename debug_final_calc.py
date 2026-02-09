import pandas as pd

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

try:
    print("READING DATA TECNICA...")
    # Header logic from app.py: header=13 for Data Tecnica? No, process_data_tecnica_full loads it.
    # Let's peek row 12 which usually has headers.
    df_dt = pd.read_excel(file_path, sheet_name='Data Tecnica', header=None)
    
    # We need Total Movement (Sum of Phases) for Jan 2026.
    # Jan 2026 is likely col Index 3 (based on Planta sheet consistency).
    
    # We need to sum specific rows. Usually "Total Movimiento" is a summary row.
    # Let's find it.
    row_mov_total = None
    for i in range(50):
        val = str(df_dt.iloc[i, 0])
        val2 = str(df_dt.iloc[i, 1])
        if "Mov" in val and "Total" in val:
             print(f"FOUND MOV TOTAL ROW: {i} | {val} | {val2}")
             row_mov_total = i
             break
    
    # Costs
    df_costos = pd.read_excel(file_path, sheet_name='Costos', header=2)
    # Jan 26 is row 0
    cost_mina = df_costos.iloc[0]['Costo Mina']
    cost_planta = df_costos.iloc[0]['Costo Planta']
    
    # Planta Trat
    df_planta = pd.read_excel(file_path, sheet_name='Planta', header=None)
    # Row 14, Col 3 (Index 3 hopefully)
    trat_jan = df_planta.iloc[14, 3] # Check if index 3 is indeed Jan 26.
    
    # Get Mov Total Value
    mov_total_jan = df_dt.iloc[row_mov_total, 3] if row_mov_total else 0
    
    # Logic from app.py: Mov Total is in kTon. Convert to Ton.
    mov_ton = mov_total_jan * 1000
    
    print("\n--- FINAL CALC ---")
    print(f"Mov Total (Ton): {mov_ton:,.0f}")
    print(f"Trat Planta (Ton): {trat_jan:,.0f}")
    print(f"Costo Mina ($/t mov): {cost_mina:.2f}")
    print(f"Costo Planta ($/t trat): {cost_planta:.2f}")
    
    cost_m = mov_ton * cost_mina
    cost_p = trat_jan * cost_planta
    total = cost_m + cost_p
    
    unit_cost_eq = total / trat_jan
    print(f"Equivalent Unit Cost ($/t milled): {unit_cost_eq:.2f}")

except Exception as e:
    print(f"Error: {e}")
