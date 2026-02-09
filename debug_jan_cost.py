import pandas as pd

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

try:
    # 1. Load Data
    df_planta = pd.read_excel(file_path, sheet_name='Planta', header=None)
    df_costos = pd.read_excel(file_path, sheet_name='Costos', header=2)
    
    # 2. Extract Jan 2026 Physicals
    # Row 10 is 'Mov Total' (Index 10 in pandas? let's look at previous output which showed Row 10)
    # Previous output: Index 10 is 'Planta'.... wait.
    # Previous output showed:
    # 14 Planta Total ... 496300 (Trat Jan 26)
    # The output didn't clearly label 'Mov Total'. I need to find the specific row for Total Movement.
    # Usually it is row 10 or 11.
    
    # Let's verify row 10 (Index 9 or 10) from previous output...
    # Previous output showed Row 10 as "Planta Stock ... 76.25". That's not Mov Total.
    
    # I will search for "Mov Total" or similar in column A.
    
    row_mov_total = None
    row_trat = None
    
    col_jan_26 = 3 # From previous output, Jan 26 was column index 3 (value 496300 in row 14)
    
    for i in range(20):
        val = str(df_planta.iloc[i, 0])
        val2 = str(df_planta.iloc[i, 1])
        if "Mov" in val or "Mov" in val2:
            print(f"Found Mov at Row {i}: {val} | {val2}")
            if "Total" in val or "Total" in val2:
                row_mov_total = i
                
        if "Trat" in val or "Total" in val2: # Row 14 was Planta Total
             if i == 14: row_trat = 14
             
    # 3. Extract Values
    mov_total = df_planta.iloc[row_mov_total, col_jan_26] if row_mov_total is not None else 0
    trat_planta = df_planta.iloc[row_trat, col_jan_26] if row_trat is not None else 0
    
    # Costos
    # Jan 2026 is Row 0 in df_costos based on previous output
    cost_mina_unit = df_costos.iloc[0]['Costo Mina']
    cost_planta_unit = df_costos.iloc[0]['Costo Planta']
    
    print(f"\n--- CALCULATION DATA (Jan 26) ---")
    print(f"Mov Total (kTon?): {mov_total}")
    print(f"Trat Planta (Ton): {trat_planta}")
    print(f"Costo Mina ($/t mov): {cost_mina_unit}")
    print(f"Costo Planta ($/t trat): {cost_planta_unit}")
    
    # 4. Do the Math
    # Assuming Mov Total is kTon (from app.py logic `safe_sum('Mov_Total')*1000`)
    # But let's check magnitude. If it is 2460.66, that's likely kTon (2.4M Ton).
    
    mov_ton = mov_total * 1000
    cost_mina_total = mov_ton * cost_mina_unit
    cost_planta_total = trat_planta * cost_planta_unit
    
    total_cost = cost_mina_total + cost_planta_total
    equiv_unit_cost = total_cost / trat_planta
    
    print(f"\nTotal Cost Mina: ${cost_mina_total:,.0f}")
    print(f"Total Cost Planta: ${cost_planta_total:,.0f}")
    print(f"Grand Total Cost: ${total_cost:,.0f}")
    print(f"Equivalent Unit Cost ($/t milled): ${equiv_unit_cost:.2f}")

except Exception as e:
    print(f"Error: {e}")
