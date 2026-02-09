import pandas as pd

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

try:
    print("READING PLANTA...")
    df_planta = pd.read_excel(file_path, sheet_name='Planta', header=None)
    
    # Print first 60 rows, col A and B
    for i in range(60):
        val1 = str(df_planta.iloc[i, 0]).strip()
        val2 = str(df_planta.iloc[i, 1]).strip()
        # Clean newlines
        val1 = val1.replace('\n', ' ')
        val2 = val2.replace('\n', ' ')
        print(f"Row {i}: {val1} | {val2}")

    print("\nREADING COSTOS...")
    df_costos = pd.read_excel(file_path, sheet_name='Costos', header=2)
    print(df_costos.head(10))

except Exception as e:
    print(f"Error: {e}")
