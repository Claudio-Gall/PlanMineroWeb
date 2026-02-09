import pandas as pd

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

try:
    print("LOADING COSTOS SHEET...")
    # Load with header at row 2 (0-indexed) as per app.py
    df_costos = pd.read_excel(file_path, sheet_name='Costos', header=2, engine='openpyxl')
    
    print("\n--- RAW COLUMNS ---")
    print(df_costos.columns.tolist())
    
    if 'Año' in df_costos.columns:
        df_costos['Año'] = df_costos['Año'].ffill()
        df_costos['Year'] = pd.to_numeric(df_costos['Año'], errors='coerce').fillna(0).astype(int)
        
        print("\n--- YEARS FOUND ---")
        print(df_costos['Year'].unique())
        
        # Check 2028/2029
        for y in [2028, 2029]:
            print(f"\n--- DATA FOR {y} ---")
            df_y = df_costos[df_costos['Year'] == y]
            if df_y.empty:
                print(f"No data for {y}")
            else:
                print("Unique Periodos found:")
                if 'Periodo' in df_y.columns:
                    print(df_y['Periodo'].unique())
                else:
                    print(f"Column 'Periodo' not found. Available: {df_y.columns}")
                
                print("Sample Rows:")
                cols_to_show = ['Year', 'Periodo', 'Costo Mina', 'Costo Planta']
                found_cols = [c for c in cols_to_show if c in df_y.columns]
                print(df_y[found_cols].head())

except Exception as e:
    print(f"Error: {e}")
