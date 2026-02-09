import pandas as pd
import fleet_loader

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

try:
    print("LOADING FLEET DATA...")
    df_fleet = fleet_loader.load_fleet_clean(file_path)
    
    if not df_fleet.empty:
        print("\n--- COLUMNS ---")
        print(df_fleet.columns.tolist())
        
        print("\n--- SAMPLE DATA (2026) ---")
        df_2026 = df_fleet[df_fleet['Year'] == 2026]
        if not df_2026.empty:
             print(df_2026[['Month', 'Equipo', 'Fase', 'Kton']].head(20))
             
             print("\n--- UNIQUE FASES ---")
             print(df_2026['Fase'].unique())
             
             print("\n--- UNIQUE EQUIPOS ---")
             print(df_2026['Equipo'].unique())
        else:
            print("No data for 2026 found.")
            print("Years available:", df_fleet['Year'].unique())

except Exception as e:
    print(f"Error: {e}")
