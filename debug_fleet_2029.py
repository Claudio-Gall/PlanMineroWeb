import pandas as pd
import fleet_loader

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

try:
    print("LOADING FLEET DATA...")
    df_fleet = fleet_loader.load_fleet_clean(file_path)
    
    year_target = 2029
    print(f"\n--- DATA FOR {year_target} ---")
    df_year = df_fleet[df_fleet['Year'] == year_target]
    
    if df_year.empty:
        print(f"No data found for {year_target}")
    else:
        print("Unique Months:", df_year['Month'].unique())
        print("Unique Periodos:", df_year['Periodo'].unique())
        print("\nSample Data:")
        print(df_year[['Month', 'Periodo', 'Equipo', 'Fase', 'Ton']].head(10))

except Exception as e:
    print(f"Error: {e}")
