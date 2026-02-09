import pandas as pd
import fleet_loader

file_path = r"c:\PROYECTOS\Proyecto_Plan_Minero\plan_budget_real.xlsx"

try:
    print("LOADING FLEET DATA...")
    df_fleet = fleet_loader.load_fleet_clean(file_path)
    
    year_target = 2027
    print(f"\n--- DATA FOR {year_target} ---")
    df_year = df_fleet[df_fleet['Year'] == year_target]
    
    if df_year.empty:
        print(f"No data found for {year_target}")
    else:
        valid_loaders = ['Pala 06', 'Pala 05', 'Pala 04', 'Pala 03']
        df_filt = df_year[df_year['Equipo'].isin(valid_loaders)]
        
        print(f"Rows for Palas: {len(df_filt)}")
        if not df_filt.empty:
            print("Max Ton:", df_filt['Ton'].max())
            print("Min Ton:", df_filt['Ton'].min())
            print(df_filt[['Month', 'Equipo', 'Fase', 'Ton']].head(10))
        else:
            print("No Palas found in 2027 (maybe they are retired or renamed?)")
            print("Available Equipos in 2027:", df_year['Equipo'].unique())

except Exception as e:
    print(f"Error: {e}")
