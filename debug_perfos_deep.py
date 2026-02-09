import pandas as pd
import fleet_v3
import os

def check_perfos_deep():
    file_path = "plan_budget_real.xlsx"
    df_aux = fleet_v3.load_kpi_perfos_servicios(file_path)
    df_perfos = df_aux[df_aux['Category'] == 'Perfos'].copy()
    
    print(f"Total Rows: {len(df_perfos)}")
    print(f"Unique Years: {sorted(df_perfos['Year'].unique())}")
    
    print("\n--- Unique MetricCategories ---")
    cats = sorted(df_perfos['MetricCategory'].unique())
    for c in cats:
        print(f"'{c}'")

    print("\n--- Unique Items (Equipos) ---")
    items = sorted(df_perfos['Item'].unique())
    for i in items:
        print(f"'{i}'")

if __name__ == "__main__":
    check_perfos_deep()
