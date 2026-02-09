import pandas as pd
import fleet_v3
import os

def check_pivot_logic():
    file_path = "plan_budget_real.xlsx"
    print(f"📂 Cargando datos...")
    
    # 1. Load Data
    df_aux = fleet_v3.load_kpi_perfos_servicios(file_path)
    df = df_aux[df_aux['Category'] == 'Perfos'].copy()
    
    # 2. Filter 2026
    df_2026 = df[df['Year'] == 2026].copy()
    print(f"Registros 2026: {len(df_2026)}")

    # 3. Simulate The Split Aggregation Logic
    print("\n🔄 Ejecutando Lógica de Pivote Dividido...")
    
    # Filter SUM metrics
    mask_sum = df_2026['MetricCategory'].str.contains('Horas|Ton|Mts', case=False, na=False)
    metrics_sum = df_2026[mask_sum]
    print(f"  - Registros SUM (Horas, etc): {len(metrics_sum)}")
    
    df_sum = metrics_sum.pivot_table(index='Item', columns='MetricCategory', values='Value', aggfunc='sum')
    print(f"  - Columnas SUM resultantes: {df_sum.columns.tolist()}")

    # Filter MEAN metrics
    mask_mean = df_2026['MetricCategory'].str.contains('Uso|Disp|%', case=False, na=False)
    metrics_mean = df_2026[mask_mean]
    print(f"  - Registros MEAN (Uso, Disp): {len(metrics_mean)}")
    
    df_mean = metrics_mean.pivot_table(index='Item', columns='MetricCategory', values='Value', aggfunc='mean')
    print(f"  - Columnas MEAN resultantes: {df_mean.columns.tolist()}")

    # Join
    result = df_sum.join(df_mean, how='outer')
    result = result.reset_index()
    
    print("\n✅ Tabla Final (Columnas):")
    print(result.columns.tolist())
    
    print("\n🔍 Primeras 5 filas:")
    print(result.head().to_string())

if __name__ == "__main__":
    check_pivot_logic()
