import pandas as pd
import fleet_v3
import os

def check_perfos_data():
    file_path = "plan_budget_real.xlsx"
    if not os.path.exists(file_path):
        print("❌ Archivo Excel no encontrado.")
        return

    print(f"📂 Cargando datos desde {file_path}...")
    try:
        # Load using the same function app.py uses
        df_aux = fleet_v3.load_kpi_perfos_servicios(file_path)
        
        if df_aux.empty:
            print("⚠️ El DataFrame devuelto por load_kpi_perfos_servicios está VACÍO.")
            return

        # Filter for Perfos
        df_perfos = df_aux[df_aux['Category'] == 'Perfos'].copy()
        
        print(f"\n✅ Total Rows in Perfos: {len(df_perfos)}")
        
        if len(df_perfos) > 0:
            print("\n🔍 Muestra de Datos (Head):")
            print(df_perfos.head().to_string())
            
            print("\n📅 Años Disponibles:")
            print(df_perfos['Year'].unique())
            
            print("\n🏷️ Categorías de Métricas (MetricCategory):")
            print(df_perfos['MetricCategory'].unique())
            
            print("\n📄 Métricas Específicas (Metric):")
            print(df_perfos['Metric'].unique()[:20]) # Show first 20
            
            # Simulate the User's Query Logic
            print("\n🧪 Simulando Filtro 2026 + Horas/Uso:")
            
            # Check 2026
            df_2026 = df_perfos[df_perfos['Year'] == 2026]
            print(f"  - Registros 2026: {len(df_2026)}")
            
            if not df_2026.empty:
                # Check 'Horas'
                df_horas = df_2026[df_2026['MetricCategory'].str.contains('Horas', case=False, na=False)]
                print(f"  - Registros 'Horas': {len(df_horas)}")
                if len(df_horas) > 0:
                     print(f"    Ejemplos: {df_horas['MetricCategory'].unique()}")

                # Check 'Uso' / 'Disponibilidad'
                df_uso = df_2026[df_2026['MetricCategory'].str.contains('Uso|Disponibilidad', case=False, na=False)]
                print(f"  - Registros 'Uso/Disp': {len(df_uso)}")
                if len(df_uso) > 0:
                     print(f"    Ejemplos: {df_uso['MetricCategory'].unique()}")
        else:
            print("⚠️ No hay datos de categoría 'Perfos'.")

    except Exception as e:
        print(f"❌ Error cargando datos: {e}")

if __name__ == "__main__":
    check_perfos_data()
