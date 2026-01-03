"""
Fleet V3 - Auxiliary Functions
Provides support functions for Perfos and Servicios data loading
"""
import pandas as pd
import numpy as np


def load_kpi_perfos_servicios(file_path):
    """
    Load Perfos and Servicios data from their respective sheets
    These sheets have a PIVOTED structure: years/periods in columns, metrics in rows
    Returns DataFrame with columns: Year, Month, Quarter, Category, Item, Metric, Value, Sheet
    """
    result_data = []
    
    # Month mapping
    month_map = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 
        'MAYO': 5, 'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8,
        'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    
    # Load and transform KPI-Perfos
    try:
        df = pd.read_excel(file_path, sheet_name='KPI-Perfos', header=None)
        
        # Row 0: Years, Row 1: Periods, Row 2+: Metrics
        years_row = df.iloc[0, 2:].tolist()  # Skip first 2 columns
        periods_row = df.iloc[1, 2:].tolist()
        
        # Build column mapping
        col_map = []
        for i, (year_val,period_val) in enumerate(zip(years_row, periods_row)):
            # Skip Total columns
            if pd.isna(year_val) or pd.isna(period_val):
                continue
            if 'TOTAL' in str(year_val).upper() or 'TOTAL' in str(period_val).upper():
                continue
            
            try:
                year = int(year_val) if isinstance(year_val, (int, float)) else int(str(year_val).replace('.0', ''))
            except:
                continue  # Skip if can't convert to year
            
            period_str = str(period_val).upper().strip()
            
            month_num = None
            quarter = None
            
            if period_str in month_map:
                month_num = month_map[period_str]
                quarter = f"Q{(month_num - 1) // 3 + 1}"
            elif 'TRIMESTRE' in period_str:
                if '1ER' in period_str:
                    month_num, quarter = 1, 'Q1'
                elif '2DO' in period_str:
                    month_num, quarter = 4, 'Q2'
                elif '3ER' in period_str:
                    month_num, quarter = 7, 'Q3'
                elif '4TO' in period_str:
                    month_num, quarter = 10, 'Q4'
            
            if month_num:
                col_map.append({
                    'col_idx': i + 2,  # +2 because we skipped first 2 cols
                    'year': year,
                    'month': month_num,
                    'quarter': quarter
                })
        
        # Extract data from each metric row (starting from row 2)
        for row_idx in range(2, min(df.shape[0], 100)):
            # Column A: Main category (e.g., "Horas Operativas", "Mts Producción")
            category_name = df.iloc[row_idx, 0]
            # Column B: Sub-item/equipment (e.g., "PV-5", "DMM3-03", "Total Flota")
            subitem_name = df.iloc[row_idx, 1]
            
            # Skip if both are empty
            if pd.isna(category_name) and pd.isna(subitem_name):
                continue
            
            # Build metric name
            if pd.notna(category_name) and pd.notna(subitem_name):
                metric_name = f"{str(category_name).strip()} - {str(subitem_name).strip()}"
                item_name = str(subitem_name).strip()
                metric_category = str(category_name).strip()
            elif pd.notna(subitem_name):
                metric_name = str(subitem_name).strip()
                item_name = metric_name
                metric_category = "General"
            else:
                continue
            
            # Extract value for each time period
            for col_info in col_map:
                val = pd.to_numeric(df.iloc[row_idx, col_info['col_idx']], errors='coerce')
                
                if pd.notna(val) and val != 0:
                    result_data.append({
                        'Year': col_info['year'],
                        'Month': col_info['month'],
                        'Quarter': col_info['quarter'],
                        'Category': 'Perfos',
                        'Item': item_name,
                        'Metric': metric_name,
                        'MetricCategory': metric_category,
                        'Value': float(val),
                        'Sheet': 'KPI-Perfos'
                    })
        
        print(f"  Perfos: {len([r for r in result_data if r['Category']=='Perfos'])} records")
    except Exception as e:
        print(f"  Perfos: Error loading ({str(e)})")
    
    # Load and transform KPI-Servicios (different structure: years in row 1, periods in row 2)
    try:
        df = pd.read_excel(file_path, sheet_name='KPI-Servicios', header=None)
        
        # Row 1: Years (skip first 2 cols which have "Año")
        # Row 2: Periods (skip first 2 cols which have "Periodo")
        years_row = df.iloc[1, 2:].tolist()
        periods_row = df.iloc[2, 2:].tolist()
        
        col_map = []
        for i, (year_val, period_val) in enumerate(zip(years_row, periods_row)):
            # Skip Total columns
            if pd.isna(year_val) or pd.isna(period_val):
                continue
            if 'TOTAL' in str(year_val).upper() or 'TOTAL' in str(period_val).upper():
                continue
            
            try:
                year = int(year_val) if isinstance(year_val, (int, float)) else int(str(year_val).replace('.0', ''))
            except:
                continue
            
            period_str = str(period_val).upper().strip()
            
            month_num = None
            quarter = None
            
            if period_str in month_map:
                month_num = month_map[period_str]
                quarter = f"Q{(month_num - 1) // 3 + 1}"
            elif 'TRIMESTRE' in period_str:
                if '1ER' in period_str:
                    month_num, quarter = 1, 'Q1'
                elif '2DO' in period_str:
                    month_num, quarter = 4, 'Q2'
                elif '3ER' in period_str:
                    month_num, quarter = 7, 'Q3'
                elif '4TO' in period_str:
                    month_num, quarter = 10, 'Q4'
            
            if month_num:
                col_map.append({
                    'col_idx': i + 2,
                    'year': year,
                    'month': month_num,
                    'quarter': quarter
                })
        
        # Equipment data starts around row 44
        # Column B: Category (e.g., "Horas Operativas")
        # Column C: Equipment name
        for row_idx in range(3, min(df.shape[0], 100)):
            # Column B: Category
            category = df.iloc[row_idx, 1]
            # Column C: Equipment name
            equip_cell = df.iloc[row_idx, 2]
            
            if pd.isna(equip_cell):
                continue
                
            equip_str = str(equip_cell).strip()
            
            # Skip if contains "Total" or is empty
            if 'TOTAL' in equip_str.upper() or not equip_str:
                continue
            
            # Only process "Horas Operativas" rows
            if pd.notna(category) and 'HORAS OPERATIVAS' in str(category).upper():
                item_name = equip_str
                
                for col_info in col_map:
                    val = pd.to_numeric(df.iloc[row_idx, col_info['col_idx']], errors='coerce')
                    
                    if pd.notna(val) and val != 0:
                        result_data.append({
                            'Year': col_info['year'],
                            'Month': col_info['month'],
                            'Quarter': col_info['quarter'],
                            'Category': 'Servicios',
                            'Item': item_name,
                            'Metric': f"Horas - {item_name}",
                            'MetricCategory': 'Horas Operativas',
                            'Value': float(val),
                            'Sheet': 'KPI-Servicios'
                        })
        
        print(f"  Servicios: {len([r for r in result_data if r['Category']=='Servicios'])} records")
    except Exception as e:
        print(f"  Servicios: Error loading ({str(e)})")
    
    return pd.DataFrame(result_data)


def load_fleet_data_v3_hybrid(file_path):
    """
    Legacy function - redirects to fleet_loader for compatibility
    """
    try:
        import fleet_loader
        return fleet_loader.load_fleet_clean(file_path)
    except:
        return pd.DataFrame()
