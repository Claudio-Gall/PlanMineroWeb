import pandas as pd
import io
import numpy as np

def load_sheet_smart(file_path, sheet_name):
    """
    Loads an Excel sheet in 'Smart Raw' mode for AI Context.
    1. Detects if it's Vertical (KPI style) or Horizontal (Time-series).
    2. Cleans extra headers and empty space.
    3. Optimizes for token usage (Markdown format).
    """
    try:
        # SPECIAL CASE: KPI-Palas has vertical structure with proper headers in row 2
        # SPECIAL CASE: KPI-Palas has vertical structure with COMPLEX headers (Rows 0, 1, 2)
        if 'KPI-Palas' in sheet_name or 'KPI-Servicios' in sheet_name or 'KPI-Perfos' in sheet_name or 'KPI-Camiones' in sheet_name:
            # Read first 3 rows to understand structure
            # Row 0: "Pala", "Pala"...
            # Row 1: "P06", "P06"... (Specific Equipment Name)
            # Row 2: "Ton", "%UsoReal"... (Metrics)
            df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            # Extract header rows
            row_eq_type = df_raw.iloc[0] # e.g. Pala
            row_eq_name = df_raw.iloc[1] # e.g. P06
            row_metrics = df_raw.iloc[2] # e.g. %UsoReal
            
            new_cols = []
            last_valid_name = "Unknown"
            
            for i in range(len(df_raw.columns)):
                metric = str(row_metrics[i]).strip()
                eq_name = str(row_eq_name[i]).strip()
                
                # Handle merged cells logic (fill forward)
                if eq_name == 'nan' or eq_name == '':
                    eq_name = last_valid_name  # Inherit from left if merged
                else:
                    last_valid_name = eq_name
                
                # Construct unique column name
                if metric in ['Año', 'Periodo', 'N° Dias']:
                    new_cols.append(metric) # Keep base columns simple
                elif eq_name and eq_name not in ['nan', 'Unknown']:
                    # Combine Name + Metric -> P06_%UsoReal
                    clean_metric = metric.replace(' ', '').replace('%', 'Pct')
                    new_cols.append(f"{eq_name}_{metric}")
                else:
                    new_cols.append(metric)
                    
            # Reload with proper header skip
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
            df.columns = new_cols
            
            # Drop completely NaN rows/cols
            df.dropna(how='all', axis=0, inplace=True)
            df.dropna(how='all', axis=1, inplace=True)
            
            return df
        
        # Read raw for other sheets
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 1. Detect Structure
        # Check first 5 rows for Time markers (Año/Periodo)
        time_rows = []
        for idx in range(min(5, len(df))):
            row_vals = [str(x).upper() for x in df.iloc[idx] if pd.notna(x)]
            if any(k in row_vals for k in ["AÑO", "YEAR", "PERIODO", "MES", "MONTH"]):
                time_rows.append(idx)
        
        # 2. Logic Selection
        if len(time_rows) >= 2 or ("PLANTA" in sheet_name.upper() or "FASE" in sheet_name.upper()):
            # HORIZONTAL MODE (Time on Columns)
            # Find year row (usually first with time marker)
            yr_idx = time_rows[0] if time_rows else 0
            pd_idx = time_rows[1] if len(time_rows) > 1 else yr_idx + 1
            
            row_yr = df.iloc[yr_idx].ffill()
            row_pd = df.iloc[pd_idx]
            
            new_cols = []
            for i in range(df.shape[1]):
                y = str(row_yr[i]).split('.')[0] if pd.notna(row_yr[i]) else ""
                p = str(row_pd[i]).strip() if pd.notna(row_pd[i]) else ""
                
                if y.lower() in ["nan", "año", "year"]: y = ""
                if p.lower() in ["nan", "periodo", "mes", "month"]: p = ""
                
                if y and p: col = f"{y}_{p}"
                elif y: col = y
                else: col = p if p else f"Col_{i}"
                new_cols.append(col)
            
            df.columns = new_cols
            df_clean = df.iloc[max(yr_idx, pd_idx)+1:].copy()
        else:
            # VERTICAL MODE (Standard Table)
            # Find header row (first non-empty row)
            header_idx = 0
            for idx in range(len(df)):
                if not df.iloc[idx].isnull().all():
                    header_idx = idx
                    break
            
            headers = [str(x).strip() for x in df.iloc[header_idx]]
            df.columns = headers
            df_clean = df.iloc[header_idx+1:].copy()

        # 3. Final Cleanup
        # Deduplicate columns to avoid 'Multi-column' Series returns
        cols = pd.Series(df_clean.columns).astype(str)
        for i in range(len(cols)):
            if (cols == cols[i]).sum() > 1:
                # Append count to duplicates
                occurrences = (cols[:i+1] == cols[i]).sum()
                if occurrences > 1:
                    cols[i] = f"{cols[i]}_{occurrences-1}"
        df_clean.columns = cols

        # Drop completely NaN rows/cols
        df_clean.dropna(how='all', axis=0, inplace=True)
        df_clean.dropna(how='all', axis=1, inplace=True)
        
        # Convert numeric to clean strings (no .0)
        for col in df_clean.columns:
            series = df_clean[col]
            if pd.api.types.is_float_dtype(series):
                # Only if they are basically ints
                if series.dropna().apply(lambda x: x % 1 == 0).all():
                    df_clean[col] = series.astype(str).str.replace('.0', '', regex=False)

        return df_clean
        
    except Exception as e:
        print(f"Error loading smart sheet {sheet_name}: {e}")
        return pd.DataFrame()

def get_ai_context(file_path):
    """
    Aggregates ALL 10 sheets for the AI.
    """
    context_data = {}
    try:
        xl = pd.ExcelFile(file_path)
        sheets = xl.sheet_names
        print(f"Loading AI Context (10 Sheets): {sheets}")
        
        for s in sheets:
            df = load_sheet_smart(file_path, s)
            if not df.empty:
                context_data[s] = df
    except Exception as e:
        print(f"Error aggregating sheets: {e}")

    return context_data
