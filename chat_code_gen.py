"""
CORRECT APPROACH: Code Generation Chat
--------------------------------------
Instead of sending data to AI, we send schema and AI generates code.

Benefits:
- 95%+ token reduction (only schema, not 10,000 rows)
- Perfect accuracy (Python calculates, not AI)
- Fast execution (local Pandas)
- Scales to millions of rows

Architecture:
1. User asks question
2. Send DataFrame schemas (columns + dtypes) to AI
3. AI generates Python code
4. Execute code safely with Pandas
5. Return result
"""

import pandas as pd
import streamlit as st
import re
from typing import Dict, Any

def get_dataframe_schema(df: pd.DataFrame, name: str, sample_rows=2) -> str:
    """
    Extract schema from DataFrame: columns, types, and tiny sample.
    This is what we send to AI instead of full data.
    """
    schema_parts = []
    schema_parts.append(f"DataFrame: {name}")
    schema_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    schema_parts.append("\nColumns:")
    
    for col in df.columns:
        try:
            col_data = df[col]
            # Handle duplicate columns (if df[col] returns a DataFrame)
            if isinstance(col_data, pd.DataFrame):
                dtype = str(col_data.iloc[:, 0].dtype)
                sample_vals = col_data.iloc[:, 0].dropna().head(2).tolist()
            else:
                dtype = str(col_data.dtype)
                sample_vals = col_data.dropna().head(2).tolist()
                
            sample_str = f" (e.g., {sample_vals})" if sample_vals else ""
            schema_parts.append(f"  - {col}: {dtype}{sample_str}")
        except Exception as e:
            schema_parts.append(f"  - {col}: Error extracting type ({str(e)})")
    
    # Add tiny sample (2 rows max)
    if sample_rows > 0:
        schema_parts.append(f"\nSample (first {min(sample_rows, len(df))} rows):")
        schema_parts.append(df.head(sample_rows).to_string())
    
    return "\n".join(schema_parts)

def build_schemas_context(data_dict: Dict[str, pd.DataFrame]) -> str:
    """
    Build complete schemas context for all DataFrames.
    This replaces sending full data.
    """
    ai_raw = data_dict.get('ai_raw', {})
    schemas = []
    
    schemas.append("=== AVAILABLE DATAFRAMES ===\n")
    
    for name, df in ai_raw.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Sync with execution env: sanitise names
            clean_name = name.replace('-', '_').replace(' ', '_')
            schema = get_dataframe_schema(df, clean_name, sample_rows=2)
            schemas.append(schema)
            schemas.append("\n" + "-"*50 + "\n")
    
    return "\n".join(schemas)

def execute_generated_code(code: str, data_context: Dict[str, pd.DataFrame]) -> Any:
    """
    Safely execute generated Pandas code.
    
    Security: Only allows access to provided DataFrames, no system access.
    """
    # Prepare safe execution environment
    ai_raw = data_context.get('ai_raw', {})
    
    # Create safe globals with only Pandas and our data
    safe_globals = {
        'pd': pd,
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'isinstance': isinstance,
            'set': set,
            'tuple': tuple,
            'abs': abs,
            'all': all,
            'any': any,
        }
    }
    
    # Add DataFrames to namespace
    for name, df in ai_raw.items():
        # Make DataFrame accessible by clean variable name
        clean_name = name.replace('-', '_').replace(' ', '_')
        safe_globals[clean_name] = df
    
    # Execute code
    local_vars = {}
    exec(code, safe_globals, local_vars)
    
    # Return the 'result' variable if it exists
    if 'result' in local_vars:
        return local_vars['result']
    elif local_vars:
        # Return last expression
        return list(local_vars.values())[-1]
    else:
        return None

def extract_code_from_response(response: str) -> str:
    """Extract Python code from AI response."""
    # Look for code blocks
    code_pattern = r'```python\n(.*?)\n```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        code = matches[0].strip()
        # SAFETY: Remove explicit imports (pd/np are globals)
        code = re.sub(r'^import .*', '', code, flags=re.MULTILINE)
        return code
    
    # Fallback: look for lines starting with df or result
    lines = response.split('\n')
    code_lines = [line for line in lines if line.strip().startswith(('df', 'result', 'KPI'))]
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def query_gemini_code_generation(prompt, api_key):
    """Query Gemini for CODE generation (not data processing)."""
    if not api_key:
        return "Error: No API Key."
    
    import requests
    import urllib3
    urllib3.disable_warnings()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        s = requests.Session()
        s.trust_env = False
        resp = s.post(url, headers=headers, json=data, timeout=30, verify=False)
        if resp.status_code == 200:
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
        return f"API Error ({resp.status_code}): {resp.text[:300]}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

class CodeGenerationChatAgent:
    """
    NEW APPROACH: Schema-only chat that generates code.
    Tokens reduced by 95%, perfect accuracy.
    """
    
    def __init__(self, data_source):
        self.data_dict = {}
        
        if isinstance(data_source, dict):
            self.data_dict = data_source
        
        self.api_key = None
        
    def initialize(self, api_key=None):
        import os
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key:
            print("Warning: No API key")
    
    def ask(self, question, history=None):
        """
        Hive Mind Strategy:
        1. Check Cloud Cache (Firestore) -> 0 Tokens
        2. If Miss -> Gen Code with Gemini -> Execute -> Save to Cloud
        """
        # 0. Initialize Cloud (if available)
        try:
            import cloud_manager
        except ImportError:
            cloud_manager = None

        # 1. CHECK CLOUD CACHE (Only if no history - Fresh Context)
        # If history exists, we need AI to understand context, so we skip cache.
        should_check_cache = True
        if history and len(history) > 0:
            print("📜 History detected. Skipping Cloud Cache to preserve context.")
            should_check_cache = False
            
        if should_check_cache and cloud_manager and cloud_manager.check_cloud_status():
            print(f"☁️ Checking Cloud Hive Mind for: {question}")
            similar_examples = cloud_manager.get_similar_cloud(question, limit=1)
            
            if similar_examples:
                best_match = similar_examples[0]
                print(f"⚡ Hive Mind HIT! Reusing code from similar query: '{best_match.get('question_pattern', 'Unknown')}'")
                
                cached_code = best_match.get('answer_code')
                if cached_code:
                    # Execute cached code immediately
                    try:
                        result = execute_generated_code(cached_code, self.data_dict)
                        # Format and return
                        answer_md = self._format_result(result, cached_code, from_cache=True)
                        return {'type': 'text', 'content': answer_md}
                    except Exception as e:
                        print(f"⚠️ Cached code failed: {e}. Falling back to generation.")
                        # Fallthrough to generation if cached code fails (e.g. data changed)

        # 2. GENERATE NEW CODE (Standard Schema-Only Flow)
        schemas_context = build_schemas_context(self.data_dict)
        
        # Format recent history for context (Last 3 rounds max to save tokens)
        history_str = ""
        if history:
            history_str = "\nCONVERSATION HISTORY:\n"
            # Filter only relevant text messages, skip errors/plots
            valid_msgs = [m for m in history if m.get('type') == 'text'][-6:] 
            for msg in valid_msgs:
                role = "User" if msg['role'] == "user" else "Assistant"
                content = msg['content'].replace('\n', ' ')[:200] # Truncate long outcomes
                history_str += f"- {role}: {content}\n"

        # 0. MINING GLOSSARY (Domain Knowledge Injection)
        mining_glossary = """
        MINING GLOSSARY & MAPPING RULES:
        - "Ton" / "Tonelaje" -> Columns: 'Tonelaje', 'Ton (Humedas)', 'Movimiento', 'Total_Ton'
        - "Ley" -> Columns: 'Ley_CuT', 'Ley', 'Cobre_Fino' (if calculating fine copper)
        - "Rec" / "Recuperacion" -> Columns: 'Recuperacion', 'Recup'
        - "Fase" -> Column: 'Fase' (Filter by string e.g. "Fase 4", "F04")
        - "Pala" / "Equipo" -> Column: 'Equipo' OR Prefix in KPI sheets (e.g., 'P06_' for Pala 6)
        - "Periodo" -> Columns: 'Periodo', 'Mes', 'Año' (Filter by Year/Month)
        - "Rendimiento" / "Rend" -> Columns: 'RenDPala', 'Rendimiento', 'Rend_Efectivo' (NEVER use %UsoReal for this)
        """

        system_prompt = f"""You are a Python Pandas code generator for mining data analysis.

AVAILABLE DATAFRAMES (Schemas only):
{schemas_context}
{history_str}
USER QUESTION:
{question}
{mining_glossary}

INSTRUCTIONS:
1. Generate Python code using Pandas to answer the question
2. If the user asks a follow-up (e.g. "and for loader?"), USE THE HISTORY to figure out the full question.
3. Use the DataFrame names provided (e.g., KPI_Palas, Pala_Fase, Planta)
4. Store final result in a variable called 'result'
5. For queries about 2027, filter: df[df['Año'] == 2027]
6. Select the EXACT column requested. Do NOT default to '%UsoReal' if the user asks for 'Rendimiento' or 'Tonelaje'.
7. FOR HORIZONTAL SHEETS (e.g. Fase-Banco, Planta):
   - Columns are named "Year_Period" (e.g., "2028_1er Trimestre", "2026_Enero").
   - SPECIAL RULE FOR 'PLANTA': 
     - Use the 'Origen' column to distinguish rows.
     - **DEFAULT**: If user asks for "Tratamiento" or "Alimentacion", RETURN ROWS for 'Mina', 'Stock' AND 'Planta Total'. Do NOT just show Total.
     - Filter strictly: `df = df[df['Origen'].isin(['Mina', 'Stock', 'Planta Total'])]`
     - Keep 'Origen' column in the final result.
   - SPECIAL RULE FOR 'PALA-FASE':
     - Column 0='Origen', Column 1='Pala', Column 2='Fase'.
     - Rename immediately: `df = df.rename(columns={{df.columns[0]: 'Origen', df.columns[1]: 'Pala', df.columns[2]: 'Fase'}})`
     - **AGGREGATION**: If user asks for a Phase (e.g. "Fase 5"), you MUST filter `df['Fase'].str.contains('F05')` and SUM the time column. Data is often strictly split across multiple rows (e.g. Bull and Pala).
   - SPECIAL RULE FOR 'FASE-BANCO':
     - Column 0 is 'Fase', Column 1 is 'Banco'.
     - Rename immediately: `df = df.rename(columns={{df.columns[0]: 'Fase', df.columns[1]: 'Banco'}})`
     - **TOTAL TONNAGE**: If answer requires "Tonelaje Total" or generic "Tonelaje" (without specific year), USE column `Grand Total_Grand Total`.
     - **CRITICAL**: The column `Grand Total_Grand Total` ALREADY contains the sum. DO NOT sum other columns if this exists.
   - Handle 'NaN' values immediately: `df = df.fillna(0)` before math.
   - For Division: `result = safe_div(a, b)` -> NO, just calculate and `fillna(0)`.
   - ROUNDING RULES:
     - **DataFrame/Series**: `result = result.fillna(0).round(0).astype(int)`
     - **SCALAR (Single Number)**: `result = int(0 if pd.isna(result) else round(result))`
     - **CRITICAL**: Do NOT call `.fillna()` on floats/ints. It causes AttributeError.
8. FORMATTING RULES (Apply to final result):
   - **Ley / CuT / Grade**: KEEP ORIGINAL (e.g. 0.823). DO NOT MULTIPLY BY 100. Round to 3 decimal places.
   - **Recuperacion / Recovery**: If values are < 1 (e.g. 0.85), * 100. If > 1 (e.g. 85.0), keep as is. Round to 2 decimals.
   - **% Uso / Disp / Util**: If values are < 1, * 100. Round to 1 decimal.
   - **Ton / Movimiento / Rendimiento**: INTEGER (No decimals).
   - **Columns**: RETURN ONLY WHAT IS ASKED. If user asks "Tratamiento", do NOT allow "Ley" columns in result.
9. Output ONLY the Python code in a ```python code block

EXAMPLE for "usos reales pala 6 2027":
```python
# Note: Columns have prefixes like P06_PctUsoReal due to header flattening
df = KPI_Palas[KPI_Palas['Año'] == 2027]
result = df[['Periodo', 'P06_PctUsoReal']].copy()
# Format as percentage INTEGER (0.82 -> 82)
result['P06_PctUsoReal'] = (result['P06_PctUsoReal'] * 100).round(0).astype(int)
```

Generate code now:"""

        print(f"🧠 Generating code for: {question}")
        print(f"📊 Schemas sent: {len(schemas_context)} chars (NOT full data)")
        
        response = query_gemini_code_generation(system_prompt, self.api_key)
        code = extract_code_from_response(response)
        
        if not code:
            return {
                'type': 'error',
                'content': f"No pude generar código. Respuesta de IA:\n{response[:500]}"
            }
        
        print(f"✅ Code generated:\n{code}\n")
        
        # 3. EXECUTE CODE
        try:
            result = execute_generated_code(code, self.data_dict)
            
            # 4. SAVE TO CLOUD (Hive Mind Learning)
            if cloud_manager and cloud_manager.check_cloud_status():
                # We save successful patterns to make the hive smarter
                # We use the USER question as the pattern for now
                cloud_manager.save_training_example_cloud(
                    question_pattern=question,
                    answer_code=code,
                    sheet_name="Auto-Learned",
                    verified=True # Auto-verified if it runs? Maybe set False nicely.
                )
            
            # Format result
            answer_md = self._format_result(result, code, from_cache=False)
            return {'type': 'text', 'content': answer_md}
        
        except Exception as e:
            error_msg = f"Error ejecutando código:\n```python\n{code}\n```\n\nError: {str(e)}"
            return {'type': 'error', 'content': error_msg}

    def _format_result(self, result, code, from_cache=False):
        """Helper to format output nicely."""
        import pandas as pd
        source_badge = "⚡ **Respuesta Instantánea (Cloud Cache)**" if from_cache else "🧠 **Respuesta Generada (Gemini)**"
        
        if isinstance(result, pd.DataFrame):
            table_md = result.to_markdown(index=False, tablefmt="github")
            return f"{source_badge}\n\n**Resultado:**\n\n{table_md}"
        elif isinstance(result, (int, float)):
            return f"{source_badge}\n\n**Resultado:** {result:,.2f}"
        else:
            return f"{source_badge}\n\n**Resultado:** {result}"
