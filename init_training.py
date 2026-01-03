"""
Initialization script for Chat IA Training Database
Creates predefined examples for common mining queries
"""

import db_manager

# Initialize database
db_manager.init_db()

# Training examples for each sheet type
training_examples = [
    # KPI-Palas examples
    {
        "question_pattern": "usos reales pala mes año",
        "answer_template": "Look in KPI-Palas sheet, filter by Pala name and year, show %UsoReal column by month in a table",
        "sheet_name": "KPI-Palas",
        "example_query": "dame los usos reales por mes para la pala 6 el 2027"
    },
    {
        "question_pattern": "rendimiento pala mes año",
        "answer_template": "Look in KPI-Palas sheet, filter by equipment and year, show Rend column",
        "sheet_name": "KPI-Palas",
        "example_query": "cual es el rendimiento de la pala 5 en marzo 2026"
    },
    {
        "question_pattern": "disponibilidad pala año",
        "answer_template": "Look in KPI-Palas sheet, show %Disp column for the equipment",
        "sheet_name": "KPI-Palas",
        "example_query": "muestra la disponibilidad de la PV6 en 2027"
    },
    
    # Pala-Fase examples
    {
        "question_pattern": "tonelaje banco fase",
        "answer_template": "Look in Pala-Fase or Fase-Banco sheet, filter by phase and banco, show tonnage",
        "sheet_name": "Pala-Fase",
        "example_query": "cuál es el tonelaje del banco 800 de la fase 5"
    },
    {
        "question_pattern": "movimiento fase mes año",
        "answer_template": "Look in Pala-Fase sheet, filter by phase, month and year, sum tonnage",
        "sheet_name": "Pala-Fase",
        "example_query": "cual es el movimiento de la fase 5 en marzo 2026"
    },
    
    # Planta examples
    {
        "question_pattern": "tratamiento año",
        "answer_template": "Look in Planta sheet, filter by year, show 'Ton Tratamiento' or similar column by month in a table",
        "sheet_name": "Planta",
        "example_query": "muestra en una tabla elegante los tratamientos del 2026"
    },
    {
        "question_pattern": "cobre fino mes año",
        "answer_template": "Look in Planta sheet, filter by month and year, show 'Cobre Fino' value",
        "sheet_name": "Planta",
        "example_query": "cuánto cobre fino se produjo en enero 2027"
    },
    {
        "question_pattern": "ley media año",
        "answer_template": "Look in Planta sheet, show 'Ley CuT' or '%' column  for the year",
        "sheet_name": "Planta",
        "example_query": "cual fue la ley media en 2026"
    },
    
    # KPI-Camiones examples
    {
        "question_pattern": "número camiones mes año",
        "answer_template": "Look in KPI-Camiones sheet, show 'N° Camiones' for the period",
        "sheet_name": "KPI-Camiones",
        "example_query": "cuántos camiones operaron en junio 2027"
    },
    
    # KPI-Perfos examples
    {
        "question_pattern": "metros perforación año",
        "answer_template": "Look in KPI-Perfos sheet, filter by drill and year, show 'Metros' column",
        "sheet_name": "KPI-Perfos",
        "example_query": "cuántos metros perforó la PV-5 en 2026"
    },
    
    # Distancias examples
    {
        "question_pattern": "distancia acarreo fase",
        "answer_template": "Look in Distancias Por Tramos sheet, filter by phase, show distance in km",
        "sheet_name": "Distancias Por Tramos",
        "example_query": "cual es la distancia de acarreo para la fase 3"
    },
]

print("=" * 70)
print("INITIALIZING CHAT IA TRAINING DATABASE")
print("=" * 70)

for ex in training_examples:
    db_manager.add_training_example(
        question_pattern=ex["question_pattern"],
        answer_template=ex["answer_template"],
        sheet_name=ex["sheet_name"],
        example_query=ex["example_query"]
    )

print(f"\n✅ Added {len(training_examples)} training examples")
print("\nExamples by sheet:")
sheet_counts = {}
for ex in training_examples:
    sheet = ex["sheet_name"]
    sheet_counts[sheet] = sheet_counts.get(sheet, 0) + 1

for sheet, count in sorted(sheet_counts.items()):
    print(f"  {sheet}: {count} examples")

print("\n" + "=" * 70)
print("DATABASE READY FOR CHAT IA TRAINING")
print("=" * 70)
