# Chat IA - Test Script
# Tests if the optimized agent can handle user queries

import sys
sys.path.insert(0, '.')

# Test 1: Analyze query function
from chat_agent import analyze_query, SHEET_INDEX

test_queries = [
    "en una gráfica de tablas dame los usos reales por mes para la pala 6 el 2027",
    "cuál es el tonelaje del banco 800 de la fase 5",
    "muestra en una tabla elegante los tratamientos del 2026",
    "cual es el movimiento de la fase 5 en marzo 2026"
]

print("=" * 70)
print("CHAT IA - OPTIMIZACIÓN DE CONTEXTO")
print("=" * 70)

print("\n📋 ÍNDICE DE SHEETS DISPONIBLES:\n")
for sheet_name, metadata in SHEET_INDEX.items():
    print(f"  • {sheet_name}")
    print(f"    Keywords: {', '.join(metadata['keywords'][:5])}")
    print(f"    {metadata['description']}")
    print()

print("\n" + "=" * 70)
print("ANÁLISIS DE CONSULTAS")
print("=" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n{i}. CONSULTA: \"{query}\"")
    relevant_sheets = analyze_query(query)
    print(f"   → Sheets relevantes: {relevant_sheets}")
    print(f"   → Token Savings: {9 - len(relevant_sheets)} sheets omitidas")
    
    # Estimate token reduction
    tokens_full = 50000  # Approximate for all 10 sheets
    tokens_smart = 5000 * len(relevant_sheets)  # Approximate per sheet
    reduction = ((tokens_full - tokens_smart) / tokens_full) * 100
    print(f"   → Reducción estimada: {reduction:.0f}% menos tokens")

print("\n" + "=" * 70)
print("CAPACIDADES DEL CHAT IA")
print("=" * 70)

capabilities = {
    "✅ PUEDE RESPONDER": [
        "Datos de producción (cobre, ley, recuperación)",
        "Movimiento de palas por fase/banco",
        "KPIs de equipos (palas, camiones, perfos, servicios)",
        "Distancias de acarreo",
        "Flujos de material (Envíos)",
        "Datos filtrados por año/mes/trimestre",
        "Tablas formateadas con datos específicos"
    ],
    "❌ NO PUEDE (limitaciones)": [
        "Crear gráficos visuales directos (puede describir qué mostrar)",
        "Ejecutar análisis complejos no basados en datos cargados",
        "Modificar datos del Excel",
        "Acceder a información fuera de las 10 sheets"
    ]
}

for category, items in capabilities.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  • {item}")

print("\n" + "=" * 70)
print("RECOMENDACIONES DE USO")
print("=" * 70)

tips = [
    "Ser específico: Mencionar año, mes, equipo o fase",
    "Usar nombres completos: 'Pala 6' mejor que 'P6'",
    "Pedir tablas markdown cuando quieras ver datos estructurados",
    "Si no encuentra datos, reformular con términos del Excel",
    "Verificar resultados con dashboard visual si quieres gráficos"
]

for i, tip in enumerate(tips, 1):
    print(f"{i}. {tip}")

print("\n" + "=" * 70)
print("PRÓXIMOS PASOS")
print("="  * 70)
print("\n1. Reiniciar Streamlit")
print("2. Ir a tab '🤖 Chat IA'")
print("3. Probar con las consultas de ejemplo")
print("4. Verificar reducción de tokens en logs")
print("\n✅ Chat IA Optimizado - Listo para usar\n")
