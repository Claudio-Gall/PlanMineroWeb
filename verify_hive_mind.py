import firebase_admin
from firebase_admin import credentials, firestore
import os

def check_hive_mind():
    print("🐝 Conectando con la Mente Colmena (Firestore)...")
    
    # 1. Connect
    if os.path.exists("firestore-key.json"):
        cred = credentials.Certificate("firestore-key.json")
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
    else:
        print("❌ No encontré firestore-key.json")
        return

    # 2. Check Training Examples (The "Brain")
    print("\n🧠 RECUERDOS ALMACENADOS (training_examples):")
    
    # Fetch documents and convert to a list to check for emptiness and iterate
    examples = list(db.collection("training_examples").order_by("created_at", direction=firestore.Query.DESCENDING).limit(5).stream())
    
    if not examples:
        print("  ⚠️ No hay recuerdos aún. La IA está 'en blanco'.")
    else:
        print(f"✅ Se encontraron al menos {len(examples)} recuerdos recientes.")
        print("   Estos se usarán para NO gastar dinero en preguntas repetidas.")
        print("\n  🔹 Detalles de los recuerdos recientes:")
        
        for doc in examples:
            data = doc.to_dict()
            q = data.get('question_pattern', 'N/A')
            code = data.get('answer_code', 'N/A')
            sheet = data.get('sheet_name', 'N/A')
            
            print(f"- [{doc.id}] '{q}' -> Sheet: {sheet}")
            if "Empty DataFrame" in code or "No data found" in code:
                print(f"  ⚠️ WARNING: Cached code might be returning empty results!")
            print("-" * 40)

if __name__ == "__main__":
    check_hive_mind()
