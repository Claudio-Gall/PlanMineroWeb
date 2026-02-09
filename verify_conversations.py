import firebase_admin
from firebase_admin import credentials, firestore
import os
import datetime

def check_conversations():
    print("☁️ Verificando Consultas en Google Cloud (Firestore)...")
    
    # Connect (using existing credential logic)
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

    # Check CONVERSATIONS (The Log)
    print("\n📜 ÚLTIMAS 5 CONSULTAS REGISTRADAS:")
    print("-" * 50)
    
    try:
        # Order by timestamp descending
        docs = db.collection("conversations")\
                 .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                 .limit(5)\
                 .stream()
        
        count = 0
        for doc in docs:
            count += 1
            data = doc.to_dict()
            q = data.get('question', 'N/A')
            ts = data.get('timestamp', 'N/A')
            
            # Format timestamp nicely if possible
            ts_str = str(ts)
            
            print(f"[{count}] 📅 {ts_str}")
            print(f"    👤 Pregunta: '{q}'")
            print("-" * 50)
        
        if count == 0:
            print("⚠️ No encontré consultas registradas en la colección 'conversations'.")
            print("   (Tal vez aún no has hecho preguntas al Chat IA después de activar el Cloud Manager)")
        else:
             print(f"\n✅ Total de {count} consultas recientes recuperadas desde Google Cloud.")

    except Exception as e:
        print(f"Error leyendo 'conversations': {e}")

if __name__ == "__main__":
    check_conversations()
