import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import toml

def clear_entry():
    # 1. Initialize Firestore
    if not firebase_admin._apps:
        try:
            secrets = toml.load(".streamlit/secrets.toml")
            cred_dict = secrets["firestore"]
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            print(f"❌ Error initializing Firestore: {e}")
            return

    db = firestore.client()
    
    # 2. Delete Specific Document
    doc_id = "9DsWGTvKGiLcQURSBEJ4" # Latest missing usage entry
    
    try:
        db.collection("training_examples").document(doc_id).delete()
        print(f"✅ Documento {doc_id} eliminado correctamente.")
        print("🧹 La memoria caché para esa consulta ha sido borrada.")
    except Exception as e:
        print(f"❌ Error eliminando documento: {e}")

if __name__ == "__main__":
    clear_entry()
