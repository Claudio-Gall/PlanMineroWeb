
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

def test_local_connection():
    print("🕵️ STARTING LOCAL KEY TEST...")
    
    key_file = "firestore-key.json"
    
    if not os.path.exists(key_file):
        print(f"❌ ERROR: File '{key_file}' not found in current directory.")
        return

    try:
        # 1. Print Key Specs (Local)
        with open(key_file, "r") as f:
            key_data = json.load(f)
            pk = key_data.get("private_key", "")
            print(f"📄 Local Key ID: {key_data.get('private_key_id')}")
            print(f"📄 Local Key Length: {len(pk)}")
            print(f"📄 Local Key Ends With: {repr(pk[-20:])}")

        # 2. Attempt Connection
        print("\n🔌 Attempting Firebase Connection...")
        cred = credentials.Certificate(key_file)
        
        # Initialize only if not already initialized
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            
        db = firestore.client()
        
        # 3. Attempt Write
        print("✍️  Attempting Write Test...")
        db.collection("diagnostics").add({
            "test": "local_script_verification",
            "status": "success",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        
        print("\n✅ SUCCESS! The local json file is VALID and WORKING.")
        print("➡️  Problem is definitely in Streamlit Cloud Secrets (Copy/Paste error).")
        
    except Exception as e:
        print(f"\n❌ FAIL! The local file is BROKEN/REVOKED.")
        print(f"Error Details: {e}")
        print("➡️  Solution: You MUST generate a NEW Key in Firebase Console.")

if __name__ == "__main__":
    test_local_connection()
