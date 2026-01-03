"""
Script to Migrate Local Data to Cloud
-------------------------------------
Reads from SQLite 'chat_memory.db' and 'init_training.py' logic
and pushes it to Firebase Firestore.
"""

import db_manager
import cloud_manager
import init_training  # Access the raw list if needed, or query DB

def sync_now():
    print("="*60)
    print("🚀 STARTING CLOUD MIGRATION")
    print("="*60)
    
    if not cloud_manager.check_cloud_status():
        print("❌ Cloud not connected. Please check 'firestore-key.json' or secrets.")
        return

    # 1. Sync Training Examples from Local DB
    print("\n📦 Syncing Training Examples...")
    try:
        conn = db_manager.sqlite3.connect(db_manager.DB_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM training_examples")
        rows = c.fetchall()
        conn.close()
        
        count = 0
        for row in rows:
            # row structure: id, pattern, template, sheet, example_query, date
            # Map to cloud structure
            q_pattern = row[1]
            # Use template as code placeholder or description
            ans_code = row[2] 
            sheet = row[3]
            
            cloud_manager.save_training_example_cloud(q_pattern, ans_code, sheet, verified=True)
            count += 1
            
        print(f"✅ Synced {count} examples to Cloud.")
        
    except Exception as e:
        print(f"❌ Error syncing DB: {e}")

if __name__ == "__main__":
    sync_now()
