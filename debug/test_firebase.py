import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from cloud_manager import get_db_connection, firestore
    import datetime

    print("🔌 Iniciando prueba de conexión a Firebase...")

    # 1. Connect
    db = get_db_connection()
    
    if db:
        print("✅ Conexión establecida (Cliente Creado).")
        
        # 2. Write Test
        doc_ref = db.collection("system_status").document("connectivity_test")
        test_data = {
            "last_check": datetime.datetime.now(),
            "status": "OK",
            "agent": "Antigravity",
            "message": "Hola desde la prueba de validación"
        }
        doc_ref.set(test_data)
        print("✅ Escritura exitosa en colección 'system_status'.")
        
        # 3. Read Test
        doc_snap = doc_ref.get()
        if doc_snap.exists:
            print(f"✅ Lectura exitosa: {doc_snap.to_dict()}")
        else:
            print("⚠️ El documento se escribió pero no se pudo leer inmediatamente.")
            
    else:
        print("❌ Falló la conexión: get_db_connection devolvió None.")
        print("Verifica si existe 'firestore-key.json' o las credenciales.")

except Exception as e:
    print(f"❌ Error Crítico: {e}")
