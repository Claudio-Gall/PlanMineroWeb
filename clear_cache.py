import cloud_manager

def clear_cache():
    print("🧹 LIMPIANDO CACHE FIREBASE...")
    db = cloud_manager.get_db_connection()
    if not db:
        print("❌ No hay conexión Cloud.")
        return

    # Delete all training examples that contain 'pala 6' to force regeneration
    ref = db.collection("training_examples")
    docs = list(ref.stream())
    
    count = 0
    for doc in docs:
        data = doc.to_dict()
        pattern = data.get("question_pattern", "").lower()
        if "banco" in pattern or "fase" in pattern:
            print(f"🗑️ Eliminando: {pattern}")
            doc.reference.delete()
            count += 1
            
    print(f"✅ Se eliminaron {count} ejemplos de la caché. La próxima consulta generará código nuevo.")

if __name__ == "__main__":
    clear_cache()
