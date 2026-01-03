import cloud_manager

print("🔍 VERIFYING CLOUD DATA...")

if user_db := cloud_manager.get_db_connection():
    print("✅ Connection Established")
    
    # Check Training Examples
    ref = user_db.collection("training_examples")
    docs = list(ref.stream())
    
    print(f"📊 Found {len(docs)} cached training examples:")
    for doc in docs:
        data = doc.to_dict()
        print(f"   - Question: {data.get('question_pattern')}")
        print(f"   - Code Snippet: {data.get('answer_code')[:50]}...")
else:
    print("❌ Could not connect to Firestore")
