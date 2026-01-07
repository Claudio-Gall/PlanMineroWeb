"""
Cloud Manager for Chat IA
-------------------------
Handles connection to Firebase Firestore for the 'Hive Mind' global memory.
"""

import streamlit as st
import datetime
# Try sourcing google-cloud-firestore, warn if missing
try:
    from google.cloud import firestore
    from google.oauth2 import service_account
    import json
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False
    print("⚠️ Google Cloud Firestore modules not found. Cloud features unavailable.")

def get_db_connection():
    """Establish connection to Firestore using Streamlit secrets or local JSON."""
    if not CLOUD_AVAILABLE:
        return None

    try:
        # Option 1: Load from streamlit secrets (deploy ready)
        if "firestore" in st.secrets:
            key_dict = dict(st.secrets["firestore"])
            
            # --- DEBUG: KEY INSPECTION ---
            pk = key_dict.get("private_key", "")
            print(f"🔑 Key Debug: Length={len(pk)}")
            print(f"🔑 First 50 chars (Repr): {repr(pk[:50])}")
            print(f"🔑 Contains literal \\n: {'\\n' in pk}")
            print(f"🔑 Contains real newline: {'\n' in pk}")
            
            # Auto-Fix Attempts
            if "\\n" in pk:
                print("🔧 Fixing literal \\n to real newlines...")
                pk = pk.replace("\\n", "\n")
            
            # REMOVE Windows Carriage Returns
            pk = pk.replace("\r", "")
            
            # ENSURE Ending Newline (PEM standard)
            if not pk.endswith("\n"):
                pk += "\n"
            
            key_dict["private_key"] = pk
            # -----------------------------
            
            creds = service_account.Credentials.from_service_account_info(key_dict)
            db = firestore.Client(credentials=creds)
            return db
            
        # Option 2: Load from local JSON file (dev mode)
        import os
        if os.path.exists("firestore-key.json"):
            creds = service_account.Credentials.from_service_account_file("firestore-key.json")
            db = firestore.Client(credentials=creds)
            return db

        return None
    except Exception as e:
        print(f"Cloud Connection Error: {e}")
        return None

def get_similar_cloud(question, limit=3):
    """
    Search for similar questions in the cloud 'training_examples' collection.
    Simple keyword matching for now (upgradeable to Vector Search).
    """
    db = get_db_connection()
    if not db:
        return []

    try:
        # HEURISTIC: If question is very short ("y la 5?", "cargador?"), likely a follow-up.
        # Bypass cache to let AI use context history.
        if len(question) < 15:
            return []

        # Fetch all verified examples (caching this in st.session_state is recommended for prod)
        examples_ref = db.collection("training_examples")
        docs = examples_ref.stream()
        
        matches = []
        import re
        
        # 1. Extract numbers from user query (CRITICAL for accuracy)
        # We don't want "Pala 3" to match "Pala 5" just because words overlap.
        user_nums = set(re.findall(r'\d+', question))
        
        q_words = set(question.lower().split())
        
        for doc in docs:
            data = doc.to_dict()
            pattern = data.get("question_pattern", "").lower()
            
            # 2. Extract numbers from cached pattern
            cached_nums = set(re.findall(r'\d+', pattern))
            
            # STRICT CHECK: If numbers don't match exactly, SKIP IT.
            # "Pala 3" (set{'3'}) != "Pala 5" (set{'5'}) -> Skip
            # "2026" (set{'2026'}) != "2027" (set{'2027'}) -> Skip
            if user_nums != cached_nums:
                continue
            
            # Simple scoring: count word overlaps
            score = sum(1 for w in q_words if w in pattern and len(w) > 3)
            
            # Require decent text overlap too
            if score > 0:
                matches.append((score, data))
        
        # Sort by score and return top results
        
        # Sort by score and return top results
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    except Exception as e:
        print(f"Cloud Search Error: {e}")
        return []

def save_training_example_cloud(question_pattern, answer_code, sheet_name, verified=False):
    """Upload a new training example to the Cloud Hive Mind."""
    db = get_db_connection()
    if not db:
        return

    try:
        data = {
            "question_pattern": question_pattern,
            "answer_code": answer_code,
            "sheet_name": sheet_name,
            "verified": verified,
            "created_at": firestore.SERVER_TIMESTAMP
        }
        db.collection("training_examples").add(data)
        print(f"☁️ Uploaded to Cloud: {question_pattern}")
    except Exception as e:
        print(f"Cloud Upload Error: {e}")

def save_conversation_cloud(question, answer, user_id="anonymous"):
    """Log the conversation for analytics."""
    db = get_db_connection()
    if not db:
        return

    try:
        data = {
            "question": question,
            "answer": answer,
            "user_id": user_id,
            "timestamp": datetime.datetime.now() # Use local python time to be safe
        }
        # 1. Save to conversations
        db.collection("conversations").add(data)
        
        # 2. DEBUG: Force write to 'training_examples' so user sees it
        debug_data = {
            "question_pattern": "DEBUG_LOG: " + question[:20],
            "answer_code": "LOGGED TO CONVERSATIONS",
            "sheet_name": "DEBUG",
            "verified": True,
            "created_at": datetime.datetime.now()
        }
        db.collection("training_examples").add(debug_data)
        
    except Exception as e:
        print(f"Cloud Logging Error: {e}")

def check_cloud_status():
    """Returns True if cloud is connected."""
    db = get_db_connection()
    return db is not None

def get_key_debug_stats():
    """Returns safe debug info about the loaded private key."""
    try:
        if "firestore" in st.secrets:
            pk = dict(st.secrets["firestore"]).get("private_key", "")
            return {
                "length": len(pk),
                "has_literal_slash_n": "\\n" in pk,
                "has_real_newline": "\n" in pk,
                "first_10_chars": repr(pk[:10]),
                "last_10_chars": repr(pk[-10:])
            }
        return "No 'firestore' in secrets"
    except Exception as e:
        return f"Error analyzing key: {e}"
