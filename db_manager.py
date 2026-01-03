import sqlite3
import datetime

DB_FILE = "chat_memory.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Original memories table (deprecated, keeping for compatibility)
        c.execute('''CREATE TABLE IF NOT EXISTS memories
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      question TEXT,
                      code TEXT)''')
        
        # NEW: Training examples table (predefined Q&A pairs)
        c.execute('''CREATE TABLE IF NOT EXISTS training_examples
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      question_pattern TEXT,
                      answer_template TEXT,
                      sheet_name TEXT,
                      example_query TEXT,
                      created_date TEXT)''')
        
        # NEW: Conversation history (actual user interactions)
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      question TEXT,
                      answer TEXT,
                      sheets_used TEXT,
                      tokens_used INTEGER,
                      was_successful BOOLEAN)''')
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Init Error: {e}")

def save_memory(question, code):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        c.execute("INSERT INTO memories (timestamp, question, code) VALUES (?, ?, ?)", (timestamp, question, code))
        conn.commit()
        last_id = c.lastrowid
        conn.close()
        return last_id
    except Exception as e:
        print(f"Save Memory Error: {e}")
        return None

def get_memories(limit=10):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Get Memories Error: {e}")
        return []

def save_conversation(question, answer, sheets_used, tokens_used=0, was_successful=True):
    """Save actual user conversation for learning."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        sheets_str = ','.join(sheets_used) if isinstance(sheets_used, list) else sheets_used
        c.execute("INSERT INTO conversations (timestamp, question, answer, sheets_used, tokens_used, was_successful) VALUES (?, ?, ?, ?, ?, ?)",
                  (timestamp, question, answer, sheets_str, tokens_used, was_successful))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Save Conversation Error: {e}")

def add_training_example(question_pattern, answer_template, sheet_name, example_query):
    """Add a predefined training example."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        created_date = datetime.datetime.now().isoformat()
        c.execute("INSERT INTO training_examples (question_pattern, answer_template, sheet_name, example_query, created_date) VALUES (?, ?, ?, ?, ?)",
                  (question_pattern, answer_template, sheet_name, example_query, created_date))
        conn.commit()
        conn.close()
        print(f"✅ Training example added: {example_query}")
    except Exception as e:
        print(f"Add Training Example Error: {e}")

def get_similar_examples(question, limit=3):
    """Get similar training examples based on keyword matching."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Simple keyword matching (can be improved with embeddings later)
        question_lower = question.lower()
        c.execute("SELECT * FROM training_examples")
        all_examples = c.fetchall()
        conn.close()
        
        # Score each example by keyword overlap
        scored = []
        for ex in all_examples:
            pattern_lower = ex[1].lower() if ex[1] else ""  # question_pattern
            score = sum(1 for word in question_lower.split() if len(word) > 3 and word in pattern_lower)
            if score > 0:
                scored.append((score, ex))
        
        # Return top matches
        scored.sort(reverse=True)
        return [ex for score, ex in scored[:limit]]
    except Exception as e:
        print(f"Get Similar Examples Error: {e}")
        return []
