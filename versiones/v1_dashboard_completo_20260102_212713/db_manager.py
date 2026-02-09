import sqlite3
import datetime

DB_FILE = "chat_memory.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS memories
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      question TEXT,
                      code TEXT)''')
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
