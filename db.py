import os, json, psycopg
from dotenv import load_dotenv

load_dotenv()

def _connect():
    return psycopg.connect(os.environ["DATABASE_URL"])

def init_db():
    with _connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")   # ready for the RAG project later
        conn.execute("""CREATE TABLE IF NOT EXISTS extractions (
            id SERIAL PRIMARY KEY,
            input_text TEXT,
            result JSONB,
            created_at TIMESTAMP DEFAULT now()
        );""")
    print("DB ready.")

def save_extraction(text, result_dict):
    with _connect() as conn:
        conn.execute("INSERT INTO extractions (input_text, result) VALUES (%s, %s)",
                     (text, json.dumps(result_dict)))

def recent(limit=10):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT input_text, result, created_at FROM extractions ORDER BY id DESC LIMIT %s",
            (limit,)).fetchall()
    return [{"input_text": r[0], "result": r[1], "created_at": r[2].isoformat()} for r in rows]