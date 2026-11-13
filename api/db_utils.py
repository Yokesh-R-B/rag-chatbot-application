import sqlite3
from contextlib import contextmanager
from pathlib import Path

from settings import DB_PATH

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(Path(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()

def create_application_logs():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

def create_document_store():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

def insert_application_logs(session_id, user_query, gpt_response, model):
    with get_db_connection() as conn:
        conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                     (session_id, user_query, gpt_response, model))
        conn.commit()

def get_chat_history(session_id):
    with get_db_connection() as conn:
        cursor = conn.execute(
            'SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at, id',
            (session_id,),
        )
        messages = []
        for row in cursor.fetchall():
            messages.extend([
                {"role": "human", "content": row["user_query"]},
                {"role": "ai", "content": row["gpt_response"]},
            ])
        return messages

def insert_document_record(filename):
    with get_db_connection() as conn:
        cursor = conn.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
        conn.commit()
        return cursor.lastrowid

def delete_document_record(file_id):
    with get_db_connection() as conn:
        cursor = conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
        conn.commit()
        return cursor.rowcount > 0

def get_all_documents():
    with get_db_connection() as conn:
        cursor = conn.execute(
            'SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC, id DESC'
        )
        return [dict(doc) for doc in cursor.fetchall()]

def get_document_ids():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id FROM document_store").fetchall()
        return {row["id"] for row in rows}

# Initialize the database tables
create_application_logs()
create_document_store()
