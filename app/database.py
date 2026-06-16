import os
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

_db_path = None
_main_db = None
_local = threading.local()

DB_PATH = Path(__file__).parent.parent / "data" / "freeapi.db"
KEYS_JSON_PATH = Path(__file__).parent.parent / "keys.json"

def dict_factory(cursor, row):
    fields = [col[0] for col in cursor.description]
    return {key: value for key, value in zip(fields, row)}

class ThreadLocalConnection:
    def __init__(self, conn):
        self.conn = conn
    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

def get_db():
    global _db_path
    if _db_path is None:
        raise Exception("Database not initialized. Call init_db() first.")
    
    if not hasattr(_local, "db_wrapper"):
        conn = sqlite3.connect(str(_db_path), check_same_thread=False, uri=True)
        conn.row_factory = dict_factory
        conn.execute('PRAGMA foreign_keys = ON')
        if ':memory:' not in str(_db_path) and 'mode=memory' not in str(_db_path):
            try:
                conn.execute('PRAGMA journal_mode = WAL')
            except sqlite3.OperationalError:
                pass
        _local.db_wrapper = ThreadLocalConnection(conn)
    return _local.db_wrapper.conn

def init_db(db_path=None):
    global _db_path, _main_db
    resolved_path = Path(db_path) if db_path else DB_PATH
    
    if str(resolved_path) == ':memory:':
        resolved_path = "file:tokenlooter_memdb?mode=memory&cache=shared"
    
    _db_path = resolved_path
    is_memory = 'mode=memory' in str(resolved_path) or str(resolved_path) == ':memory:'

    if not is_memory:
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

    _main_db = sqlite3.connect(str(resolved_path), check_same_thread=False, uri=True)
    _main_db.row_factory = dict_factory
    
    if not is_memory:
        _main_db.execute('PRAGMA journal_mode = WAL')
    _main_db.execute('PRAGMA foreign_keys = ON')

    create_tables(_main_db)

    _local.db = _main_db

    print(f"Database initialized at {resolved_path}")
    return _main_db

def create_tables(db):
    # If the old requests table has integer key_id with foreign key constraints, recreate it
    try:
        cursor = db.cursor()
        cursor.execute("PRAGMA table_info(requests)")
        info = cursor.fetchall()
        # Find if key_id is integer
        for col in info:
            if col["name"] == "key_id" and "INT" in str(col["type"]).upper():
                db.execute("DROP TABLE requests")
                db.commit()
                break
    except Exception:
        pass

    db.execute("""
    CREATE TABLE IF NOT EXISTS requests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      platform TEXT NOT NULL,
      model_id TEXT NOT NULL,
      key_id TEXT,
      status TEXT NOT NULL,
      input_tokens INTEGER NOT NULL DEFAULT 0,
      output_tokens INTEGER NOT NULL DEFAULT 0,
      latency_ms INTEGER NOT NULL DEFAULT 0,
      error TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      requested_model TEXT,
      ttfb_ms INTEGER,
      request_type TEXT NOT NULL DEFAULT 'chat'
    );
    """)
    db.commit()

def load_keys_json() -> dict:
    if not KEYS_JSON_PATH.exists():
        return {"unified_api_key": "tokenlooter_secret_key_here", "providers": {}}
    try:
        with open(KEYS_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"unified_api_key": "tokenlooter_secret_key_here", "providers": {}}

def get_unified_api_key() -> str:
    data = load_keys_json()
    return data.get("unified_api_key", "tokenlooter_secret_key_here")

def get_provider_keys(platform: str) -> list:
    data = load_keys_json()
    return data.get("providers", {}).get(platform, [])

