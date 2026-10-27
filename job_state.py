"""Small SQLite store for last successful jobs and registered FPL managers."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path('data/pipeline_state.sqlite3')


@contextmanager
def open_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=60)
    db.execute('PRAGMA busy_timeout = 60000')
    db.execute('''CREATE TABLE IF NOT EXISTS jobs (
        job_name TEXT PRIMARY KEY,
        successful_at TEXT NOT NULL
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS users (
        manager_id INTEGER PRIMARY KEY,
        joined_at TEXT NOT NULL
    )''')
    db.commit()
    try:
        yield db
    finally:
        db.close()


def last_success(db, name):
    row = db.execute('SELECT successful_at FROM jobs WHERE job_name=?',
                     (name,)).fetchone()
    return datetime.fromisoformat(row[0]) if row else None


def mark_success(db, name, when=None):
    when = when or datetime.now(timezone.utc)
    db.execute('''INSERT INTO jobs(job_name, successful_at) VALUES(?, ?)
                  ON CONFLICT(job_name) DO UPDATE SET successful_at=excluded.successful_at''',
               (name, when.isoformat()))
    db.commit()


def add_user(db, manager_id):
    db.execute('INSERT OR IGNORE INTO users VALUES(?, ?)', (
        int(manager_id), datetime.now(timezone.utc).isoformat()
    ))
    db.commit()


def manager_ids(db):
    rows = db.execute('SELECT manager_id FROM users ORDER BY manager_id').fetchall()
    return [int(row[0]) for row in rows]
