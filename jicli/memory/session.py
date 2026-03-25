"""SQLite session storage — WAL mode, concurrent-safe."""

import sqlite3
import uuid
import os
from datetime import datetime, timezone


class SessionStore:
    """Persists sessions and messages in SQLite."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(os.getcwd(), "jicli.db")
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                model TEXT NOT NULL,
                cwd TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                tags TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                thinking TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                pruned INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                error_type TEXT NOT NULL,
                description TEXT NOT NULL,
                resolution TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category);
            CREATE INDEX IF NOT EXISTS idx_error_log_type ON error_log(error_type);
        """)
        conn.commit()
        conn.close()

    # ── Sessions ─────────────────────────────────────────────────

    def create_session(self, model: str, cwd: str = "") -> str:
        """Create a new session, return its ID."""
        sid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).timestamp()
        conn = self._conn()
        conn.execute(
            "INSERT INTO sessions (id, created_at, updated_at, model, cwd) VALUES (?, ?, ?, ?, ?)",
            (sid, now, now, model, cwd),
        )
        conn.commit()
        conn.close()
        return sid

    def touch_session(self, sid: str):
        """Update session's updated_at timestamp."""
        now = datetime.now(timezone.utc).timestamp()
        conn = self._conn()
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
        conn.commit()
        conn.close()

    def get_latest_session(self) -> dict:
        """Get the most recent session."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def set_session_summary(self, sid: str, summary: str):
        """Store a session summary for re-priming."""
        conn = self._conn()
        conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, sid))
        conn.commit()
        conn.close()

    def list_sessions(self, limit: int = 20) -> list:
        """List recent sessions."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT id, created_at, updated_at, model, summary FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Messages ─────────────────────────────────────────────────

    def add_message(self, sid: str, role: str, content: str,
                    thinking: str = "", token_estimate: int = 0):
        """Add a message to a session."""
        now = datetime.now(timezone.utc).timestamp()
        conn = self._conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, thinking, timestamp, token_estimate) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, role, content, thinking, now, token_estimate),
        )
        conn.commit()
        conn.close()
        self.touch_session(sid)

    def get_messages(self, sid: str, include_pruned: bool = False) -> list:
        """Get messages for a session, ordered by timestamp."""
        conn = self._conn()
        query = "SELECT role, content, thinking, token_estimate, pruned FROM messages WHERE session_id = ?"
        if not include_pruned:
            query += " AND pruned = 0"
        query += " ORDER BY timestamp ASC"
        rows = conn.execute(query, (sid,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def prune_messages(self, sid: str, keep_last: int = 4):
        """Mark old messages as pruned, keeping the last N."""
        conn = self._conn()
        conn.execute(
            """UPDATE messages SET pruned = 1 
               WHERE session_id = ? AND id NOT IN (
                   SELECT id FROM messages WHERE session_id = ? 
                   ORDER BY timestamp DESC LIMIT ?
               )""",
            (sid, sid, keep_last),
        )
        conn.commit()
        conn.close()

    def count_tokens(self, sid: str) -> int:
        """Get estimated total tokens for active (non-pruned) messages."""
        conn = self._conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM messages WHERE session_id = ? AND pruned = 0",
            (sid,),
        ).fetchone()
        conn.close()
        return row[0] if row else 0

    # ── Persistent Memory ────────────────────────────────────────

    def set_memory(self, key: str, value: str, category: str = "general"):
        """Store a persistent memory fact."""
        now = datetime.now(timezone.utc).timestamp()
        conn = self._conn()
        conn.execute(
            """INSERT INTO memory (key, value, category, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=?, category=?, updated_at=?""",
            (key, value, category, now, now, value, category, now),
        )
        conn.commit()
        conn.close()

    def get_memory(self, key: str) -> str:
        """Retrieve a memory fact by key."""
        conn = self._conn()
        row = conn.execute("SELECT value FROM memory WHERE key = ?", (key,)).fetchone()
        if row:
            conn.execute(
                "UPDATE memory SET access_count = access_count + 1 WHERE key = ?", (key,)
            )
            conn.commit()
        conn.close()
        return row[0] if row else None

    def search_memory(self, query: str, category: str = None, limit: int = 10) -> list:
        """Search persistent memory by key/value content."""
        conn = self._conn()
        sql = "SELECT key, value, category, access_count FROM memory WHERE (key LIKE ? OR value LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY access_count DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def list_memory(self, category: str = None, limit: int = 50) -> list:
        """List persistent memory facts."""
        conn = self._conn()
        if category:
            rows = conn.execute(
                "SELECT key, value, category FROM memory WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT key, value, category FROM memory WHERE 1 ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_memory(self, key: str):
        """Delete a memory fact."""
        conn = self._conn()
        conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        conn.commit()
        conn.close()

    # ── Error Learning ───────────────────────────────────────────

    def log_error(self, error_type: str, description: str,
                  resolution: str = "", session_id: str = None):
        """Log an error for future avoidance."""
        now = datetime.now(timezone.utc).timestamp()
        conn = self._conn()
        conn.execute(
            "INSERT INTO error_log (session_id, error_type, description, resolution, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, error_type, description, resolution, now),
        )
        conn.commit()
        conn.close()

    def get_common_errors(self, limit: int = 10) -> list:
        """Get frequently recurring error patterns."""
        conn = self._conn()
        rows = conn.execute(
            """SELECT error_type, description, resolution, COUNT(*) as count
               FROM error_log GROUP BY error_type, description
               ORDER BY count DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Cleanup ──────────────────────────────────────────────────

    def cleanup_old_sessions(self, keep_days: int = 30):
        """Remove sessions older than keep_days."""
        cutoff = datetime.now(timezone.utc).timestamp() - (keep_days * 86400)
        conn = self._conn()
        conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
        conn.commit()
        conn.close()
