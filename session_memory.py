"""
Apex Harness — Session Memory Engine (session_memory.py)
Stores decisions, touched files, TODOs, and architectural notes across sessions
in a dedicated SQLite database (~/.apex_sessions/sessions.db).
"""

import os
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


class SessionMemory:
    """Manages persistent session history, decisions, and task continuity."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = Path(db_path).resolve()
        else:
            base = os.environ.get("APEX_SESSION_DIR", str(Path.home() / ".apex_sessions"))
            p = Path(base).resolve()
            p.mkdir(parents=True, exist_ok=True)
            self.db_path = p / "sessions.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.db_path.parent / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    model TEXT DEFAULT '',
                    summary TEXT DEFAULT ''
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);")
            conn.commit()

    def ensure_session_id(self, session_id: Optional[str] = None, model: str = "") -> str:
        """Ensure a stable session ID exists for this terminal session and persist it in the environment."""
        current = session_id or os.environ.get("APEX_SESSION_ID")
        if not current:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            current = f"apex-{stamp}-{uuid.uuid4().hex[:8]}"
        os.environ["APEX_SESSION_ID"] = current
        self.open_session(current, model=model)
        return current

    def open_session(self, session_id: str, model: str = "") -> None:
        """Register or update an active session."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO sessions (id, started_at, last_active, model, summary)
                VALUES (?, ?, ?, ?, '')
                ON CONFLICT(id) DO UPDATE SET last_active = excluded.last_active, model = CASE WHEN excluded.model != '' THEN excluded.model ELSE model END;
            """, (session_id, now, now, model))
            conn.commit()

    def log_event(self, session_id: str, event_type: str, content: str) -> int:
        """Log an event (decision, file_touched, todo_open, todo_done, note) for the session."""
        self.open_session(session_id)
        now = datetime.now(timezone.utc).isoformat()
        clean_content = content.strip()

        with self._get_connection() as conn:
            # Auto-generate summary for session if it doesn't have one and this is the first user message
            if event_type == "user_message":
                cur_sum = conn.execute("SELECT summary FROM sessions WHERE id = ?;", (session_id,)).fetchone()
                if cur_sum and (not cur_sum["summary"] or cur_sum["summary"].startswith("Sessão ")):
                    first_line = clean_content.split("\n")[0].strip()
                    auto_title = (first_line[:50] + "…") if len(first_line) > 50 else first_line
                    if auto_title:
                        conn.execute("UPDATE sessions SET summary = ? WHERE id = ?;", (auto_title, session_id))

            cur = conn.execute("""
                INSERT INTO events (session_id, ts, event_type, content)
                VALUES (?, ?, ?, ?);
            """, (session_id, now, event_type, clean_content))
            conn.commit()
            last_id = cur.lastrowid

        # Persist session to JSON file on disk
        try:
            self.export_session_file(session_id)
        except Exception:
            pass

        return last_id

    def export_session_file(self, session_id: str) -> Optional[Path]:
        """Export session metadata, messages, and events to a standalone JSON file on disk."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE id = ?;", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            sess_dict = dict(row)

        events = self.get_events(session_id)
        messages = []
        for ev in events:
            if ev["event_type"] in ("user_message", "assistant_message"):
                role = "user" if ev["event_type"] == "user_message" else "assistant"
                messages.append({
                    "role": role,
                    "content": ev["content"],
                    "ts": ev["ts"]
                })

        data = {
            "id": sess_dict["id"],
            "started_at": sess_dict["started_at"],
            "last_active": sess_dict["last_active"],
            "model": sess_dict["model"],
            "summary": sess_dict["summary"],
            "messages": messages,
            "events": events
        }

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        target = self.sessions_dir / f"{session_id}.json"
        tmp_target = self.sessions_dir / f"{session_id}.json.tmp"
        with open(tmp_target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_target.replace(target)
        return target

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full session data including messages, either from SQLite or disk."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE id = ?;", (session_id,))
            row = cur.fetchone()
            if row:
                sess = dict(row)
                events = self.get_events(session_id)
                messages = []
                for ev in events:
                    if ev["event_type"] in ("user_message", "assistant_message"):
                        role = "user" if ev["event_type"] == "user_message" else "assistant"
                        messages.append({
                            "role": role,
                            "content": ev["content"],
                            "ts": ev["ts"]
                        })
                sess["messages"] = messages
                sess["events"] = events
                return sess

        # Fallback to disk JSON file
        target = self.sessions_dir / f"{session_id}.json"
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Re-index into sqlite
                self.open_session(data["id"], model=data.get("model", ""))
                if data.get("summary"):
                    self.update_summary(data["id"], data["summary"])
                for ev in data.get("events", []):
                    with self._get_connection() as conn:
                        conn.execute("""
                            INSERT INTO events (session_id, ts, event_type, content)
                            VALUES (?, ?, ?, ?);
                        """, (data["id"], ev.get("ts", datetime.now(timezone.utc).isoformat()), ev.get("event_type", "note"), ev.get("content", "")))
                        conn.commit()
                return data
            except Exception:
                pass
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete session from SQLite and remove its JSON file from disk."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM events WHERE session_id = ?;", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?;", (session_id,))
            conn.commit()

        target = self.sessions_dir / f"{session_id}.json"
        if target.exists():
            try:
                target.unlink()
            except Exception:
                pass

        if os.environ.get("APEX_SESSION_ID") == session_id:
            os.environ.pop("APEX_SESSION_ID", None)
        return True

    def log_decision(self, session_id: str, text: str) -> int:
        """Record an architectural or design decision."""
        return self.log_event(session_id, "decision", text)

    def log_file_touched(self, session_id: str, file_path: str) -> int:
        """Record a file read or modified during the session."""
        resolved = str(Path(file_path).resolve())
        return self.log_event(session_id, "file_touched", resolved)

    def add_todo(self, session_id: str, text: str) -> int:
        """Add an open task item for the session."""
        return self.log_event(session_id, "todo_open", text)

    def close_todo(self, session_id: str, todo_id: int) -> bool:
        """Mark a TODO as completed."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT content FROM events WHERE id = ? AND session_id = ?;
            """, (todo_id, session_id))
            row = cur.fetchone()
            if not row:
                return False
            conn.execute("""
                UPDATE events SET event_type = 'todo_done' WHERE id = ?;
            """, (todo_id,))
            conn.commit()
            return True

    def add_note(self, session_id: str, text: str) -> int:
        """Add a general session note."""
        return self.log_event(session_id, "note", text)

    def update_summary(self, session_id: str, summary: str) -> None:
        """Update overall session summary text."""
        with self._get_connection() as conn:
            conn.execute("UPDATE sessions SET summary = ? WHERE id = ?;", (summary, session_id))
            conn.commit()
        try:
            self.export_session_file(session_id)
        except Exception:
            pass

    def get_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events for a given session."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT id, ts, event_type, content FROM events
                WHERE session_id = ? ORDER BY id ASC;
            """, (session_id,))
            return [dict(r) for r in cur.fetchall()]

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List past sessions ordered by most recently active, syncing from disk."""
        # Sync any standalone JSON session files in sessions_dir
        if self.sessions_dir.exists():
            try:
                for json_file in self.sessions_dir.glob("*.json"):
                    sid = json_file.stem
                    with self._get_connection() as conn:
                        cur = conn.execute("SELECT 1 FROM sessions WHERE id = ?;", (sid,))
                        if not cur.fetchone():
                            try:
                                with open(json_file, "r", encoding="utf-8") as f:
                                    sdata = json.load(f)
                                self.open_session(sdata["id"], model=sdata.get("model", ""))
                                if sdata.get("summary"):
                                    self.update_summary(sdata["id"], sdata["summary"])
                            except Exception:
                                pass
            except Exception:
                pass

        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT s.id, s.started_at, s.last_active, s.model, s.summary,
                       COUNT(e.id) as event_count,
                       SUM(CASE WHEN e.event_type IN ('user_message', 'assistant_message') THEN 1 ELSE 0 END) as message_count
                FROM sessions s
                LEFT JOIN events e ON s.id = e.session_id
                GROUP BY s.id
                ORDER BY s.last_active DESC
                LIMIT ?;
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_active_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Compatibility wrapper for the web UI and CLI status screens."""
        return self.list_sessions(limit=limit)

    def resume_session(self, session_id: str) -> str:
        """Build a comprehensive textual context digest of a prior session."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE id = ?;", (session_id,))
            sess = cur.fetchone()
            if not sess:
                return f"Session '{session_id}' not found."

        events = self.get_events(session_id)
        if not events:
            return f"Session '{session_id}' has no recorded history."

        decisions = [e["content"] for e in events if e["event_type"] == "decision"]
        files = sorted(list({e["content"] for e in events if e["event_type"] == "file_touched"}))
        open_todos = [(e["id"], e["content"]) for e in events if e["event_type"] == "todo_open"]
        done_todos = [e["content"] for e in events if e["event_type"] == "todo_done"]
        notes = [e["content"] for e in events if e["event_type"] == "note"]

        lines = [
            f"=== RESUMED SESSION: {session_id} ===",
            f"Started: {sess['started_at']} | Last active: {sess['last_active']} | Model: {sess['model'] or 'N/A'}"
        ]
        if sess["summary"]:
            lines.extend(["", f"Summary: {sess['summary']}"])

        if decisions:
            lines.extend(["", "Decisions Made:"])
            for d in decisions:
                lines.append(f"  • {d}")

        if files:
            lines.extend(["", "Files Touched:"])
            for f in files:
                lines.append(f"  • {f}")

        if open_todos:
            lines.extend(["", "Open Tasks (TODOs):"])
            for tid, t in open_todos:
                lines.append(f"  [ ] (#{tid}) {t}")

        if done_todos:
            lines.extend(["", "Completed Tasks:"])
            for t in done_todos:
                lines.append(f"  [✓] {t}")

        if notes:
            lines.extend(["", "Notes:"])
            for n in notes:
                lines.append(f"  - {n}")

        return "\n".join(lines)


# Global singleton instance
_GLOBAL_SESSION_MEMORY: Optional[SessionMemory] = None

def get_session_memory(db_path: Optional[str] = None) -> SessionMemory:
    global _GLOBAL_SESSION_MEMORY
    if _GLOBAL_SESSION_MEMORY is None or db_path is not None:
        _GLOBAL_SESSION_MEMORY = SessionMemory(db_path)
    return _GLOBAL_SESSION_MEMORY
