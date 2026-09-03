"""
context/memory_store.py

SQLite backend for multi-user / multi-conversation / ACL memory.

Tables
------
  users            identity + role (guest | user | agent | admin)
  conversations    a dialogue owned by a user
  stm_turns        short-term window: turns belonging to a conversation query
  ltm_records      durable memories, scoped global | user | conversation
  ltm_acl          extra read/write grants: principal = user:<id> | role:<role>
  cache            namespaced KV (embed vectors, index snapshots, recall hints)

The store is the source of truth. Markdown files under the memory dir remain a
human-readable cache/export, not the primary index.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    role           TEXT NOT NULL DEFAULT 'user',
    created_at     REAL NOT NULL,
    password_hash  TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    title      TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stm_turns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    query_index      INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stm_conv ON stm_turns(conversation_id, query_index);

CREATE TABLE IF NOT EXISTS ltm_records (
    name             TEXT PRIMARY KEY,
    owner_user_id    TEXT NOT NULL,
    conversation_id  TEXT,
    scope            TEXT NOT NULL DEFAULT 'user',
    visibility       TEXT NOT NULL DEFAULT 'private',
    kind             TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    text             TEXT NOT NULL,
    tags             TEXT NOT NULL DEFAULT '[]',
    files            TEXT NOT NULL DEFAULT '[]',
    source           TEXT NOT NULL DEFAULT '',
    outcome          TEXT NOT NULL DEFAULT '',
    importance       REAL NOT NULL DEFAULT 0.5,
    content_hash     TEXT NOT NULL,
    created_at       REAL NOT NULL,
    last_access      REAL NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL DEFAULT 'approved'
);
CREATE INDEX IF NOT EXISTS idx_ltm_owner ON ltm_records(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_ltm_hash ON ltm_records(content_hash);
CREATE INDEX IF NOT EXISTS idx_ltm_conv ON ltm_records(conversation_id);

CREATE TABLE IF NOT EXISTS ltm_acl (
    record_name  TEXT NOT NULL,
    principal    TEXT NOT NULL,
    perm         TEXT NOT NULL,
    PRIMARY KEY (record_name, principal, perm)
);

CREATE TABLE IF NOT EXISTS cache (
    namespace   TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       BLOB NOT NULL,
    expires_at  REAL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_cache_exp ON cache(expires_at);
"""

ROLES = ("guest", "user", "agent", "admin")
ROLE_RANK = {"guest": 0, "user": 1, "agent": 2, "admin": 3}
SCOPES = ("global", "user", "conversation")
VISIBILITIES = ("public", "shared", "private")


@dataclass(frozen=True)
class Actor:
    """Who is reading/writing memory right now."""
    user_id: str = "default"
    role: str = "agent"
    conversation_id: str | None = None

    def principal(self) -> str:
        return f"user:{self.user_id}"

    def role_principal(self) -> str:
        return f"role:{self.role}"


@dataclass
class UserRow:
    id: str
    name: str
    role: str
    created_at: float


@dataclass
class ConversationRow:
    id: str
    user_id: str
    title: str
    created_at: float
    updated_at: float


def _json_list(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(list(value or []), ensure_ascii=False)


def _parse_list(raw: str) -> list[str]:
    try:
        val = json.loads(raw or "[]")
        return list(val) if isinstance(val, list) else []
    except Exception:
        return []


class MemoryStore:
    """SQLite persistence for users, conversations, STM turns, LTM, ACL, cache."""

    kind = "sqlite"

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time,
                 kv: Any | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._kv = kv
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self.ensure_user("default", name="default", role="agent")

    def ping(self) -> bool:
        self._conn.execute("SELECT 1").fetchone()
        return True

    def _hot(self):
        kv = getattr(self, "_kv", None)
        return kv if kv is not None and getattr(kv, "kind", "") == "redis" else None

    def _migrate(self) -> None:
        rec_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(ltm_records)")}
        if "status" not in rec_cols:
            self._conn.execute(
                "ALTER TABLE ltm_records ADD COLUMN status TEXT NOT NULL DEFAULT 'approved'"
            )
        user_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(users)")}
        if "password_hash" not in user_cols:
            self._conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _now(self) -> float:
        return float(self._clock())

    # -- users --------------------------------------------------------------

    def ensure_user(self, user_id: str, *, name: str | None = None, role: str = "user") -> UserRow:
        user_id = (user_id or "default").strip() or "default"
        if role not in ROLES:
            role = "user"
        row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            return UserRow(row["id"], row["name"], row["role"], row["created_at"])
        now = self._now()
        self._conn.execute(
            "INSERT INTO users (id, name, role, created_at) VALUES (?, ?, ?, ?)",
            (user_id, name or user_id, role, now),
        )
        self._conn.commit()
        return UserRow(user_id, name or user_id, role, now)

    def get_user(self, user_id: str) -> UserRow | None:
        row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return UserRow(row["id"], row["name"], row["role"], row["created_at"])

    def list_users(self) -> list[UserRow]:
        rows = self._conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [UserRow(r["id"], r["name"], r["role"], r["created_at"]) for r in rows]

    def set_role(self, user_id: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        self.ensure_user(user_id, role=role)
        self._conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        self._conn.commit()

    def get_password_hash(self, user_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        val = row["password_hash"]
        return str(val) if val else None

    def set_password_hash(self, user_id: str, password_hash: str, *, name: str | None = None) -> UserRow:
        self.ensure_user(user_id, name=name, role="user")
        self._conn.execute(
            "UPDATE users SET password_hash = ?, name = COALESCE(?, name) WHERE id = ?",
            (password_hash, name, user_id),
        )
        self._conn.commit()
        user = self.get_user(user_id)
        assert user is not None
        return user

    # -- conversations ------------------------------------------------------

    def create_conversation(self, user_id: str, title: str = "") -> ConversationRow:
        self.ensure_user(user_id)
        cid = uuid.uuid4().hex[:12]
        now = self._now()
        self._conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, user_id, title, now, now),
        )
        self._conn.commit()
        return ConversationRow(cid, user_id, title, now, now)

    def get_conversation(self, conversation_id: str) -> ConversationRow | None:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if not row:
            return None
        return ConversationRow(row["id"], row["user_id"], row["title"],
                               row["created_at"], row["updated_at"])

    def list_conversations(self, user_id: str) -> list[ConversationRow]:
        rows = self._conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [ConversationRow(r["id"], r["user_id"], r["title"],
                                r["created_at"], r["updated_at"]) for r in rows]

    def touch_conversation(self, conversation_id: str) -> None:
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (self._now(), conversation_id),
        )
        self._conn.commit()

    # -- short-term turns ---------------------------------------------------

    def stm_append(self, conversation_id: str, query_index: int,
                   role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO stm_turns (conversation_id, query_index, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, int(query_index), role, content, self._now()),
        )
        self.touch_conversation(conversation_id)

    def stm_load(self, conversation_id: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT query_index, role, content, created_at FROM stm_turns "
            "WHERE conversation_id = ? ORDER BY query_index, id",
            (conversation_id,),
        ).fetchall())

    def stm_trim(self, conversation_id: str, keep_from_index: int) -> None:
        self._conn.execute(
            "DELETE FROM stm_turns WHERE conversation_id = ? AND query_index < ?",
            (conversation_id, int(keep_from_index)),
        )
        self._conn.commit()

    def stm_clear(self, conversation_id: str) -> None:
        self._conn.execute(
            "DELETE FROM stm_turns WHERE conversation_id = ?", (conversation_id,)
        )
        self._conn.commit()

    # -- long-term records --------------------------------------------------

    def ltm_upsert(self, rec: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO ltm_records (
                name, owner_user_id, conversation_id, scope, visibility, kind,
                description, text, tags, files, source, outcome, importance,
                content_hash, created_at, last_access, access_count, status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                owner_user_id=excluded.owner_user_id,
                conversation_id=excluded.conversation_id,
                scope=excluded.scope,
                visibility=excluded.visibility,
                kind=excluded.kind,
                description=excluded.description,
                text=excluded.text,
                tags=excluded.tags,
                files=excluded.files,
                source=excluded.source,
                outcome=excluded.outcome,
                importance=excluded.importance,
                content_hash=excluded.content_hash,
                created_at=excluded.created_at,
                last_access=excluded.last_access,
                access_count=excluded.access_count,
                status=excluded.status
            """,
            (
                rec["name"], rec["owner_user_id"], rec.get("conversation_id") or None,
                rec.get("scope", "user"), rec.get("visibility", "private"),
                rec["kind"], rec.get("description", ""), rec["text"],
                _json_list(rec.get("tags")), _json_list(rec.get("files")),
                rec.get("source", ""), rec.get("outcome", ""),
                float(rec.get("importance", 0.5)), rec["content_hash"],
                float(rec.get("created_at", 0.0)), float(rec.get("last_access", 0.0)),
                int(rec.get("access_count", 0)),
                rec.get("status", "approved") or "approved",
            ),
        )
        self._conn.commit()

    def ltm_get(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM ltm_records WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def ltm_all(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM ltm_records").fetchall()
        return [self._row_to_record(r) for r in rows]

    def ltm_find_hash(self, content_hash: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        if owner_user_id:
            row = self._conn.execute(
                "SELECT * FROM ltm_records WHERE content_hash = ? AND owner_user_id = ?",
                (content_hash, owner_user_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM ltm_records WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def ltm_delete(self, name: str) -> None:
        self._conn.execute("DELETE FROM ltm_acl WHERE record_name = ?", (name,))
        self._conn.execute("DELETE FROM ltm_records WHERE name = ?", (name,))
        self._conn.commit()

    def ltm_replace_all(self, records: Iterable[dict[str, Any]]) -> None:
        keep = {r["name"] for r in records}
        existing = [row["name"] for row in self._conn.execute("SELECT name FROM ltm_records")]
        for name in existing:
            if name not in keep:
                self.ltm_delete(name)
        for rec in records:
            self.ltm_upsert(rec)

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "name": row["name"],
            "owner_user_id": row["owner_user_id"],
            "conversation_id": row["conversation_id"] or "",
            "scope": row["scope"],
            "visibility": row["visibility"],
            "kind": row["kind"],
            "description": row["description"],
            "text": row["text"],
            "tags": _parse_list(row["tags"]),
            "files": _parse_list(row["files"]),
            "source": row["source"],
            "outcome": row["outcome"],
            "importance": row["importance"],
            "content_hash": row["content_hash"],
            "created_at": row["created_at"],
            "last_access": row["last_access"],
            "access_count": row["access_count"],
            "status": row["status"] if "status" in row.keys() else "approved",
        }

    # -- ACL ----------------------------------------------------------------

    def acl_grant(self, record_name: str, principal: str, perm: str = "read") -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO ltm_acl (record_name, principal, perm) VALUES (?, ?, ?)",
            (record_name, principal, perm),
        )
        self._conn.commit()

    def acl_revoke(self, record_name: str, principal: str, perm: str | None = None) -> None:
        if perm:
            self._conn.execute(
                "DELETE FROM ltm_acl WHERE record_name = ? AND principal = ? AND perm = ?",
                (record_name, principal, perm),
            )
        else:
            self._conn.execute(
                "DELETE FROM ltm_acl WHERE record_name = ? AND principal = ?",
                (record_name, principal),
            )
        self._conn.commit()

    def acl_for(self, record_name: str) -> set[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT principal, perm FROM ltm_acl WHERE record_name = ?", (record_name,)
        ).fetchall()
        return {(r["principal"], r["perm"]) for r in rows}

    def acl_map(self) -> dict[str, set[tuple[str, str]]]:
        out: dict[str, set[tuple[str, str]]] = {}
        for r in self._conn.execute("SELECT record_name, principal, perm FROM ltm_acl"):
            out.setdefault(r["record_name"], set()).add((r["principal"], r["perm"]))
        return out

    # -- cache --------------------------------------------------------------

    def cache_get(self, namespace: str, key: str) -> bytes | None:
        hot = self._hot()
        if hot is not None:
            hit = hot.get(namespace, key)
            if hit is not None:
                return hit
        row = self._conn.execute(
            "SELECT value, expires_at FROM cache WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
        if not row:
            return None
        exp = row["expires_at"]
        if exp is not None and exp < self._now():
            self._conn.execute(
                "DELETE FROM cache WHERE namespace = ? AND key = ?", (namespace, key)
            )
            self._conn.commit()
            return None
        value = bytes(row["value"])
        if hot is not None:
            hot.set(namespace, key, value)
        return value

    def cache_set(self, namespace: str, key: str, value: bytes,
                  *, ttl_s: float | None = None) -> None:
        payload = bytes(value)
        hot = self._hot()
        if hot is not None:
            hot.set(namespace, key, payload, ttl_s=ttl_s)
        exp = (self._now() + ttl_s) if ttl_s else None
        self._conn.execute(
            """INSERT INTO cache (namespace, key, value, expires_at, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(namespace, key) DO UPDATE SET
                 value=excluded.value, expires_at=excluded.expires_at, created_at=excluded.created_at
            """,
            (namespace, key, payload, exp, self._now()),
        )
        self._conn.commit()

    def cache_delete(self, namespace: str, key: str | None = None) -> None:
        hot = self._hot()
        if hot is not None:
            hot.delete(namespace, key)
        if key is None:
            self._conn.execute("DELETE FROM cache WHERE namespace = ?", (namespace,))
        else:
            self._conn.execute(
                "DELETE FROM cache WHERE namespace = ? AND key = ?", (namespace, key)
            )
        self._conn.commit()

    def cache_purge_expired(self) -> int:
        cur = self._conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (self._now(),),
        )
        self._conn.commit()
        return int(cur.rowcount or 0)


def can_read(actor: Actor, rec: dict[str, Any],
             acl: set[tuple[str, str]] | None = None) -> bool:
    """Permission check for recalling a long-term record."""
    if actor.role == "admin":
        return True
    if rec.get("visibility") == "public":
        return True
    if rec.get("owner_user_id") == actor.user_id:
        return True
    if (rec.get("scope") == "conversation"
            and rec.get("conversation_id")
            and rec.get("conversation_id") == actor.conversation_id):
        return True
    if rec.get("visibility") == "shared" and acl:
        principals = {actor.principal(), actor.role_principal()}
        for principal, perm in acl:
            if principal in principals and perm in ("read", "write"):
                return True
    return False


def can_write(actor: Actor, rec: dict[str, Any],
              acl: set[tuple[str, str]] | None = None) -> bool:
    if actor.role == "admin":
        return True
    if actor.role == "guest":
        return False
    if rec.get("owner_user_id") == actor.user_id:
        return True
    if rec.get("visibility") == "shared" and acl:
        principals = {actor.principal(), actor.role_principal()}
        for principal, perm in acl:
            if principal in principals and perm == "write":
                return True
    return False
