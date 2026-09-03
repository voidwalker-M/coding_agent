"""
context/pg_store.py

Postgres backend with the same public API as MemoryStore.
Activated when MEMORY_DATABASE_URL / DATABASE_URL starts with postgres.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Iterable

from context.memory_store import (
    ROLES,
    ConversationRow,
    UserRow,
    _json_list,
    _parse_list,
)

logger = logging.getLogger(__name__)

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    role           TEXT NOT NULL DEFAULT 'user',
    created_at     DOUBLE PRECISION NOT NULL,
    password_hash  TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    title      TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS stm_turns (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    query_index      INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    created_at       DOUBLE PRECISION NOT NULL
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
    importance       DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    content_hash     TEXT NOT NULL,
    created_at       DOUBLE PRECISION NOT NULL,
    last_access      DOUBLE PRECISION NOT NULL,
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
    value       BYTEA NOT NULL,
    expires_at  DOUBLE PRECISION,
    created_at  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_cache_exp ON cache(expires_at);
"""


def _as_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except Exception:
        return {}


class PostgresMemoryStore:
    """Postgres persistence for users, conversations, STM, LTM, ACL, cache."""

    kind = "postgres"

    def __init__(
        self,
        url: str,
        *,
        clock: Callable[[], float] = time.time,
        retries: int = 30,
        delay_s: float = 1.0,
        kv=None,
    ) -> None:
        import psycopg
        from psycopg.rows import dict_row

        self.url = url
        self._clock = clock
        self._kv = kv
        self._lock = threading.RLock()
        last: Exception | None = None
        conn = None
        for attempt in range(max(1, retries)):
            try:
                conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
                conn.execute("SELECT 1")
                break
            except Exception as exc:
                last = exc
                if attempt + 1 < retries:
                    time.sleep(delay_s)
        if conn is None:
            raise ConnectionError(f"Postgres not reachable: {last}") from last
        self._conn = conn
        with self._lock:
            for stmt in PG_SCHEMA.split(";"):
                stmt = stmt.strip()
                if stmt:
                    self._conn.execute(stmt)
            self._conn.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT"
            )
        self.ensure_user("default", name="default", role="agent")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def ping(self) -> bool:
        with self._lock:
            self._conn.execute("SELECT 1")
        return True

    def _now(self) -> float:
        return float(self._clock())

    def _hot(self):
        kv = self._kv
        return kv if kv is not None and getattr(kv, "kind", "") == "redis" else None

    # -- users --------------------------------------------------------------

    def ensure_user(self, user_id: str, *, name: str | None = None, role: str = "user") -> UserRow:
        user_id = (user_id or "default").strip() or "default"
        if role not in ROLES:
            role = "user"
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
            if row:
                return UserRow(row["id"], row["name"], row["role"], row["created_at"])
            now = self._now()
            self._conn.execute(
                "INSERT INTO users (id, name, role, created_at) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (user_id, name or user_id, role, now),
            )
            row = self._conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
        return UserRow(row["id"], row["name"], row["role"], row["created_at"])

    def get_user(self, user_id: str) -> UserRow | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
        if not row:
            return None
        return UserRow(row["id"], row["name"], row["role"], row["created_at"])

    def list_users(self) -> list[UserRow]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [UserRow(r["id"], r["name"], r["role"], r["created_at"]) for r in rows]

    def set_role(self, user_id: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        self.ensure_user(user_id, role=role)
        with self._lock:
            self._conn.execute("UPDATE users SET role = %s WHERE id = %s", (role, user_id))

    def get_password_hash(self, user_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT password_hash FROM users WHERE id = %s", (user_id,)
            ).fetchone()
        if not row:
            return None
        val = row["password_hash"]
        return str(val) if val else None

    def set_password_hash(self, user_id: str, password_hash: str, *, name: str | None = None) -> UserRow:
        self.ensure_user(user_id, name=name, role="user")
        with self._lock:
            self._conn.execute(
                "UPDATE users SET password_hash = %s, name = COALESCE(%s, name) WHERE id = %s",
                (password_hash, name, user_id),
            )
        user = self.get_user(user_id)
        assert user is not None
        return user

    # -- conversations ------------------------------------------------------

    def create_conversation(self, user_id: str, title: str = "") -> ConversationRow:
        self.ensure_user(user_id)
        cid = uuid.uuid4().hex[:12]
        now = self._now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (cid, user_id, title, now, now),
            )
        return ConversationRow(cid, user_id, title, now, now)

    def get_conversation(self, conversation_id: str) -> ConversationRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversations WHERE id = %s", (conversation_id,)
            ).fetchone()
        if not row:
            return None
        return ConversationRow(row["id"], row["user_id"], row["title"],
                               row["created_at"], row["updated_at"])

    def list_conversations(self, user_id: str) -> list[ConversationRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM conversations WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [ConversationRow(r["id"], r["user_id"], r["title"],
                                r["created_at"], r["updated_at"]) for r in rows]

    def touch_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET updated_at = %s WHERE id = %s",
                (self._now(), conversation_id),
            )

    # -- short-term turns ---------------------------------------------------

    def stm_append(self, conversation_id: str, query_index: int,
                   role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO stm_turns (conversation_id, query_index, role, content, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (conversation_id, int(query_index), role, content, self._now()),
            )
        self.touch_conversation(conversation_id)

    def stm_load(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT query_index, role, content, created_at FROM stm_turns "
                "WHERE conversation_id = %s ORDER BY query_index, id",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stm_trim(self, conversation_id: str, keep_from_index: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM stm_turns WHERE conversation_id = %s AND query_index < %s",
                (conversation_id, int(keep_from_index)),
            )

    def stm_clear(self, conversation_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM stm_turns WHERE conversation_id = %s", (conversation_id,)
            )

    # -- long-term records --------------------------------------------------

    def ltm_upsert(self, rec: dict[str, Any]) -> None:
        params = (
            rec["name"], rec["owner_user_id"], rec.get("conversation_id") or None,
            rec.get("scope", "user"), rec.get("visibility", "private"),
            rec["kind"], rec.get("description", ""), rec["text"],
            _json_list(rec.get("tags")), _json_list(rec.get("files")),
            rec.get("source", ""), rec.get("outcome", ""),
            float(rec.get("importance", 0.5)), rec["content_hash"],
            float(rec.get("created_at", 0.0)), float(rec.get("last_access", 0.0)),
            int(rec.get("access_count", 0)),
            rec.get("status", "approved") or "approved",
        )
        with self._lock:
            self._conn.execute(
                """INSERT INTO ltm_records (
                    name, owner_user_id, conversation_id, scope, visibility, kind,
                    description, text, tags, files, source, outcome, importance,
                    content_hash, created_at, last_access, access_count, status
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (name) DO UPDATE SET
                    owner_user_id=EXCLUDED.owner_user_id,
                    conversation_id=EXCLUDED.conversation_id,
                    scope=EXCLUDED.scope,
                    visibility=EXCLUDED.visibility,
                    kind=EXCLUDED.kind,
                    description=EXCLUDED.description,
                    text=EXCLUDED.text,
                    tags=EXCLUDED.tags,
                    files=EXCLUDED.files,
                    source=EXCLUDED.source,
                    outcome=EXCLUDED.outcome,
                    importance=EXCLUDED.importance,
                    content_hash=EXCLUDED.content_hash,
                    created_at=EXCLUDED.created_at,
                    last_access=EXCLUDED.last_access,
                    access_count=EXCLUDED.access_count,
                    status=EXCLUDED.status
                """,
                params,
            )

    def ltm_get(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM ltm_records WHERE name = %s", (name,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def ltm_all(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM ltm_records").fetchall()
        return [self._row_to_record(r) for r in rows]

    def ltm_find_hash(self, content_hash: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if owner_user_id:
                row = self._conn.execute(
                    "SELECT * FROM ltm_records WHERE content_hash = %s AND owner_user_id = %s",
                    (content_hash, owner_user_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM ltm_records WHERE content_hash = %s", (content_hash,)
                ).fetchone()
        return self._row_to_record(row) if row else None

    def ltm_delete(self, name: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM ltm_acl WHERE record_name = %s", (name,))
            self._conn.execute("DELETE FROM ltm_records WHERE name = %s", (name,))

    def ltm_replace_all(self, records: Iterable[dict[str, Any]]) -> None:
        keep = {r["name"] for r in records}
        with self._lock:
            existing = [row["name"] for row in self._conn.execute("SELECT name FROM ltm_records")]
        for name in existing:
            if name not in keep:
                self.ltm_delete(name)
        for rec in records:
            self.ltm_upsert(rec)

    def _row_to_record(self, row: Any) -> dict[str, Any]:
        row = _as_dict(row)
        return {
            "name": row["name"],
            "owner_user_id": row["owner_user_id"],
            "conversation_id": row.get("conversation_id") or "",
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
            "status": row.get("status") or "approved",
        }

    # -- ACL ----------------------------------------------------------------

    def acl_grant(self, record_name: str, principal: str, perm: str = "read") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO ltm_acl (record_name, principal, perm) VALUES (%s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                (record_name, principal, perm),
            )

    def acl_revoke(self, record_name: str, principal: str, perm: str | None = None) -> None:
        with self._lock:
            if perm:
                self._conn.execute(
                    "DELETE FROM ltm_acl WHERE record_name = %s AND principal = %s AND perm = %s",
                    (record_name, principal, perm),
                )
            else:
                self._conn.execute(
                    "DELETE FROM ltm_acl WHERE record_name = %s AND principal = %s",
                    (record_name, principal),
                )

    def acl_for(self, record_name: str) -> set[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT principal, perm FROM ltm_acl WHERE record_name = %s", (record_name,)
            ).fetchall()
        return {(r["principal"], r["perm"]) for r in rows}

    def acl_map(self) -> dict[str, set[tuple[str, str]]]:
        out: dict[str, set[tuple[str, str]]] = {}
        with self._lock:
            rows = self._conn.execute("SELECT record_name, principal, perm FROM ltm_acl").fetchall()
        for r in rows:
            out.setdefault(r["record_name"], set()).add((r["principal"], r["perm"]))
        return out

    # -- cache --------------------------------------------------------------

    def cache_get(self, namespace: str, key: str) -> bytes | None:
        hot = self._hot()
        if hot is not None:
            hit = hot.get(namespace, key)
            if hit is not None:
                return hit
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache WHERE namespace = %s AND key = %s",
                (namespace, key),
            ).fetchone()
        if not row:
            return None
        exp = row["expires_at"]
        if exp is not None and exp < self._now():
            self.cache_delete(namespace, key)
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
        with self._lock:
            self._conn.execute(
                """INSERT INTO cache (namespace, key, value, expires_at, created_at)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (namespace, key) DO UPDATE SET
                     value=EXCLUDED.value, expires_at=EXCLUDED.expires_at,
                     created_at=EXCLUDED.created_at
                """,
                (namespace, key, payload, exp, self._now()),
            )

    def cache_delete(self, namespace: str, key: str | None = None) -> None:
        hot = self._hot()
        if hot is not None:
            hot.delete(namespace, key)
        with self._lock:
            if key is None:
                self._conn.execute("DELETE FROM cache WHERE namespace = %s", (namespace,))
            else:
                self._conn.execute(
                    "DELETE FROM cache WHERE namespace = %s AND key = %s", (namespace, key)
                )

    def cache_purge_expired(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < %s",
                (self._now(),),
            )
        return int(cur.rowcount or 0)
