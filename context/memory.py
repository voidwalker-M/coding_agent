"""
context/memory.py

Cursor / Claude Code-shaped memory for a coding agent:

  Project rules     — instructions *you* wrote (AGENTS.md, CLAUDE.md,
                      .cursor/rules). Always-on, not this module. See rules.py.
  ShortTermMemory   — the current conversation: last *n* user queries in
                      process memory (the thread *is* STM; SQLite persist is optional).
  LongTermMemory    — small durable *facts* (Memories). Explicit remember() is
                      approved immediately. Auto-proposed notes can sit pending
                      until approve(). Episodic run logs are stored but not
                      stuffed into every prompt.

SQLite (`memory.db`) is the source of truth for memories; markdown is an export.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from context.memory_store import (
    SCOPES,
    VISIBILITIES,
    Actor,
    MemoryStore,
    can_read,
    can_write,
)

logger = logging.getLogger(__name__)

# A code/text friendly tokenizer (local, so this module has no numpy import at
# load time — the optional dense path imports numpy lazily).
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|[0-9]+")
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "into", "use",
    "was", "were", "are", "has", "have", "not", "but", "all", "any", "you",
    "your", "our", "its", "then", "than", "when", "what", "which", "who",
})

# Memory types (Claude Code's four + our two run-derived kinds).
KINDS: frozenset[str] = frozenset({
    "user", "feedback", "project", "reference", "episodic", "semantic",
    "conversation",  # searchable archived turns (overflow / chat history)
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, split identifiers on camelCase/snake_case, drop stopwords."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        for part in _CAMEL_RE.findall(raw) or [raw]:
            p = part.lower()
            if len(p) >= 2 and p not in _STOPWORDS:
                out.append(p)
    return out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:48].strip("-")
    return slug or fallback


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _rrf(rankings: Sequence[Sequence[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over id-rankings (mirrors context/rag.py)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, rid in enumerate(ranking):
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# ShortTermMemory — conversation window of n queries
# ---------------------------------------------------------------------------

@dataclass
class ConversationQuery:
    """One user query in a conversation, plus the agent replies that followed it."""
    index: int
    user_text: str
    responses: list[str] = field(default_factory=list)
    created_at: float = 0.0


class ShortTermMemory:
    """Conversation-scoped short-term memory: last *n* user queries.

    This is the chat window, not the ReAct step history. Each `append_query`
    starts a new slot; older slots fall out of the window. A small scratchpad
    (notes / files examined / overflow summary) still rides along so a single
    query's ReAct loop does not silently lose trimmed turns.

    Bind a MemoryStore + conversation_id to persist turns across process
    restarts; without a store it is in-memory only (tests, single run).

    The *prompt* only shows the last n queries (working window). Older turns
    stay in the store and are distilled into LongTermMemory (facts + searchable
    conversation snippets) when `on_overflow` is set — same split ChatGPT uses
    between the open thread and Memory / chat history.
    """

    def __init__(
        self,
        window_queries: int = 10,
        max_notes: int = 12,
        max_summary_chars: int = 1500,
        *,
        store: MemoryStore | None = None,
        user_id: str = "default",
        conversation_id: str | None = None,
        clock: Callable[[], float] = time.time,
        on_overflow: Callable[[list], None] | None = None,
    ) -> None:
        self.window_queries = max(1, int(window_queries))
        self._notes: list[str] = []
        self._files: list[str] = []
        self._summary: str = ""
        self._max_notes = max_notes
        self._max_summary = max_summary_chars
        self._queries: list[ConversationQuery] = []
        self._store = store
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._clock = clock
        self._on_overflow = on_overflow
        if store is not None and conversation_id:
            self._hydrate()

    # -- conversation window ------------------------------------------------

    def bind_conversation(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        if self._store is not None:
            self._hydrate()

    def begin_query(self, text: str) -> None:
        """Start a query slot; no-op if the current slot already has this text."""
        text = (text or "").strip()
        if not text:
            return
        if self._queries and self._queries[-1].user_text == text:
            return
        self.append_query(text)

    def append_query(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        idx = (self._queries[-1].index + 1) if self._queries else 0
        q = ConversationQuery(index=idx, user_text=text, created_at=self._clock())
        self._queries.append(q)
        self._persist_turn(idx, "user", text)
        self._trim_window()

    def append_response(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if not self._queries:
            self.append_query("(context)")
        self._queries[-1].responses.append(text)
        self._persist_turn(self._queries[-1].index, "assistant", text)

    def _trim_window(self) -> None:
        overflow = len(self._queries) - self.window_queries
        if overflow <= 0:
            return
        dropped = self._queries[:overflow]
        self._queries = self._queries[overflow:]
        # Keep the full transcript in the store (chat-history layer). Only the
        # in-memory prompt window shrinks. Distill durable facts via callback.
        if self._on_overflow is not None:
            try:
                self._on_overflow(dropped)
            except Exception as exc:
                logger.debug("stm overflow distill failed: %s", exc)
        if dropped:
            hint = f"[archived {len(dropped)} earlier quer" + ("y]" if len(dropped) == 1 else "ies]")
            self.fold(hint)

    def _persist_turn(self, query_index: int, role: str, content: str) -> None:
        if self._store is None or not self.conversation_id:
            return
        try:
            self._store.stm_append(self.conversation_id, query_index, role, content)
        except Exception as exc:
            logger.debug("stm persist failed: %s", exc)

    def _hydrate(self) -> None:
        if self._store is None or not self.conversation_id:
            return
        rows = self._store.stm_load(self.conversation_id)
        by_idx: dict[int, ConversationQuery] = {}
        for row in rows:
            idx = int(row["query_index"])
            q = by_idx.get(idx)
            if q is None:
                q = ConversationQuery(index=idx, user_text="", created_at=float(row["created_at"]))
                by_idx[idx] = q
            if row["role"] == "user" and not q.user_text:
                q.user_text = row["content"]
            else:
                q.responses.append(row["content"])
        self._queries = [by_idx[i] for i in sorted(by_idx)]
        if len(self._queries) > self.window_queries:
            self._queries = self._queries[-self.window_queries:]

    @property
    def queries(self) -> list[ConversationQuery]:
        return list(self._queries)

    # -- within-query scratchpad (ReAct overflow) ---------------------------

    def add_note(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._notes.append(text)
        if len(self._notes) > self._max_notes:
            self._notes = self._notes[-self._max_notes:]

    def note_file(self, path: str) -> None:
        path = (path or "").strip()
        if path and path not in self._files:
            self._files.append(path)

    def fold(self, evicted_text: str) -> None:
        """Fold an evicted ReAct history turn into the overflow summary (bounded)."""
        evicted_text = (evicted_text or "").strip()
        if not evicted_text:
            return
        snippet = evicted_text.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        self._summary = (self._summary + " " + snippet).strip()
        if len(self._summary) > self._max_summary:
            self._summary = "…" + self._summary[-self._max_summary:]

    def render(self) -> str:
        """Compact block for the system prompt; empty string when nothing to show.

        Prior queries are one-liners (the current query already lives in
        ConversationHistory — do not duplicate replies).
        """
        prior = self._queries[:-1] if self._queries else []
        if not (prior or self._notes or self._files or self._summary):
            return ""
        lines: list[str] = []
        if prior:
            lines.append(f"## Conversation window (last {self.window_queries} queries)")
            for q in prior:
                text = q.user_text.replace("\n", " ")
                if len(text) > 120:
                    text = text[:120] + "…"
                lines.append(f"- Q{q.index}: {text}")
        elif self._notes or self._files or self._summary:
            lines.append("## Working memory (this task so far)")
        if self._summary:
            lines.append(f"Earlier steps (summarized): {self._summary}")
        if self._files:
            lines.append("Files examined: " + ", ".join(self._files[-12:]))
        if self._notes:
            lines.append("Notes:")
            lines.extend(f"- {n}" for n in self._notes)
        return "\n".join(lines)

    def make_evict_callback(self) -> Callable[[dict], None]:
        """A callback for ConversationHistory(on_evict=…) that folds drops in."""
        def _cb(message: dict) -> None:
            self.fold(message.get("content", ""))
        return _cb

    def clear(self) -> None:
        self._notes.clear()
        self._files.clear()
        self._summary = ""
        self._queries.clear()
        if self._store is not None and self.conversation_id:
            try:
                self._store.stm_clear(self.conversation_id)
            except Exception as exc:
                logger.debug("stm clear failed: %s", exc)

    def to_state(self) -> dict:
        return {
            "notes": list(self._notes),
            "files": list(self._files),
            "summary": self._summary,
            "window_queries": self.window_queries,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "queries": [
                {
                    "index": q.index,
                    "user_text": q.user_text,
                    "responses": list(q.responses),
                    "created_at": q.created_at,
                }
                for q in self._queries
            ],
        }

    def load_state(self, data: dict) -> None:
        self._notes = list(data.get("notes", []))
        self._files = list(data.get("files", []))
        self._summary = str(data.get("summary", ""))
        self.window_queries = int(data.get("window_queries", self.window_queries) or self.window_queries)
        self.user_id = str(data.get("user_id") or self.user_id)
        if data.get("conversation_id"):
            self.conversation_id = str(data["conversation_id"])
        self._queries = [
            ConversationQuery(
                index=int(q.get("index", i)),
                user_text=str(q.get("user_text", "")),
                responses=list(q.get("responses") or []),
                created_at=float(q.get("created_at", 0.0)),
            )
            for i, q in enumerate(data.get("queries") or [])
        ]


# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """One long-term memory, persisted in SQLite (markdown is a cache/export)."""
    name: str                     # slug / filename stem, also the index key
    description: str              # one-line summary shown in the MEMORY.md index
    kind: str                     # one of KINDS
    text: str                     # the memory body (markdown)
    tags: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    source: str = ""              # task id / "user" / repo
    outcome: str = ""             # "success" | "failure" | "" (episodic only)
    importance: float = 0.5       # 0..1 salience prior
    created_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0
    owner_user_id: str = "default"
    conversation_id: str = ""
    scope: str = "user"           # global | user | conversation
    visibility: str = "private"   # public | shared | private
    status: str = "approved"      # approved | pending

    @property
    def content_hash(self) -> str:
        return _sha(self.text.strip().lower())

    def effective_importance(self, now: float, half_life_s: float) -> float:
        """Importance decayed by age — used for ranking tie-breaks and eviction."""
        age = max(0.0, now - self.created_at)
        decay = 0.5 ** (age / half_life_s) if half_life_s > 0 else 1.0
        recall_boost = 1.0 + min(0.5, 0.05 * self.access_count)  # recalled → resists decay
        return self.importance * decay * recall_boost

    def search_text(self) -> str:
        """Everything used for lexical/dense matching (name, meta, and body)."""
        head = " ".join([self.name, self.description] + self.tags + self.files)
        return f"{self.kind} {head}\n{self.text}".strip()

    def index_line(self) -> str:
        return f"- [{self.name}]({self.name}.md) — {self.description}"

    # -- frontmatter (de)serialization -------------------------------------

    _FRONT_KEYS = ("name", "description", "kind", "tags", "files", "source",
                   "outcome", "importance", "created_at", "last_access", "access_count",
                   "owner_user_id", "conversation_id", "scope", "visibility", "status")

    def to_markdown(self) -> str:
        meta = {k: getattr(self, k) for k in self._FRONT_KEYS}
        return f"---\n{_dump_frontmatter(meta)}---\n\n{self.text.strip()}\n"

    @classmethod
    def from_markdown(cls, raw: str, *, default_name: str) -> "MemoryRecord":
        meta, body = _parse_markdown(raw)
        return cls(
            name=str(meta.get("name") or default_name),
            description=str(meta.get("description", "")),
            kind=str(meta.get("kind", "semantic")),
            text=body.strip(),
            tags=list(meta.get("tags", []) or []),
            files=list(meta.get("files", []) or []),
            source=str(meta.get("source", "")),
            outcome=str(meta.get("outcome", "")),
            importance=float(meta.get("importance", 0.5)),
            created_at=float(meta.get("created_at", 0.0)),
            last_access=float(meta.get("last_access", 0.0)),
            access_count=int(meta.get("access_count", 0)),
            owner_user_id=str(meta.get("owner_user_id", "default") or "default"),
            conversation_id=str(meta.get("conversation_id", "") or ""),
            scope=str(meta.get("scope", "user") or "user"),
            visibility=str(meta.get("visibility", "private") or "private"),
            status=str(meta.get("status", "approved") or "approved"),
        )

    def to_store_dict(self) -> dict:
        d = asdict(self)
        d["content_hash"] = self.content_hash
        return d

    @classmethod
    def from_store_dict(cls, d: dict) -> "MemoryRecord":
        known = cls.__dataclass_fields__
        payload = {k: v for k, v in d.items() if k in known}
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frontmatter helpers (PyYAML when available, tiny fallback otherwise)
# ---------------------------------------------------------------------------

def _dump_frontmatter(meta: dict) -> str:
    try:
        import yaml
        return yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    except Exception:
        lines = []
        for k, v in meta.items():
            if isinstance(v, list):
                lines.append(f"{k}: [{', '.join(json.dumps(x) for x in v)}]")
            else:
                lines.append(f"{k}: {json.dumps(v)}")
        return "\n".join(lines) + "\n"


def _parse_markdown(raw: str) -> tuple[dict, str]:
    """Split a `---`-delimited frontmatter block from the markdown body."""
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            front_raw, body = parts[1], parts[2]
            try:
                import yaml
                meta = yaml.safe_load(front_raw) or {}
            except Exception:
                meta = _parse_frontmatter_fallback(front_raw)
            return (meta if isinstance(meta, dict) else {}), body
    return {}, raw


def _parse_frontmatter_fallback(front_raw: str) -> dict:
    meta: dict = {}
    for line in front_raw.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            meta[key] = [x.strip().strip('"') for x in inner.split(",") if x.strip()] if inner else []
        else:
            try:
                meta[key] = json.loads(val)
            except Exception:
                meta[key] = val.strip('"')
    return meta


def shard_path(root: Path, record_id: str, name: str) -> Path:
    """Scale-out layout: records/<first-2-hex-of-id>/<name>.md (documented option)."""
    return root / "records" / record_id[:2] / f"{name}.md"


# selector(query, catalog_lines) -> list[str] of memory names to load
LlmSelector = Callable[[str, "list[str]"], "list[str]"]


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

class LongTermMemory:
    """Persistent, ACL-filtered memory (SQLite source of truth + markdown cache).

    Usage:
        mem = LongTermMemory(mem_dir, user_id="alice", role="user").load()
        mem.remember("repo uses pytest -m slow", kind="reference", visibility="private")
        mem.as_actor("bob", role="user").recall("pytest")  # bob cannot see alice's private
    """

    INDEX_FILE = "MEMORY.md"
    META_FILE = ".memory_meta.json"
    DB_FILE = "memory.db"

    def __init__(
        self,
        mem_dir: str | Path,
        *,
        embeddings=None,
        max_records: int = 500,
        decay_half_life_days: float = 30.0,
        consolidate_threshold: int = 40,
        consolidate_min_interval_s: float = 86_400.0,
        consolidate_min_new: int = 5,
        clock: Callable[[], float] = time.time,
        user_id: str = "default",
        role: str = "agent",
        conversation_id: str | None = None,
        store: MemoryStore | None = None,
        auto_approve: bool = True,
    ) -> None:
        self._dir = Path(mem_dir).expanduser()
        self._embeddings = embeddings
        self._max_records = max_records
        self._half_life_s = decay_half_life_days * 86400.0
        self._consolidate_threshold = consolidate_threshold
        self._consolidate_min_interval = consolidate_min_interval_s
        self._consolidate_min_new = consolidate_min_new
        self._clock = clock
        self._actor = Actor(user_id=user_id, role=role, conversation_id=conversation_id)
        self._store = store
        self.auto_approve = auto_approve

        self._records: list[MemoryRecord] = []
        self._by_name: dict[str, MemoryRecord] = {}
        self._by_hash: dict[str, str] = {}
        self._inverted: dict[str, set[str]] = {}
        self._acl: dict[str, set[tuple[str, str]]] = {}
        self._vectors = None
        self._dirty_dense = True
        self.stats: dict = {}

    # -- identity -----------------------------------------------------------

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            self._dir.mkdir(parents=True, exist_ok=True)
            from context.store_factory import open_memory_store
            self._store = open_memory_store(self._dir / self.DB_FILE, clock=self._clock)
        return self._store

    @property
    def actor(self) -> Actor:
        return self._actor

    @property
    def user_id(self) -> str:
        return self._actor.user_id

    def as_actor(
        self,
        user_id: str,
        role: str = "user",
        conversation_id: str | None = None,
    ) -> "LongTermMemory":
        """Switch the calling identity and reload the visible subset."""
        self.store.ensure_user(user_id, role=role)
        self._actor = Actor(user_id=user_id, role=role, conversation_id=conversation_id)
        self._rebuild_visible()
        return self

    def new_conversation(self, title: str = "") -> str:
        conv = self.store.create_conversation(self._actor.user_id, title=title)
        self._actor = Actor(
            user_id=self._actor.user_id,
            role=self._actor.role,
            conversation_id=conv.id,
        )
        return conv.id

    def grant(self, name: str, principal: str, perm: str = "read") -> None:
        """Share a record with a user (`user:alice`) or role (`role:agent`)."""
        rec = self.store.ltm_get(name)
        if rec is None:
            raise KeyError(name)
        if not can_write(self._actor, rec, self.store.acl_for(name)):
            raise PermissionError(f"{self._actor.user_id} cannot grant on {name}")
        self.store.acl_grant(name, principal, perm)
        # Promote private → shared so ACL actually applies.
        if rec.get("visibility") == "private":
            rec["visibility"] = "shared"
            self.store.ltm_upsert(rec)
        self._rebuild_visible()

    # -- properties ---------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def backend_info(self) -> str:
        store_kind = getattr(self.store, "kind", "sqlite")
        parts = [
            f"store={store_kind}",
            f"records={len(self._records)}",
            f"user={self._actor.user_id}",
            f"role={self._actor.role}",
            "index=MEMORY.md",
        ]
        if self._actor.conversation_id:
            parts.append(f"conversation={self._actor.conversation_id}")
        if self._embeddings is not None:
            parts.append(f"dense={getattr(self._embeddings, 'name', 'on')}")
        return ", ".join(parts)

    # -- persistence --------------------------------------------------------

    def load(self) -> "LongTermMemory":
        self.store.ensure_user(self._actor.user_id, role=self._actor.role)
        self._import_legacy_markdown()
        self._rebuild_visible()
        logger.info(
            "LongTermMemory loaded: %d visible / store at %s (user=%s role=%s)",
            len(self._records), self._dir, self._actor.user_id, self._actor.role,
        )
        return self

    def _import_legacy_markdown(self) -> None:
        """One-shot: if SQLite is empty, ingest existing *.md records."""
        if self.store.ltm_all():
            return
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.md")):
            if path.name == self.INDEX_FILE:
                continue
            try:
                rec = MemoryRecord.from_markdown(
                    path.read_text("utf-8", errors="replace"), default_name=path.stem)
            except Exception as exc:
                logger.debug("skipping unreadable memory %s: %s", path, exc)
                continue
            if not rec.owner_user_id:
                rec.owner_user_id = self._actor.user_id
            self.store.ltm_upsert(rec.to_store_dict())

    def _rebuild_visible(self) -> None:
        self._records, self._by_name, self._by_hash, self._inverted = [], {}, {}, {}
        self._acl = self.store.acl_map()
        for raw in self.store.ltm_all():
            if not can_read(self._actor, raw, self._acl.get(raw["name"])):
                continue
            rec = MemoryRecord.from_store_dict(raw)
            self._index(rec)
        self._dirty_dense = True

    def _write_record(self, rec: MemoryRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self.store.ltm_upsert(rec.to_store_dict())
        (self._dir / f"{rec.name}.md").write_text(rec.to_markdown(), encoding="utf-8")

    def _write_index(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Memory index (long-term)", ""]
        now = self._clock()
        ordered = sorted(
            self._records,
            key=lambda r: r.effective_importance(now, self._half_life_s), reverse=True)
        lines.extend(r.index_line() for r in ordered)
        blob = "\n".join(lines) + "\n"
        (self._dir / self.INDEX_FILE).write_text(blob, encoding="utf-8")
        try:
            self.store.cache_set("index", f"{self._actor.user_id}:{self._actor.role}",
                                 blob.encode("utf-8"), ttl_s=300)
        except Exception:
            pass

    def index_block(self, max_chars: int = 2_500) -> str:
        """Tiny approved-fact catalog for the prompt (Claude-style bound).

        Episodic run logs and pending proposals are omitted — they are recalled
        on demand, not stuffed into every turn.
        """
        facts = [
            r for r in self._records
            if r.status == "approved" and r.kind != "episodic"
        ]
        if not facts:
            return ""
        now = self._clock()
        ordered = sorted(
            facts,
            key=lambda r: r.effective_importance(now, self._half_life_s), reverse=True)
        lines = ["## Memories (approved facts; recall for detail)"]
        used = len(lines[0])
        n = 0
        for rec in ordered:
            line = rec.index_line()
            if used + len(line) > max_chars or n >= 40:
                lines.append(f"- … ({len(ordered) - n} more)")
                break
            lines.append(line)
            used += len(line)
            n += 1
        return "\n".join(lines)

    # -- indexing -----------------------------------------------------------

    def _index(self, rec: MemoryRecord) -> None:
        self._records.append(rec)
        self._by_name[rec.name] = rec
        self._by_hash[rec.content_hash] = rec.name
        for tok in set(_tokenize(rec.search_text())):
            self._inverted.setdefault(tok, set()).add(rec.name)
        self._dirty_dense = True

    def _unique_name(self, base: str) -> str:
        name = base
        i = 2
        while name in self._by_name or self.store.ltm_get(name) is not None:
            name = f"{base}-{i}"
            i += 1
        return name

    # -- writing ------------------------------------------------------------

    def add(self, rec: MemoryRecord) -> MemoryRecord:
        """Add a record, deduping on (owner, content). Returns the stored record."""
        if rec.scope not in SCOPES:
            rec.scope = "user"
        if rec.visibility not in VISIBILITIES:
            rec.visibility = "private"
        if self._actor.role == "guest":
            raise PermissionError("guest cannot write long-term memory")
        if rec.scope == "global" and self._actor.role not in ("agent", "admin"):
            rec.scope = "user"
        existing_raw = self.store.ltm_find_hash(rec.content_hash, rec.owner_user_id)
        if existing_raw is not None:
            existing = MemoryRecord.from_store_dict(existing_raw)
            if not can_write(self._actor, existing.to_store_dict(),
                             self.store.acl_for(existing.name)):
                raise PermissionError(f"{self._actor.user_id} cannot update {existing.name}")
            existing.access_count += 1
            existing.last_access = self._clock()
            existing.importance = min(1.0, existing.importance + 0.05)
            self._write_record(existing)
            self._rebuild_visible()
            return self._by_name.get(existing.name, existing)
        rec.name = self._unique_name(rec.name)
        rec.owner_user_id = rec.owner_user_id or self._actor.user_id
        if rec.scope == "conversation" and not rec.conversation_id:
            rec.conversation_id = self._actor.conversation_id or ""
        self._write_record(rec)
        self._index(rec)
        self._write_index()
        self.maybe_consolidate()
        return rec

    def remember(
        self,
        text: str,
        *,
        kind: str = "semantic",
        description: str = "",
        name: str = "",
        tags: Iterable[str] = (),
        files: Iterable[str] = (),
        source: str = "",
        outcome: str = "",
        importance: float = 0.5,
        scope: str = "user",
        visibility: str = "private",
        owner_user_id: str = "",
        conversation_id: str = "",
        status: str = "approved",
    ) -> MemoryRecord:
        now = self._clock()
        text = text.strip()
        if kind not in KINDS:
            kind = "semantic"
        desc = (description or text.splitlines()[0] if text else "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "…"
        base = _slugify(name or desc, fallback=_sha(text)[:8])
        rec = MemoryRecord(
            name=base, description=desc, kind=kind, text=text,
            tags=list(tags), files=list(files), source=source, outcome=outcome,
            importance=_clamp(importance), created_at=now, last_access=now, access_count=0,
            owner_user_id=owner_user_id or self._actor.user_id,
            conversation_id=conversation_id or (self._actor.conversation_id or ""),
            scope=scope if scope in SCOPES else "user",
            visibility=visibility if visibility in VISIBILITIES else "private",
            status=status if status in ("approved", "pending") else "approved",
        )
        return self.add(rec)

    def propose(
        self,
        text: str,
        *,
        kind: str = "semantic",
        description: str = "",
        tags: Iterable[str] = (),
        files: Iterable[str] = (),
        source: str = "propose",
        importance: float = 0.45,
        scope: str = "user",
        visibility: str = "public",
    ) -> MemoryRecord:
        """Cursor-style: stage a memory. Auto-approved when `auto_approve` is on."""
        status = "approved" if self.auto_approve else "pending"
        return self.remember(
            text, kind=kind, description=description, tags=tags, files=files,
            source=source, importance=importance, scope=scope, visibility=visibility,
            status=status,
        )

    def approve(self, name: str) -> MemoryRecord:
        rec = self._by_name.get(name)
        if rec is None:
            raw = self.store.ltm_get(name)
            if not raw:
                raise KeyError(name)
            rec = MemoryRecord.from_store_dict(raw)
        rec.status = "approved"
        rec.last_access = self._clock()
        self._write_record(rec)
        self._rebuild_visible()
        return self._by_name.get(name, rec)

    def reject(self, name: str) -> None:
        raw = self.store.ltm_get(name)
        if raw is None:
            raise KeyError(name)
        if not can_write(self._actor, raw, self.store.acl_for(name)):
            raise PermissionError(f"{self._actor.user_id} cannot reject {name}")
        self.store.ltm_delete(name)
        (self._dir / f"{name}.md").unlink(missing_ok=True)
        self._rebuild_visible()
        self._write_index()

    def pending(self) -> list[MemoryRecord]:
        return [r for r in self._records if r.status == "pending"]

    def record_episode(
        self,
        task: str,
        *,
        outcome: str,
        files: Iterable[str] = (),
        summary: str = "",
        source: str = "",
        tags: Iterable[str] = (),
    ) -> MemoryRecord:
        """Capture a finished run as an episodic memory (called at end of run)."""
        files = list(files)
        one_line = task.strip().splitlines()[0] if task.strip() else "task"
        body = f"Task: {task.strip()}"
        if summary:
            body += f"\n\nSolution: {summary.strip()}"
        if files:
            body += f"\n\nFiles changed: {', '.join(files)}"
        body += f"\n\nOutcome: {outcome}"
        importance = 0.7 if outcome == "success" else 0.4   # wins are worth recalling
        return self.remember(
            body, kind="episodic", description=f"{outcome}: {one_line}"[:120],
            tags=list(tags), files=files, source=source, outcome=outcome,
            importance=importance,
            visibility="private",
        )

    def observe_turn(self, user_text: str, assistant_text: str = "") -> list[MemoryRecord]:
        """Extract ChatGPT-style saved memories from one user turn (no transcript dump)."""
        from context.memory_extract import extract_durable_facts
        written: list[MemoryRecord] = []
        for text, kind, importance in extract_durable_facts(user_text, assistant_text):
            rec = self.remember(
                text, kind=kind, importance=importance, source="auto-extract",
                visibility="private",
            )
            written.append(rec)
        return written

    def ingest_overflow(self, queries: list) -> list[MemoryRecord]:
        """STM window overflow: keep facts + searchable chat-history snippets."""
        from context.memory_extract import distill_queries
        written: list[MemoryRecord] = []
        for text, kind, importance in distill_queries(queries):
            rec = self.remember(
                text, kind=kind, importance=importance,
                source="stm-overflow", visibility="private",
            )
            written.append(rec)
        return written

    # -- retrieval ----------------------------------------------------------

    def select(
        self, query: str, k: int = 5, *, kind: str | None = None,
        selector: LlmSelector | None = None,
        include_pending: bool = False,
        exclude_kinds: Iterable[str] = (),
    ) -> list[tuple[MemoryRecord, float]]:
        """Rank records by relevance to `query`.

        With `selector` (a cheap LLM side-query taking the catalog and returning
        names), the model's picks are honored, ordered by the lexical engine and
        padded from it if the model under-selects. Without it, pure engine ranking.
        Pending proposals are skipped unless `include_pending`.
        """
        if not self._records:
            return []
        skip = set(exclude_kinds or ())

        def _ok(rec: MemoryRecord) -> bool:
            if rec.kind in skip:
                return False
            if rec.status != "approved" and not include_pending:
                return False
            return True

        engine = [(r, s) for r, s in self._engine_rank(query, k=max(k * 3, 15), kind=kind) if _ok(r)]

        if selector is not None:
            try:
                catalog = [r.index_line() for r, _ in engine[:30]]
                picks = [p for p in selector(query, catalog) if p in self._by_name]
            except Exception as exc:
                logger.debug("memory LLM selector failed, using engine: %s", exc)
                picks = []
            if picks:
                pick_set = set(picks)
                chosen = [(self._by_name[p], 1.0) for p in picks if _ok(self._by_name[p])]
                for rec, score in engine:
                    if rec.name not in pick_set:
                        chosen.append((rec, score))
                    if len(chosen) >= k:
                        break
                return self._touch(chosen[:k])

        return self._touch(engine[:k])

    def _touch(self, hits: list[tuple[MemoryRecord, float]]) -> list[tuple[MemoryRecord, float]]:
        """Reinforce recalled records — and persist it.

        access_count / last_access feed effective_importance (decay resistance and
        eviction order), so they must survive a reload; keeping them in memory only
        would make a frequently-recalled memory look never-used to the next session.
        """
        now = self._clock()
        for rec, _ in hits:
            rec.access_count += 1
            rec.last_access = now
            try:
                self._write_record(rec)
            except Exception as exc:      # never let bookkeeping break a recall
                logger.debug("memory touch persist failed for %s: %s", rec.name, exc)
        return hits

    def recall(self, query: str, k: int = 5, max_chars: int = 4000,
               *, selector: LlmSelector | None = None,
               for_prompt: bool = False) -> str:
        """Formatted block of the top-k relevant memories for prompt injection.

        `for_prompt=True` skips episodic *run logs* (agent task outcomes).
        Conversation archives and user/project facts stay eligible — that is
        the ChatGPT split between "saved memory" and "chat history search".
        """
        exclude = ("episodic",) if for_prompt else ()
        hits = self.select(query, k=k, selector=selector, exclude_kinds=exclude)
        if not hits:
            return ""
        lines = ["## Relevant memory (recalled from past sessions)"]
        used = 0
        for rec, score in hits:
            tag = f" [{', '.join(rec.tags)}]" if rec.tags else ""
            body = rec.text if len(rec.text) <= 1200 else rec.text[:1200] + "…"
            block = f"\n### {rec.kind}: {rec.name}{tag}\n{body}"
            if used + len(block) > max_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines)

    def _engine_rank(self, query: str, k: int, kind: str | None) -> list[tuple[MemoryRecord, float]]:
        lexical = self._lexical_rank(query, k * 3, kind)
        dense = self._dense_rank(query, k * 3, kind)
        if dense:
            fused = _rrf([[n for n, _ in lexical], [n for n, _ in dense]])
        else:
            fused = lexical
        now = self._clock()
        scored: list[tuple[MemoryRecord, float]] = []
        for name, base in fused:
            rec = self._by_name.get(name)
            if rec is None:
                continue
            eff = rec.effective_importance(now, self._half_life_s)
            age_s = max(0.0, now - rec.created_at)
            recency = 0.5 ** (age_s / (14.0 * 86400.0)) if self._half_life_s > 0 else 1.0
            scored.append((rec, base * (0.55 + 0.45 * recency) * (1.0 + 0.4 * eff)))
        scored.sort(key=lambda rs: rs[1], reverse=True)
        return scored[:k]

    def _lexical_rank(self, query: str, pool: int, kind: str | None) -> list[tuple[str, float]]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        n = len(self._records)
        scores: dict[str, float] = {}
        for term in set(q_terms):
            ids = self._inverted.get(term)
            if not ids:
                continue
            idf = math.log(1 + n / (1 + len(ids)))     # rare terms discriminate
            for name in ids:
                scores[name] = scores.get(name, 0.0) + idf
        if kind is not None:
            scores = {nm: s for nm, s in scores.items() if self._by_name[nm].kind == kind}
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:pool]

    def _dense_rank(self, query: str, pool: int, kind: str | None) -> list[tuple[str, float]]:
        if self._embeddings is None:
            return []
        try:
            import numpy as np
        except ImportError:
            return []
        self._ensure_dense()
        if self._vectors is None or self._vectors.shape[0] != len(self._records):
            return []
        try:
            q = self._embeddings.embed([query])[0].reshape(-1)
        except Exception as exc:
            logger.debug("memory dense query failed: %s", exc)
            return []
        sims = self._vectors @ q
        order = np.argsort(-sims)[: pool * 2]
        out: list[tuple[str, float]] = []
        for i in order:
            rec = self._records[int(i)]
            if kind is not None and rec.kind != kind:
                continue
            out.append((rec.name, float(sims[int(i)])))
            if len(out) >= pool:
                break
        return out

    def _ensure_dense(self) -> None:
        if not self._dirty_dense or self._embeddings is None:
            return
        try:
            import numpy as np
            dim = int(getattr(self._embeddings, "dim", 0) or 0)
            vecs: list[np.ndarray] = []
            missing: list[tuple[int, str]] = []
            for i, rec in enumerate(self._records):
                cached = self.store.cache_get("embed", rec.content_hash)
                if cached:
                    arr = np.frombuffer(cached, dtype=np.float32)
                    if dim and arr.size != dim:
                        missing.append((i, rec.search_text()))
                        vecs.append(np.zeros(dim, dtype=np.float32))
                    else:
                        vecs.append(arr)
                else:
                    missing.append((i, rec.search_text()))
                    vecs.append(np.zeros(dim or 1, dtype=np.float32))
            if missing:
                texts = [t for _, t in missing]
                embedded = self._embeddings.embed(texts)
                if dim == 0 and len(embedded):
                    dim = int(embedded.shape[1])
                for (i, _), vec in zip(missing, embedded):
                    vecs[i] = vec
                    try:
                        self.store.cache_set("embed", self._records[i].content_hash, vec.astype(np.float32).tobytes())
                    except Exception:
                        pass
            self._vectors = np.vstack(vecs) if vecs else None
            self._dirty_dense = False
        except Exception as exc:
            logger.debug("memory dense build skipped: %s", exc)
            self._vectors = None

    # -- consolidation (reflection) -----------------------------------------

    def _read_meta(self) -> dict:
        p = self._dir / self.META_FILE
        if p.exists():
            try:
                return json.loads(p.read_text("utf-8"))
            except Exception:
                return {}
        return {}

    def _write_meta(self, meta: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / self.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    def maybe_consolidate(self, reducer: Optional[Callable] = None) -> Optional[dict]:
        """Consolidate only when all gates pass (count / interval / new records).

        Mirrors Claude Code's multi-layer gating so consolidation is rare and cheap.
        """
        if len(self._records) < self._consolidate_threshold:
            return None
        meta = self._read_meta()
        now = self._clock()
        last = float(meta.get("last_consolidation", 0.0))
        seen_at_last = int(meta.get("count_at_last", 0))
        if now - last < self._consolidate_min_interval:
            return None
        if len(self._records) - seen_at_last < self._consolidate_min_new:
            return None
        return self.consolidate(reducer=reducer)

    def consolidate(self, reducer: Optional[Callable] = None) -> dict:
        """Dedup, apply decay, cap to max_records (evict weakest), rewrite files.

        `reducer(records) -> records` optionally lets an LLM resolve contradictions
        / merge; without it the pass is deterministic.
        """
        now = self._clock()
        # Dedup by content hash, keeping the most-reinforced copy.
        best: dict[str, MemoryRecord] = {}
        for rec in self._records:
            cur = best.get(rec.content_hash)
            if cur is None or rec.access_count > cur.access_count:
                best[rec.content_hash] = rec
        survivors = list(best.values())
        removed = len(self._records) - len(survivors)

        if reducer is not None:
            try:
                survivors = list(reducer(survivors)) or survivors
            except Exception as exc:
                logger.warning("memory reducer failed, keeping deterministic result: %s", exc)

        if len(survivors) > self._max_records:
            survivors.sort(key=lambda r: r.effective_importance(now, self._half_life_s), reverse=True)
            removed += len(survivors) - self._max_records
            survivors = survivors[: self._max_records]

        surviving_names = {r.name for r in survivors}
        for rec in list(self._records):
            if rec.name in surviving_names:
                continue
            raw = rec.to_store_dict()
            if can_write(self._actor, raw, self.store.acl_for(rec.name)):
                self.store.ltm_delete(rec.name)
                (self._dir / f"{rec.name}.md").unlink(missing_ok=True)

        for rec in survivors:
            self._write_record(rec)
        self._rebuild_visible()
        self._write_index()
        self._write_meta({"last_consolidation": now, "count_at_last": len(self._records)})
        self.stats = {"records": len(self._records), "removed": removed}
        logger.info("LongTermMemory consolidated: %s", self.stats)
        return self.stats
