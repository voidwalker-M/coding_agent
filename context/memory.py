"""
context/memory.py

Two-tier agent memory (feature #2), modeled on how Claude Code itself does
memory (see https://learn.shareai.run/en/s09/) and upgraded with a proper
retrieval engine.

The agent already has a *sliding-window* conversation history (context/history.py)
— raw working context that is lossy: when the window overflows the oldest turns
are dropped outright. Memory is the layer that "keeps what compression throws
away". Two tiers:

  ShortTermMemory  — within-episode / session working memory. A small, bounded
                     scratchpad the agent accumulates during one run: distilled
                     notes, the files it has looked at, and a *rolling summary* of
                     the turns history evicted (so nothing is silently lost across
                     a window trim / auto-compaction). Rendered into the prompt and
                     discarded at the end of the run.

  LongTermMemory   — persistent memory across sessions, on disk. This mirrors
                     Claude Code's user-memory design:
                       * one Markdown file **per memory** with YAML frontmatter,
                         under `<mem_dir>/` — human-readable and hand-editable;
                       * a single **MEMORY.md index** (one line per memory) that is
                         cheap and stable, so it lives in the SYSTEM prompt and
                         stays prompt-cache friendly;
                       * memory **content injected on demand** — only the handful of
                         records relevant to the current task are pulled in full,
                         capped per file, so the catalog stays cheap.
                     Typed like Claude Code — user / feedback / project / reference
                     — plus `episodic` (a captured past run) and `semantic` (a
                     free fact).

## Retrieval (two paths, as in the reference — upgraded)

  1. Index path — `index_block()` returns the MEMORY.md catalog for the system
     prompt (always present, cache-stable).
  2. On-demand path — `select(query, k)` ranks records and returns the top few,
     whose bodies `recall()` injects. Ranking is a real engine, not just keyword
     match: an inverted index + idf lexical score, recency **decay** + **importance**
     weighting, optional **dense** embeddings fused via reciprocal-rank-fusion, and
     an optional pluggable **LLM selector** side-query (`select(..., selector=fn)`)
     for the "ask a cheap model which memories matter" path — falling back to the
     lexical engine when it is absent or errors.

## Consolidation (reflection)

`maybe_consolidate()` is gated like Claude Code's real implementation — a file-count
threshold AND a minimum time interval AND a minimum number of new records since the
last pass — then dedups by content, applies decay, and caps to `max_records`
(evicting the weakest). An optional `reducer` hook lets an LLM do contradiction
resolution; without it the pass is deterministic.

## Storing many records (the on-disk design)

Default: file-per-memory + `MEMORY.md` index (readable, cache-friendly, the right
scale for a per-user/per-repo agent). At very large scale a flat directory of
hundreds of thousands of files hurts the filesystem, so records shard cleanly into
`records/<first-2-hex-of-id>/<name>.md` — see `shard_path()` and docs/MEMORY.md.

Everything degrades gracefully: with no numpy / no embedding backend the store is
lexical-only and fully functional; with no PyYAML it falls back to a minimal
frontmatter parser.
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
# ShortTermMemory (session / working memory)
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """Bounded within-episode scratchpad: notes, files seen, rolling summary.

    Rendered into the system prompt each step so the agent keeps a compact,
    lossless-enough trace even after the raw history window has trimmed old
    turns — the session-memory tier that survives compaction in the reference.
    """

    def __init__(self, max_notes: int = 12, max_summary_chars: int = 1500) -> None:
        self._notes: list[str] = []
        self._files: list[str] = []          # ordered, de-duplicated
        self._summary: str = ""
        self._max_notes = max_notes
        self._max_summary = max_summary_chars

    def add_note(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._notes.append(text)
        if len(self._notes) > self._max_notes:
            self._notes = self._notes[-self._max_notes:]   # keep the working set

    def note_file(self, path: str) -> None:
        path = (path or "").strip()
        if path and path not in self._files:
            self._files.append(path)

    def fold(self, evicted_text: str) -> None:
        """Fold an evicted history turn into the rolling summary (bounded)."""
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
        """Compact block for the system prompt; empty string when nothing to show."""
        if not (self._notes or self._files or self._summary):
            return ""
        lines = ["## Working memory (this task so far)"]
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


# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """One long-term memory, persisted as a Markdown file with YAML frontmatter."""
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
                   "outcome", "importance", "created_at", "last_access", "access_count")

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
        )

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
    """Persistent, retrieval-backed memory store (markdown files + MEMORY.md index).

    Usage:
        mem = LongTermMemory(mem_dir="~/.coding_agent/memory").load()
        mem.remember("repo uses pytest -m slow for slow tests", kind="reference")
        catalog = mem.index_block()           # -> put in the SYSTEM prompt
        ctx = mem.recall("run the slow tests") # -> inject the relevant bodies
    """

    INDEX_FILE = "MEMORY.md"
    META_FILE = ".memory_meta.json"

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
    ) -> None:
        self._dir = Path(mem_dir).expanduser()
        self._embeddings = embeddings                 # optional EmbeddingBackend
        self._max_records = max_records
        self._half_life_s = decay_half_life_days * 86400.0
        self._consolidate_threshold = consolidate_threshold
        self._consolidate_min_interval = consolidate_min_interval_s
        self._consolidate_min_new = consolidate_min_new
        self._clock = clock

        self._records: list[MemoryRecord] = []
        self._by_name: dict[str, MemoryRecord] = {}
        self._by_hash: dict[str, str] = {}            # content_hash -> name (dedup)
        self._inverted: dict[str, set[str]] = {}      # token -> {record name}
        self._vectors = None                          # np.ndarray aligned with _records
        self._dirty_dense = True
        self.stats: dict = {}

    # -- properties ---------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def backend_info(self) -> str:
        parts = ["store=markdown", f"records={len(self._records)}", "index=MEMORY.md"]
        if self._embeddings is not None:
            parts.append(f"dense={getattr(self._embeddings, 'name', 'on')}")
        return ", ".join(parts)

    # -- persistence --------------------------------------------------------

    def load(self) -> "LongTermMemory":
        self._records, self._by_name, self._by_hash, self._inverted = [], {}, {}, {}
        if self._dir.exists():
            for path in sorted(self._dir.glob("*.md")):
                if path.name == self.INDEX_FILE:
                    continue
                try:
                    rec = MemoryRecord.from_markdown(
                        path.read_text("utf-8", errors="replace"), default_name=path.stem)
                except Exception as exc:
                    logger.debug("skipping unreadable memory %s: %s", path, exc)
                    continue
                self._index(rec)
        self._dirty_dense = True
        logger.info("LongTermMemory loaded: %d records from %s", len(self._records), self._dir)
        return self

    def _write_record(self, rec: MemoryRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{rec.name}.md").write_text(rec.to_markdown(), encoding="utf-8")

    def _write_index(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Memory index (long-term)", ""]
        # Most salient first, so a truncated catalog keeps the best.
        now = self._clock()
        ordered = sorted(
            self._records,
            key=lambda r: r.effective_importance(now, self._half_life_s), reverse=True)
        lines.extend(r.index_line() for r in ordered)
        (self._dir / self.INDEX_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def index_block(self, max_chars: int = 3000) -> str:
        """The MEMORY.md catalog for the SYSTEM prompt (stable → cache friendly)."""
        if not self._records:
            return ""
        now = self._clock()
        ordered = sorted(
            self._records,
            key=lambda r: r.effective_importance(now, self._half_life_s), reverse=True)
        lines = ["## Memory index (ask to recall any of these by topic)"]
        used = len(lines[0])
        for rec in ordered:
            line = rec.index_line()
            if used + len(line) > max_chars:
                lines.append(f"- … ({len(ordered) - (len(lines) - 1)} more)")
                break
            lines.append(line)
            used += len(line)
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
        while name in self._by_name:
            name = f"{base}-{i}"
            i += 1
        return name

    # -- writing ------------------------------------------------------------

    def add(self, rec: MemoryRecord) -> MemoryRecord:
        """Add a record, deduping on content. Returns the stored record."""
        existing_name = self._by_hash.get(rec.content_hash)
        if existing_name is not None:            # same content → reinforce, don't duplicate
            existing = self._by_name[existing_name]
            existing.access_count += 1
            existing.last_access = self._clock()
            existing.importance = min(1.0, existing.importance + 0.05)
            self._write_record(existing)
            return existing
        rec.name = self._unique_name(rec.name)
        self._index(rec)
        self._write_record(rec)
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
        )
        return self.add(rec)

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
        )

    # -- retrieval ----------------------------------------------------------

    def select(
        self, query: str, k: int = 5, *, kind: str | None = None,
        selector: LlmSelector | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Rank records by relevance to `query`.

        With `selector` (a cheap LLM side-query taking the catalog and returning
        names), the model's picks are honored, ordered by the lexical engine and
        padded from it if the model under-selects. Without it, pure engine ranking.
        """
        if not self._records:
            return []
        engine = self._engine_rank(query, k=max(k * 3, 15), kind=kind)

        if selector is not None:
            try:
                catalog = [r.index_line() for r, _ in self._engine_rank(query, k=30, kind=kind)]
                picks = [p for p in selector(query, catalog) if p in self._by_name]
            except Exception as exc:
                logger.debug("memory LLM selector failed, using engine: %s", exc)
                picks = []
            if picks:
                pick_set = set(picks)
                chosen = [(self._by_name[p], 1.0) for p in picks]
                # Pad with the engine's best that the model didn't name.
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
               *, selector: LlmSelector | None = None) -> str:
        """Formatted block of the top-k relevant memories for prompt injection."""
        hits = self.select(query, k=k, selector=selector)
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
            scored.append((rec, base * (1.0 + 0.5 * eff)))
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
            import numpy as np  # noqa: F401
            texts = [r.search_text() for r in self._records]
            self._vectors = self._embeddings.embed(texts) if texts else None
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
        # Delete files that no longer survive.
        if self._dir.exists():
            for path in self._dir.glob("*.md"):
                if path.name != self.INDEX_FILE and path.stem not in surviving_names:
                    path.unlink(missing_ok=True)

        # Rebuild indices + rewrite survivors.
        self._records, self._by_name, self._by_hash, self._inverted = [], {}, {}, {}
        for rec in survivors:
            self._index(rec)
            self._write_record(rec)
        self._write_index()
        self._write_meta({"last_consolidation": now, "count_at_last": len(self._records)})
        self.stats = {"records": len(self._records), "removed": removed}
        logger.info("LongTermMemory consolidated: %s", self.stats)
        return self.stats
