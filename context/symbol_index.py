"""
context/symbol_index.py

A persistent, incremental **symbol index** (a ctags-style inverted index for code)
— the concrete search/index optimization for feature #3.

## The problem it solves

`tools/search_tool.py::FindSymbolTool` (and grep-style `search_text`) re-walk the
whole repo and re-run a regex over every file *on every single call*. In an agent
loop that looks up dozens of symbols, that is O(files × calls) of repeated disk
reads and scanning — the same files parsed again and again.

The symbol index inverts that: parse each file **once** into a
`name → [locations]` map, then answer every later lookup in O(1) (exact) or
O(distinct-names) (prefix/substring) with **no disk access**. It is:

  * **accurate** — reuses repo_map's tree-sitter extraction (regex only as a
    fallback), so it understands functions / classes / methods across languages,
    not just `^def`/`^class` lines;
  * **incremental** — each file's symbols are cached keyed by a content hash
    (mirroring the RAG cache). A rebuild re-parses only files whose bytes changed;
    unchanged files reuse cached symbols and deleted files drop out;
  * **cheap & dependency-free** — pure stdlib + repo_map; no numpy, no network.

## Index structures

    _by_name    : exact name        -> [SymbolLoc]      (O(1) exact lookup)
    _by_lower   : lowercased name   -> [SymbolLoc]      (case-insensitive)
    _names      : sorted list of distinct names         (prefix/substring scan)

## Scale-out

For a repo far larger than fits comfortably in memory, the same per-file cache
shards onto disk (one JSON per file, keyed by path hash) and the in-memory maps
become an LMDB/SQLite table — see docs/SEARCH_AND_INDEX.md. This module ships the
in-memory + single-JSON-cache implementation, which is the right size for the
per-repo agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from context.repo_map import _SKIP_DIRS, _extract_symbols

logger = logging.getLogger(__name__)

# Source extensions we index (superset of tree-sitter-supported; regex covers the rest).
_INDEXED_EXTS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".cpp", ".c",
    ".h", ".hpp", ".rb",
})
_MAX_FILE_BYTES = 500_000


@dataclass(frozen=True)
class SymbolLoc:
    """One definition site of a symbol."""
    name: str
    kind: str          # "function" | "class" | "method"
    file: str          # repo-relative path
    line: int
    indent: int = 0

    def format(self) -> str:
        scope = "method" if self.indent > 0 else "top-level"
        return f"{self.file}:{self.line}: {self.kind} {self.name} ({scope})"

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "file": self.file,
                "line": self.line, "indent": self.indent}

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolLoc":
        return cls(name=d["name"], kind=d["kind"], file=d["file"],
                   line=int(d["line"]), indent=int(d.get("indent", 0)))


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _iter_source_files(root: Path):
    """Yield (relative_path_str, content) for indexable source files."""
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in _INDEXED_EXTS:
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        yield str(path.relative_to(root)), content


class SymbolIndex:
    """Build-once, incremental symbol lookup over a repository.

    Usage:
        idx = SymbolIndex(repo_path, cache_dir=".symbol_cache").build()
        for loc in idx.lookup("parse", mode="prefix"):
            print(loc.format())
    """

    CACHE_FILE = "symbols.json"

    def __init__(self, repo_path: str | Path, cache_dir: str | Path | None = None) -> None:
        self._root = Path(repo_path).resolve()
        self._cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self._by_name: dict[str, list[SymbolLoc]] = {}
        self._by_lower: dict[str, list[SymbolLoc]] = {}
        self._names: list[str] = []
        self.stats: dict = {}

    # -- properties ---------------------------------------------------------

    @property
    def root(self) -> Path:
        """Repo root the index is built over (indexed paths are relative to it)."""
        return self._root

    @property
    def symbol_count(self) -> int:
        return sum(len(v) for v in self._by_name.values())

    @property
    def name_count(self) -> int:
        return len(self._by_name)

    def all_names(self) -> list[str]:
        """Sorted list of every distinct symbol name in the index."""
        return list(self._names)

    # -- build (incremental) ------------------------------------------------

    def build(self) -> "SymbolIndex":
        t0 = time.time()
        cached = self._load_cache()               # {rel: (hash, [SymbolLoc])}
        per_file: dict[str, tuple[str, list[SymbolLoc]]] = {}
        reused = reparsed = 0

        for rel, content in _iter_source_files(self._root):
            h = _sha(content)
            hit = cached.get(rel)
            if hit is not None and hit[0] == h:
                per_file[rel] = hit
                reused += 1
            else:
                syms = self._extract(rel, content)
                per_file[rel] = (h, syms)
                reparsed += 1

        self._reindex(per_file)
        self._save_cache(per_file)
        self.stats = {
            "files": len(per_file),
            "symbols": self.symbol_count,
            "names": self.name_count,
            "reused_files": reused,
            "reparsed_files": reparsed,
            "build_seconds": round(time.time() - t0, 4),
            "cache": bool(self._cache_dir),
        }
        logger.info("SymbolIndex built: %s", self.stats)
        return self

    def _extract(self, rel: str, content: str) -> list[SymbolLoc]:
        out: list[SymbolLoc] = []
        p = Path(rel)
        try:
            for s in _extract_symbols(content, p, p.suffix.lower()):
                out.append(SymbolLoc(name=s.name, kind=s.kind, file=rel,
                                     line=s.line, indent=s.indent))
        except Exception:
            pass
        return out

    def _reindex(self, per_file: dict[str, tuple[str, list[SymbolLoc]]]) -> None:
        self._by_name = {}
        self._by_lower = {}
        for _rel, (_h, syms) in per_file.items():
            for loc in syms:
                self._by_name.setdefault(loc.name, []).append(loc)
                self._by_lower.setdefault(loc.name.lower(), []).append(loc)
        self._names = sorted(self._by_name)

    # -- lookup -------------------------------------------------------------

    def lookup(
        self, symbol: str, *, mode: str = "prefix", kind: str | None = None,
        case_sensitive: bool = False, path_prefix: str | None = None, limit: int = 50,
    ) -> list[SymbolLoc]:
        """Return matching definition sites.

        mode: "exact" | "prefix" | "substring". `kind` filters function/class/method.
        Results are ordered top-level defs first (usually the intended target).
        """
        if not symbol:
            return []
        results: list[SymbolLoc] = []

        if mode == "exact":
            table = self._by_name if case_sensitive else self._by_lower
            key = symbol if case_sensitive else symbol.lower()
            results = list(table.get(key, []))
        else:
            needle = symbol if case_sensitive else symbol.lower()
            for name in self._names:
                hay = name if case_sensitive else name.lower()
                if (mode == "prefix" and hay.startswith(needle)) or \
                   (mode == "substring" and needle in hay):
                    results.extend(self._by_name[name])

        if kind is not None:
            results = [r for r in results if r.kind == kind]
        if path_prefix:
            results = [r for r in results if r.file.startswith(path_prefix)]

        # Top-level defs are more likely the intended symbol; then by path/line.
        results.sort(key=lambda r: (r.indent > 0, r.file, r.line))
        return results[:limit]

    # -- persistence --------------------------------------------------------

    def _load_cache(self) -> dict[str, tuple[str, list[SymbolLoc]]]:
        if not self._cache_dir:
            return {}
        path = self._cache_dir / self.CACHE_FILE
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text("utf-8"))
            if data.get("version") != 1:
                return {}
            out: dict[str, tuple[str, list[SymbolLoc]]] = {}
            for rel, info in data.get("files", {}).items():
                out[rel] = (info["hash"], [SymbolLoc.from_dict(d) for d in info["symbols"]])
            return out
        except Exception as exc:
            logger.warning("SymbolIndex cache load failed (%s) → rebuild", exc)
            return {}

    def _save_cache(self, per_file: dict[str, tuple[str, list[SymbolLoc]]]) -> None:
        if not self._cache_dir:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            files = {rel: {"hash": h, "symbols": [s.to_dict() for s in syms]}
                     for rel, (h, syms) in per_file.items()}
            (self._cache_dir / self.CACHE_FILE).write_text(
                json.dumps({"version": 1, "files": files}), encoding="utf-8")
        except Exception as exc:
            logger.warning("SymbolIndex cache save failed: %s", exc)
