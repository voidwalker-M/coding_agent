"""
context/rag.py

RAG (Retrieval-Augmented Generation) retriever: chunks the entire repo, vectorizes it,
builds an index, retrieves the most relevant code snippets for a task description,
and injects them into the system prompt.

Difference from repo_map.py:
- repo_map: static structural summary (symbol list for all files), always injected
- rag: dynamically retrieves top-k relevant code chunks per query, prioritizing semantic relevance

## Production-grade pipeline (every layer has a pure numpy/stdlib fallback — runs offline/without keys)

1. Chunking (Chunker / SyntaxChunker)
   - SyntaxChunker: syntax-aware chunking. Python uses stdlib `ast` to split at function/class
     boundaries; other languages use repo_map symbols (tree-sitter, regex fallback) as boundaries;
     degrades to line-window on failure.
   - Chunker: fixed line-window (default 40 lines, 10 overlap), used as the final fallback.
   - Each chunk is annotated with its language and contained symbols.

2. Vectorization (EmbeddingBackend)
   - OpenAIEmbeddings: text-embedding-3-small (default when an API key is present)
   - HashingEmbeddings: offline deterministic bag-of-tokens hashing (no key / testing)

3. Hybrid retrieval (dense + sparse)
   - Dense: vector index (faiss HNSW/Flat, falls back to numpy dot-product)
   - Sparse: pure-numpy BM25 (exact keyword/identifier matching, important for code)
   - Fused with Reciprocal Rank Fusion (RRF)

4. Reranking (optional)
   - MMRReranker: pure-numpy Maximal Marginal Relevance, improves result diversity
   - CrossEncoderReranker: sentence-transformers cross-encoder (used when installed)

5. Persistence + incremental updates
   - Chunks / vectors / per-file content hashes are saved to cache_dir
   - On subsequent build() calls, only files whose content changed are re-chunked and re-embedded;
     unchanged files reuse cached vectors; deleted files are removed

6. Metadata filtering + observability
   - Filter candidates by language / path prefix
   - stats tracks chunk count, reused/re-embedded file counts, backends used, elapsed time
   - evaluate_recall() provides offline recall@k / MRR evaluation

Dependencies (all optional):
    pip install faiss-cpu              # vector index; degrades to numpy without it
    pip install sentence-transformers # cross-encoder reranking; degrades to MMR without it
    # OpenAI embeddings reuse the existing openai SDK
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from context.repo_map import _SKIP_DIRS, _extract_symbols

logger = logging.getLogger(__name__)


# Source file extensions included in retrieval → language label
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".cpp": "cpp", ".c": "c",
    ".h": "c", ".hpp": "cpp", ".rb": "ruby", ".md": "markdown", ".txt": "text",
    ".yaml": "yaml", ".yml": "yaml",
}
_SOURCE_EXTS: frozenset[str] = frozenset(_EXT_TO_LANG)

_MAX_FILE_BYTES = 500_000
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
# Split camelCase / snake_case identifiers for query expansion
_SUBTOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _language_of(ext: str) -> str:
    return _EXT_TO_LANG.get(ext.lower(), "")


def _tokenize(text: str) -> list[str]:
    """Code-friendly tokenizer: keep identifiers whole and also split sub-words (fooBar→foo,bar)."""
    toks: list[str] = []
    for m in _TOKEN_RE.findall(text.lower()):
        toks.append(m)
    return toks


# ---------------------------------------------------------------------------
# CodeChunk
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A code fragment that has been chunked and is ready for vectorization."""
    file: str
    start_line: int
    end_line: int
    text: str
    symbols: list[str] = field(default_factory=list)
    language: str = ""

    @property
    def id(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line}"

    def header(self) -> str:
        sym = f" [{', '.join(self.symbols)}]" if self.symbols else ""
        return f"{self.file}:{self.start_line}-{self.end_line}{sym}"

    def embed_text(self) -> str:
        """Text fed to the embedding model / BM25: includes file path and symbol names to improve recall."""
        head = self.file
        if self.symbols:
            head += " " + " ".join(self.symbols)
        return f"{head}\n{self.text}"

    def to_dict(self) -> dict:
        return {
            "file": self.file, "start_line": self.start_line, "end_line": self.end_line,
            "text": self.text, "symbols": self.symbols, "language": self.language,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodeChunk":
        return cls(
            file=d["file"], start_line=d["start_line"], end_line=d["end_line"],
            text=d["text"], symbols=list(d.get("symbols", [])), language=d.get("language", ""),
        )


# ---------------------------------------------------------------------------
# Chunker (line-window fallback)
# ---------------------------------------------------------------------------

def _iter_source_files(root: Path):
    """Iterate over source files in the repo that participate in retrieval; yield (Path, relative_str, content)."""
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in _SOURCE_EXTS:
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        yield path, str(path.relative_to(root)), content


# External file library (#2): index docs / dependency source / reference material
# that lives OUTSIDE the repo. Allow common doc formats in addition to source code.
_DOC_EXTS: frozenset[str] = frozenset({
    ".md", ".markdown", ".rst", ".txt", ".text",
})
_EXTERNAL_EXTS: frozenset[str] = _SOURCE_EXTS | _DOC_EXTS


def _iter_external_files(paths):
    """
    Iterate over files in an external library (dirs and/or single files) that
    participate in retrieval. Yields (Path, relative_str, content) where the
    relative path is namespaced under 'external/<label>/...' so external chunks
    never collide with repo paths and are clearly marked in retrieved context.
    """
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            files = [(p, p.name)]
        elif p.is_dir():
            files = [(c, f"{p.name}/{c.relative_to(p)}") for c in sorted(p.rglob("*"))]
        else:
            logger.warning("RAG external path not found, skipping: %s", p)
            continue
        for path, rel in files:
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in _EXTERNAL_EXTS:
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            yield path, f"external/{rel}", content


class Chunker:
    """Chunk repo files into fixed line-window chunks (fallback chunker)."""

    def __init__(self, chunk_lines: int = 40, overlap: int = 10) -> None:
        self._chunk_lines = chunk_lines
        self._overlap = max(0, min(overlap, chunk_lines - 1))

    def chunk_repo(self, root: Path) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for path, rel, content in _iter_source_files(root):
            chunks.extend(self.chunk_file(content, Path(rel)))
        return chunks

    def chunk_file(self, content: str, rel: Path) -> list[CodeChunk]:
        lines = content.splitlines()
        if not lines:
            return []
        sym_by_line = _symbol_map(content, rel)
        lang = _language_of(rel.suffix)

        out: list[CodeChunk] = []
        step = self._chunk_lines - self._overlap
        for start in range(0, len(lines), step):
            window = lines[start:start + self._chunk_lines]
            if not window:
                break
            start_line = start + 1
            end_line = start + len(window)
            chunk = _make_chunk(lines, start_line, end_line, rel, lang, sym_by_line)
            if chunk:
                out.append(chunk)
            if start + self._chunk_lines >= len(lines):
                break
        return out

    # Alias for old tests
    _chunk_file = chunk_file


def _symbol_map(content: str, rel: Path) -> dict[int, str]:
    """{line_no: 'kind name'} — reuses repo_map's symbol extraction (tree-sitter / regex)."""
    sym_by_line: dict[int, str] = {}
    try:
        for s in _extract_symbols(content, rel, rel.suffix.lower()):
            sym_by_line[s.line] = f"{s.kind} {s.name}"
    except Exception:
        pass
    return sym_by_line


def _make_chunk(
    lines: list[str], start_line: int, end_line: int,
    rel: Path, lang: str, sym_by_line: dict[int, str],
) -> Optional[CodeChunk]:
    text = "\n".join(lines[start_line - 1:end_line]).strip()
    if not text:
        return None
    syms = [sym_by_line[ln] for ln in range(start_line, end_line + 1) if ln in sym_by_line]
    return CodeChunk(
        file=str(rel), start_line=start_line, end_line=end_line,
        text=text, symbols=syms, language=lang,
    )


# ---------------------------------------------------------------------------
# SyntaxChunker (syntax-aware chunking)
# ---------------------------------------------------------------------------

class SyntaxChunker:
    """
    Syntax-aware chunker: splits along function/class boundaries where possible,
    avoiding cutting a single function into two chunks.

    - .py              → stdlib `ast` (precise; no tree-sitter dependency)
    - Other languages  → repo_map symbol lines as boundaries (tree-sitter; regex fallback)
    - Parse failure / no symbols → degrades to line-window (Chunker)
    """

    def __init__(self, chunk_lines: int = 40, overlap: int = 10, max_chunk_lines: int = 120) -> None:
        self._fallback = Chunker(chunk_lines=chunk_lines, overlap=overlap)
        self._max = max_chunk_lines

    def chunk_repo(self, root: Path) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for path, rel, content in _iter_source_files(root):
            chunks.extend(self.chunk_file(content, Path(rel)))
        return chunks

    def chunk_file(self, content: str, rel: Path) -> list[CodeChunk]:
        if not content.strip():
            return []
        ext = rel.suffix.lower()
        spans: Optional[list[tuple[int, int]]] = None
        if ext == ".py":
            spans = self._python_spans(content)
        if spans is None:
            spans = self._symbol_spans(content, rel, ext)
        if not spans:
            return self._fallback.chunk_file(content, rel)

        lines = content.splitlines()
        sym_by_line = _symbol_map(content, rel)
        lang = _language_of(ext)
        out: list[CodeChunk] = []
        for start, end in spans:
            out.extend(self._emit(lines, start, end, rel, lang, sym_by_line))
        return out

    # Sub-divide large spans into line-windows to prevent any single chunk from being too large
    def _emit(self, lines, start, end, rel, lang, sym_by_line) -> list[CodeChunk]:
        end = min(end, len(lines))
        if end < start:
            return []
        if end - start + 1 <= self._max:
            c = _make_chunk(lines, start, end, rel, lang, sym_by_line)
            return [c] if c else []
        out: list[CodeChunk] = []
        for s in range(start, end + 1, self._max):
            e = min(s + self._max - 1, end)
            c = _make_chunk(lines, s, e, rel, lang, sym_by_line)
            if c:
                out.append(c)
        return out

    def _python_spans(self, content: str) -> Optional[list[tuple[int, int]]]:
        """Use ast to split the module into spans of [simple statements | functions | classes]."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None
        n = len(content.splitlines())
        spans: list[tuple[int, int]] = []
        buf_start: Optional[int] = None
        buf_end = 0

        def flush():
            nonlocal buf_start
            if buf_start is not None:
                spans.append((buf_start, buf_end))
                buf_start = None

        defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        for node in tree.body:
            start = node.lineno
            for dec in getattr(node, "decorator_list", []):
                start = min(start, dec.lineno)
            end = getattr(node, "end_lineno", start) or start
            if isinstance(node, defs):
                flush()
                spans.append((start, end))
            else:
                if buf_start is None:
                    buf_start = start
                buf_end = end
        flush()
        if not spans:
            return [(1, n)] if n else []
        # Merge any lines before the first span (comments/shebang) into the first segment
        if spans[0][0] > 1:
            spans[0] = (1, spans[0][1])
        return spans

    def _symbol_spans(self, content, rel, ext) -> Optional[list[tuple[int, int]]]:
        """Use symbol lines as split points (for non-Python languages)."""
        sym_by_line = _symbol_map(content, rel)
        if not sym_by_line:
            return None
        n = len(content.splitlines())
        cuts = sorted({1, *(ln for ln in sym_by_line if 1 < ln <= n)})
        cuts.append(n + 1)
        return [(cuts[i], cuts[i + 1] - 1) for i in range(len(cuts) - 1) if cuts[i + 1] - 1 >= cuts[i]]


def create_chunker(syntax_aware: bool = True, chunk_lines: int = 40, overlap: int = 10):
    return SyntaxChunker(chunk_lines, overlap) if syntax_aware else Chunker(chunk_lines, overlap)


# ---------------------------------------------------------------------------
# EmbeddingBackend
# ---------------------------------------------------------------------------

class EmbeddingBackend(ABC):
    """Abstract interface: text → vector. Returns L2-normalized vectors (cosine == dot product)."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbeddings(EmbeddingBackend):
    """
    Offline deterministic embedding: bag-of-tokens hashed to a fixed dimension (feature hashing).
    No API key, no internet connection, reproducible — for testing and keyless environments.
    Cosine similarity ≈ token overlap; sufficient for demonstrating retrieval recall.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                mat[i, idx] += sign
        return _l2_normalize(mat)


class OpenAIEmbeddings(EmbeddingBackend):
    """OpenAI embeddings (text-embedding-3-small). Reuses the installed openai SDK."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dim: int = 1536,
        batch_size: int = 128,
    ) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = dim
        self._batch = batch_size

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = list(texts[start:start + self._batch])
            resp = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.size == 0:
            mat = np.zeros((0, self._dim), dtype=np.float32)
        return _l2_normalize(mat)


def create_embedding_backend(
    provider: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
) -> EmbeddingBackend:
    """
    Select the embedding backend:
        "auto"    → OpenAI when OPENAI_API_KEY is set, otherwise HashingEmbeddings (offline)
        "openai"  → force OpenAI
        "hashing" → force offline hashing embeddings
    """
    provider = (provider or "auto").lower()
    if provider == "hashing":
        return HashingEmbeddings()

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if provider == "openai" or (provider == "auto" and key):
        try:
            return OpenAIEmbeddings(api_key=key or None, base_url=base_url)
        except Exception as exc:
            logger.warning("OpenAI embeddings unavailable (%s), falling back to hashing", exc)
    return HashingEmbeddings()


# ---------------------------------------------------------------------------
# VectorIndex (dense)
# ---------------------------------------------------------------------------

class VectorIndex(ABC):
    """Similarity index for normalized vectors (inner product == cosine)."""

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class NumpyIndex(VectorIndex):
    """numpy dot-product fallback index (used when faiss is not installed)."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._mat = np.zeros((0, dim), dtype=np.float32)

    @property
    def name(self) -> str:
        return "numpy"

    def add(self, vectors: np.ndarray) -> None:
        if vectors.size:
            self._mat = np.vstack([self._mat, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._mat.shape[0] == 0:
            return []
        scores = self._mat @ query.reshape(-1).astype(np.float32)
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


class FaissIndex(VectorIndex):
    """faiss IndexFlatIP (inner product == cosine for normalized vectors)."""

    def __init__(self, dim: int) -> None:
        import faiss
        self._index = faiss.IndexFlatIP(dim)
        self._dim = dim

    @property
    def name(self) -> str:
        return "faiss-flat"

    def add(self, vectors: np.ndarray) -> None:
        if vectors.size:
            self._index.add(np.ascontiguousarray(vectors.astype(np.float32)))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        scores, idx = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


class HnswIndex(VectorIndex):
    """faiss IndexHNSWFlat — large-scale ANN approximate index."""

    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        import faiss
        self._index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = ef_construction
        self._index.hnsw.efSearch = ef_search
        self._dim = dim

    @property
    def name(self) -> str:
        return "faiss-hnsw"

    def add(self, vectors: np.ndarray) -> None:
        if vectors.size:
            self._index.add(np.ascontiguousarray(vectors.astype(np.float32)))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        scores, idx = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


def create_vector_index(dim: int, kind: str = "auto") -> VectorIndex:
    """
    Select the dense index:
        "auto"/"flat" → FaissIndex(Flat) when faiss is available, otherwise NumpyIndex
        "hnsw"        → HnswIndex (ANN) when faiss is available, otherwise NumpyIndex
        "numpy"       → force numpy
    """
    kind = (kind or "auto").lower()
    if kind == "numpy":
        return NumpyIndex(dim)
    try:
        if kind == "hnsw":
            return HnswIndex(dim)
        return FaissIndex(dim)
    except Exception:
        return NumpyIndex(dim)


# ---------------------------------------------------------------------------
# BM25 (sparse, pure numpy)
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Pure-numpy BM25 sparse retrieval (Okapi BM25).
    In code retrieval, exact identifier/keyword matching is often more reliable
    than dense semantics; used as one branch of hybrid retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs_tokens: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_len: np.ndarray = np.zeros(0, dtype=np.float32)
        self._avg_len: float = 0.0
        self._tf: list[dict[str, int]] = []

    @property
    def name(self) -> str:
        return "bm25"

    def build(self, documents: Sequence[str]) -> "BM25Index":
        self._docs_tokens = [_tokenize(d) for d in documents]
        self._tf = []
        self._df = {}
        for toks in self._docs_tokens:
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self._tf.append(counts)
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1
        n = len(self._docs_tokens)
        self._doc_len = np.array([len(t) for t in self._docs_tokens], dtype=np.float32)
        self._avg_len = float(self._doc_len.mean()) if n else 0.0
        # BM25+ idf (guaranteed non-negative)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }
        return self

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self._tf:
            return []
        q_terms = _tokenize(query)
        scores = np.zeros(len(self._tf), dtype=np.float32)
        for term in q_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self._tf):
                f = counts.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self._doc_len[i] / (self._avg_len or 1.0))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        nz = np.nonzero(scores)[0]
        if nz.size == 0:
            return []
        k = min(k, nz.size)
        top = nz[np.argsort(-scores[nz])[:k]]
        return [(int(i), float(scores[i])) for i in top]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion: fuse multiple ranked lists (each is a list of doc ids in descending relevance order).
    score(d) = Σ_l 1 / (rrf_k + rank_l(d)). Returns (doc_id, fused_score) in descending order.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    chunk: CodeChunk
    vector: np.ndarray
    score: float


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, query_vec: np.ndarray,
               candidates: list[Candidate], k: int) -> list[tuple[CodeChunk, float]]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class MMRReranker(Reranker):
    """
    Maximal Marginal Relevance (pure numpy): balances relevance and diversity,
    avoiding a top-k that consists entirely of near-duplicate snippets.
    Higher lambda favors relevance over diversity.
    """

    def __init__(self, lambda_: float = 0.7) -> None:
        self._lambda = lambda_

    @property
    def name(self) -> str:
        return f"mmr-{self._lambda}"

    def rerank(self, query, query_vec, candidates, k):
        if not candidates:
            return []
        q = query_vec.reshape(-1).astype(np.float32)
        mat = np.vstack([c.vector.reshape(1, -1) for c in candidates]).astype(np.float32)
        rel = mat @ q  # relevance to the query
        selected: list[int] = []
        remaining = list(range(len(candidates)))
        k = min(k, len(candidates))
        while remaining and len(selected) < k:
            if not selected:
                best = int(remaining[int(np.argmax(rel[remaining]))])
            else:
                sel_mat = mat[selected]
                best, best_score = remaining[0], -1e9
                for idx in remaining:
                    diversity = float(np.max(sel_mat @ mat[idx]))
                    mmr = self._lambda * float(rel[idx]) - (1 - self._lambda) * diversity
                    if mmr > best_score:
                        best_score, best = mmr, idx
            selected.append(best)
            remaining.remove(best)
        return [(candidates[i].chunk, float(rel[i])) for i in selected]


class CrossEncoderReranker(Reranker):
    """sentence-transformers cross-encoder reranker (used when installed)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"cross-encoder:{self._model_name}"

    def rerank(self, query, query_vec, candidates, k):
        if not candidates:
            return []
        pairs = [(query, c.chunk.embed_text()) for c in candidates]
        scores = self._model.predict(pairs)
        order = np.argsort(-np.asarray(scores))[:k]
        return [(candidates[i].chunk, float(scores[i])) for i in order]


def create_reranker(kind: str = "none", **kwargs) -> Optional[Reranker]:
    """
        "none"          → None (no reranking)
        "mmr"           → MMRReranker
        "cross-encoder" → CrossEncoderReranker; degrades to MMR if unavailable
    """
    kind = (kind or "none").lower()
    if kind in ("", "none"):
        return None
    if kind == "mmr":
        return MMRReranker(**kwargs)
    if kind in ("cross-encoder", "cross_encoder", "ce"):
        try:
            return CrossEncoderReranker(**kwargs)
        except Exception as exc:
            logger.warning("cross-encoder unavailable (%s), falling back to MMR", exc)
            return MMRReranker()
    return None


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(query: str, max_extra: int = 1) -> list[str]:
    """
    Lightweight query expansion: original query + one keyword query built from identifier sub-tokens.
    (HyDE / multi-query can be built on top of this by passing an LLM-generated hypothetical document.)
    """
    queries = [query]
    subtokens: list[str] = []
    for tok in _TOKEN_RE.findall(query):
        parts = _SUBTOKEN_RE.findall(tok)
        subtokens.extend(p.lower() for p in parts if len(p) > 1)
    if subtokens:
        extra = " ".join(dict.fromkeys(subtokens))
        if extra and extra != query.lower():
            queries.append(extra)
    return queries[:1 + max_extra]


# ---------------------------------------------------------------------------
# Persistent cache
# ---------------------------------------------------------------------------

def _sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _load_cache(cache_dir: Path, emb_name: str, dim: int):
    """Return {rel: (hash, [CodeChunk], vectors)}; returns {} when missing or incompatible."""
    manifest_p = cache_dir / "manifest.json"
    chunks_p = cache_dir / "chunks.jsonl"
    vec_p = cache_dir / "vectors.npy"
    if not (manifest_p.exists() and chunks_p.exists() and vec_p.exists()):
        return {}
    try:
        manifest = json.loads(manifest_p.read_text("utf-8"))
        if manifest.get("embedding") != emb_name or manifest.get("dim") != dim:
            logger.info("RAG cache invalid (backend/dim changed) → full rebuild")
            return {}
        vectors = np.load(vec_p)
        chunk_dicts = [json.loads(line) for line in chunks_p.read_text("utf-8").splitlines() if line]
        chunks = [CodeChunk.from_dict(d) for d in chunk_dicts]
        if len(chunks) != vectors.shape[0]:
            return {}
        state: dict[str, tuple] = {}
        offset = 0
        for rel, info in manifest.get("files", {}).items():
            n = info["n"]
            state[rel] = (info["hash"], chunks[offset:offset + n], vectors[offset:offset + n])
            offset += n
        return state
    except Exception as exc:
        logger.warning("RAG cache load failed (%s) → full rebuild", exc)
        return {}


def _save_cache(cache_dir: Path, emb_name: str, dim: int,
                ordered: list[tuple[str, str, list[CodeChunk], np.ndarray]]) -> None:
    """ordered: [(rel, hash, chunks, vectors)] in file order."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        files_meta: dict[str, dict] = {}
        all_chunks: list[CodeChunk] = []
        vec_parts: list[np.ndarray] = []
        for rel, h, chunks, vecs in ordered:
            files_meta[rel] = {"hash": h, "n": len(chunks)}
            all_chunks.extend(chunks)
            if len(chunks):
                vec_parts.append(vecs)
        vectors = np.vstack(vec_parts) if vec_parts else np.zeros((0, dim), dtype=np.float32)
        (cache_dir / "manifest.json").write_text(
            json.dumps({"version": 1, "embedding": emb_name, "dim": dim, "files": files_meta}),
            encoding="utf-8",
        )
        with (cache_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(cache_dir / "vectors.npy", vectors)
    except Exception as exc:
        logger.warning("RAG cache save failed: %s", exc)


# ---------------------------------------------------------------------------
# RagRetriever
# ---------------------------------------------------------------------------

class RagRetriever:
    """
    End-to-end RAG retriever (hybrid retrieval + optional reranking + persistent incremental cache).

    Usage:
        rag = RagRetriever(repo_path="/repo", cache_dir=".rag_cache", reranker="mmr")
        rag.build()                                    # incremental chunking + embedding + index build
        ctx = rag.retrieve("fix the parser bug", k=5)  # string to inject into the prompt
    """

    def __init__(
        self,
        repo_path: str | Path,
        embeddings: EmbeddingBackend | None = None,
        chunk_lines: int = 40,
        overlap: int = 10,
        *,
        syntax_aware: bool = True,
        hybrid: bool = True,
        rrf_k: int = 60,
        index_kind: str = "auto",
        reranker: "str | Reranker | None" = None,
        cache_dir: str | Path | None = None,
        multi_query: bool = False,
        extra_paths: "list[str | Path] | None" = None,
    ) -> None:
        self._root = Path(repo_path).resolve()
        # External file library (#2): extra dirs/files indexed alongside the repo.
        self._extra_paths = list(extra_paths) if extra_paths else []
        self._embeddings = embeddings or create_embedding_backend()
        self._chunker = create_chunker(syntax_aware, chunk_lines, overlap)
        self._hybrid = hybrid
        self._rrf_k = rrf_k
        self._index_kind = index_kind
        self._multi_query = multi_query
        self._cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if isinstance(reranker, str):
            self._reranker = create_reranker(reranker)
        else:
            self._reranker = reranker

        self._chunks: list[CodeChunk] = []
        self._vectors: np.ndarray = np.zeros((0, self._embeddings.dim), dtype=np.float32)
        self._index: VectorIndex | None = None
        self._bm25: BM25Index | None = None
        self.stats: dict = {}

    # -- Read-only properties -----------------------------------------------

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def backend_info(self) -> str:
        idx = self._index.name if self._index else "none"
        parts = [f"embeddings={self._embeddings.name}", f"index={idx}"]
        if self._hybrid:
            parts.append("sparse=bm25")
        if self._reranker:
            parts.append(f"reranker={self._reranker.name}")
        if self._cache_dir:
            parts.append("cache=on")
        return ", ".join(parts)

    # -- Build (incremental) ------------------------------------------------

    def build(self) -> "RagRetriever":
        t0 = time.time()
        emb_name, dim = self._embeddings.name, self._embeddings.dim
        cached = _load_cache(self._cache_dir, emb_name, dim) if self._cache_dir else {}

        ordered: list[tuple[str, str, list[CodeChunk], np.ndarray]] = []
        reused = reembedded = embedded_chunks = 0
        pending: list[tuple[int, int]] = []   # (ordered index, chunk count)
        pending_texts: list[str] = []

        # Repo files first, then any external library files (#2). External chunks
        # use 'external/<label>/...' rel paths, so they cache and dedupe cleanly.
        source_iter = _iter_source_files(self._root)
        if self._extra_paths:
            import itertools
            source_iter = itertools.chain(source_iter, _iter_external_files(self._extra_paths))

        for _path, rel, content in source_iter:
            h = _sha(content)
            if rel in cached and cached[rel][0] == h:
                _, chunks, vecs = cached[rel]
                ordered.append((rel, h, chunks, vecs))
                reused += 1
            else:
                chunks = self._chunker.chunk_file(content, Path(rel))
                ordered.append((rel, h, chunks, np.zeros((0, dim), dtype=np.float32)))
                if chunks:
                    pending.append((len(ordered) - 1, len(chunks)))
                    pending_texts.extend(c.embed_text() for c in chunks)
                reembedded += 1

        # One batch embedding call for all changed files' chunks, then scatter back by file (incremental core)
        if pending_texts:
            all_vecs = self._embeddings.embed(pending_texts)
            off = 0
            for slot, n in pending:
                rel, h, chunks, _ = ordered[slot]
                ordered[slot] = (rel, h, chunks, all_vecs[off:off + n])
                off += n
                embedded_chunks += n

        # Aggregate
        self._chunks = [c for _, _, chunks, _ in ordered for c in chunks]
        vec_parts = [v for _, _, chunks, v in ordered if len(chunks)]
        self._vectors = np.vstack(vec_parts) if vec_parts else np.zeros((0, dim), dtype=np.float32)

        # Dense index
        self._index = create_vector_index(dim, self._index_kind)
        if self._vectors.size:
            self._index.add(self._vectors)
        # Sparse index
        self._bm25 = BM25Index().build([c.embed_text() for c in self._chunks]) if self._hybrid else None

        if self._cache_dir:
            _save_cache(self._cache_dir, emb_name, dim, ordered)

        self.stats = {
            "chunks": len(self._chunks),
            "files": len(ordered),
            "reused_files": reused,
            "reembedded_files": reembedded,
            "embedded_chunks": embedded_chunks,
            "build_seconds": round(time.time() - t0, 4),
            "backend": self.backend_info,
        }
        logger.info("RAG built: %s", self.stats)
        return self

    # -- Retrieval ----------------------------------------------------------

    def retrieve(self, query: str, k: int = 5, max_chars: int = 6_000,
                 *, language: str | None = None, path_prefix: str | None = None) -> str:
        hits = self.retrieve_chunks(query, k=k, language=language, path_prefix=path_prefix)
        if not hits:
            return ""
        lines = ["## Retrieved code (most relevant to the task)"]
        used = 0
        for chunk, score in hits:
            block = f"\n### {chunk.header()}  (score {score:.2f})\n```\n{chunk.text}\n```"
            if used + len(block) > max_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines)

    def retrieve_chunks(self, query: str, k: int = 5,
                        *, language: str | None = None,
                        path_prefix: str | None = None) -> list[tuple[CodeChunk, float]]:
        if self._index is None or not self._chunks:
            return []

        pool = max(k * 5, 50)
        queries = expand_query(query) if self._multi_query else [query]

        # Dense: possibly multiple queries
        dense_rankings: list[list[int]] = []
        q_vecs = self._embeddings.embed(queries)
        for i in range(q_vecs.shape[0]):
            hits = self._index.search(q_vecs[i], pool)
            dense_rankings.append([idx for idx, _ in hits])
        primary_qvec = q_vecs[0] if q_vecs.shape[0] else self._embeddings.embed([query])[0]

        # Fusion
        if self._hybrid and self._bm25 is not None:
            sparse = [idx for idx, _ in self._bm25.search(query, pool)]
            fused = reciprocal_rank_fusion(dense_rankings + [sparse], self._rrf_k)
        elif len(dense_rankings) > 1:
            fused = reciprocal_rank_fusion(dense_rankings, self._rrf_k)
        else:
            # Pure dense: use cosine scores directly
            fused = [(idx, score) for idx, score in self._index.search(primary_qvec, pool)]

        # Metadata filtering
        ranked = [(i, s) for i, s in fused if self._passes_filter(self._chunks[i], language, path_prefix)]
        if not ranked:
            return []

        # Reranking
        if self._reranker is not None:
            cands = [Candidate(self._chunks[i], self._vectors[i], s) for i, s in ranked[:pool]]
            return self._reranker.rerank(query, primary_qvec, cands, k)
        return [(self._chunks[i], float(s)) for i, s in ranked[:k]]

    def _passes_filter(self, chunk: CodeChunk, language, path_prefix) -> bool:
        if language and chunk.language != language:
            return False
        if path_prefix and not chunk.file.startswith(path_prefix):
            return False
        return True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_recall(
    retriever: RagRetriever,
    cases: Sequence[dict],
    k: int = 5,
    relevance_key: str = "relevant_files",
) -> dict:
    """
    Offline retrieval evaluation. cases: [{"query": str, "relevant_files": {...}}],
    where relevant_files is the set of relevant file names for that query.
    Returns recall@k / MRR / hit@k.
    """
    if not cases:
        return {"recall@k": 0.0, "mrr": 0.0, "hit@k": 0.0, "n": 0, "k": k}
    recalls, rr, hits = [], [], 0
    for case in cases:
        gold = set(case.get(relevance_key, []))
        if not gold:
            continue
        retrieved = retriever.retrieve_chunks(case["query"], k=k)
        files = [c.file for c, _ in retrieved]
        found = [f for f in files if f in gold]
        recalls.append(len(set(found) & gold) / len(gold))
        if found:
            hits += 1
            first = next(i for i, f in enumerate(files) if f in gold)
            rr.append(1.0 / (first + 1))
        else:
            rr.append(0.0)
    n = len(recalls)
    return {
        "recall@k": round(sum(recalls) / n, 4) if n else 0.0,
        "mrr": round(sum(rr) / n, 4) if n else 0.0,
        "hit@k": round(hits / n, 4) if n else 0.0,
        "n": n, "k": k,
    }
