"""
tests/test_rag.py

RAG retriever tests. All tests use offline HashingEmbeddings + NumpyIndex;
no OpenAI key, faiss, or network access required — fast and deterministic.

Covers:
- Chunker splitting
- HashingEmbeddings normalization / determinism
- NumpyIndex / FaissIndex top-k
- create_embedding_backend fallback when no API key is present
- RagRetriever end-to-end retrieval relevance
- Integration with agent/core.py (retrieved context injected into system prompt)
"""

from pathlib import Path

import numpy as np
import pytest

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, Task, ToolCall
from context.rag import (
    BM25Index,
    Candidate,
    Chunker,
    CodeChunk,
    HashingEmbeddings,
    MMRReranker,
    NumpyIndex,
    RagRetriever,
    SyntaxChunker,
    create_embedding_backend,
    create_reranker,
    create_vector_index,
    evaluate_recall,
    expand_query,
    reciprocal_rank_fusion,
)
from llm.base import MockBackend
from tools.base import NoopTool, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def code_repo(tmp_path):
    """Two source files with different topics, for testing retrieval relevance."""
    (tmp_path / "parser.py").write_text(
        "def parse_tokens(source):\n"
        "    tokenizer = Tokenizer(source)\n"
        "    return tokenizer.parse()\n"
        "\n"
        "class Tokenizer:\n"
        "    def parse(self):\n"
        "        return self.tokens\n"
    )
    (tmp_path / "network.py").write_text(
        "def connect_socket(host, port):\n"
        "    sock = Socket(host, port)\n"
        "    sock.connect()\n"
        "    return sock\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def test_chunker_splits_file_into_windows(tmp_path):
    (tmp_path / "big.py").write_text("\n".join(f"line_{i} = {i}" for i in range(100)))
    chunks = Chunker(chunk_lines=40, overlap=10).chunk_repo(tmp_path)
    assert len(chunks) >= 2
    assert all(isinstance(c, CodeChunk) for c in chunks)
    assert all(c.file == "big.py" for c in chunks)
    # windows overlap: second chunk start_line <= first chunk end_line
    assert chunks[1].start_line <= chunks[0].end_line


def test_chunker_labels_symbols(code_repo):
    chunks = Chunker(chunk_lines=40, overlap=0).chunk_repo(code_repo)
    parser_chunks = [c for c in chunks if c.file == "parser.py"]
    all_syms = [s for c in parser_chunks for s in c.symbols]
    assert any("parse_tokens" in s for s in all_syms)
    assert any("Tokenizer" in s for s in all_syms)


def test_chunker_skips_non_source(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "code.py").write_text("x = 1\n")
    files = {c.file for c in Chunker().chunk_repo(tmp_path)}
    assert "code.py" in files
    assert "image.png" not in files


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def test_hashing_embeddings_normalized():
    emb = HashingEmbeddings(dim=128)
    vecs = emb.embed(["hello world foo", "another piece of text"])
    assert vecs.shape == (2, 128)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hashing_embeddings_deterministic():
    emb = HashingEmbeddings(dim=64)
    a = emb.embed(["parse the tokens"])
    b = emb.embed(["parse the tokens"])
    assert np.array_equal(a, b)


def test_hashing_embeddings_similarity_reflects_overlap():
    emb = HashingEmbeddings(dim=512)
    v = emb.embed([
        "tokenizer parse source tokens",      # 0: query-like
        "tokenizer parse source tokens code", # 1: high overlap
        "socket connect host port network",   # 2: unrelated
    ])
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


def test_empty_input_embeds_to_empty_matrix():
    emb = HashingEmbeddings(dim=32)
    out = emb.embed([])
    assert out.shape == (0, 32)


def test_create_embedding_backend_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = create_embedding_backend(provider="auto")
    assert isinstance(backend, HashingEmbeddings)


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------

def test_numpy_index_topk_sorted():
    idx = NumpyIndex(dim=3)
    idx.add(np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32))
    results = idx.search(np.array([1, 0, 0], dtype=np.float32), k=2)
    assert len(results) == 2
    # sorted by descending score: row 0 is most similar, row 2 is next
    assert results[0][0] == 0
    assert results[0][1] >= results[1][1]


def test_numpy_index_empty_returns_nothing():
    idx = NumpyIndex(dim=4)
    assert idx.search(np.zeros(4, dtype=np.float32), k=3) == []


def test_create_vector_index_returns_working_index():
    idx = create_vector_index(dim=2)
    idx.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
    res = idx.search(np.array([1, 0], dtype=np.float32), k=1)
    assert res and res[0][0] == 0


# ---------------------------------------------------------------------------
# RagRetriever end-to-end
# ---------------------------------------------------------------------------

def test_retriever_finds_relevant_chunk(code_repo):
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings()).build()
    assert rag.chunk_count >= 2
    hits = rag.retrieve_chunks("fix the parser tokenizer bug", k=1)
    assert hits
    top_chunk, _score = hits[0]
    assert top_chunk.file == "parser.py"


def test_retrieve_returns_formatted_string(code_repo):
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings()).build()
    text = rag.retrieve("tokenizer parse", k=2)
    assert "Retrieved code" in text
    assert "```" in text
    assert "parser.py" in text


def test_retriever_empty_repo(tmp_path):
    rag = RagRetriever(tmp_path, embeddings=HashingEmbeddings()).build()
    assert rag.chunk_count == 0
    assert rag.retrieve("anything") == ""


def test_retrieve_before_build_is_safe(code_repo):
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings())
    # calling retrieve before build should not crash
    assert rag.retrieve("x") == ""


# ---------------------------------------------------------------------------
# Integration with agent core
# ---------------------------------------------------------------------------

def test_core_injects_rag_context_into_system_prompt(code_repo, tmp_path):
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings()).build()
    backend = MockBackend([
        Action(ActionType.FINISH, thought="done", message="done"),
    ])
    registry = ToolRegistry().register(NoopTool("shell"))
    cfg = AgentConfig(max_steps=2, retriever=rag)
    agent = Agent(backend, registry, cfg)

    task = Task(description="fix the parser tokenizer bug",
                repo_path=str(code_repo), max_steps=2)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        result = agent.run(task, log)

    assert result.is_success()
    # The first LLM call's system message should contain the retrieved code
    system_msg = backend.received_messages[0][0]
    assert system_msg.role == "system"
    assert "Retrieved code" in system_msg.content
    assert "parser.py" in system_msg.content


def test_core_without_retriever_has_no_retrieved_section(code_repo, tmp_path):
    backend = MockBackend([Action(ActionType.FINISH, thought="done", message="done")])
    registry = ToolRegistry().register(NoopTool("shell"))
    agent = Agent(backend, registry, AgentConfig(max_steps=2))  # retriever=None

    task = Task(description="anything", repo_path=str(code_repo), max_steps=2)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        agent.run(task, log)

    system_msg = backend.received_messages[0][0]
    assert "Retrieved code" not in system_msg.content


# ---------------------------------------------------------------------------
# Syntax-aware chunking (SyntaxChunker)
# ---------------------------------------------------------------------------

def test_syntax_chunker_splits_python_at_def_boundaries(code_repo):
    chunks = SyntaxChunker().chunk_file((code_repo / "parser.py").read_text(), Path("parser.py"))
    # functions and classes each get their own chunk, not mixed together
    func = [c for c in chunks if any("parse_tokens" in s for s in c.symbols)]
    cls = [c for c in chunks if any("Tokenizer" in s for s in c.symbols)]
    assert func and cls
    assert func[0] is not cls[0]
    assert all(c.language == "python" for c in chunks)


def test_syntax_chunker_falls_back_on_unparseable_python(tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n  this is not valid python !!!\n")
    chunks = SyntaxChunker().chunk_file((tmp_path / "broken.py").read_text(), Path("broken.py"))
    # parse failure should still produce output (falls back to line windows), not crash
    assert chunks
    assert all(isinstance(c, CodeChunk) for c in chunks)


def test_syntax_chunker_large_function_subsplit(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    (tmp_path / "big.py").write_text(f"def huge():\n{body}\n")
    chunks = SyntaxChunker(max_chunk_lines=80).chunk_file(
        (tmp_path / "big.py").read_text(), Path("big.py"))
    assert len(chunks) >= 2  # oversized function is sub-split
    assert all((c.end_line - c.start_line + 1) <= 80 for c in chunks)


# ---------------------------------------------------------------------------
# BM25 sparse retrieval
# ---------------------------------------------------------------------------

def test_bm25_ranks_keyword_match_first():
    bm = BM25Index().build([
        "tokenizer parse source tokens",
        "socket connect host port network",
        "logging configure handler format",
    ])
    hits = bm.search("tokenizer parse", k=2)
    assert hits
    assert hits[0][0] == 0  # doc 0 matches keywords best


def test_bm25_empty_query_or_index():
    assert BM25Index().build([]).search("anything", k=3) == []
    bm = BM25Index().build(["alpha beta"])
    assert bm.search("zzz nomatch", k=3) == []  # no hits returns empty


# ---------------------------------------------------------------------------
# RRF fusion / query expansion
# ---------------------------------------------------------------------------

def test_rrf_rewards_agreement_across_rankings():
    # doc 2 ranks first in both lists → should rank first after fusion
    fused = reciprocal_rank_fusion([[2, 0, 1], [2, 1, 0]], rrf_k=60)
    assert fused[0][0] == 2
    # scores in descending order
    assert all(fused[i][1] >= fused[i + 1][1] for i in range(len(fused) - 1))


def test_expand_query_splits_identifiers():
    qs = expand_query("fix parseTokens in Tokenizer")
    assert qs[0] == "fix parseTokens in Tokenizer"          # original query preserved
    assert any("parse" in q and "tokens" in q for q in qs)  # sub-words extracted


# ---------------------------------------------------------------------------
# Reranking (MMR / factory)
# ---------------------------------------------------------------------------

def test_mmr_reranker_returns_k_diverse_chunks():
    emb = HashingEmbeddings(dim=64)
    texts = ["parse tokens", "parse tokens again", "socket network host"]
    vecs = emb.embed(texts)
    chunks = [CodeChunk(f"f{i}.py", 1, 1, texts[i]) for i in range(3)]
    cands = [Candidate(chunks[i], vecs[i], 1.0) for i in range(3)]
    qv = emb.embed(["parse tokens"])[0]
    out = MMRReranker(lambda_=0.5).rerank("parse tokens", qv, cands, k=2)
    assert len(out) == 2
    assert all(isinstance(c, CodeChunk) for c, _ in out)


def test_create_reranker_factory():
    assert create_reranker("none") is None
    assert isinstance(create_reranker("mmr"), MMRReranker)
    # cross-encoder falls back to MMR when unavailable (does not raise)
    assert isinstance(create_reranker("cross-encoder"), MMRReranker)


# ---------------------------------------------------------------------------
# Hybrid retrieval + metadata filtering
# ---------------------------------------------------------------------------

def test_hybrid_retrieval_finds_relevant(code_repo):
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings(), hybrid=True).build()
    hits = rag.retrieve_chunks("fix the parser tokenizer bug", k=1)
    assert hits and hits[0][0].file == "parser.py"
    assert "sparse=bm25" in rag.backend_info


def test_metadata_filter_by_path_and_language(code_repo):
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings()).build()
    hits = rag.retrieve_chunks("connect socket", k=5, path_prefix="network")
    assert hits and all(c.file.startswith("network") for c, _ in hits)
    none = rag.retrieve_chunks("connect socket", k=5, language="rust")
    assert none == []  # no rust files in the repo


def test_reranker_string_arg_is_constructed(code_repo):
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings(), reranker="mmr").build()
    assert "reranker=mmr" in rag.backend_info
    assert rag.retrieve_chunks("tokenizer", k=2)


# ---------------------------------------------------------------------------
# Persistence + incremental updates
# ---------------------------------------------------------------------------

def test_persistence_writes_cache(code_repo, tmp_path):
    cache = tmp_path / "ragcache"
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings(), cache_dir=cache).build()
    assert (cache / "manifest.json").exists()
    assert (cache / "chunks.jsonl").exists()
    assert (cache / "vectors.npy").exists()
    assert rag.stats["embedded_chunks"] == rag.chunk_count


def test_incremental_reuses_unchanged_files(code_repo, tmp_path):
    cache = tmp_path / "ragcache"
    first = RagRetriever(code_repo, embeddings=HashingEmbeddings(), cache_dir=cache).build()
    assert first.stats["reembedded_files"] == 2
    # second build: nothing changed → all files reused, zero re-embeddings
    second = RagRetriever(code_repo, embeddings=HashingEmbeddings(), cache_dir=cache).build()
    assert second.stats["reused_files"] == 2
    assert second.stats["embedded_chunks"] == 0
    assert second.chunk_count == first.chunk_count


def test_incremental_reembeds_only_changed_file(code_repo, tmp_path):
    cache = tmp_path / "ragcache"
    RagRetriever(code_repo, embeddings=HashingEmbeddings(), cache_dir=cache).build()
    (code_repo / "network.py").write_text("def changed():\n    return 1\n")
    third = RagRetriever(code_repo, embeddings=HashingEmbeddings(), cache_dir=cache).build()
    assert third.stats["reused_files"] == 1       # parser.py reused
    assert third.stats["reembedded_files"] == 1   # only network.py re-embedded
    assert third.stats["embedded_chunks"] >= 1


def test_cache_invalidated_when_embedding_changes(code_repo, tmp_path):
    cache = tmp_path / "ragcache"
    RagRetriever(code_repo, embeddings=HashingEmbeddings(dim=128), cache_dir=cache).build()
    # different embedding dimension → cache incompatible → full rebuild
    rebuilt = RagRetriever(code_repo, embeddings=HashingEmbeddings(dim=256), cache_dir=cache).build()
    assert rebuilt.stats["reused_files"] == 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_recall_perfect_and_empty(code_repo):
    rag = RagRetriever(code_repo, embeddings=HashingEmbeddings()).build()
    cases = [
        {"query": "parser tokenizer", "relevant_files": {"parser.py"}},
        {"query": "connect socket host", "relevant_files": {"network.py"}},
    ]
    metrics = evaluate_recall(rag, cases, k=2)
    assert metrics["recall@k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["n"] == 2
    assert evaluate_recall(rag, [], k=2)["n"] == 0
