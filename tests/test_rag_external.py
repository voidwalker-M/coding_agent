"""
tests/test_rag_external.py

Tests for the RAG external file library (#2): indexing dirs/files outside the
repo (docs, dependency source). Skipped automatically when numpy is unavailable,
consistent with the rest of the RAG test suite.
"""

import pytest

pytest.importorskip("numpy")

from context.rag import RagRetriever, _iter_external_files  # noqa: E402


def test_iter_external_files_dir_namespaces_and_filters(tmp_path):
    ext = tmp_path / "libdocs"
    ext.mkdir()
    (ext / "guide.md").write_text("# Guide\nuse the API\n")
    (ext / "helper.py").write_text("def helper():\n    pass\n")
    (ext / "image.bin").write_bytes(b"\x00\x01\x02")   # unsupported ext, skipped

    rels = sorted(rel for _, rel, _ in _iter_external_files([str(ext)]))
    assert rels == ["external/libdocs/guide.md", "external/libdocs/helper.py"]


def test_iter_external_files_single_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("some reference notes")
    out = list(_iter_external_files([str(f)]))
    assert len(out) == 1
    assert out[0][1] == "external/notes.md"


def test_iter_external_files_missing_path_skipped(tmp_path):
    assert list(_iter_external_files([str(tmp_path / "does_not_exist")])) == []


def test_retriever_indexes_external_library(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    return 1\n")

    ext = tmp_path / "docs"
    ext.mkdir()
    (ext / "api.md").write_text("The frobnicate function reticulates splines.\n" * 6)

    rag = RagRetriever(str(repo), extra_paths=[str(ext)]).build()
    ctx = rag.retrieve("how does frobnicate work", k=5)
    # The external doc must be retrievable and clearly namespaced.
    assert "external/docs/api.md" in ctx
