"""
tests/test_symbol_index.py

Tests for the persistent, incremental SymbolIndex (feature #3) and its wiring
into FindSymbolTool. Pure-Python — no numpy, no API.
"""

import pytest

from context.symbol_index import SymbolIndex, SymbolLoc
from tools.search_tool import FindSymbolTool


def _make_repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text(
        "def parse_input(x):\n"
        "    return x\n"
        "\n"
        "class Parser:\n"
        "    def parse(self):\n"
        "        return 1\n"
        "\n"
        "def helper():\n"
        "    return 2\n"
    )
    (tmp_path / "util.py").write_text(
        "def parse_config():\n"
        "    return {}\n"
        "\n"
        "class Widget:\n"
        "    pass\n"
    )
    # A skipped dir must not be indexed.
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text("def ghost():\n    pass\n")
    return tmp_path


def test_build_indexes_symbols(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    assert idx.symbol_count >= 6            # parse_input, Parser, parse, helper, parse_config, Widget
    names = set(idx.all_names())
    assert "parse_input" in names
    assert "ghost" not in names             # __pycache__ skipped


def test_exact_lookup(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    hits = idx.lookup("Parser", mode="exact")
    assert len(hits) == 1
    assert hits[0].kind == "class"
    assert hits[0].file == "pkg/core.py"


def test_prefix_lookup_finds_all_parse(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    names = {loc.name for loc in idx.lookup("parse", mode="prefix")}
    assert {"parse_input", "parse", "parse_config"} <= names


def test_kind_filter(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    classes = idx.lookup("W", mode="prefix", kind="class")
    assert any(l.name == "Widget" for l in classes)
    assert all(l.kind == "class" for l in classes)


def test_path_prefix_filter(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    hits = idx.lookup("parse", mode="prefix", path_prefix="pkg/")
    assert hits and all(h.file.startswith("pkg/") for h in hits)


def test_toplevel_ordered_first(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    # "parse" prefix returns a top-level def (parse_input/parse_config) and a
    # method (Parser.parse); the top-level ones should sort ahead.
    hits = idx.lookup("parse", mode="prefix")
    assert hits[0].indent == 0


def test_incremental_cache_reuses_unchanged(tmp_path):
    repo = _make_repo(tmp_path)
    cache = str(tmp_path / ".symcache")
    first = SymbolIndex(repo, cache_dir=cache).build()
    assert first.stats["reparsed_files"] == 2
    assert first.stats["reused_files"] == 0

    # Rebuild without changes → everything reused, nothing reparsed.
    second = SymbolIndex(repo, cache_dir=cache).build()
    assert second.stats["reused_files"] == 2
    assert second.stats["reparsed_files"] == 0
    assert second.symbol_count == first.symbol_count

    # Change one file → only that file reparsed.
    (repo / "util.py").write_text("def parse_config():\n    return {'a': 1}\n\ndef added():\n    pass\n")
    third = SymbolIndex(repo, cache_dir=cache).build()
    assert third.stats["reparsed_files"] == 1
    assert third.stats["reused_files"] == 1
    assert any(l.name == "added" for l in third.lookup("added", mode="exact"))


# ---------------------------------------------------------------------------
# FindSymbolTool integration
# ---------------------------------------------------------------------------

def test_find_symbol_tool_uses_index(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    tool = FindSymbolTool(index=idx)
    res = tool.execute({"symbol": "parse"})
    assert res.success
    assert "parse_input" in res.output
    assert "parse_config" in res.output


def test_find_symbol_tool_falls_back_without_index(tmp_path):
    _make_repo(tmp_path)
    tool = FindSymbolTool()                 # no index → regex scan
    res = tool.execute({"symbol": "Parser", "path": str(tmp_path)})
    assert res.success
    assert "Parser" in res.output


@pytest.mark.parametrize("path", [".", "./", "", None])
def test_find_symbol_tool_root_path_is_not_a_filter(tmp_path, path):
    """Regression: the old tool took `path` as a search ROOT defaulting to '.', so a
    model may pass '.' — it must not be treated as a path prefix (which matches
    nothing and would wrongly report the symbol as missing)."""
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    tool = FindSymbolTool(index=idx)
    params = {"symbol": "parse"}
    if path is not None:
        params["path"] = path
    res = tool.execute(params)
    assert res.success
    assert "parse_input" in res.output and "parse_config" in res.output


def test_find_symbol_tool_absolute_repo_path_is_not_a_filter(tmp_path):
    repo = _make_repo(tmp_path)
    tool = FindSymbolTool(index=SymbolIndex(repo).build())
    res = tool.execute({"symbol": "parse", "path": str(repo)})
    assert "parse_input" in res.output and "parse_config" in res.output


def test_find_symbol_tool_real_subdir_still_filters(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    tool = FindSymbolTool(index=idx)
    res = tool.execute({"symbol": "parse", "path": "pkg/"})
    assert "parse_input" in res.output          # in pkg/
    assert "parse_config" not in res.output     # in util.py, filtered out


def test_find_symbol_tool_no_match(tmp_path):
    idx = SymbolIndex(_make_repo(tmp_path)).build()
    tool = FindSymbolTool(index=idx)
    res = tool.execute({"symbol": "nonexistent_symbol"})
    assert res.success
    assert "No definition found" in res.output


def test_symbol_loc_roundtrip():
    loc = SymbolLoc(name="f", kind="function", file="a.py", line=3, indent=0)
    assert SymbolLoc.from_dict(loc.to_dict()) == loc
