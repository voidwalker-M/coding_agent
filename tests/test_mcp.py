"""Tests for MCP server/client bridge."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from config.schema import load_config
from entry.cli import _build_registry, _build_symbol_index
from context.mcp_bridge import DEFAULT_MCP_TOOL_NAMES, build_mcp_server


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "hello.py").write_text("print('hi')\n")
    return tmp_path


def test_mcp_server_exposes_registry_tools(repo, monkeypatch):
    monkeypatch.chdir(repo)
    cfg = load_config()
    reg = _build_registry(cfg, symbol_index=_build_symbol_index(str(repo)), repo_path=str(repo))
    server = build_mcp_server(reg, repo_path=str(repo))

    async def _check():
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert names >= set(DEFAULT_MCP_TOOL_NAMES)
        result = await server.call_tool("file_read", {"path": "hello.py"})
        text = result.content[0].text
        assert "print('hi')" in text

    asyncio.run(_check())
