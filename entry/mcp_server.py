"""
entry/mcp_server.py

Run the coding-agent tool registry as an MCP server (stdio or HTTP).

Usage:
  agent mcp --repo .
  agent mcp --repo . --transport http --port 8767

Connect from Claude Desktop / Cursor (stdio example):
  {
    "mcpServers": {
      "coding-agent": {
        "command": "python",
        "args": ["-m", "entry.mcp_server", "--repo", "/path/to/project"]
      }
    }
  }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding Agent MCP Server")
    parser.add_argument("--repo", "-r", default=".", help="Repository root path")
    parser.add_argument(
        "--transport", "-t", default="stdio",
        choices=["stdio", "http", "streamable-http"],
        help="MCP transport (default: stdio for IDE integration)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--include-writes", action="store_true",
                        help="Also expose file_edit/file_write/pytest (use with care)")
    args = parser.parse_args()

    repo = str(Path(args.repo).resolve())
    import os
    os.chdir(repo)
    from config.schema import load_config
    from entry.cli import _build_registry, _build_symbol_index
    from context.mcp_bridge import build_mcp_server, run_mcp_server_blocking

    cfg = load_config()
    symbol_index = _build_symbol_index(repo)
    registry = _build_registry(cfg, symbol_index=symbol_index, repo_path=repo)
    server = build_mcp_server(
        registry, repo_path=repo, include_writes=args.include_writes,
    )
    run_mcp_server_blocking(
        server, transport=args.transport, host=args.host, port=args.port,
    )


if __name__ == "__main__":
    main()
