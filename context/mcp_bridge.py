"""
context/mcp_bridge.py

Model Context Protocol (MCP) integration:

- **Server**: expose the agent ToolRegistry to external MCP hosts (Claude Desktop,
  Cursor, TRAE, etc.) over stdio or Streamable HTTP.
- **Client**: attach tools from an external MCP server into the local registry so
  the ReAct loop can call third-party capabilities through the same Tool Calling path.

Design: reuse existing BaseTool.execute() — MCP is a transport/protocol layer, not
a parallel tool system.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from tools.base import BaseTool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Read-only tools safe to expose over MCP by default (no shell / file writes).
DEFAULT_MCP_TOOL_NAMES = (
    "file_read", "file_view", "search_text", "find_files", "find_symbol",
    "git_status", "git_diff", "skill", "web_fetch",
)


def _require_mcp():
    try:
        from mcp.server import MCPServer  # noqa: F401
        from mcp.client.stdio import stdio_client  # noqa: F401
        from mcp import ClientSession, StdioServerParameters  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MCP SDK not installed. Run: pip install 'coding-agent[server]'"
        ) from exc


def build_mcp_server(
    registry: ToolRegistry,
    *,
    repo_path: str,
    tool_names: tuple[str, ...] | None = None,
    include_writes: bool = False,
) -> "Any":
    """Build an MCPServer that delegates tool calls to the local ToolRegistry."""
    _require_mcp()
    from mcp.server import MCPServer

    names = tool_names or DEFAULT_MCP_TOOL_NAMES
    if include_writes:
        names = tuple(set(names) | {"file_edit", "file_write", "pytest"})

    server = MCPServer(
        name="coding-agent",
        title="Coding Agent MCP",
        description=(
            f"Code exploration and editing tools for repository at {repo_path}. "
            "Wraps the same ToolRegistry used by the ReAct agent."
        ),
        version="0.1.0",
        instructions=(
            "Use file_read/file_view to inspect code, search_text/find_symbol to locate "
            "symbols, git_diff/git_status to review changes, skill to load SOP playbooks, "
            "web_fetch to read http(s) pages. Write tools are off by default."
        ),
    )

    for name in names:
        tool = registry.get_tool(name)
        if tool is None:
            continue
        server.add_tool(_wrap_registry_tool(tool), name=name, description=tool.description)

    @server.resource("repo://summary")
    def repo_summary() -> str:
        """Static repo-map summary for MCP hosts."""
        try:
            from context.repo_map import RepoMap
            return RepoMap(repo_path).build(budget=4000)
        except Exception as exc:
            return f"(repo summary unavailable: {exc})"

    return server


def _wrap_registry_tool(tool: BaseTool) -> Callable[..., str]:
    """Convert a BaseTool into a plain function MCPServer can expose."""
    props = tool.parameters_schema.get("properties", {})
    required = set(tool.parameters_schema.get("required", []))

    def _run(params: dict[str, Any]) -> str:
        clean = {k: v for k, v in params.items() if v is not None}
        result = tool.execute(clean)
        if not result.success:
            raise RuntimeError(result.error or f"tool {tool.name} failed")
        return result.output

    if not props:
        def _handler() -> str:
            return _run({})

        _handler.__name__ = tool.name
        _handler.__doc__ = tool.description
        return _handler

    # Build explicit signature so MCP infers per-field schema (not a single **dict).
    params = list(props.keys())
    args = ", ".join(f"{p}: str" + ("" if p in required else "=None") for p in params)
    body = f"return _run({{{', '.join(f'\"{p}\": {p}' for p in params)}}})"
    namespace: dict[str, Any] = {"_run": _run, "str": str}
    src = f"def _handler({args}):\n    {body}\n"
    exec(src, namespace)  # noqa: S102 — small controlled signature synthesis
    handler = namespace["_handler"]
    handler.__name__ = tool.name
    handler.__doc__ = tool.description
    return handler


async def _call_mcp_tool(session, name: str, params: dict) -> ToolResult:
    try:
        result = await session.call_tool(name, params)
        parts = []
        for block in result.content or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        output = "\n".join(parts) if parts else json.dumps(
            result.model_dump() if hasattr(result, "model_dump") else str(result),
            ensure_ascii=False,
        )
        return ToolResult(success=not result.isError, output=output,
                          error=None if not result.isError else output)
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


async def discover_mcp_tools(
    *,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """List tools exposed by an external MCP server (stdio)."""
    _require_mcp()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args or [], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            if not isinstance(tools, list):
                tools = list(getattr(tools, "tools", tools))
            out = []
            for t in tools:
                schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}
                out.append({
                    "name": t.name,
                    "description": t.description or t.name,
                    "input_schema": schema,
                })
            return out


class McpOneShotTool(BaseTool):
    """Call a remote MCP tool by spawning a short-lived stdio client session."""

    def __init__(
        self,
        *,
        remote_name: str,
        local_name: str,
        description: str,
        parameters_schema: dict,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._remote_name = remote_name
        self._local_name = local_name
        self._description = description
        self._schema = parameters_schema
        self._command = command
        self._args = args or []
        self._env = env

    @property
    def name(self) -> str:
        return self._local_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._schema

    def execute(self, params: dict[str, Any]) -> ToolResult:
        return asyncio.run(_call_remote_tool_once(
            command=self._command, args=self._args, env=self._env,
            tool_name=self._remote_name, params=params,
        ))


async def _call_remote_tool_once(
    *,
    command: str,
    args: list[str],
    env: dict[str, str] | None,
    tool_name: str,
    params: dict[str, Any],
) -> ToolResult:
    _require_mcp()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    sp = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(sp) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await _call_mcp_tool(session, tool_name, params)


def register_mcp_client_tools(
    registry: ToolRegistry,
    *,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    prefix: str = "mcp_",
) -> list[str]:
    """Discover tools on an external MCP server and register them locally."""
    tools = asyncio.run(discover_mcp_tools(command=command, args=args, env=env))
    attached: list[str] = []
    for t in tools:
        local = f"{prefix}{t['name']}"
        if registry.get_tool(local) is not None:
            local = f"{prefix}{t['name']}_ext"
        registry.register(McpOneShotTool(
            remote_name=t["name"],
            local_name=local,
            description=t["description"],
            parameters_schema=t.get("input_schema") or {"type": "object", "properties": {}},
            command=command,
            args=args,
            env=env,
        ))
        attached.append(local)
    return attached


async def run_mcp_stdio(server) -> None:
    await server.run_stdio_async()


async def run_mcp_streamable_http(server, *, host: str = "127.0.0.1", port: int = 8767) -> None:
    await server.run_streamable_http_async(host=host, port=port)


def run_mcp_server_blocking(
    server,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8767,
) -> None:
    """Blocking entry for CLI."""
    if transport == "stdio":
        asyncio.run(run_mcp_stdio(server))
    elif transport in ("http", "streamable-http", "sse"):
        asyncio.run(run_mcp_streamable_http(server, host=host, port=port))
    else:
        raise ValueError(f"unknown MCP transport: {transport}")
