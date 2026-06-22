"""
tools/base.py

Tool layer infrastructure:
- ToolResult     result of a tool execution
- BaseTool       abstract base class for all tools
- ToolRegistry   tool registry; core.py uses it to execute tools and generate schemas

To add a new tool:
    1. Subclass BaseTool, implement execute() and the schema properties
    2. Call registry.register(MyTool())
    No other code needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from agent.task import Observation, ObservationStatus
from llm.base import LLMToolSchema


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    Raw result returned by each Tool.execute().
    core.py converts it to an Observation before writing it to EventLog.
    """
    success: bool
    output: str                         # text output from the tool, already truncated
    error: str | None = None            # error message on failure

    def to_observation(self, tool_name: str) -> Observation:
        """Convert to an Observation for core.py to write to EventLog and inject into context."""
        return Observation(
            status=ObservationStatus.SUCCESS if self.success else ObservationStatus.ERROR,
            output=self.output,
            tool_name=tool_name,
            error=self.error,
        )


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Subclasses must implement:
    - name:     tool name (matches the function name used in LLM function calling)
    - schema:   JSON Schema description telling the LLM how to use this tool
    - execute(): the actual execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name, e.g. "shell", "file_read". Must be globally unique."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool capability description, injected into the LLM's system prompt and tool schema."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """
        JSON Schema for the parameters. Example:
        {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to run"},
            },
            "required": ["cmd"],
        }
        """
        ...

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool and return a ToolResult. Never raise; wrap errors in ToolResult.error."""
        ...

    def to_llm_schema(self) -> LLMToolSchema:
        """Generate the schema for LLM use; called by ToolRegistry."""
        return LLMToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Tool registry. core.py holds one registry instance and uses it to:
    1. Look up and execute tools (execute_tool)
    2. Generate schema lists to inject into the LLM (get_schemas)

    Thread safety: v1 is single-threaded; no locking.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """
        Register a tool. Supports method chaining:
            registry.register(ShellTool()).register(FileTool())
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        return self

    @property
    def tool_names(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools)

    def subset(self, names) -> "ToolRegistry":
        """
        Return a new registry containing only the named tools (missing names are
        skipped). Used by the multi-agent orchestrator to give each role a
        restricted tool set (e.g. read-only for the planner/reviewer).
        """
        r = ToolRegistry()
        for n in names:
            if n in self._tools:
                r._tools[n] = self._tools[n]
        return r

    def execute_tool(self, name: str, params: dict[str, Any]) -> ToolResult:
        """
        Look up a tool by name and execute it.
        Returns an error ToolResult if the tool is not found (does not raise),
        so the agent can continue running.
        """
        if name not in self._tools:
            available = ", ".join(self._tools.keys()) or "none"
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool '{name}'. Available tools: {available}",
            )

        tool = self._tools[name]
        try:
            return tool.execute(params)
        except Exception as exc:
            # Uncaught exception inside the tool; degrade to an error result
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' raised an unexpected error: {exc}",
            )

    def get_schemas(self) -> list[LLMToolSchema]:
        """Return schemas for all registered tools, to be injected into the LLM."""
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def get_tool(self, name: str) -> "BaseTool | None":
        """Retrieve a registered tool instance by name (used by LangChain adapters etc.)."""
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"


# ---------------------------------------------------------------------------
# NoopTool — test helper
# ---------------------------------------------------------------------------

class NoopTool(BaseTool):
    """
    Test-only tool whose execute() always returns success without doing anything.
    Used to test core.py flows without depending on the real filesystem or shell.
    """

    def __init__(self, tool_name: str = "noop", output: str = "ok") -> None:
        self._name = tool_name
        self._output = output
        self.call_count = 0
        self.last_params: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"No-op tool '{self._name}' for testing."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Anything"},
            },
            "required": [],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        self.call_count += 1
        self.last_params = params
        return ToolResult(success=True, output=self._output)


class FailingTool(BaseTool):
    """
    Test-only tool whose execute() always returns failure.
    Used to test the Reflection trigger (test-failure path).
    """

    def __init__(self, tool_name: str = "test", error_msg: str = "AssertionError: 1 != 2") -> None:
        self._name = tool_name
        self._error_msg = error_msg
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Always-failing tool '{self._name}' for testing."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, params: dict[str, Any]) -> ToolResult:
        self.call_count += 1
        return ToolResult(
            success=False,
            output=self._error_msg,
            error=self._error_msg,
        )
