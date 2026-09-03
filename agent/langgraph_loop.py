"""
agent/langgraph_loop.py

LangGraph implementation of the ReAct main loop — a drop-in replacement for agent/core.py.

Models the plan → act → observe → reflect cycle as a LangGraph StateGraph:

        ┌─────────┐  tool_call   ┌─────────┐
        │  agent  │ ───────────▶ │  tools  │
        │  node   │ ◀─────────── │  node   │
        └────┬────┘              └─────────┘
             │ finish / give_up / max_steps
             ▼
            END

Reuses existing components to ensure behavioral consistency and output compatibility:
- LLMBackend:       decisions are still made via backend.complete(), compatible across providers
- ToolRegistry:     wrapped as LangChain StructuredTools; tool execution goes through the LangChain interface
- EventLog:         each node writes action/observation; real-time CLI printing and replay are fully compatible
- RagRetriever / RepoMap / TokenBudget: same context-building logic as core.py

Dependencies (optional extra):
    pip install langgraph langchain-core pydantic
When not installed, import raises a friendly error with install instructions.

Design intent: demonstrates that the same agent can be driven by either a from-scratch
engine or a mainstream orchestration framework (LangGraph), while both share the same
tool layer and context layer.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Callable, Optional, TypedDict

from agent.core import AgentConfig, _make_history_compactor
from agent.decision import DecisionEngine, DecisionKind, is_clean_git_repo
from agent.event_log import EventLog
from agent.plan import Plan
from agent.prompt import (
    build_system_prompt,
    build_task_prompt,
)
from agent.task import (
    Action, ActionType, RunResult, RunStatus, Task,
)
from context.history import ConversationHistory
from context.repo_map import RepoMap
from context.token_budget import TokenBudget
from context.workspace_sync import record_edit
from llm.base import LLMBackend, LLMMessage
from tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolRegistry → LangChain tools
# ---------------------------------------------------------------------------

_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def to_langchain_tools(registry: ToolRegistry) -> dict[str, Any]:
    """
    Wrap each tool in the ToolRegistry as a LangChain StructuredTool.

    Returns {tool_name: StructuredTool}. Each tool's JSON Schema parameters are
    converted to a pydantic args_schema; on invocation, parameters are validated
    and then delegated to registry.execute_tool(), returning a native ToolResult
    (preserving success/error for EventLog recording).
    """
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    tools: dict[str, Any] = {}
    for name in registry.tool_names:
        base = registry.get_tool(name)
        if base is None:
            continue
        schema = base.parameters_schema or {}
        props: dict[str, dict] = schema.get("properties", {})
        required = set(schema.get("required", []))

        fields: dict[str, tuple] = {}
        for pname, pspec in props.items():
            pytype = _JSON_TO_PY.get(pspec.get("type", "string"), str)
            desc = pspec.get("description", "")
            if pname in required:
                fields[pname] = (pytype, Field(description=desc))
            else:
                fields[pname] = (Optional[pytype], Field(default=None, description=desc))

        args_model = create_model(f"{name.capitalize()}Args", **fields)  # type: ignore[call-overload]

        def make_run(tool_name: str) -> Callable[..., ToolResult]:
            def _run(**kwargs: Any) -> ToolResult:
                params = {k: v for k, v in kwargs.items() if v is not None}
                return registry.execute_tool(tool_name, params)
            return _run

        tools[name] = StructuredTool(
            name=name,
            description=base.description,
            args_schema=args_model,
            func=make_run(name),
        )
    return tools


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """State passed between nodes in LangGraph. history is a mutable object; updated in-place."""
    history: ConversationHistory
    step: int
    total_tokens: int
    steps_without_edit: int
    status: str            # "running" | "success" | "gave_up"
    summary: str
    last_action: Action
    noop_finish_rejections: int
    verified_since_edit: bool
    located: bool
    retry_agent: bool


# ---------------------------------------------------------------------------
# LangGraphAgent
# ---------------------------------------------------------------------------

class LangGraphAgent:
    """
    Same interface as agent.core.Agent (run(task, log) -> RunResult),
    but driven internally by a LangGraph StateGraph.
    The CLI can switch to this via --engine langgraph.
    """

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._cfg = config or AgentConfig()
        self._compactor = _make_history_compactor(self._backend, self._cfg)
        self._decision = DecisionEngine.from_config(self._cfg)
        plan_tool = self._registry.get_tool("plan")
        self._plan: Plan = getattr(plan_tool, "plan", None) or Plan()
        # Wrap as LangChain tools; fall back to direct registry use if wrapping fails
        try:
            self._lc_tools = to_langchain_tools(registry)
        except Exception as exc:  # langchain missing or schema error
            logger.warning("LangChain tool wrapping failed (%s); using registry directly", exc)
            self._lc_tools = {}
        self._dirty_paths: set[str] = set()
        self._repo_summary = ""
        self._rag_context = ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, task: Task, log: EventLog) -> RunResult:
        try:
            from langgraph.graph import StateGraph, END
        except ImportError as exc:
            raise ImportError(
                "langgraph not installed. Run: pip install langgraph langchain-core"
            ) from exc

        log.log_task_start(task)
        logger.info("LangGraphAgent starting task %s", task.task_id)

        # Build context (system prompt: repo-map + optional RAG)
        repo_map = RepoMap(task.repo_path)
        token_budget = TokenBudget(total=self._cfg.budget_tokens)
        self._live_repo_map = repo_map
        self._live_token_budget = token_budget
        self._live_task = task
        self._repo_summary = repo_map.build(
            budget=token_budget.default_plan().repo_map,
            query=task.description or None,
        )
        self._rag_context = self._build_rag_context(task.description)
        try:
            from context.rules import load_project_rules
            rules_context = load_project_rules(task.repo_path)
        except Exception:
            rules_context = ""
        try:
            from context.skills import load_skills_prompt
            skills_context = load_skills_prompt(task.repo_path)
        except Exception:
            skills_context = ""
        if skills_context:
            rules_context = "\n\n".join(p for p in (rules_context, skills_context) if p)
        schemas = self._registry.get_schemas()

        # Initialize conversation history
        history = ConversationHistory(max_messages=self._cfg.history_max_messages)
        history.add(LLMMessage(
            role="user",
            content=build_task_prompt(task.description, task.repo_path, task.issue_url),
        ))

        max_steps = task.max_steps

        # ── node: agent ─────────────────────────────────────────────────
        def agent_node(state: AgentState) -> dict:
            step = state["step"] + 1
            compact_tokens = 0
            if self._compactor is not None:
                cr = self._compactor.maybe_compact(state["history"])
                compact_tokens = cr.input_tokens + cr.output_tokens
            self._flush_dirty_indexes()
            messages = self._build_messages(
                state["history"], token_budget, schemas,
                task.repo_path, self._repo_summary, self._rag_context, rules_context,
            )
            response = self._backend.complete(messages, schemas)
            action = response.action
            log.log_action(step=step, action=action, raw_content=response.raw_content)
            logger.info("Step %d: %r", step, action)

            updates: dict = {
                "step": step,
                "total_tokens": state["total_tokens"] + response.total_tokens + compact_tokens,
                "last_action": action,
                "retry_agent": False,
            }

            need_git = action.action_type == ActionType.FINISH and (
                self._cfg.require_edit_before_finish
                or self._cfg.require_test_before_finish
            )
            git_clean = need_git and is_clean_git_repo(task.repo_path)
            decision = self._decision.after_action(
                action, log, git_clean=git_clean,
                noop_finish_rejections=state.get("noop_finish_rejections", 0),
                verified_since_edit=state.get("verified_since_edit", True),
                located=state.get(
                    "located", not self._cfg.require_locate_before_edit
                ),
            )
            if decision.kind in (DecisionKind.ABORT_LOOP, DecisionKind.ABORT_NOOP_FINISH):
                log.log_task_failed(steps=step, reason=decision.reason)
                updates["status"] = "gave_up"
                updates["summary"] = decision.reason
                return updates
            if decision.kind in (DecisionKind.REJECT_FINISH, DecisionKind.REJECT_ACTION):
                history = state["history"]
                if decision.kind == DecisionKind.REJECT_FINISH:
                    history.add(LLMMessage(role="assistant", content=action.message or "(finish)"))
                else:
                    history.add(LLMMessage(role="assistant", content=self._fmt_action(action)))
                history.add(LLMMessage(role="user", content=decision.prompt))
                log.log_reflection(step=step, reason=decision.reason, prompt=decision.prompt)
                updates["history"] = history
                if decision.reason == "noop_finish":
                    updates["noop_finish_rejections"] = (
                        state.get("noop_finish_rejections", 0) + 1
                    )
                updates["retry_agent"] = True
                return updates

            if action.action_type == ActionType.FINISH:
                summary = action.message or "Task complete."
                log.log_task_complete(steps=step, summary=summary)
                updates["status"] = "success"
                updates["summary"] = summary
            elif action.action_type == ActionType.GIVE_UP:
                reason = action.message or "Agent gave up."
                log.log_task_failed(steps=step, reason=reason)
                updates["status"] = "gave_up"
                updates["summary"] = reason
            return updates

        # ── node: tools ─────────────────────────────────────────────────
        def tool_node(state: AgentState) -> dict:
            action = state["last_action"]
            tc = action.tool_call
            history = state["history"]

            result = self._execute_tool(tc.name, tc.params)
            observation = result.to_observation(tc.name)
            log.log_observation(step=state["step"], observation=observation)
            record_edit(
                self._dirty_paths, tc.name, tc.params,
                succeeded=observation.is_success(),
            )

            history.add(LLMMessage(role="assistant", content=self._fmt_action(action)))
            history.add(LLMMessage(role="user", content=self._fmt_obs(observation)))

            swe = self._decision.next_steps_without_edit(
                tc.name, state.get("steps_without_edit", 0)
            )
            verified = self._decision.next_verified_since_edit(
                tc.name, observation, state.get("verified_since_edit", True)
            )
            located = self._decision.next_located(
                tc.name, observation,
                state.get("located", not self._cfg.require_locate_before_edit),
            )

            reflect = self._decision.after_observation(tc.name, observation, swe)
            if reflect.kind in (DecisionKind.REFLECT_TEST_FAILED, DecisionKind.REFLECT_NO_EDIT):
                log.log_reflection(step=state["step"], reason=reflect.reason, prompt=reflect.prompt)
                history.add(LLMMessage(role="user", content=reflect.prompt))
                if reflect.kind == DecisionKind.REFLECT_NO_EDIT:
                    swe = 0

            return {
                "steps_without_edit": swe,
                "verified_since_edit": verified,
                "located": located,
                "history": history,
            }

        # ── routing ──────────────────────────────────────────────────────
        def route_after_agent(state: AgentState) -> str:
            if state.get("status", "running") != "running":
                return "end"
            if state["step"] >= max_steps:
                return "end"
            if state.get("retry_agent"):
                return "retry"
            action = state["last_action"]
            if action.action_type == ActionType.TOOL_CALL and action.tool_call:
                return "tools"
            return "end"

        # ── compile graph ────────────────────────────────────────────────
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent", route_after_agent,
            {"tools": "tools", "retry": "agent", "end": END},
        )
        graph.add_edge("tools", "agent")
        app = graph.compile()

        initial: AgentState = {
            "history": history,
            "step": 0,
            "total_tokens": 0,
            "steps_without_edit": 0,
            "noop_finish_rejections": 0,
            "verified_since_edit": True,
            "located": not self._cfg.require_locate_before_edit,
            "status": "running",
            "summary": "",
        }

        try:
            final = app.invoke(initial, config={"recursion_limit": max_steps * 2 + 5})
        except Exception as exc:
            logger.error("LangGraph run failed: %s", exc)
            log.log_task_failed(steps=0, reason=f"LangGraph error: {exc}")
            return RunResult(
                task_id=task.task_id, status=RunStatus.FAILED,
                summary=f"LangGraph run failed: {exc}", steps_taken=0, error=str(exc),
            )

        return self._to_run_result(task, final)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, params: dict) -> ToolResult:
        """Prefer LangChain tool (.invoke); fall back to registry when not available."""
        lc_tool = self._lc_tools.get(name)
        if lc_tool is not None:
            try:
                return lc_tool.invoke(params)
            except Exception as exc:
                logger.warning("LangChain tool '%s' invoke failed (%s); falling back", name, exc)
        return self._registry.execute_tool(name, params)

    def _build_rag_context(self, query: str) -> str:
        retriever = self._cfg.retriever
        if retriever is None:
            return ""
        try:
            if getattr(retriever, "chunk_count", 0) == 0 and hasattr(retriever, "build"):
                retriever.build()
            return retriever.retrieve(query, k=self._cfg.rag_top_k)
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return ""

    def _flush_dirty_indexes(self) -> None:
        if not self._dirty_paths:
            return
        retriever = getattr(self._cfg, "retriever", None)
        if retriever is not None and hasattr(retriever, "build"):
            try:
                retriever.build()
            except Exception as exc:
                logger.warning("RAG rebuild after edits failed: %s", exc)
        tool = self._registry.get_tool("find_symbol")
        index = getattr(tool, "_index", None) if tool is not None else None
        if index is not None and hasattr(index, "build"):
            try:
                index.build()
            except Exception as exc:
                logger.warning("Symbol index rebuild after edits failed: %s", exc)
        task = getattr(self, "_live_task", None)
        repo_map = getattr(self, "_live_repo_map", None)
        token_budget = getattr(self, "_live_token_budget", None)
        if repo_map is not None and token_budget is not None:
            self._repo_summary = repo_map.build(
                budget=token_budget.default_plan().repo_map,
                query=(task.description if task is not None else None),
            )
        if task is not None:
            self._rag_context = self._build_rag_context(task.description)
        logger.info("LangGraph refreshed indexes after edits to %s", sorted(self._dirty_paths))
        self._dirty_paths.clear()

    def _build_messages(
        self, history, token_budget, schemas, repo_path, repo_summary, rag_context,
        rules_context: str = "",
    ) -> list[LLMMessage]:
        system_content = build_system_prompt(
            repo_path=repo_path,
            tools=schemas,
            repo_summary=repo_summary,
            retrieved_context=rag_context or None,
            rules_context=rules_context or None,
            plan_context=(self._plan.render_for_prompt() or None),
        )
        trimmed = token_budget.trim_history(
            history.to_dicts(), token_budget.default_plan().history
        )
        messages = [LLMMessage(role="system", content=system_content)]
        for d in trimmed:
            messages.append(LLMMessage(role=d["role"], content=d["content"]))
        return messages

    def _fmt_action(self, action: Action) -> str:
        import json
        parts = [f"Thought: {action.thought}"]
        if action.tool_call:
            parts.append(f"Action: {action.tool_call.name}")
            parts.append(f"Params: {json.dumps(action.tool_call.params, ensure_ascii=False)}")
        elif action.message:
            parts.append(f"Message: {action.message}")
        return "\n".join(parts)

    def _fmt_obs(self, observation) -> str:
        status = "SUCCESS" if observation.is_success() else "ERROR"
        lines = [f"[Tool: {observation.tool_name} | {status}]"]
        if observation.output:
            lines.append(observation.output)
        if observation.error and not observation.is_success():
            lines.append(f"Error: {observation.error}")
        return "\n".join(lines)

    def _to_run_result(self, task: Task, final: dict) -> RunResult:
        status_str = final.get("status", "running")
        steps = final.get("step", 0)
        tokens = final.get("total_tokens", 0)

        if status_str == "success":
            return RunResult(
                task_id=task.task_id, status=RunStatus.SUCCESS,
                summary=final.get("summary", "Task complete."),
                steps_taken=steps, total_tokens=tokens,
                patch=self._git_diff(task.repo_path),
            )
        if status_str == "gave_up":
            return RunResult(
                task_id=task.task_id, status=RunStatus.GAVE_UP,
                summary=final.get("summary", "Agent gave up."),
                steps_taken=steps, total_tokens=tokens,
            )
        # Still "running" but graph ended → max_steps limit reached
        return RunResult(
            task_id=task.task_id, status=RunStatus.MAX_STEPS,
            summary=f"Reached max_steps limit ({task.max_steps})",
            steps_taken=steps, total_tokens=tokens,
        )

    def _git_diff(self, repo_path: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True, text=True, timeout=10, cwd=repo_path,
            )
            diff = proc.stdout.strip()
            return diff or None
        except Exception:
            return None
