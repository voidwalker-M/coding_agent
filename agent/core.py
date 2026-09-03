"""
agent/core.py

ReAct main loop. The brain of the entire agent.

Responsibilities (only these, nothing else):
- Maintain conversation history; assemble messages each step to call the LLM
- Dispatch Actions to ToolRegistry for execution after receiving them
- Write Action + Observation into EventLog
- Apply the decision engine (loop abort, finish gate, reflection nudges)
- Return RunResult

Not responsible for:
- Any LLM details (delegated to LLMBackend)
- Any tool implementation (delegated to Tool)
- Context compression (handled by context/compaction.py + context/token_budget.py)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from agent.decision import DecisionEngine, DecisionKind, is_clean_git_repo
from agent.event_log import EventLog
from agent.plan import Plan
from context.history import ConversationHistory
from context.repo_map import RepoMap
from context.token_budget import TokenBudget
from context.workspace_sync import note_for_prompt, record_edit
from agent.prompt import (
    build_system_prompt,
    build_task_prompt,
)
from agent.task import (
    Action, ActionType, Observation, RunResult, RunStatus, Task,
)
from llm.base import LLMBackend, LLMMessage, LLMToolSchema
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# Sentinel distinguishing "argument not supplied" from an explicit None.
_UNSET: object = object()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Agent runtime configuration, loaded from config/default.yaml and passed in."""
    max_steps: int = 40
    reflection_no_edit_steps: int = 6   # trigger Reflection after N consecutive steps with no file writes
    loop_detection_window: int = 3       # declare a dead loop when N consecutive actions are identical
    test_tool_names: tuple[str, ...] = ("test", "pytest")  # tool names that trigger Reflection
    budget_tokens: int = 80_000            # total token budget
    history_max_messages: int = 40         # maximum number of history messages to keep
    llm_max_retries: int = 3               # maximum retries on LLM call failure
    llm_retry_delay: float = 2.0           # retry interval in seconds (exponential back-off)
    stream: bool = False                   # whether to enable streaming output
    stream_callback: object = None         # StreamCallback for the final answer stream
    thought_callback: object = None        # StreamCallback for reasoning-process stream (reasoning models only)
    confirm_dangerous: bool = False        # whether to require user confirmation for dangerous commands
    confirm_callback: object = None        # ConfirmCallback; None = skip confirmation
    retriever: object = None               # RagRetriever; None = RAG retrieval disabled
    rag_top_k: int = 5                     # number of chunks injected per RAG retrieval
    long_term_memory: object = None        # LongTermMemory; None = long-term memory disabled
    short_term_memory: object = None       # ShortTermMemory; None = auto-create per run when memory on
    memory_top_k: int = 4                  # number of long-term memories recalled per task
    memory_window_queries: int = 10        # STM conversation window (n user queries)
    capture_episodes: bool = True          # capture an episodic memory at end of run (when memory on)
    compaction_enabled: bool = False       # LLM summarization when history exceeds trigger
    compaction_trigger_tokens: int = 24_000
    compaction_keep_recent: int = 8        # recent messages kept verbatim after compaction
    compaction_min_messages: int = 4       # minimum middle turns before compacting
    # When True, a FINISH that left the working tree unchanged (clean git repo) is
    # rejected up to a few times, pushing the model to actually apply its fix
    # instead of just narrating it. For code-fix tasks on a git repo (eval,
    # github-issue); OFF by default so chat / no-change tasks finish normally.
    require_edit_before_finish: bool = False
    # When True, a FINISH on a dirty tree is rejected until a test/pytest tool
    # has succeeded since the last edit. SWE-bench only; chat stays off so
    # "explain this" turns can finish without running a suite.
    require_test_before_finish: bool = False
    # When True, refuse file_write/file_edit until a successful search_text or
    # find_symbol this run. Stops "edit the nearby file" on SWE-bench.
    require_locate_before_edit: bool = False
    max_noop_finish_rejections: int = 3


def _make_history_compactor(backend: LLMBackend, cfg: AgentConfig):
    """Build HistoryCompactor from AgentConfig, or None when disabled."""
    if not cfg.compaction_enabled:
        return None
    from context.compaction import CompactionSettings, HistoryCompactor
    return HistoryCompactor(
        backend,
        CompactionSettings(
            enabled=True,
            trigger_tokens=cfg.compaction_trigger_tokens,
            keep_recent_messages=cfg.compaction_keep_recent,
            min_messages_to_compact=cfg.compaction_min_messages,
        ),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    ReAct main loop implementation.

    Usage:
        agent = Agent(backend, registry, config)
        result = agent.run(task, log)
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
        # Paths mutated this session; flushed into repo-map / RAG / symbols
        # before the next LLM call so the prompt does not describe a stale tree.
        self._dirty_paths: set[str] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        task: Task,
        log: EventLog,
        *,
        checkpoint_dir: str | None = None,
        resume_from: str | None = None,
    ) -> RunResult:
        """
        Execute one complete agent run.

        Args:
            task: task description
            log:  an initialized EventLog (created and passed in by the caller)
            checkpoint_dir: when set, write a resumable checkpoint after each step
            resume_from: path to a checkpoint JSON to continue a prior run

        Returns:
            RunResult containing the final status and statistics
        """
        self._current_repo_path = task.repo_path
        self._current_task_desc = task.description
        # Isolate repo_map cache per repo_path; rebuild automatically when the repo changes
        cache_key = task.repo_path
        if getattr(self, "_repo_map_cache_key", None) != cache_key:
            if hasattr(self, "_repo_map_cache"):
                del self._repo_map_cache
            if hasattr(self, "_rag_context_cache"):
                del self._rag_context_cache
            if hasattr(self, "_memory_context_cache"):
                del self._memory_context_cache
            if hasattr(self, "_rules_context_cache"):
                del self._rules_context_cache
            self._repo_map_cache_key = cache_key
            self._dirty_paths = set()
        # New task description → recall fresh memories (the cache is keyed on the
        # repo above, but the recall query is the task; clear it every run).
        if getattr(self, "_memory_query", None) != task.description:
            if hasattr(self, "_memory_context_cache"):
                del self._memory_context_cache
            self._memory_query = task.description
        if getattr(self, "_memory_query", None) != task.description:
            if hasattr(self, "_memory_context_cache"):
                del self._memory_context_cache
            self._memory_query = task.description

        resumed_from_step = 0
        ttft_samples: list[float] = []
        verified_since_edit = True
        located = not self._cfg.require_locate_before_edit

        # ── Resume from checkpoint when requested ─────────────────────
        if resume_from:
            from agent.checkpoint import load_checkpoint, short_term_from_dict

            cp = load_checkpoint(resume_from)
            task = Task(**cp.task)
            self._current_repo_path = task.repo_path
            self._current_task_desc = task.description
            resumed_from_step = cp.step
            history = ConversationHistory.from_dicts(
                cp.history, max_messages=self._cfg.history_max_messages)
            total_tokens = cp.total_tokens
            llm_time_total = cp.llm_time_total
            tool_time_total = cp.tool_time_total
            steps_without_edit = cp.steps_without_edit
            noop_finish_rejections = cp.noop_finish_rejections
            self._short_term = short_term_from_dict(cp.short_term)
            if self._short_term is not None and history.has_evict_callback() is False:
                history.set_evict_callback(self._short_term.make_evict_callback())
            logger.info("Resuming task %s from step %d", task.task_id, resumed_from_step)
        else:
            log.log_task_start(task)
            logger.info("Agent starting task %s", task.task_id)
            total_tokens = 0
            llm_time_total = 0.0
            tool_time_total = 0.0
            steps_without_edit = 0
            noop_finish_rejections = 0

            # Short-term memory: a conversation window of n queries. Caller-owned
            # (chat session) is never wiped; otherwise a fresh window per run.
            if self._cfg.short_term_memory is not None:
                self._short_term = self._cfg.short_term_memory
            elif self._cfg.long_term_memory is not None:
                from context.memory import ShortTermMemory
                ltm = self._cfg.long_term_memory
                self._short_term = ShortTermMemory(
                    window_queries=self._cfg.memory_window_queries,
                    on_overflow=getattr(ltm, "ingest_overflow", None),
                )
            else:
                self._short_term = None
            if self._short_term is not None and hasattr(self._short_term, "begin_query"):
                self._short_term.begin_query(task.description)

            # Initialize context managers.
            # If the caller (ChatSession) injected a shared history, reuse it;
            # otherwise create a fresh one (single-run mode).
            if hasattr(self, "_pending_history") and self._pending_history is not None:
                history = self._pending_history
                # The caller built this history, so attach the fold hook here —
                # otherwise trimmed turns would be lost in chat mode.
                if (self._short_term is not None
                        and hasattr(history, "set_evict_callback")
                        and not history.has_evict_callback()):
                    history.set_evict_callback(self._short_term.make_evict_callback())
            else:
                on_evict = self._short_term.make_evict_callback() if self._short_term is not None else None
                history = ConversationHistory(
                    max_messages=self._cfg.history_max_messages, on_evict=on_evict)
                # Single-run mode: add the task description as the first user message
                from agent.prompt import build_task_prompt
                history.add(LLMMessage(
                    role="user",
                    content=build_task_prompt(task.description, task.repo_path, task.issue_url),
                ))

        token_budget = TokenBudget(total=self._cfg.budget_tokens)
        repo_map = RepoMap(task.repo_path)
        start_step = resumed_from_step + 1 if resume_from else 1

        def _persist_checkpoint(step_done: int, status: str = "running") -> str | None:
            if not checkpoint_dir:
                return None
            from agent.checkpoint import RunCheckpoint, save_checkpoint, short_term_to_dict
            cp = RunCheckpoint.from_task(
                task,
                step=step_done,
                total_tokens=total_tokens,
                llm_time_total=llm_time_total,
                tool_time_total=tool_time_total,
                steps_without_edit=steps_without_edit,
                noop_finish_rejections=noop_finish_rejections,
                history=history.to_dicts(),
                short_term=short_term_to_dict(self._short_term),
                log_path=str(log.path),
                status=status,
            )
            path = save_checkpoint(cp, checkpoint_dir)
            log.log_checkpoint(step=step_done, checkpoint_path=str(path), status=status)
            return str(path)

        def _latency_fields() -> dict:
            from agent.event_log import _percentile
            if not ttft_samples:
                return {}
            ordered = sorted(ttft_samples)
            return {
                "avg_ttft_ms": round(sum(ttft_samples) / len(ttft_samples), 2),
                "p95_ttft_ms": round(_percentile(ordered, 95), 2),
            }

        for step in range(start_step, task.max_steps + 1):
            logger.debug("Step %d/%d", step, task.max_steps)

            # ── 1. Assemble messages and call the LLM ──────────────────
            total_tokens += self._maybe_compact_history(history)
            messages = self._build_messages(history, token_budget, repo_map)
            tools = self._registry.get_schemas()

            t_llm = time.time()
            try:
                response, step_ttft_ms = self._call_with_retry(messages, tools)
            except Exception as exc:
                # Count the failed attempt's time too — retries are real latency.
                llm_time_total += time.time() - t_llm
                logger.error("LLM call failed at step %d after retries: %s", step, exc)
                log.log_task_failed(steps=step, reason=f"LLM error: {exc}")
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.FAILED,
                    summary=f"LLM call failed: {exc}",
                    steps_taken=step,
                    total_tokens=total_tokens,
                    error=str(exc),
                    llm_time=llm_time_total,
                    tool_time=tool_time_total,
                )
            llm_elapsed = time.time() - t_llm
            llm_time_total += llm_elapsed
            e2e_ms = llm_elapsed * 1000
            if step_ttft_ms is not None:
                ttft_samples.append(step_ttft_ms)

            total_tokens += response.total_tokens
            action = response.action

            # ── 2. Write Action event ───────────────────────────────────
            log.log_action(
                step=step, action=action, raw_content=response.raw_content,
                duration_ms=e2e_ms, ttft_ms=step_ttft_ms, e2e_ms=e2e_ms,
            )
            logger.info("Step %d: %r (llm %.2fs)", step, action, llm_elapsed)

            # ── 3. Decision engine: loop / premature finish ────────────
            need_git = action.action_type == ActionType.FINISH and (
                self._cfg.require_edit_before_finish
                or self._cfg.require_test_before_finish
            )
            decision = self._decision.after_action(
                action, log,
                git_clean=need_git and is_clean_git_repo(task.repo_path),
                noop_finish_rejections=noop_finish_rejections,
                verified_since_edit=verified_since_edit,
                located=located,
            )
            if decision.kind == DecisionKind.ABORT_LOOP:
                logger.warning(decision.reason)
                log.log_task_failed(steps=step, reason=decision.reason)
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.GAVE_UP,
                    summary=decision.reason,
                    steps_taken=step,
                    total_tokens=total_tokens,
                    llm_time=llm_time_total,
                    tool_time=tool_time_total,
                )
            if decision.kind == DecisionKind.ABORT_NOOP_FINISH:
                logger.warning(decision.reason)
                log.log_task_failed(steps=step, reason=decision.reason)
                self._capture_episode(task, outcome="failure", summary=decision.reason)
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.GAVE_UP,
                    summary=decision.reason,
                    steps_taken=step,
                    total_tokens=total_tokens,
                    llm_time=llm_time_total,
                    tool_time=tool_time_total,
                )
            if decision.kind == DecisionKind.REJECT_ACTION:
                log.log_reflection(step=step, reason=decision.reason, prompt=decision.prompt)
                history.add(LLMMessage(
                    role="assistant",
                    content=self._format_action_for_history(action),
                ))
                history.add(LLMMessage(role="user", content=decision.prompt))
                logger.info("Rejected action at step %d (%s)", step, decision.reason)
                continue

            # ── 4. Terminal actions ─────────────────────────────────────
            if action.action_type == ActionType.FINISH:
                patch = self._get_git_diff(task.repo_path)
                if decision.kind == DecisionKind.REJECT_FINISH:
                    if decision.reason == "noop_finish":
                        noop_finish_rejections += 1
                    log.log_reflection(step=step, reason=decision.reason, prompt=decision.prompt)
                    history.add(LLMMessage(role="assistant", content=action.message or "(finish)"))
                    history.add(LLMMessage(role="user", content=decision.prompt))
                    logger.info("Rejected finish at step %d (%s, noop %d/%d)",
                                step, decision.reason, noop_finish_rejections,
                                self._cfg.max_noop_finish_rejections)
                    continue

                summary = action.message or "Task complete."
                log.log_task_complete(steps=step, summary=summary)
                self._capture_episode(task, outcome="success", summary=summary, patch=patch)
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.SUCCESS,
                    summary=summary,
                    steps_taken=step,
                    total_tokens=total_tokens,
                    patch=patch,
                    llm_time=llm_time_total,
                    tool_time=tool_time_total,
                    resumed_from_step=resumed_from_step,
                    **_latency_fields(),
                )

            if action.action_type == ActionType.GIVE_UP:
                reason = action.message or "Agent gave up."
                log.log_task_failed(steps=step, reason=reason)
                # patch omitted on purpose — computed lazily only if memory is on.
                self._capture_episode(task, outcome="failure", summary=reason)
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.GAVE_UP,
                    summary=reason,
                    steps_taken=step,
                    total_tokens=total_tokens,
                    llm_time=llm_time_total,
                    tool_time=tool_time_total,
                )

            # ── 5. Execute tool ─────────────────────────────────────────
            if action.action_type == ActionType.TOOL_CALL and action.tool_call:
                tc = action.tool_call
                t_tool = time.time()
                result = self._registry.execute_tool(tc.name, tc.params)
                tool_elapsed = time.time() - t_tool
                tool_time_total += tool_elapsed
                observation = result.to_observation(tc.name)

                # Track whether a file write operation occurred. `undo` counts:
                # it changes the code on disk, so it is real progress rather than
                # the idle exploration the no-edit reflection is meant to catch.
                steps_without_edit = self._decision.next_steps_without_edit(
                    tc.name, steps_without_edit
                )
                verified_since_edit = self._decision.next_verified_since_edit(
                    tc.name, observation, verified_since_edit
                )
                located = self._decision.next_located(tc.name, observation, located)

                # Short-term memory: remember which files the agent has examined
                # so it doesn't re-open them after the window trims (also feeds the
                # efficiency guidance in the system prompt).
                if self._short_term is not None and tc.name in (
                    "file_read", "file_view", "file_write", "file_edit", "edit", "undo"
                ):
                    fp = tc.params.get("path") or tc.params.get("file") or tc.params.get("filename")
                    if fp:
                        self._short_term.note_file(str(fp))

                record_edit(
                    self._dirty_paths, tc.name, tc.params,
                    succeeded=observation.is_success(),
                )

                log.log_observation(step=step, observation=observation,
                                    duration_ms=tool_elapsed * 1000)

                # Add action and observation to the conversation history
                history.add(LLMMessage(
                    role="assistant",
                    content=self._format_action_for_history(action),
                ))
                history.add(LLMMessage(
                    role="user",
                    content=self._format_observation_for_history(observation),
                ))

                # ── 6. Reflection trigger check ─────────────────────────
                reflect = self._decision.after_observation(
                    tc.name, observation, steps_without_edit
                )
                if reflect.kind == DecisionKind.REFLECT_TEST_FAILED:
                    log.log_reflection(step=step, reason=reflect.reason, prompt=reflect.prompt)
                    history.add(LLMMessage(role="user", content=reflect.prompt))
                    logger.debug("Reflection triggered: test_failed at step %d", step)
                elif reflect.kind == DecisionKind.REFLECT_NO_EDIT:
                    log.log_reflection(step=step, reason=reflect.reason, prompt=reflect.prompt)
                    history.add(LLMMessage(role="user", content=reflect.prompt))
                    steps_without_edit = 0  # reset counter to avoid triggering every step
                    logger.debug("Reflection triggered: no_edit at step %d", step)

            elif action.action_type == ActionType.REFLECTION:
                # LLM-initiated reflection (reserved; MockBackend does not produce this)
                history.add(LLMMessage(
                    role="assistant",
                    content=action.thought,
                ))

            _persist_checkpoint(step)

        # ── 7. Exceeded max steps ───────────────────────────────────────
        ckpt = _persist_checkpoint(task.max_steps, status="interrupted")
        reason = f"Reached max_steps limit ({task.max_steps})"
        log.log_task_failed(steps=task.max_steps, reason=reason)
        return RunResult(
            task_id=task.task_id,
            status=RunStatus.MAX_STEPS,
            summary=reason,
            steps_taken=task.max_steps,
            total_tokens=total_tokens,
            llm_time=llm_time_total,
            tool_time=tool_time_total,
            checkpoint_path=ckpt,
            resumed_from_step=resumed_from_step,
            **_latency_fields(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_compact_history(self, history: ConversationHistory) -> int:
        """Run LLM history compaction when enabled. Returns tokens consumed."""
        if self._compactor is None:
            return 0
        result = self._compactor.maybe_compact(history)
        return result.input_tokens + result.output_tokens

    def _flush_dirty_indexes(self) -> None:
        """Rebuild cached repo-map / RAG / symbols after successful file edits.

        Called once per LLM turn. Unchanged files reuse content-hash caches inside
        RagRetriever.build / SymbolIndex.build; we only drop the *prompt* caches
        so the next assemble sees the on-disk tree.
        """
        if not self._dirty_paths:
            return
        note = note_for_prompt(self._dirty_paths)
        if hasattr(self, "_repo_map_cache"):
            del self._repo_map_cache
        if hasattr(self, "_rag_context_cache"):
            del self._rag_context_cache
        retriever = getattr(self._cfg, "retriever", None)
        if retriever is not None and hasattr(retriever, "build"):
            try:
                retriever.build()
            except Exception as exc:
                logger.warning("RAG rebuild after edits failed: %s", exc)
        self._refresh_symbol_index()
        stm = getattr(self, "_short_term", None)
        if note and stm is not None and hasattr(stm, "add_note"):
            stm.add_note(note)
        logger.info("Refreshed indexes after edits to %s", sorted(self._dirty_paths))
        self._dirty_paths.clear()

    def _refresh_symbol_index(self) -> None:
        tool = self._registry.get_tool("find_symbol")
        index = getattr(tool, "_index", None) if tool is not None else None
        if index is None or not hasattr(index, "build"):
            return
        try:
            index.build()
        except Exception as exc:
            logger.warning("Symbol index rebuild after edits failed: %s", exc)

    def _build_messages(
        self,
        history: ConversationHistory,
        token_budget: TokenBudget,
        repo_map: RepoMap,
    ) -> list[LLMMessage]:
        """Assemble the full message list to send to the LLM, with token trimming."""
        schemas = self._registry.get_schemas()

        self._flush_dirty_indexes()

        # Build repo-map (cached until a file write dirties the workspace).
        # Pass the task description as the query so files relevant to the current
        # task are ranked first.
        if not hasattr(self, "_repo_map_cache"):
            self._repo_map_cache = repo_map.build(
                budget=token_budget.default_plan().repo_map,
                query=getattr(self, "_current_task_desc", "") or None,
            )

        # RAG retrieval (cached until a file write dirties the workspace).
        if not hasattr(self, "_rag_context_cache"):
            self._rag_context_cache = self._build_rag_context()

        if not hasattr(self, "_rules_context_cache"):
            self._rules_context_cache = self._build_rules_context()

        # Recalled memories only (not the full catalog). Rules are a separate cache.
        if not hasattr(self, "_memory_context_cache"):
            self._memory_context_cache = self._build_memory_context()

        # Short-term conversation window is dynamic (grows each query) → not cached.
        working_memory = self._short_term.render() if self._short_term is not None else ""
        plan_context = self._plan.render_for_prompt() if self._plan is not None else ""

        system_content = build_system_prompt(
            repo_path=getattr(self, "_current_repo_path", "."),
            tools=schemas,
            repo_summary=self._repo_map_cache,
            retrieved_context=self._rag_context_cache or None,
            memory_context=self._memory_context_cache or None,
            working_memory=working_memory or None,
            rules_context=getattr(self, "_rules_context_cache", None) or None,
            plan_context=plan_context or None,
        )

        # Trim history
        trimmed_history_dicts = token_budget.trim_history(
            history.to_dicts(),
            token_budget.default_plan().history,
        )

        # Assemble: system + trimmed history
        messages = [LLMMessage(role="system", content=system_content)]
        for d in trimmed_history_dicts:
            messages.append(LLMMessage(role=d["role"], content=d["content"]))
        return messages

    def _build_rag_context(self) -> str:
        """
        Use the RAG retriever to find code chunks relevant to the task description.
        Returns an empty string (silent fallback) if the retriever is not configured,
        fails to build, or returns no results.
        """
        retriever = self._cfg.retriever
        if retriever is None:
            return ""
        try:
            # Rebuild when empty *or* after a write flushed the prompt cache
            # (chunk_count stays > 0; build() itself is content-hash incremental).
            if hasattr(retriever, "build") and (
                getattr(retriever, "chunk_count", 0) == 0
                or not hasattr(self, "_rag_context_cache")
            ):
                # First assemble: lazy-build. After edits: _flush_dirty_indexes
                # already called build(); retrieve() below is enough. If this is
                # the first call, build now.
                if getattr(retriever, "chunk_count", 0) == 0:
                    retriever.build()
            query = getattr(self, "_current_task_desc", "") or ""
            return retriever.retrieve(query, k=self._cfg.rag_top_k)
        except Exception as exc:
            logger.warning("RAG retrieval failed, continuing without it: %s", exc)
            return ""

    def _build_rules_context(self) -> str:
        """Always-on project rules plus a Skills catalog (named SOP playbooks)."""
        repo = getattr(self, "_current_repo_path", ".") or "."
        parts: list[str] = []
        try:
            from context.rules import load_project_rules
            parts.append(load_project_rules(repo))
        except Exception as exc:
            logger.warning("Project rules load failed: %s", exc)
        try:
            from context.skills import load_skills_prompt
            parts.append(load_skills_prompt(repo))
        except Exception as exc:
            logger.warning("Skills load failed: %s", exc)
        return "\n\n".join(p for p in parts if p)

    def _build_memory_context(self) -> str:
        """Recalled approved memories for this task — not the full catalog."""
        ltm = self._cfg.long_term_memory
        if ltm is None:
            return ""
        try:
            query = getattr(self, "_current_task_desc", "") or ""
            return ltm.recall(query, k=self._cfg.memory_top_k, for_prompt=True) or ""
        except Exception as exc:
            logger.warning("Memory recall failed, continuing without it: %s", exc)
            return ""

    def _capture_episode(self, task: Task, *, outcome: str, summary: str,
                         patch: "str | None | object" = _UNSET) -> None:
        """Persist this run as an episodic memory (best-effort; never raises).

        `patch` is optional: when omitted it is computed here, *after* the
        memory-disabled guard, so a run with memory off never pays for the extra
        `git diff` subprocess.
        """
        ltm = self._cfg.long_term_memory
        if ltm is None or not self._cfg.capture_episodes:
            return
        if patch is _UNSET:
            patch = self._get_git_diff(task.repo_path)
        try:
            files = self._changed_files(patch)
            ltm.record_episode(
                task.description, outcome=outcome, files=files,
                summary=summary, source=task.task_id,
            )
        except Exception as exc:
            logger.warning("Episodic memory capture failed: %s", exc)

    @staticmethod
    def _changed_files(patch: str | None) -> list[str]:
        """Extract changed file paths from a unified diff ('+++ b/<path>' lines)."""
        if not patch:
            return []
        files: list[str] = []
        for line in patch.splitlines():
            if line.startswith("+++ b/"):
                files.append(line[len("+++ b/"):].strip())
            elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
                files.append(line[len("+++ "):].strip())
        # de-dup, preserve order
        seen: set[str] = set()
        return [f for f in files if not (f in seen or seen.add(f))]

    def _format_action_for_history(self, action: Action) -> str:
        """Format an Action as an assistant message for the conversation history."""
        parts = [f"Thought: {action.thought}"]
        if action.tool_call:
            parts.append(f"Action: {action.tool_call.name}")
            parts.append(f"Params: {json.dumps(action.tool_call.params, ensure_ascii=False)}")
        elif action.message:
            parts.append(f"Message: {action.message}")
        return "\n".join(parts)

    def _format_observation_for_history(self, observation: Observation) -> str:
        """Format an Observation as a user message for the conversation history."""
        status = "SUCCESS" if observation.is_success() else "ERROR"
        lines = [f"[Tool: {observation.tool_name} | {status}]"]
        if observation.output:
            lines.append(observation.output)
        if observation.error and not observation.is_success():
            lines.append(f"Error: {observation.error}")
        return "\n".join(lines)

    def _is_looping(self, log: EventLog) -> bool:
        """Dead-loop check; delegated to DecisionEngine (kept for tests / replay)."""
        return self._decision.is_looping(log)

    def _call_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSchema],
    ) -> tuple:
        """
        LLM call with exponential back-off retry.
        Returns (LLMResponse, ttft_ms) where ttft_ms is time-to-first-token in
        milliseconds when streaming, else None for non-streaming calls.
        """
        import time as _time

        last_exc: Exception | None = None
        delay = self._cfg.llm_retry_delay

        for attempt in range(1, self._cfg.llm_max_retries + 1):
            try:
                t0 = _time.time()
                first_token_at: list[float | None] = [None]

                def _wrap_cb(cb):
                    if cb is None:
                        return None
                    def wrapped(text: str) -> None:
                        if first_token_at[0] is None:
                            first_token_at[0] = _time.time()
                        cb(text)
                    return wrapped

                if self._cfg.stream:
                    cb = self._cfg.stream_callback
                    thought_cb = self._cfg.thought_callback
                    if hasattr(self._backend, "stream"):
                        response = self._backend.stream(
                            messages, tools,
                            on_text=_wrap_cb(cb),
                            on_thought=_wrap_cb(thought_cb),
                        )
                        ttft_ms = None
                        if first_token_at[0] is not None:
                            ttft_ms = (first_token_at[0] - t0) * 1000
                        return response, ttft_ms
                response = self._backend.complete(messages, tools)
                return response, None
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if any(kw in exc_str for kw in (
                    "401", "403", "invalid api key", "authentication",
                    "400", "bad request",
                    "ceiling", "budget exceeded",   # cost ceiling — stop, don't retry
                )):
                    raise
                if attempt < self._cfg.llm_max_retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, self._cfg.llm_max_retries, exc, delay,
                    )
                    _time.sleep(delay)
                    delay *= 2

        raise last_exc  # type: ignore[misc]

    def _get_git_diff(self, repo_path: str) -> str | None:
        """Fetch `git diff HEAD` as a patch; silently return None on failure."""
        import subprocess
        try:
            proc = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True, text=True, timeout=10, cwd=repo_path,
            )
            diff = proc.stdout.strip()
            return diff if diff else None
        except Exception:
            return None

    def _is_clean_git_repo(self, repo_path: str) -> bool:
        """Kept for tests / callers; delegates to the decision-engine helper."""
        return is_clean_git_repo(repo_path)
