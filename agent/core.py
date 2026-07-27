"""
agent/core.py

ReAct main loop. The brain of the entire agent.

Responsibilities (only these, nothing else):
- Maintain conversation history; assemble messages each step to call the LLM
- Dispatch Actions to ToolRegistry for execution after receiving them
- Write Action + Observation into EventLog
- Detect three termination / Reflection trigger conditions
- Return RunResult

Not responsible for:
- Any LLM details (delegated to LLMBackend)
- Any tool implementation (delegated to Tool)
- Context compression (handled by the context/ module)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from agent.event_log import EventLog
from context.history import ConversationHistory
from context.repo_map import RepoMap
from context.token_budget import TokenBudget
from agent.prompt import (
    build_system_prompt,
    build_task_prompt,
    reflection_no_edit,
    reflection_test_failed,
)
from agent.task import (
    Action, ActionType, Event, EventType,
    Observation, ObservationStatus, RunResult, RunStatus, Task, ToolCall,
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
    capture_episodes: bool = True          # capture an episodic memory at end of run (when memory on)
    # When True, a FINISH that left the working tree unchanged (clean git repo) is
    # rejected up to a few times, pushing the model to actually apply its fix
    # instead of just narrating it. For code-fix tasks on a git repo (eval,
    # github-issue); OFF by default so chat / no-change tasks finish normally.
    require_edit_before_finish: bool = False


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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, task: Task, log: EventLog) -> RunResult:
        """
        Execute one complete agent run.

        Args:
            task: task description
            log:  an initialized EventLog (created and passed in by the caller)

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
            self._repo_map_cache_key = cache_key
        # New task description → recall fresh memories (the cache is keyed on the
        # repo above, but the recall query is the task; clear it every run).
        if getattr(self, "_memory_query", None) != task.description:
            if hasattr(self, "_memory_context_cache"):
                del self._memory_context_cache
            self._memory_query = task.description
        log.log_task_start(task)
        logger.info("Agent starting task %s", task.task_id)

        # Short-term (working) memory for this run: notes, files examined, and a
        # rolling summary of turns the history window evicts.
        if self._cfg.short_term_memory is not None:
            # Caller-owned (e.g. a chat session spanning rounds) — never wipe it.
            self._short_term = self._cfg.short_term_memory
        elif self._cfg.long_term_memory is not None:
            from context.memory import ShortTermMemory
            self._short_term = ShortTermMemory()   # fresh per run; no leak between tasks
        else:
            self._short_term = None

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

        total_tokens = 0
        llm_time_total = 0.0
        tool_time_total = 0.0
        steps_without_edit = 0
        noop_finish_rejections = 0
        max_noop_finish_rejections = 3

        for step in range(1, task.max_steps + 1):
            logger.debug("Step %d/%d", step, task.max_steps)

            # ── 1. Assemble messages and call the LLM ──────────────────
            messages = self._build_messages(history, token_budget, repo_map)
            tools = self._registry.get_schemas()

            t_llm = time.time()
            try:
                response = self._call_with_retry(messages, tools)
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

            total_tokens += response.total_tokens
            action = response.action

            # ── 2. Write Action event ───────────────────────────────────
            log.log_action(step=step, action=action, raw_content=response.raw_content,
                           duration_ms=llm_elapsed * 1000)
            logger.info("Step %d: %r (llm %.2fs)", step, action, llm_elapsed)

            # ── 3. Detect dead loop (consecutive identical actions) ─────
            if self._is_looping(log):
                reason = f"Loop detected: same action repeated {self._cfg.loop_detection_window} times"
                logger.warning(reason)
                log.log_task_failed(steps=step, reason=reason)
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.GAVE_UP,
                    summary=reason,
                    steps_taken=step,
                    total_tokens=total_tokens,
                    llm_time=llm_time_total,
                    tool_time=tool_time_total,
                )

            # ── 4. Terminal actions ─────────────────────────────────────
            if action.action_type == ActionType.FINISH:
                patch = self._get_git_diff(task.repo_path)
                # Guard: reject a "finish" that changed nothing. Models (notably
                # gpt-oss) sometimes narrate a fix and declare completion without
                # ever calling an edit tool — the message claims a file was
                # modified while `git diff` is empty. For a code task a no-op
                # finish is always wrong, so push back and make the model actually
                # apply its change. Capped, after which we accept and let the
                # verifier judge (preserves generality for genuine no-change tasks).
                if (self._cfg.require_edit_before_finish
                        and noop_finish_rejections < max_noop_finish_rejections
                        and self._is_clean_git_repo(task.repo_path)):
                    noop_finish_rejections += 1
                    correction = (
                        "You called finish, but the working tree has NO changes — you have "
                        "not edited any file yet. Describing the fix is not enough; you must "
                        "APPLY it. Use the file_edit tool to make the change to the target "
                        "file (or file_write to rewrite it), then verify with git diff. Do "
                        "not finish until your change is actually on disk."
                    )
                    log.log_reflection(step=step, reason="noop_finish", prompt=correction)
                    history.add(LLMMessage(role="assistant", content=action.message or "(finish)"))
                    history.add(LLMMessage(role="user", content=correction))
                    logger.info("Rejected no-op finish at step %d (%d/%d)",
                                step, noop_finish_rejections, max_noop_finish_rejections)
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
                if tc.name in ("file_write", "file_edit", "edit", "undo"):
                    steps_without_edit = 0
                else:
                    steps_without_edit += 1

                # Short-term memory: remember which files the agent has examined
                # so it doesn't re-open them after the window trims (also feeds the
                # efficiency guidance in the system prompt).
                if self._short_term is not None and tc.name in (
                    "file_read", "file_view", "file_write", "file_edit", "edit", "undo"
                ):
                    fp = tc.params.get("path") or tc.params.get("file") or tc.params.get("filename")
                    if fp:
                        self._short_term.note_file(str(fp))

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

                # Condition A: test tool failed
                if (
                    tc.name in self._cfg.test_tool_names
                    and not observation.is_success()
                ):
                    reflect_prompt = reflection_test_failed()
                    log.log_reflection(
                        step=step,
                        reason="test_failed",
                        prompt=reflect_prompt,
                    )
                    history.add(LLMMessage(role="user", content=reflect_prompt))
                    logger.debug("Reflection triggered: test_failed at step %d", step)

                # Condition B: N consecutive steps without any file edit
                elif steps_without_edit >= self._cfg.reflection_no_edit_steps:
                    reflect_prompt = reflection_no_edit(steps_without_edit)
                    log.log_reflection(
                        step=step,
                        reason="no_edit",
                        prompt=reflect_prompt,
                    )
                    history.add(LLMMessage(role="user", content=reflect_prompt))
                    steps_without_edit = 0  # reset counter to avoid triggering every step
                    logger.debug("Reflection triggered: no_edit at step %d", step)

            elif action.action_type == ActionType.REFLECTION:
                # LLM-initiated reflection (reserved; MockBackend does not produce this)
                history.add(LLMMessage(
                    role="assistant",
                    content=action.thought,
                ))

        # ── 7. Exceeded max steps ───────────────────────────────────────
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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        history: ConversationHistory,
        token_budget: TokenBudget,
        repo_map: RepoMap,
    ) -> list[LLMMessage]:
        """Assemble the full message list to send to the LLM, with token trimming."""
        schemas = self._registry.get_schemas()

        # Build repo-map (cached: generated once on the first step, reused afterward).
        # Pass the task description as the query so files relevant to the current
        # task are ranked first; this is constant within a task, so caching is safe.
        if not hasattr(self, "_repo_map_cache"):
            self._repo_map_cache = repo_map.build(
                budget=token_budget.default_plan().repo_map,
                query=getattr(self, "_current_task_desc", "") or None,
            )

        # RAG retrieval (cached: retrieved once per task description)
        if not hasattr(self, "_rag_context_cache"):
            self._rag_context_cache = self._build_rag_context()

        # Long-term memory: MEMORY.md index + records recalled for this task
        # (cached per task, like RAG — stable, so prompt-cache friendly).
        if not hasattr(self, "_memory_context_cache"):
            self._memory_context_cache = self._build_memory_context()

        # Short-term working memory is dynamic (changes each step) → not cached.
        working_memory = self._short_term.render() if self._short_term is not None else ""

        system_content = build_system_prompt(
            repo_path=getattr(self, "_current_repo_path", "."),
            tools=schemas,
            repo_summary=self._repo_map_cache,
            retrieved_context=self._rag_context_cache or None,
            memory_context=self._memory_context_cache or None,
            working_memory=working_memory or None,
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
            # Lazy-build the retriever in case the CLI didn't pre-build it
            if getattr(retriever, "chunk_count", 0) == 0 and hasattr(retriever, "build"):
                retriever.build()
            query = getattr(self, "_current_task_desc", "") or ""
            return retriever.retrieve(query, k=self._cfg.rag_top_k)
        except Exception as exc:
            logger.warning("RAG retrieval failed, continuing without it: %s", exc)
            return ""

    def _build_memory_context(self) -> str:
        """Long-term memory block for the system prompt: index + recalled records.

        Returns "" (silent fallback) when memory is disabled/empty or errors.
        """
        ltm = self._cfg.long_term_memory
        if ltm is None:
            return ""
        try:
            query = getattr(self, "_current_task_desc", "") or ""
            parts = []
            recalled = ltm.recall(query, k=self._cfg.memory_top_k)
            if recalled:
                parts.append(recalled)
            index = ltm.index_block()
            if index:
                parts.append(index)
            return "\n\n".join(parts)
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
        """
        Detect a dead loop: the last N actions are all identical.
        Compares (tool_name, params) tuples.
        """
        n = self._cfg.loop_detection_window
        actions = log.get_actions()
        if len(actions) < n:
            return False

        recent = actions[-n:]
        # Only check TOOL_CALL type actions
        if not all(a.action_type == ActionType.TOOL_CALL for a in recent):
            return False
        if not all(a.tool_call for a in recent):
            return False

        first = recent[0].tool_call
        return all(
            a.tool_call.name == first.name and a.tool_call.params == first.params
            for a in recent[1:]
        )

    def _call_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSchema],
    ):
        """
        LLM call with exponential back-off retry.
        When stream=True, uses backend.stream(); otherwise uses complete().
        Not retried: authentication failures (401/403), bad requests (400).
        """
        import time as _time

        last_exc: Exception | None = None
        delay = self._cfg.llm_retry_delay

        for attempt in range(1, self._cfg.llm_max_retries + 1):
            try:
                if self._cfg.stream:
                    cb = self._cfg.stream_callback
                    thought_cb = self._cfg.thought_callback
                    if hasattr(self._backend, "stream"):
                        return self._backend.stream(
                            messages, tools,
                            on_text=cb,
                            on_thought=thought_cb,
                        )
                return self._backend.complete(messages, tools)
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
        """True only when repo_path is definitively a git work tree with NO changes
        (tracked or untracked). False if it has changes OR is not a git repo /
        can't be determined — so callers never mistake "not a git repo" for
        "clean" and wrongly block a legitimate finish."""
        import subprocess
        try:
            chk = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, timeout=10, cwd=repo_path,
            )
            if chk.returncode != 0 or chk.stdout.strip() != "true":
                return False
            st = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, cwd=repo_path,
            )
            if st.returncode != 0:
                return False
            return not st.stdout.strip()   # clean iff porcelain output is empty
        except Exception:
            return False
