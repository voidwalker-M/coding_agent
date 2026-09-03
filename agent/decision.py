"""
agent/decision.py

Decision engine for the ReAct / LangGraph loops.

The heuristics that used to live inline in `agent/core.py` (dead-loop abort,
no-op finish rejection, test-failed / no-edit reflection) are the agent's
control policy: they decide whether the latest model action is progress,
a stall, or a premature "I'm done". Lifting them here makes the policy:

- testable without spinning a full Agent
- shared by the native loop and the LangGraph engine
- the place to extend closed-loop rules (e.g. require tests before finish)

The engine is mostly pure: counters (`steps_without_edit`, noop-finish
rejections) stay with the loop so checkpoints keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent.prompt import (
    locate_before_edit_correction,
    noop_finish_correction,
    reflection_no_edit,
    reflection_test_failed,
    unverified_finish_correction,
)
from agent.task import Action, ActionType, Observation
from context.workspace_sync import EDIT_TOOLS

DEFAULT_MAX_NOOP_FINISH_REJECTIONS = 3
LOCATE_TOOLS = frozenset({"search_text", "find_symbol"})


def is_clean_git_repo(repo_path: str) -> bool:
    """True only for a git work tree with no tracked or untracked changes."""
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
        return not st.stdout.strip()
    except Exception:
        return False


class DecisionKind(str, Enum):
    CONTINUE = "continue"
    ABORT_LOOP = "abort_loop"
    REJECT_FINISH = "reject_finish"
    REJECT_ACTION = "reject_action"
    ABORT_NOOP_FINISH = "abort_noop_finish"
    REFLECT_TEST_FAILED = "reflect_test_failed"
    REFLECT_NO_EDIT = "reflect_no_edit"


@dataclass
class Decision:
    kind: DecisionKind
    reason: str = ""
    prompt: str = ""

    def is_continue(self) -> bool:
        return self.kind == DecisionKind.CONTINUE


class DecisionEngine:
    """Policy object for loop guards, finish gates, and reflection nudges."""

    def __init__(
        self,
        *,
        loop_detection_window: int = 3,
        reflection_no_edit_steps: int = 6,
        test_tool_names: tuple[str, ...] = ("test", "pytest"),
        require_edit_before_finish: bool = False,
        require_test_before_finish: bool = False,
        require_locate_before_edit: bool = False,
        max_noop_finish_rejections: int = DEFAULT_MAX_NOOP_FINISH_REJECTIONS,
    ) -> None:
        self.loop_detection_window = loop_detection_window
        self.reflection_no_edit_steps = reflection_no_edit_steps
        self.test_tool_names = test_tool_names
        self.require_edit_before_finish = require_edit_before_finish
        self.require_test_before_finish = require_test_before_finish
        self.require_locate_before_edit = require_locate_before_edit
        self.max_noop_finish_rejections = max_noop_finish_rejections

    @classmethod
    def from_config(cls, cfg) -> "DecisionEngine":
        return cls(
            loop_detection_window=getattr(cfg, "loop_detection_window", 3),
            reflection_no_edit_steps=getattr(cfg, "reflection_no_edit_steps", 6),
            test_tool_names=tuple(getattr(cfg, "test_tool_names", ("test", "pytest"))),
            require_edit_before_finish=getattr(cfg, "require_edit_before_finish", False),
            require_test_before_finish=getattr(cfg, "require_test_before_finish", False),
            require_locate_before_edit=getattr(cfg, "require_locate_before_edit", False),
            max_noop_finish_rejections=getattr(
                cfg, "max_noop_finish_rejections", DEFAULT_MAX_NOOP_FINISH_REJECTIONS
            ),
        )

    # ------------------------------------------------------------------
    # After the model returns an Action
    # ------------------------------------------------------------------

    def is_looping(self, log) -> bool:
        """True when the last N tool calls are identical (name + params)."""
        n = self.loop_detection_window
        actions = log.get_actions()
        if len(actions) < n:
            return False
        recent = actions[-n:]
        if not all(a.action_type == ActionType.TOOL_CALL for a in recent):
            return False
        if not all(a.tool_call for a in recent):
            return False
        first = recent[0].tool_call
        return all(
            a.tool_call.name == first.name and a.tool_call.params == first.params
            for a in recent[1:]
        )

    def after_action(
        self,
        action: Action,
        log,
        *,
        git_clean: bool = False,
        noop_finish_rejections: int = 0,
        verified_since_edit: bool = True,
        located: bool = True,
    ) -> Decision:
        if self.is_looping(log):
            n = self.loop_detection_window
            return Decision(
                DecisionKind.ABORT_LOOP,
                reason=f"Loop detected: same action repeated {n} times",
            )
        if (
            self.require_locate_before_edit
            and not located
            and action.action_type == ActionType.TOOL_CALL
            and action.tool_call
            and action.tool_call.name in EDIT_TOOLS
        ):
            return Decision(
                DecisionKind.REJECT_ACTION,
                reason="edit_before_locate",
                prompt=locate_before_edit_correction(),
            )
        if action.action_type == ActionType.FINISH:
            if self.require_edit_before_finish and git_clean:
                # Never accept a no-op as success. Keep rejecting (do not abort)
                # so remaining steps can still be used to search and edit —
                # aborting after N nudges cut off two previously-resolved
                # SWE-Lite tasks that had not edited yet.
                return Decision(
                    DecisionKind.REJECT_FINISH,
                    reason="noop_finish",
                    prompt=noop_finish_correction(),
                )
            if (
                self.require_test_before_finish
                and not git_clean
                and not verified_since_edit
            ):
                return Decision(
                    DecisionKind.REJECT_FINISH,
                    reason="unverified_finish",
                    prompt=unverified_finish_correction(),
                )
        return Decision(DecisionKind.CONTINUE)

    # ------------------------------------------------------------------
    # After a tool Observation
    # ------------------------------------------------------------------

    @staticmethod
    def next_steps_without_edit(tool_name: str, steps_without_edit: int) -> int:
        if tool_name in EDIT_TOOLS:
            return 0
        return steps_without_edit + 1

    def next_located(
        self,
        tool_name: str,
        observation: Observation,
        located: bool,
    ) -> bool:
        """True after a successful search_text / find_symbol."""
        if tool_name in LOCATE_TOOLS and observation.is_success():
            return True
        return located

    def next_verified_since_edit(
        self,
        tool_name: str,
        observation: Observation,
        verified_since_edit: bool,
    ) -> bool:
        """False after a successful edit; True after a successful test tool."""
        if tool_name in EDIT_TOOLS and observation.is_success():
            return False
        if tool_name in self.test_tool_names and observation.is_success():
            return True
        return verified_since_edit

    def after_observation(
        self,
        tool_name: str,
        observation: Observation,
        steps_without_edit: int,
    ) -> Decision:
        if tool_name in self.test_tool_names and not observation.is_success():
            return Decision(
                DecisionKind.REFLECT_TEST_FAILED,
                reason="test_failed",
                prompt=reflection_test_failed(),
            )
        if steps_without_edit >= self.reflection_no_edit_steps:
            return Decision(
                DecisionKind.REFLECT_NO_EDIT,
                reason="no_edit",
                prompt=reflection_no_edit(steps_without_edit),
            )
        return Decision(DecisionKind.CONTINUE)
