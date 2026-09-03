"""Decision engine policy: loop abort, noop-finish, reflection nudges."""

from agent.core import AgentConfig
from agent.decision import DecisionEngine, DecisionKind, EDIT_TOOLS
from agent.event_log import EventLog
from agent.task import Action, ActionType, Observation, ObservationStatus, Task, ToolCall


def _log_with_actions(tmp_path, actions: list[Action]) -> EventLog:
    task = Task(description="x", repo_path=str(tmp_path), task_id="dec")
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    log.log_task_start(task)
    for i, action in enumerate(actions, start=1):
        log.log_action(step=i, action=action)
    return log


def test_loop_detection_requires_identical_tool_calls(tmp_path):
    engine = DecisionEngine(loop_detection_window=3)
    same = Action(ActionType.TOOL_CALL, "t", tool_call=ToolCall("shell", {"cmd": "echo"}))
    log = _log_with_actions(tmp_path, [same, same, same])
    assert engine.is_looping(log) is True
    log.close()

    mixed = [
        Action(ActionType.TOOL_CALL, "t", tool_call=ToolCall("shell", {"cmd": "a"})),
        Action(ActionType.TOOL_CALL, "t", tool_call=ToolCall("shell", {"cmd": "b"})),
        Action(ActionType.TOOL_CALL, "t", tool_call=ToolCall("shell", {"cmd": "a"})),
    ]
    log2 = _log_with_actions(tmp_path, mixed)
    assert engine.is_looping(log2) is False
    log2.close()


def test_reject_noop_finish_when_git_clean(tmp_path):
    engine = DecisionEngine.from_config(AgentConfig(require_edit_before_finish=True))
    task = Task(description="x", repo_path=str(tmp_path), task_id="fin")
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    log.log_task_start(task)
    action = Action(ActionType.FINISH, "done", message="fixed")
    log.log_action(step=1, action=action)
    decision = engine.after_action(action, log, git_clean=True, noop_finish_rejections=0)
    assert decision.kind == DecisionKind.REJECT_FINISH
    assert "not edited" in decision.prompt.lower() or "NO changes" in decision.prompt
    log.close()


def test_reflection_on_test_failure_and_no_edit():
    engine = DecisionEngine(reflection_no_edit_steps=3)
    fail = Observation(ObservationStatus.ERROR, "boom", "pytest")
    d = engine.after_observation("pytest", fail, steps_without_edit=1)
    assert d.kind == DecisionKind.REFLECT_TEST_FAILED

    ok = Observation(ObservationStatus.SUCCESS, "ok", "file_read")
    d2 = engine.after_observation("file_read", ok, steps_without_edit=3)
    assert d2.kind == DecisionKind.REFLECT_NO_EDIT
    d3 = engine.after_observation("file_read", ok, steps_without_edit=1)
    assert d3.kind == DecisionKind.CONTINUE


def test_edit_tools_reset_counter():
    engine = DecisionEngine()
    assert engine.next_steps_without_edit("file_write", 9) == 0
    assert engine.next_steps_without_edit("file_read", 2) == 3
    assert "undo" in EDIT_TOOLS


def test_exhausted_noop_finish_still_rejected_not_success(tmp_path):
    """After the nudge budget, finish on a clean tree is still REJECT — never
    CONTINUE (false success) and never ABORT (that wasted remaining steps)."""
    engine = DecisionEngine.from_config(
        AgentConfig(require_edit_before_finish=True, max_noop_finish_rejections=3)
    )
    task = Task(description="x", repo_path=str(tmp_path), task_id="fin")
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    log.log_task_start(task)
    action = Action(ActionType.FINISH, "done", message="fixed")
    log.log_action(step=1, action=action)
    decision = engine.after_action(
        action, log, git_clean=True, noop_finish_rejections=3
    )
    assert decision.kind == DecisionKind.REJECT_FINISH
    assert "NO changes" in decision.prompt or "not edited" in decision.prompt.lower()
    log.close()


def test_reject_unverified_finish_when_dirty_and_untested(tmp_path):
    engine = DecisionEngine.from_config(
        AgentConfig(require_edit_before_finish=True, require_test_before_finish=True)
    )
    task = Task(description="x", repo_path=str(tmp_path), task_id="fin")
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    log.log_task_start(task)
    action = Action(ActionType.FINISH, "done", message="fixed")
    log.log_action(step=1, action=action)
    decision = engine.after_action(
        action, log, git_clean=False, verified_since_edit=False
    )
    assert decision.kind == DecisionKind.REJECT_FINISH
    assert decision.reason == "unverified_finish"
    assert "test" in decision.prompt.lower()
    log.close()


def test_allow_finish_when_dirty_and_verified(tmp_path):
    engine = DecisionEngine.from_config(
        AgentConfig(require_edit_before_finish=True, require_test_before_finish=True)
    )
    task = Task(description="x", repo_path=str(tmp_path), task_id="fin")
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    log.log_task_start(task)
    action = Action(ActionType.FINISH, "done", message="fixed")
    log.log_action(step=1, action=action)
    decision = engine.after_action(
        action, log, git_clean=False, verified_since_edit=True
    )
    assert decision.kind == DecisionKind.CONTINUE
    log.close()


def test_verified_flag_toggles_on_edit_and_test():
    engine = DecisionEngine()
    ok = Observation(ObservationStatus.SUCCESS, "ok", "file_write")
    fail = Observation(ObservationStatus.ERROR, "boom", "test")
    test_ok = Observation(ObservationStatus.SUCCESS, "passed", "test")
    assert engine.next_verified_since_edit("file_write", ok, True) is False
    assert engine.next_verified_since_edit("test", fail, False) is False
    assert engine.next_verified_since_edit("test", test_ok, False) is True
    assert engine.next_verified_since_edit("file_read", ok, False) is False


def test_reject_edit_before_locate(tmp_path):
    engine = DecisionEngine.from_config(
        AgentConfig(require_locate_before_edit=True)
    )
    task = Task(description="x", repo_path=str(tmp_path), task_id="loc")
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    log.log_task_start(task)
    action = Action(
        ActionType.TOOL_CALL, "edit",
        tool_call=ToolCall("file_edit", {"path": "a.py", "old_string": "a", "new_string": "b"}),
    )
    log.log_action(step=1, action=action)
    decision = engine.after_action(action, log, located=False)
    assert decision.kind == DecisionKind.REJECT_ACTION
    assert decision.reason == "edit_before_locate"
    assert "search" in decision.prompt.lower()
    log.close()


def test_allow_edit_after_locate(tmp_path):
    engine = DecisionEngine.from_config(
        AgentConfig(require_locate_before_edit=True)
    )
    task = Task(description="x", repo_path=str(tmp_path), task_id="loc")
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    log.log_task_start(task)
    action = Action(
        ActionType.TOOL_CALL, "edit",
        tool_call=ToolCall("file_edit", {"path": "a.py", "old_string": "a", "new_string": "b"}),
    )
    log.log_action(step=1, action=action)
    decision = engine.after_action(action, log, located=True)
    assert decision.kind == DecisionKind.CONTINUE
    log.close()


def test_located_flag_set_by_search():
    engine = DecisionEngine()
    ok = Observation(ObservationStatus.SUCCESS, "hits", "search_text")
    miss = Observation(ObservationStatus.ERROR, "none", "search_text")
    assert engine.next_located("search_text", ok, False) is True
    assert engine.next_located("search_text", miss, False) is False
    assert engine.next_located("file_read", ok, False) is False
    assert engine.next_located("find_symbol", ok, False) is True
