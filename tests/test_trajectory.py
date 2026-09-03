"""Process metrics from EventLog trajectories."""

from agent.event_log import EventLog
from agent.task import Action, ActionType, Observation, ObservationStatus, Task, ToolCall
from eval.trajectory import merge_trajectory, trajectory_metrics


def test_trajectory_counts_errors_duplicates_and_first_edit(tmp_path):
    task = Task(description="x", repo_path=str(tmp_path), task_id="traj")
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    log.log_task_start(task)
    read = Action(ActionType.TOOL_CALL, "r", tool_call=ToolCall("file_read", {"path": "a.py"}))
    write = Action(ActionType.TOOL_CALL, "w", tool_call=ToolCall("file_write", {"path": "a.py", "content": "x"}))
    log.log_action(step=1, action=read)
    log.log_observation(
        step=1,
        observation=Observation(ObservationStatus.ERROR, "", "file_read", error="missing"),
        duration_ms=100,
    )
    log.log_action(step=2, action=read)
    log.log_observation(
        step=2,
        observation=Observation(ObservationStatus.SUCCESS, "ok", "file_read"),
        duration_ms=50,
    )
    log.log_action(step=3, action=write)
    log.log_observation(
        step=3,
        observation=Observation(ObservationStatus.SUCCESS, "wrote", "file_write"),
        duration_ms=200,
    )
    stats = trajectory_metrics(log)
    log.close()
    assert stats["tool_errors"] == 1
    assert stats["duplicate_actions"] == 1  # two identical file_read calls in a row
    assert stats["edited"] is True
    assert stats["time_to_first_edit"] == 0.35  # 0.1 + 0.05 + 0.2


def test_merge_trajectory_sums_and_mins():
    a = {"tool_ok": 1, "tool_errors": 1, "duplicate_actions": 0, "reflections": 1,
         "time_to_first_edit": 0.5, "edited": True, "tool_error_rate": 0.5}
    b = {"tool_ok": 2, "tool_errors": 0, "duplicate_actions": 3, "reflections": 0,
         "time_to_first_edit": 0.2, "edited": True, "tool_error_rate": 0.0}
    m = merge_trajectory(a, b)
    assert m["tool_errors"] == 1
    assert m["duplicate_actions"] == 3
    assert m["reflections"] == 1
    assert m["time_to_first_edit"] == 0.2
