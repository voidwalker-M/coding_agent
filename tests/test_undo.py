"""
tests/test_undo.py

Undo / rollback tests. Uses the real filesystem (tmp_path) like test_day3.py —
the point of these tools is that they actually restore bytes on disk.

Deliberately never creates a git repo: snapshot-based undo must work on a plain
directory, which is what `run --repo` accepts.
"""

from pathlib import Path

import pytest

from tools.file_tool import FileEditTool, FileWriteTool
from tools.undo_tool import Mutation, UndoManager, UndoTool


# ---------------------------------------------------------------------------
# UndoManager — snapshot and restore
# ---------------------------------------------------------------------------

def test_restores_previous_content(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("original\n")
    mgr = UndoManager()

    mgr.snapshot_from_disk("file_write", str(f))
    f.write_text("changed\n")
    reverted = mgr.pop()

    assert f.read_text() == "original\n"
    assert len(reverted) == 1
    assert reverted[0].existed_before is True


def test_deletes_file_that_did_not_exist_before(tmp_path):
    f = tmp_path / "new.py"
    mgr = UndoManager()

    mgr.snapshot_from_disk("file_write", str(f))
    f.write_text("created\n")
    mgr.pop()

    assert not f.exists()


def test_pop_is_last_in_first_out(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v0\n")
    mgr = UndoManager()

    for version in ("v1", "v2", "v3"):
        mgr.snapshot_from_disk("file_write", str(f))
        f.write_text(f"{version}\n")

    mgr.pop()
    assert f.read_text() == "v2\n"
    mgr.pop(steps=2)
    assert f.read_text() == "v0\n"


def test_pop_more_steps_than_stack_reverts_what_it_can(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("original\n")
    mgr = UndoManager()
    mgr.snapshot_from_disk("file_write", str(f))
    f.write_text("changed\n")

    reverted = mgr.pop(steps=99)

    assert len(reverted) == 1
    assert f.read_text() == "original\n"
    assert mgr.pop() == []


def test_pop_filtered_by_path_leaves_other_files_alone(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("a0\n")
    b.write_text("b0\n")
    mgr = UndoManager()

    mgr.snapshot_from_disk("file_write", str(a))
    a.write_text("a1\n")
    mgr.snapshot_from_disk("file_write", str(b))
    b.write_text("b1\n")

    mgr.pop(path=str(a))

    assert a.read_text() == "a0\n"
    assert b.read_text() == "b1\n"       # untouched
    assert len(mgr.stack) == 1           # b's snapshot survives


def test_snapshot_from_disk_never_raises_on_unreadable_path(tmp_path):
    mgr = UndoManager()
    mgr.snapshot_from_disk("file_write", str(tmp_path))   # a directory, not a file
    assert mgr.stack == []


# ---------------------------------------------------------------------------
# Sidecar persistence
# ---------------------------------------------------------------------------

def test_sidecar_roundtrip(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("original\n")
    sidecar = tmp_path / "logs" / "task1_20250101_000000.undo.jsonl"

    with UndoManager(sidecar_path=sidecar) as mgr:
        mgr.snapshot_from_disk("file_write", str(f))
        f.write_text("changed\n")

    loaded = UndoManager.load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].path == str(f)
    assert loaded[0].content_before == "original\n"
    assert loaded[0].tool == "file_write"


def test_sidecar_is_flushed_before_close(tmp_path):
    """A crashed run must still leave a usable sidecar — hence flush-per-write."""
    sidecar = tmp_path / "s.undo.jsonl"
    mgr = UndoManager(sidecar_path=sidecar)
    f = tmp_path / "a.py"
    f.write_text("x\n")
    mgr.snapshot_from_disk("file_write", str(f))

    assert len(UndoManager.load_sidecar(sidecar)) == 1   # readable, not yet closed


def test_find_sidecar_and_sidecar_for_log_path(tmp_path):
    (tmp_path / "abc12345_20250101_000000.undo.jsonl").write_text("")
    (tmp_path / "abc12345_20250101_000000.jsonl").write_text("")

    assert UndoManager.find_sidecar(tmp_path, "abc12345") is not None
    assert UndoManager.find_sidecar(tmp_path, "nope") is None

    found = UndoManager.sidecar_for_log_path(tmp_path / "abc12345_20250101_000000.jsonl")
    assert found is not None and found.name.endswith(".undo.jsonl")


# ---------------------------------------------------------------------------
# rollback_plan — CLI semantics (earliest snapshot per path)
# ---------------------------------------------------------------------------

def test_rollback_plan_takes_earliest_snapshot_per_path():
    muts = [
        Mutation(seq=1, tool="file_write", path="a.py", existed_before=True, content_before="a0"),
        Mutation(seq=2, tool="file_edit", path="a.py", existed_before=True, content_before="a1"),
        Mutation(seq=3, tool="file_write", path="b.py", existed_before=False, content_before=None),
    ]
    plan = UndoManager.rollback_plan(muts)

    assert [m.path for m in plan] == ["a.py", "b.py"]
    assert plan[0].content_before == "a0"    # pre-run state, not the last edit


def test_rollback_plan_filters_by_path():
    muts = [
        Mutation(seq=1, tool="file_write", path="a.py", existed_before=True, content_before="a0"),
        Mutation(seq=2, tool="file_write", path="b.py", existed_before=True, content_before="b0"),
    ]
    plan = UndoManager.rollback_plan(muts, path="b.py")
    assert [m.path for m in plan] == ["b.py"]


def test_rollback_plan_restores_whole_run(tmp_path):
    """End-to-end: several edits across two files, then roll the run back."""
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("a0\n")
    sidecar = tmp_path / "run.undo.jsonl"

    with UndoManager(sidecar_path=sidecar) as mgr:
        write, edit = FileWriteTool(undo_manager=mgr), FileEditTool(undo_manager=mgr)
        write.execute({"path": str(a), "content": "a1\n"})
        edit.execute({"path": str(a), "old_string": "a1", "new_string": "a2"})
        write.execute({"path": str(b), "content": "b1\n"})   # b did not exist before

    for m in UndoManager.rollback_plan(UndoManager.load_sidecar(sidecar)):
        UndoManager._restore(m)

    assert a.read_text() == "a0\n"
    assert not b.exists()


# ---------------------------------------------------------------------------
# File tools snapshot before mutating
# ---------------------------------------------------------------------------

def test_file_write_snapshots_before_overwriting(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("original\n")
    mgr = UndoManager()

    FileWriteTool(undo_manager=mgr).execute({"path": str(f), "content": "changed\n"})

    assert f.read_text() == "changed\n"
    assert mgr.stack[0].content_before == "original\n"
    mgr.pop()
    assert f.read_text() == "original\n"


def test_file_edit_snapshots_before_replacing(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    mgr = UndoManager()

    FileEditTool(undo_manager=mgr).execute(
        {"path": str(f), "old_string": "x = 1", "new_string": "x = 2"})

    assert f.read_text() == "x = 2\n"
    mgr.pop()
    assert f.read_text() == "x = 1\n"


def test_failed_edit_records_no_snapshot(tmp_path):
    """A rejected edit changed nothing, so it must not consume an undo slot."""
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    mgr = UndoManager()

    result = FileEditTool(undo_manager=mgr).execute(
        {"path": str(f), "old_string": "not-present", "new_string": "y"})

    assert not result.success
    assert mgr.stack == []


def test_file_tools_work_without_an_injected_manager(tmp_path):
    """Zero-arg construction stays valid (existing call sites rely on it)."""
    f = tmp_path / "a.py"
    result = FileWriteTool().execute({"path": str(f), "content": "hi\n"})
    assert result.success and f.read_text() == "hi\n"


# ---------------------------------------------------------------------------
# UndoTool — the LLM-facing surface
# ---------------------------------------------------------------------------

def test_undo_tool_reverts_and_describes(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("original\n")
    mgr = UndoManager()
    FileWriteTool(undo_manager=mgr).execute({"path": str(f), "content": "changed\n"})

    result = UndoTool(mgr).execute({})

    assert result.success
    assert str(f) in result.output
    assert "restored previous content" in result.output
    assert f.read_text() == "original\n"


def test_undo_tool_reports_deletion(tmp_path):
    f = tmp_path / "new.py"
    mgr = UndoManager()
    FileWriteTool(undo_manager=mgr).execute({"path": str(f), "content": "hi\n"})

    result = UndoTool(mgr).execute({})

    assert result.success
    assert "deleted" in result.output
    assert not f.exists()


def test_undo_tool_errors_when_nothing_to_undo():
    result = UndoTool(UndoManager()).execute({})
    assert not result.success
    assert "Nothing to undo" in result.error


@pytest.mark.parametrize("steps", [0, -1])
def test_undo_tool_rejects_non_positive_steps(steps):
    result = UndoTool(UndoManager()).execute({"steps": steps})
    assert not result.success
    assert "steps must be >= 1" in result.error


def test_undo_tool_rejects_non_integer_steps():
    result = UndoTool(UndoManager()).execute({"steps": "many"})
    assert not result.success
    assert "integer" in result.error


def test_undo_tool_multiple_steps(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v0\n")
    mgr = UndoManager()
    write = FileWriteTool(undo_manager=mgr)
    write.execute({"path": str(f), "content": "v1\n"})
    write.execute({"path": str(f), "content": "v2\n"})

    result = UndoTool(mgr).execute({"steps": 2})

    assert result.success
    assert f.read_text() == "v0\n"


def test_undo_tool_registered_in_default_registry():
    from config.schema import AppConfig
    from entry.cli import _build_registry

    registry = _build_registry(AppConfig())
    assert "undo" in registry


def test_registry_shares_one_manager_across_tools(tmp_path):
    """file_write and undo must see the same stack when routed via the registry."""
    from config.schema import AppConfig
    from entry.cli import _build_registry

    f = tmp_path / "a.py"
    f.write_text("original\n")
    registry = _build_registry(AppConfig())

    registry.execute_tool("file_write", {"path": str(f), "content": "changed\n"})
    result = registry.execute_tool("undo", {})

    assert result.success
    assert f.read_text() == "original\n"
