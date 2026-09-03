"""Guards that keep SWE agents out of tests/."""

from tools.file_tool import FileEditTool, FileWriteTool
from tools.path_policy import is_test_path
from tools.undo_tool import UndoManager


def test_is_test_path_detects_common_layouts():
    assert is_test_path("tests/test_foo.py")
    assert is_test_path("pkg/tests/test_foo.py")
    assert is_test_path("test/test_foo.py")
    assert is_test_path("test_foo.py")
    assert is_test_path("foo_test.py")
    assert not is_test_path("src/flask/blueprints.py")
    assert not is_test_path("seaborn/_core/properties.py")
    assert not is_test_path("src/flask/testing.py")  # library helper, not a suite


def test_file_write_blocks_tests_when_enabled(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    target = repo / "tests" / "test_x.py"
    tool = FileWriteTool(UndoManager(), block_test_edits=True)
    result = tool.execute({"path": str(target), "content": "def test_x():\n    pass\n"})
    assert result.success is False
    assert "test" in (result.error or "").lower()
    assert not target.exists()


def test_file_edit_blocks_tests_when_enabled(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    target = repo / "tests" / "test_x.py"
    target.write_text("x = 1\n", encoding="utf-8")
    tool = FileEditTool(UndoManager(), block_test_edits=True)
    result = tool.execute({
        "path": str(target), "old_string": "x = 1", "new_string": "x = 2",
    })
    assert result.success is False
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_file_write_allows_library_when_blocked(tmp_path):
    target = tmp_path / "lib.py"
    tool = FileWriteTool(UndoManager(), block_test_edits=True)
    result = tool.execute({"path": str(target), "content": "ok = True\n"})
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "ok = True\n"
