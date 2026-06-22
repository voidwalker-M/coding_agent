"""
tests/test_web.py

Tests for the frontend chat box (#1). Exercises the pure event-conversion
helper, the single-page HTML, and an in-process chat round driven by
MockBackend (no live HTTP server, no API).
"""

import os
import queue
from types import SimpleNamespace

from agent.task import Action, ActionType, EventType, ToolCall
from entry.web import ChatWebApp, INDEX_HTML, event_to_web, _DONE
from llm.base import MockBackend
from tools.base import ToolRegistry
from tools.file_tool import FileWriteTool
from tools.test_tool import PytestTool


class _Ev:
    def __init__(self, etype, payload):
        self.event_type = etype
        self.payload = payload


def test_event_to_web_action_tool_call():
    ev = _Ev(EventType.ACTION, {"step": 2, "action": {
        "action_type": "tool_call", "thought": "let's look",
        "tool_call": {"name": "file_read", "params": {"path": "a.py"}}}})
    web = event_to_web(ev)
    assert web == {"type": "step", "step": 2, "tool": "file_read",
                   "thought": "let's look", "arg": "a.py"}


def test_event_to_web_observation_and_reflection():
    obs = event_to_web(_Ev(EventType.OBSERVATION,
                           {"observation": {"status": "error", "error": "boom"}}))
    assert obs == {"type": "obs", "ok": False, "text": "boom"}
    refl = event_to_web(_Ev(EventType.REFLECTION, {"reason": "test_failed"}))
    assert refl == {"type": "reflection", "reason": "test_failed"}


def test_index_html_has_chat_elements():
    assert "/api/chat" in INDEX_HTML
    assert "EventSource" in INDEX_HTML
    assert "<form" in INDEX_HTML


def _cfg(tmp_path):
    return SimpleNamespace(
        agent=SimpleNamespace(max_steps=5, budget_tokens=80_000, log_dir=str(tmp_path / "logs")),
        context=SimpleNamespace(history_window=20),
    )


def test_run_round_emits_events_and_done(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")

    script = [
        Action(ActionType.TOOL_CALL, "fix it",
               tool_call=ToolCall("file_write", {"path": "calc.py", "content": "def add(a,b):\n    return a+b\n"})),
        Action(ActionType.FINISH, "fixed the bug", message="Fixed add()."),
    ]
    registry = ToolRegistry().register(FileWriteTool()).register(PytestTool())
    app = ChatWebApp(MockBackend(script), registry, _cfg(tmp_path),
                     str(repo), str(tmp_path / "logs"))

    q: queue.Queue = queue.Queue()
    prev = os.getcwd()
    try:
        os.chdir(repo)
        app.run_round("fix the add bug", q)
    finally:
        os.chdir(prev)

    events = []
    while not q.empty():
        events.append(q.get())
    types = [e["type"] for e in events]

    assert "step" in types                     # the tool-call step was streamed
    assert any(e["type"] == "answer" for e in events)
    answer = next(e for e in events if e["type"] == "answer")
    assert answer["text"] == "Fixed add()."
    assert types[-1] == _DONE                   # round always ends with the sentinel


def test_run_round_busy_returns_error(tmp_path):
    app = ChatWebApp(MockBackend([]), ToolRegistry(), _cfg(tmp_path),
                     str(tmp_path), str(tmp_path / "logs"))
    app._lock.acquire()  # simulate an in-flight round
    q: queue.Queue = queue.Queue()
    app.run_round("hello", q)
    events = [q.get(), q.get()]
    assert events[0]["type"] == "error"
    assert events[1]["type"] == _DONE
