"""
entry/web.py

Frontend chat box (#1): a minimal web UI for the agent.

Deliberately stdlib-only (http.server + SSE) to honor the project's
"no heavy dependencies" rule — no FastAPI/uvicorn required. It serves a single
HTML page and one Server-Sent-Events endpoint that streams the agent's steps
live to the browser as it works.

Architecture mirrors ChatSession (entry/chat.py):
- one persistent Agent + shared ConversationHistory (multi-turn memory)
- each round runs in a background thread; the agent's EventLog is monkey-patched
  to push compact events onto a per-request queue, which the SSE handler drains
- a lock serializes rounds (a chat is inherently sequential)

Usage:
    agent web --repo . --port 8765
    # then open http://localhost:8765
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

# Sentinel pushed by the worker when a round ends.
_DONE = "__done__"


# ---------------------------------------------------------------------------
# Event → compact web dict
# ---------------------------------------------------------------------------

def event_to_web(event) -> dict | None:
    """Convert an EventLog event into a compact dict for the browser, or None to drop it."""
    from agent.task import EventType
    etype = event.event_type
    p = event.payload

    if etype == EventType.ACTION:
        action = p["action"]
        tc = action.get("tool_call")
        if tc:
            params = tc.get("params", {})
            key = (params.get("cmd") or params.get("path") or params.get("pattern")
                   or params.get("symbol") or params.get("message") or "")
            return {"type": "step", "step": p["step"], "tool": tc["name"],
                    "thought": (action.get("thought") or "").strip()[:300],
                    "arg": str(key)[:120]}
        if action.get("action_type") == "finish":
            return {"type": "finish_action", "step": p["step"]}
        if action.get("action_type") == "give_up":
            return {"type": "giveup_action", "step": p["step"]}
        return None

    if etype == EventType.OBSERVATION:
        obs = p["observation"]
        ok = obs.get("status") == "success"
        return {"type": "obs", "ok": ok,
                "text": ((obs.get("error") or obs.get("output") or "").strip()[:400])}

    if etype == EventType.REFLECTION:
        return {"type": "reflection", "reason": p.get("reason", "")}

    if etype == EventType.TASK_FAILED:
        return {"type": "failed", "reason": p.get("reason", "")}

    return None


# ---------------------------------------------------------------------------
# ChatWebApp — persistent agent + shared history
# ---------------------------------------------------------------------------

class ChatWebApp:
    def __init__(self, backend, registry, config, repo_path: str, log_dir: str) -> None:
        from agent.core import Agent, AgentConfig
        from context.history import ConversationHistory

        self.repo_path = repo_path
        self.log_dir = log_dir
        self.config = config
        self._lock = threading.Lock()
        self._q: queue.Queue | None = None

        agent_cfg = AgentConfig(
            max_steps=config.agent.max_steps,
            budget_tokens=config.agent.budget_tokens,
            history_max_messages=config.context.history_window * 2,
            stream=False,   # web feed is event-based, not token-based
        )
        self.agent = Agent(backend, registry, agent_cfg)
        self._history = ConversationHistory(max_messages=config.context.history_window * 2)
        self.round_count = 0

    def run_round(self, message: str, q: "queue.Queue") -> None:
        """Run one chat round, pushing web events onto q, ending with a _DONE dict."""
        from agent.event_log import EventLog
        from agent.task import Task
        from llm.base import LLMMessage

        if not self._lock.acquire(blocking=False):
            q.put({"type": "error", "message": "Agent is busy with another request."})
            q.put({"type": _DONE})
            return
        try:
            self.round_count += 1
            self._history.add(LLMMessage(role="user", content=message))
            task = Task(description=message, repo_path=self.repo_path,
                        max_steps=self.config.agent.max_steps,
                        budget_tokens=self.config.agent.budget_tokens)
            self.agent._pending_history = self._history

            with EventLog.create(task, log_dir=self.log_dir) as log:
                original_append = log._append

                def live_append(event):
                    original_append(event)
                    web = event_to_web(event)
                    if web is not None:
                        q.put(web)

                log._append = live_append
                result = self.agent.run(task, log)

            if result.summary:
                self._history.add(LLMMessage(
                    role="assistant",
                    content=f"[Round {self.round_count} complete]\n{result.summary}"))
            q.put({"type": "answer", "text": result.summary or "(no summary)",
                   "status": result.status.value, "steps": result.steps_taken,
                   "tokens": result.total_tokens})
        except Exception as exc:
            logger.exception("web round failed")
            q.put({"type": "error", "message": str(exc)})
        finally:
            if hasattr(self.agent, "_pending_history"):
                del self.agent._pending_history
            self._lock.release()
            q.put({"type": _DONE})


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _make_handler(app: ChatWebApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default stderr logging
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/chat":
                self._sse(parse_qs(parsed.query).get("message", [""])[0])
            else:
                self.send_error(404)

        def _sse(self, message: str):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            if not message.strip():
                self._write({"type": "error", "message": "empty message"})
                self._write({"type": _DONE})
                return
            q: queue.Queue = queue.Queue()
            threading.Thread(target=app.run_round, args=(message, q), daemon=True).start()
            while True:
                ev = q.get()
                if ev.get("type") == _DONE:
                    self._write({"type": "done"})
                    break
                try:
                    self._write(ev)
                except (BrokenPipeError, ConnectionResetError):
                    break

        def _write(self, obj: dict):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler


def serve(app: ChatWebApp, host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), _make_handler(app))
    logger.info("Chat web UI on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# Single-page UI
# ---------------------------------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coding Agent</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
         background:#0d1117; color:#e6edf3; display:flex; flex-direction:column; height:100vh; }
  header { padding:12px 16px; background:#161b22; border-bottom:1px solid #30363d; font-weight:600; }
  #log { flex:1; overflow-y:auto; padding:16px; }
  .msg { max-width:760px; margin:0 auto 14px; }
  .user { text-align:right; }
  .user .bubble { background:#1f6feb; display:inline-block; padding:8px 12px; border-radius:12px; }
  .agent .bubble { background:#161b22; border:1px solid #30363d; padding:10px 14px; border-radius:12px; white-space:pre-wrap; }
  .step { font:13px ui-monospace,monospace; color:#7d8590; margin:2px auto; max-width:760px; }
  .step .tool { color:#39c5cf; } .ok { color:#3fb950; } .bad { color:#f85149; }
  .reflect { color:#d29922; }
  footer { padding:12px 16px; background:#161b22; border-top:1px solid #30363d; }
  form { max-width:760px; margin:0 auto; display:flex; gap:8px; }
  input { flex:1; padding:10px 12px; border-radius:8px; border:1px solid #30363d; background:#0d1117; color:#e6edf3; }
  button { padding:10px 16px; border:0; border-radius:8px; background:#238636; color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
</style></head>
<body>
<header>🤖 Coding Agent</header>
<div id="log"></div>
<footer><form id="f">
  <input id="m" autocomplete="off" placeholder="Describe a task or ask a question…" autofocus>
  <button id="b" type="submit">Send</button>
</form></footer>
<script>
const log = document.getElementById('log'), form = document.getElementById('f'),
      input = document.getElementById('m'), btn = document.getElementById('b');
function add(cls, html){ const d=document.createElement('div'); d.className=cls; d.innerHTML=html; log.appendChild(d); log.scrollTop=log.scrollHeight; return d; }
function esc(s){ return (s||'').replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
form.onsubmit = e => {
  e.preventDefault();
  const msg = input.value.trim(); if(!msg) return;
  add('msg user', `<span class="bubble">${esc(msg)}</span>`);
  input.value=''; btn.disabled=true; input.disabled=true;
  const es = new EventSource('/api/chat?message=' + encodeURIComponent(msg));
  es.onmessage = ev => {
    const d = JSON.parse(ev.data);
    if(d.type==='step') add('step', `[${d.step}] <span class="tool">${esc(d.tool)}</span> ${esc(d.arg)}`);
    else if(d.type==='obs') add('step', `&nbsp;&nbsp;<span class="${d.ok?'ok':'bad'}">${d.ok?'✓':'✗'}</span> ${esc(d.text).slice(0,200)}`);
    else if(d.type==='reflection') add('step reflect', `⟳ reflection (${esc(d.reason)})`);
    else if(d.type==='answer') add('msg agent', `<span class="bubble">${esc(d.text)}</span><div class="step">${d.status} · ${d.steps} steps · ${d.tokens} tokens</div>`);
    else if(d.type==='failed') add('msg agent', `<span class="bubble bad">Failed: ${esc(d.reason)}</span>`);
    else if(d.type==='error') add('msg agent', `<span class="bubble bad">Error: ${esc(d.message)}</span>`);
    else if(d.type==='done'){ es.close(); btn.disabled=false; input.disabled=false; input.focus(); }
  };
  es.onerror = () => { es.close(); btn.disabled=false; input.disabled=false; };
};
</script>
</body></html>
"""
