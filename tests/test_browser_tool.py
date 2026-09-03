"""web_fetch: HTML extract + stdlib HTTP GET. No Playwright required."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tools.browser_tool import WebFetchTool, extract_html_text


def test_extract_html_strips_script_and_keeps_title():
    html = (
        "<html><head><title>Status</title></head>"
        "<body><script>secret=1</script><style>p{}</style>"
        "<p>Latency is high.</p></body></html>"
    )
    title, text = extract_html_text(html)
    assert title == "Status"
    assert "Latency is high." in text
    assert "secret=1" not in text


def test_rejects_non_http_urls():
    tool = WebFetchTool()
    for url in ("file:///etc/passwd", "ftp://example.com/a", "not-a-url", ""):
        result = tool.execute({"url": url})
        assert not result.success


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/ok":
            body = b"<html><head><title>Demo</title></head><body><p>hello agent</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/plain":
            body = b"plain text payload"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):  # noqa: ARG002
        return


@pytest.fixture
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    thread.join(timeout=2)


def test_fetch_html_page(http_server):
    result = WebFetchTool().execute({"url": f"{http_server}/ok"})
    assert result.success
    assert "Title: Demo" in result.output
    assert "hello agent" in result.output


def test_fetch_plain_text(http_server):
    result = WebFetchTool().execute({"url": f"{http_server}/plain"})
    assert result.success
    assert "plain text payload" in result.output


def test_http_error(http_server):
    result = WebFetchTool().execute({"url": f"{http_server}/missing"})
    assert not result.success
    assert "404" in (result.error or "")


def test_playwright_engine_without_package():
    try:
        import playwright  # noqa: F401
    except ImportError:
        result = WebFetchTool().execute({
            "url": "http://127.0.0.1:1/",
            "engine": "playwright",
        })
        assert not result.success
        assert "Playwright" in (result.error or "")
        return
    pytest.skip("playwright is installed; missing-package path not exercised")
