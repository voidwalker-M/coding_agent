"""
tools/browser_tool.py

Fetch a URL and return extracted text — the thin "webpage Agent" primitive.

Default engine is stdlib urllib (no extra deps, works in tests). Optional
Playwright (`engine=playwright`) renders JavaScript pages when installed:

    pip install playwright && playwright install chromium

Safety: only http/https; response size and time are capped. This is a read
tool (extract / inspect), not a general Computer-Use driver.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tools.base import BaseTool, ToolResult

_MAX_BYTES = 400_000
_TIMEOUT = 15
_USER_AGENT = "coding-agent/0.1 (+web_fetch)"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += (" " if self.title else "") + text
        else:
            self._parts.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._parts)
        return re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", raw)).strip()


def extract_html_text(html: str) -> tuple[str, str]:
    """Return (title, visible text) from an HTML document."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        stripped = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        return "", re.sub(r"\s+", " ", stripped).strip()
    return parser.title, parser.text()


def _fetch_urllib(url: str) -> tuple[str, str, str]:
    """Return (final_url, content_type, body_text)."""
    req = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain"})
    with urlopen(req, timeout=_TIMEOUT) as resp:
        raw = resp.read(_MAX_BYTES + 1)
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        final = resp.geturl()
    truncated = len(raw) > _MAX_BYTES
    body = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
    if truncated:
        body += "\n… [truncated]"
    return final, ctype, body


def _fetch_playwright(url: str) -> tuple[str, str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=_TIMEOUT * 1000)
            html = page.content()
            final = page.url
        finally:
            browser.close()
    return final, "text/html", html[:_MAX_BYTES]


class WebFetchTool(BaseTool):
    """Fetch a page and return title + extracted text."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch an http(s) URL and return the page title plus extracted text. "
            "Default engine is a static HTTP GET. Use engine=playwright for JS-rendered "
            "pages (requires the playwright extra)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http or https URL to fetch"},
                "engine": {
                    "type": "string",
                    "enum": ["http", "playwright"],
                    "description": "http=static GET (default), playwright=headless Chromium",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters of extracted text to return (default 8000)",
                },
            },
            "required": ["url"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        url = (params.get("url") or "").strip()
        if not url:
            return ToolResult(success=False, output="", error="`url` is required")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ToolResult(
                success=False, output="",
                error="only http/https URLs are allowed",
            )
        engine = (params.get("engine") or "http").strip().lower()
        max_chars = int(params.get("max_chars") or 8000)
        max_chars = max(200, min(max_chars, 20_000))
        try:
            if engine == "playwright":
                final, ctype, body = _fetch_playwright(url)
            else:
                final, ctype, body = _fetch_urllib(url)
            final_parsed = urlparse(final)
            if final_parsed.scheme not in ("http", "https"):
                return ToolResult(
                    success=False, output="",
                    error="redirect landed on a non-http(s) URL; blocked",
                )
        except ImportError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except HTTPError as exc:
            return ToolResult(success=False, output="", error=f"HTTP {exc.code}: {exc.reason}")
        except (URLError, TimeoutError, OSError) as exc:
            return ToolResult(success=False, output="", error=f"fetch failed: {exc}")
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"fetch failed: {exc}")

        if "html" in ctype or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            title, text = extract_html_text(body)
        else:
            title, text = "", body.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n… [truncated]"
        header = f"URL: {final}\nContent-Type: {ctype or 'unknown'}"
        if title:
            header += f"\nTitle: {title}"
        return ToolResult(success=True, output=f"{header}\n\n{text}")
