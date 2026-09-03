"""
context/rules.py

Project rules — the Cursor / Claude Code long-term *instruction* layer.

Memories are facts the agent learned. Rules are instructions *you* wrote and
expect on every turn: AGENTS.md, CLAUDE.md, .cursor/rules/*.mdc.

Loaded from the repo (not SQLite). Bounded so they stay prompt-cache friendly.
Independent of --memory: rules apply whenever the files exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Claude Code auto-memory style bound: keep the always-on block small.
_MAX_CHARS = 8_000
_MAX_BODY = 2_500

_ROOT_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/CLAUDE.md",
    ".cursorrules",
)


@dataclass
class RuleFile:
    path: str
    body: str
    always_apply: bool = True
    description: str = ""
    globs: str = ""


def _parse_mdc(raw: str) -> tuple[dict, str]:
    """Split optional YAML frontmatter from an .mdc / markdown rule."""
    text = raw or ""
    if not text.startswith("---"):
        return {}, text.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    front, body = parts[1], parts[2]
    meta: dict = {}
    try:
        import yaml
        parsed = yaml.safe_load(front) or {}
        if isinstance(parsed, dict):
            meta = parsed
    except Exception:
        for line in front.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')
    return meta, body.strip()


def _truthy(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes")


def load_rule_files(repo_path: str | Path) -> list[RuleFile]:
    """Discover rule files under repo_path. Missing files are skipped."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        return []
    found: list[RuleFile] = []
    seen: set[str] = set()

    def _add(path: Path, *, always: bool | None = None) -> None:
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        if rel in seen or not path.is_file():
            return
        try:
            raw = path.read_text("utf-8", errors="replace")
        except OSError as exc:
            logger.debug("skip rule %s: %s", path, exc)
            return
        meta, body = _parse_mdc(raw)
        if not body.strip():
            return
        seen.add(rel)
        if always is None:
            always = _truthy(meta.get("alwaysApply", meta.get("always_apply", True)))
            if meta.get("globs") and not _truthy(meta.get("alwaysApply", False)):
                always = False
        found.append(RuleFile(
            path=rel,
            body=body.strip(),
            always_apply=always,
            description=str(meta.get("description") or ""),
            globs=str(meta.get("globs") or ""),
        ))

    for name in _ROOT_FILES:
        _add(root / name, always=True)

    for folder in (root / ".cursor" / "rules", root / ".agent" / "rules",
                   root / ".claude" / "rules"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.mdc")) + sorted(folder.glob("*.md")):
            _add(path)

    return found


def render_rules(rules: list[RuleFile], *, max_chars: int = _MAX_CHARS) -> str:
    """Bounded prompt block. alwaysApply bodies first; others as a one-line index."""
    if not rules:
        return ""
    lines = ["## Project rules (follow these over generic habits)"]
    used = len(lines[0])
    catalog: list[str] = []
    for rule in rules:
        if not rule.always_apply:
            hint = rule.description or rule.path
            extra = f" [{rule.globs}]" if rule.globs else ""
            catalog.append(f"- {rule.path}{extra}: {hint}")
            continue
        body = rule.body
        if len(body) > _MAX_BODY:
            body = body[:_MAX_BODY] + "\n…"
        block = f"\n### {rule.path}\n{body}"
        if used + len(block) > max_chars:
            lines.append(f"- … ({rule.path} truncated)")
            break
        lines.append(block)
        used += len(block)
    if catalog and used < max_chars:
        extra = "\n### Conditional rules (pull in when relevant)\n" + "\n".join(catalog)
        if used + len(extra) <= max_chars:
            lines.append(extra)
    return "\n".join(lines) if len(lines) > 1 else ""


def load_project_rules(repo_path: str | Path, *, max_chars: int = _MAX_CHARS) -> str:
    return render_rules(load_rule_files(repo_path), max_chars=max_chars)
