"""
context/skills.py

Agent Skills — packaged SOP + instructions the model can load on demand.

Rules (`AGENTS.md`) are always-on project policy. Skills are *named playbooks*
(Claude Code SKILL.md style): a short catalog sits in the prompt; the `skill`
tool loads the full body when the task matches.

This is how platform SOPs (alert triage, change review, oncall) get wrapped
without baking every procedure into the system prompt.

Discovery (first match wins per name):
    <repo>/.agent/skills/<name>/SKILL.md
    <repo>/.claude/skills/<name>/SKILL.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from context.rules import _parse_mdc, _truthy

logger = logging.getLogger(__name__)

_MAX_CATALOG = 4_000
_MAX_BODY = 6_000

_SKILL_DIRS = (".agent/skills", ".claude/skills")


@dataclass
class Skill:
    name: str
    path: str
    description: str
    body: str
    tools: list[str] = field(default_factory=list)
    always_apply: bool = False

    def render_body(self) -> str:
        body = self.body
        if len(body) > _MAX_BODY:
            body = body[:_MAX_BODY] + "\n…"
        tools = f"\nAllowed tools: {', '.join(self.tools)}" if self.tools else ""
        return f"## Skill: {self.name}\n{self.description}{tools}\n\n{body}"


def load_skills(repo_path: str | Path) -> list[Skill]:
    """Discover SKILL.md files. Missing folders are skipped."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        return []
    found: list[Skill] = []
    seen: set[str] = set()
    for folder_rel in _SKILL_DIRS:
        folder = root / folder_rel
        if not folder.is_dir():
            continue
        for skill_md in sorted(folder.glob("*/SKILL.md")):
            try:
                raw = skill_md.read_text("utf-8", errors="replace")
            except OSError as exc:
                logger.debug("skip skill %s: %s", skill_md, exc)
                continue
            meta, body = _parse_mdc(raw)
            if not body.strip():
                continue
            name = str(meta.get("name") or skill_md.parent.name).strip()
            if not name or name in seen:
                continue
            tools_raw = meta.get("tools") or []
            if isinstance(tools_raw, str):
                tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
            elif isinstance(tools_raw, list):
                tools = [str(t).strip() for t in tools_raw if str(t).strip()]
            else:
                tools = []
            seen.add(name)
            try:
                rel = str(skill_md.relative_to(root))
            except ValueError:
                rel = str(skill_md)
            found.append(Skill(
                name=name,
                path=rel,
                description=str(meta.get("description") or "").strip(),
                body=body.strip(),
                tools=tools,
                always_apply=_truthy(meta.get("alwaysApply", meta.get("always_apply", False))),
            ))
    return found


def get_skill(repo_path: str | Path, name: str) -> Skill | None:
    want = (name or "").strip()
    for skill in load_skills(repo_path):
        if skill.name == want:
            return skill
    return None


def render_skills_catalog(skills: list[Skill], *, max_chars: int = _MAX_CATALOG) -> str:
    """Always-apply bodies + a one-line catalog of the rest (load via `skill` tool)."""
    if not skills:
        return ""
    lines = ["## Skills (named SOPs — load with the skill tool when relevant)"]
    used = len(lines[0])
    catalog: list[str] = []
    for skill in skills:
        if skill.always_apply:
            block = "\n" + skill.render_body()
            if used + len(block) > max_chars:
                catalog.append(f"- {skill.name}: {skill.description or skill.path} (truncated)")
                continue
            lines.append(block)
            used += len(block)
            continue
        hint = skill.description or skill.path
        catalog.append(f"- **{skill.name}**: {hint}")
    if catalog and used < max_chars:
        extra = "\n### Available skills\n" + "\n".join(catalog)
        if used + len(extra) <= max_chars:
            lines.append(extra)
    return "\n".join(lines) if len(lines) > 1 else ""


def load_skills_prompt(repo_path: str | Path) -> str:
    return render_skills_catalog(load_skills(repo_path))
