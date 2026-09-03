"""
tools/skill_tool.py

Load a named Skill (SOP playbook) into the conversation.

    skill name=alert-triage

The catalog of available skills is injected by context/skills.py; this tool
returns the full body when the model decides the SOP applies.
"""

from __future__ import annotations

from typing import Any

from context.skills import get_skill, load_skills
from tools.base import BaseTool, ToolResult


class SkillTool(BaseTool):
    def __init__(self, repo_path: str = ".") -> None:
        self._repo_path = repo_path

    @property
    def name(self) -> str:
        return "skill"

    @property
    def description(self) -> str:
        return (
            "Load a named Skill / SOP playbook (full instructions). "
            "Use after matching a catalog entry in the system prompt. "
            "Pass `list` as the name to see available skills."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name, or 'list' to show the catalog.",
                },
            },
            "required": ["name"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        name = (params.get("name") or "").strip()
        if not name:
            return ToolResult(success=False, output="", error="`name` is required")
        if name.lower() == "list":
            skills = load_skills(self._repo_path)
            if not skills:
                return ToolResult(success=True, output="(no skills in this repo)")
            lines = [f"- {s.name}: {s.description or s.path}" for s in skills]
            return ToolResult(success=True, output="\n".join(lines))
        skill = get_skill(self._repo_path, name)
        if skill is None:
            available = ", ".join(s.name for s in load_skills(self._repo_path)) or "none"
            return ToolResult(
                success=False, output="",
                error=f"unknown skill {name!r}. Available: {available}",
            )
        return ToolResult(success=True, output=skill.render_body())
