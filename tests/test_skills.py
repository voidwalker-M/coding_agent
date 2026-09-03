"""Skills loader (SKILL.md playbooks) + skill tool."""

from pathlib import Path

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, Task, ToolCall
from context.skills import get_skill, load_skills, load_skills_prompt, render_skills_catalog
from llm.base import MockBackend
from tools.base import ToolRegistry
from tools.skill_tool import SkillTool


def _write_skill(root: Path, name: str, body: str, *, always: bool = False, desc: str = "") -> None:
    folder = root / ".agent" / "skills" / name
    folder.mkdir(parents=True)
    front = [
        "---",
        f"name: {name}",
        f"description: {desc or name}",
        f"alwaysApply: {'true' if always else 'false'}",
        "---",
        "",
        body,
    ]
    (folder / "SKILL.md").write_text("\n".join(front), encoding="utf-8")


def test_catalog_does_not_dump_on_demand_body(tmp_path):
    _write_skill(tmp_path, "alert-triage", "Do not mutate production.", desc="Oncall SOP")
    prompt = load_skills_prompt(tmp_path)
    assert "alert-triage" in prompt
    assert "Oncall SOP" in prompt
    assert "Do not mutate production" not in prompt


def test_always_apply_dumps_body(tmp_path):
    _write_skill(tmp_path, "safety", "Never run rm -rf /.", always=True, desc="Safety SOP")
    prompt = load_skills_prompt(tmp_path)
    assert "Never run rm -rf /." in prompt


def test_claude_skills_dir_discovered(tmp_path):
    folder = tmp_path / ".claude" / "skills" / "review"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: review\ndescription: PR review SOP\n---\nCheck the diff.\n",
        encoding="utf-8",
    )
    names = [s.name for s in load_skills(tmp_path)]
    assert names == ["review"]


def test_agent_dir_wins_over_claude_same_name(tmp_path):
    _write_skill(tmp_path, "triage", "From .agent", desc="agent copy")
    claude = tmp_path / ".claude" / "skills" / "triage"
    claude.mkdir(parents=True)
    (claude / "SKILL.md").write_text(
        "---\nname: triage\ndescription: claude copy\n---\nFrom .claude\n",
        encoding="utf-8",
    )
    skill = get_skill(tmp_path, "triage")
    assert skill is not None
    assert "From .agent" in skill.body


def test_missing_repo_is_empty(tmp_path):
    assert load_skills(tmp_path / "nope") == []
    assert load_skills_prompt(tmp_path) == ""
    assert render_skills_catalog([]) == ""


def test_example_alert_triage_skill_parses(tmp_path):
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "examples" / "skills" / "alert-triage" / "SKILL.md"
    dest = tmp_path / ".agent" / "skills" / "alert-triage"
    dest.mkdir(parents=True)
    dest.joinpath("SKILL.md").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    skill = get_skill(tmp_path, "alert-triage")
    assert skill is not None
    assert "Gather evidence" in skill.body
    assert skill.always_apply is False


def test_cli_registry_includes_skill_and_web_fetch():
    from config.schema import AppConfig
    from entry.cli import _build_registry
    names = _build_registry(AppConfig()).tool_names
    assert "skill" in names
    assert "web_fetch" in names
    assert "plan" in names


def test_skill_tool_list_and_load(tmp_path):
    _write_skill(tmp_path, "alert-triage", "Gather evidence first.", desc="Oncall SOP")
    tool = SkillTool(repo_path=str(tmp_path))
    listed = tool.execute({"name": "list"})
    assert listed.success
    assert "alert-triage" in listed.output
    loaded = tool.execute({"name": "alert-triage"})
    assert loaded.success
    assert "Gather evidence first." in loaded.output
    missing = tool.execute({"name": "no-such"})
    assert not missing.success


def test_skill_catalog_injected_into_prompt(tmp_path):
    _write_skill(tmp_path, "alert-triage", "Hidden body.", desc="Oncall SOP")
    task = Task(description="triage this alert", repo_path=str(tmp_path), max_steps=3)
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    backend = MockBackend([
        Action(ActionType.FINISH, "done", message="ok"),
    ])
    Agent(backend, ToolRegistry(), AgentConfig(max_steps=3)).run(task, log)
    sys_msg = backend.received_messages[0][0].content
    assert "alert-triage" in sys_msg
    assert "Oncall SOP" in sys_msg
    assert "Hidden body." not in sys_msg
    log.close()


def test_skill_tool_load_injected_via_observation(tmp_path):
    _write_skill(tmp_path, "alert-triage", "Page the owner if p95 > 2s.", desc="Oncall SOP")
    task = Task(description="triage", repo_path=str(tmp_path), max_steps=5)
    log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    script = [
        Action(ActionType.TOOL_CALL, "skill",
               tool_call=ToolCall("skill", {"name": "alert-triage"})),
        Action(ActionType.FINISH, "done", message="ok"),
    ]
    backend = MockBackend(script)
    registry = ToolRegistry().register(SkillTool(repo_path=str(tmp_path)))
    Agent(backend, registry, AgentConfig(max_steps=5)).run(task, log)
    # Second LLM call should see the loaded SOP in the observation
    second = backend.received_messages[1]
    blob = "\n".join(m.content for m in second)
    assert "Page the owner if p95 > 2s." in blob
    log.close()
