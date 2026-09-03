"""Plan object + plan tool."""

from agent.plan import Plan, StepStatus
from tools.plan_tool import PlanTool


def test_replace_and_remaining():
    plan = Plan()
    plan.replace(["explore", "edit", "test"], goal="fix add()")
    assert plan.goal == "fix add()"
    assert [s.title for s in plan.steps] == ["explore", "edit", "test"]
    assert len(plan.remaining()) == 3
    plan.set_status(1, StepStatus.DONE)
    plan.set_status(2, StepStatus.IN_PROGRESS)
    assert [s.id for s in plan.remaining()] == [2, 3]
    rendered = plan.render()
    assert "[x] 1. explore" in rendered
    assert "[>] 2. edit" in rendered
    assert "Remaining: 2/3" in rendered


def test_render_for_prompt_empty():
    assert Plan().render_for_prompt() == ""
    plan = Plan()
    plan.replace(["do the thing"])
    assert "## Current plan" in plan.render_for_prompt()


def test_plan_tool_create_complete_list():
    tool = PlanTool()
    created = tool.execute({"action": "create", "steps": ["read", "patch"], "goal": "bugfix"})
    assert created.success
    assert "1. read" in created.output
    done = tool.execute({"action": "complete", "step_id": 1})
    assert done.success
    assert "[x] 1. read" in done.output
    listed = tool.execute({"action": "list"})
    assert "Remaining: 1/2" in listed.output


def test_plan_tool_rejects_empty_create():
    tool = PlanTool()
    result = tool.execute({"action": "create", "steps": []})
    assert result.success is False
