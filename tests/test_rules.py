"""tests/test_rules.py — project rules loader (AGENTS.md / .cursor/rules)."""

from context.rules import load_project_rules, load_rule_files, render_rules


def test_loads_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Agents\nAlways use pytest.\n")
    block = load_project_rules(tmp_path)
    assert "AGENTS.md" in block
    assert "pytest" in block


def test_loads_cursor_mdc_always_apply(tmp_path):
    rules = tmp_path / ".agent" / "rules"
    rules.mkdir(parents=True)
    (rules / "python.mdc").write_text(
        "---\nalwaysApply: true\n---\nPrefer ruff over flake8.\n"
    )
    block = load_project_rules(tmp_path)
    assert "Prefer ruff" in block
    assert "python.mdc" in block


def test_glob_rules_are_catalogued_not_dumped(tmp_path):
    rules = tmp_path / ".agent" / "rules"
    rules.mkdir(parents=True)
    (rules / "frontend.mdc").write_text(
        "---\nalwaysApply: false\nglobs: \"**/*.tsx\"\ndescription: React conventions\n---\n"
        "Never use default exports.\n"
    )
    block = load_project_rules(tmp_path)
    assert "React conventions" in block
    assert "Never use default exports" not in block


def test_missing_repo_is_empty(tmp_path):
    assert load_project_rules(tmp_path / "nope") == ""
    assert load_rule_files(tmp_path) == []
    assert render_rules([]) == ""
