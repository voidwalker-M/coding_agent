# Skills — named SOP playbooks

Rules (`AGENTS.md`, `.cursor/rules`) are always-on project policy. **Skills** are
named playbooks the agent loads when the task matches — the same split Claude Code
uses for `SKILL.md`.

This is how a platform SOP (alert triage, change review, on-call) becomes an Agent
capability without stuffing every procedure into the system prompt.

## Layout

```
<repo>/.agent/skills/<name>/SKILL.md
<repo>/.claude/skills/<name>/SKILL.md   # also discovered; .agent wins on name clash
```

Frontmatter:

```markdown
---
name: alert-triage
description: SOP for triaging a monitoring/oncall alert
tools: web_fetch, search_text, file_read
alwaysApply: false
---

# Steps
1. Identify the alert …
```

- `alwaysApply: true` — body is injected into the system prompt (keep it short).
- otherwise — a one-line catalog entry is injected; the model calls `skill name=…`
  to load the full body.

## Example

Copy the sample on-call playbook into a target repo:

```bash
mkdir -p your-repo/.agent/skills/alert-triage
cp examples/skills/alert-triage/SKILL.md your-repo/.agent/skills/alert-triage/
```

Then: `agent chat --repo your-repo` and ask it to triage an alert. The catalog
shows `alert-triage`; the `skill` tool loads the SOP.

## Wrapping platform tools (Tool / MCP / Skill)

| Layer | What it is in this repo |
| ----- | ------------------------ |
| **Tool** | `BaseTool` in `tools/` (`web_fetch`, `search_text`, `pytest`, …) |
| **MCP** | `agent mcp` exposes the same registry to Cursor / Claude Desktop (`context/mcp_bridge.py`) |
| **Skill** | SOP text that tells the model *when* and *how* to call those tools |

A risk / change-control / on-call platform would be wrapped the same way: a thin
`BaseTool` (or MCP server) around the real API, plus a Skill that encodes the SOP.
This repo does not ship fictional 风控 APIs.

## Tests

See `tests/test_skills.py` (catalog vs body, `.agent` vs `.claude`, tool list/load,
prompt injection). Run: `pytest tests/test_skills.py`.
