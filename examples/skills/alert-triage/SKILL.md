---
name: alert-triage
description: SOP for triaging a monitoring/oncall alert — gather evidence, decide severity, propose next action
tools: web_fetch, search_text, file_read, skill
alwaysApply: false
---

# Alert triage playbook

You are assisting an on-call engineer. Do **not** mutate production. Read-only
tools only unless the user explicitly asks for a change.

## Steps

1. **Identify** the alert: name, service, timestamp, symptom (error rate, latency, drop).
2. **Gather evidence** — logs, dashboards, recent deploys, related code.
   - Code: `search_text` / `find_symbol` in this repo.
   - Runbooks / status pages: `web_fetch`.
3. **Hypotheses** — list 2–3 likely causes, ranked, with what would confirm each.
4. **Decision** — one of: page the owner / roll back / mitigate / watch / false alarm.
5. **Hand-off** — a short paragraph an on-call can paste into the ticket.

Finish with the decision and the evidence that supports it. If evidence is missing, say what to fetch next instead of guessing.
