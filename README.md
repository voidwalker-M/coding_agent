# Coding Agent

[![Repo Views](https://visitor-badge.laobi.icu/badge?page_id=voidwalker-M.coding_agent&left_text=Repo%20Views)](https://github.com/voidwalker-M/coding_agent)

An autonomous coding agent. Give it a task description and it explores the codebase, edits files, and runs tests until the task is complete.

Supports **Claude, DeepSeek, OpenAI, Groq, Ollama, vLLM** and more, with built-in streaming output, Docker sandboxing, and GitHub Issue auto-fix.

---

## Quick Start

```bash
# Install
git clone <repo-url> && cd coding-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure (edit config/default.yaml with your provider and api_key)
export DEEPSEEK_API_KEY=sk-xxx   # or ANTHROPIC_API_KEY / OPENAI_API_KEY

# Verify
python smoke_test.py

# Use
cd your-project
agent chat
```

---

## Usage

### chat mode (recommended)

Continuous conversation with history retained across rounds — the closest experience to Claude Code:

```bash
agent chat                            # current directory
agent chat --repo /path/to/project   # specify directory
agent chat --model deepseek-v4-pro   # switch model
agent chat --sandbox                  # Docker sandbox
```

In-session commands: `/exit` quit, `/stats` view statistics, `/clear` clear history, `/help` help

### run mode

One-shot tasks, suitable for well-defined batch scenarios:

```bash
agent run --task "Fix all failing tests"
agent run --task-file task.txt           # read task from file
agent run --task "..." --confirm         # confirm before dangerous commands
agent run --task "..." --sandbox         # Docker sandbox
```

### GitHub Issue auto-fix

```bash
export GITHUB_TOKEN=ghp_xxx
python -m entry.github_issue \
    --repo owner/repo --issue 42 --local-path /tmp/myrepo
```

Automatically fetches the Issue → runs the agent → submits a PR.

### Multi-agent

```bash
agent multi --task "Fix failing tests" --topology pipeline   # planner → coder ⟲ reviewer
agent multi --task "..." --topology pair                     # coder ⟲ reviewer
agent multi --task "..." --topology debate                   # two plans, then coder
agent multi --task "..." --topology autonomous               # single agent
```

### Eval

```bash
agent eval --engine native --output native.json
agent eval --engine langgraph --output langgraph.json
python -m eval.compare --labels native,langgraph native.json langgraph.json
```

---

## Configuration

Edit `config/default.yaml`:

```yaml
llm:
  provider: deepseek                      # anthropic | openai | deepseek | groq | ollama | vllm
  model: deepseek-v4-flash
  api_key: ${DEEPSEEK_API_KEY}            # read from environment variable
  base_url: https://api.deepseek.com      # fill in for OpenAI-compatible providers; leave blank for anthropic

agent:
  max_steps: 40           # max steps per round
  budget_tokens: 80000    # token budget

context:
  repo_map_budget: 8000   # repo-map injection size
  history_window: 20      # number of history rounds to retain
```

---

## Project Structure

```
coding-agent/
├── agent/              # Core: ReAct main loop, event log, data structures
│   ├── core.py         # Agent class driving the entire run loop
│   ├── decision.py     # Loop / finish / reflection policy
│   ├── plan.py         # Structured numbered checklist
│   ├── orchestrator.py # Multi-agent topologies
│   ├── workflow.py     # Ticket → run → verify → memory
│   ├── task.py         # Task / Action / Observation / RunResult dataclasses
│   ├── event_log.py    # JSONL append-only event stream with replay support
│   └── prompt.py       # System prompt templates
│
├── llm/                # LLM backends
│   ├── base.py         # LLMBackend abstract base class with default stream()
│   ├── anthropic_backend.py   # Claude native (tool_use + streaming)
│   ├── openai_compat.py       # OpenAI / DeepSeek / Groq / Ollama
│   └── router.py       # Select backend from configuration
│
├── tools/              # Tool layer (operations the agent can invoke)
│   ├── base.py         # BaseTool + ToolRegistry
│   ├── file_tool.py    # File read / write / view
│   ├── shell_tool.py   # Shell execution (4-layer safety)
│   ├── search_tool.py  # Text search / file find / symbol locate
│   ├── test_tool.py    # pytest execution + structured result parsing
│   ├── git_tool.py     # git status / diff / add / commit
│   ├── plan_tool.py    # Numbered checklist
│   ├── skill_tool.py   # Load named SOP / SKILL.md playbooks
│   ├── browser_tool.py # web_fetch (http/https; optional Playwright)
│   ├── undo_tool.py    # Snapshot-based undo (git-independent rollback)
│   └── runtime.py      # LocalRuntime / DockerRuntime
│
├── context/            # Context management
│   ├── repo_map.py     # tree-sitter multi-language symbol extraction, repo summary
│   ├── token_budget.py # Token budget allocation and trimming
│   ├── history.py      # Conversation history sliding window
│   ├── rules.py        # AGENTS.md / CLAUDE.md / .cursor/rules
│   ├── skills.py       # SKILL.md SOP catalog
│   └── rag.py          # Hybrid retrieval (optional)
│
├── config/             # default.yaml + schema
├── eval/               # Harness: suite, verifiers, trajectory metrics, report compare
├── entry/              # CLI, chat, FastAPI, MCP, GitHub Issue, web UI
├── docs/               # Architecture, memory, RAG, skills, harness, testing
│
├── tests/              # pytest suite — see docs/TESTING.md for inventory and counts
├── examples/skills/    # Sample SOP playbooks (copy into a repo's .agent/skills/)
├── smoke_test.py       # End-to-end connectivity verification
├── quicksort_task.py   # Example task script
└── USAGE.md            # Full usage guide
```

---

## Key Features

**Multi-model support**
- Anthropic Claude (native tool_use)
- OpenAI, DeepSeek, Groq, Ollama, vLLM (OpenAI-compatible; local quant/serve via Ollama or vLLM)
- Models that don't support function calling (e.g. DeepSeek R1) use a text-parse fallback
- Switch with one line in the config file, or override temporarily via `--model`

**Multi-language Repo-map**
Uses tree-sitter to precisely extract symbols (functions, classes, methods) and generates a repo summary injected into the system prompt. Supports Python / JavaScript / TypeScript / Go / Rust / Java / C++ / C / Ruby.

**Streaming output**
Model thoughts are printed token by token in real time; tool calls are shown immediately — experience close to Claude Code.

**Safety (3-layer)**
- Hard blacklist: `rm -rf /`, `mkfs`, etc. are never executed
- Read-only whitelist: `ls`, `grep`, `git status`, `pytest` etc. execute directly
- Write confirmation: in `--confirm` mode, `git commit`, `pip install` etc. require y/n confirmation

**Docker sandbox**
`--sandbox` flag runs all commands inside a `python:3.11-slim` container with the repo bind-mounted for two-way sync; network disabled by default.

**Reflection mechanism**
- Test failure → automatically triggers a reflection prompt to re-analyze the error
- 6 consecutive steps without file edits → triggers reflection to break exploration loops
- 3 consecutive identical actions → detects an infinite loop and terminates automatically

**Event log**
Each run generates a JSONL log recording all actions / observations / reflections with full replay and statistical analysis support.

**Skills / MCP / web_fetch**
Named SOP playbooks (`.agent/skills/*/SKILL.md`) are catalogued in the prompt and loaded with the `skill` tool. `agent mcp` exposes the same tools to Cursor / Claude Desktop. `web_fetch` reads http(s) pages (stdlib GET; optional Playwright for JS). See [docs/SKILLS.md](docs/SKILLS.md).

**Eval harness**
`agent eval` grades tasks with an independent verifier (not the model's FINISH). Trajectory metrics and `python -m eval.compare` support iteration. See [docs/HARNESS.md](docs/HARNESS.md).

---

## Safety Notes

`--confirm` mode (`run`) and `chat` mode both require confirmation before write operations:

```
  ⚠  Agent wants to run:
     $ git commit -m "fix parser bug"
  Allow? [y/N]
```

`--sandbox` mode executes in a Docker container, fully isolated from the host environment.

---

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (use the venv interpreter — see docs/TESTING.md)
.venv/bin/python -m pytest                     # 2026-08-20: 572 passed, 11 skipped
.venv/bin/python -m pytest tests/test_skills.py
.venv/bin/python -m pytest tests/test_browser_tool.py

# Optional extras
pip install -e ".[rag]"         # numpy / faiss for RAG tests
pip install -e ".[langgraph]"   # LangGraph engine tests
pip install -e ".[server]"      # FastAPI + MCP
pip install -e ".[browser]" && playwright install chromium   # JS-rendered web_fetch

# Optional: tree-sitter support for more languages
pip install tree-sitter-javascript tree-sitter-typescript \
            tree-sitter-go tree-sitter-rust tree-sitter-java

# Optional: accurate token counting
pip install tiktoken
```

Test inventory, extras, and how we count the suite: [docs/TESTING.md](docs/TESTING.md).
Harness / eval loop: [docs/HARNESS.md](docs/HARNESS.md).
Skills / SOP wrapping: [docs/SKILLS.md](docs/SKILLS.md).

---

## Command Reference

```bash
# chat
agent chat [--repo PATH] [--model MODEL] [--sandbox] [-v]

# run
agent run --task TEXT [--repo PATH] [--task-file FILE]
          [--model MODEL] [--confirm] [--sandbox] [--no-stream] [-v]

# log
agent log list [--dir DIR]
agent log show LOG_FILE

# multi
agent multi --task TEXT [--topology pipeline|pair|debate|autonomous] [-i ITER]

# eval
agent eval [--engine native|langgraph] [-o REPORT.json]
python -m eval.compare [--labels a,b] a.json b.json

# mcp / serve / web
agent mcp --repo PATH
agent serve --repo PATH
agent web --repo PATH

# github issue
python -m entry.github_issue \
    -r owner/repo -i ISSUE_NUM -l LOCAL_PATH [--no-pr] [-v]
```

See [USAGE.md](USAGE.md) for detailed usage. Chinese feature overview: [docs/功能说明.md](docs/功能说明.md).

---

## Benchmarks

Two objective, agent-driven benchmarks under [benchmarks/](benchmarks/), both
graded by an independent re-run (the agent's self-reported success is ignored):

```bash
# HumanEval — single-function completion
python -m benchmarks.run_humaneval --limit 20

# SWE-bench Lite — fix a real GitHub issue in a real repo (read→plan→edit→test→repair)
python -m benchmarks.download_swebench --split test        # get the dataset (300 instances)
python -m benchmarks.run_swebench --mock --instances pytest-dev__pytest-5692   # validate pipeline, no key
python -m benchmarks.run_swebench --repos psf/requests --limit 3   # evaluate the model
```

SWE-bench grades with the official "resolved" criterion (all `FAIL_TO_PASS` pass
**and** all `PASS_TO_PASS` stay passing). This host has no Docker, so grading is
best-effort — see [benchmarks/README.md](benchmarks/README.md) for the exact
scope and caveats.
