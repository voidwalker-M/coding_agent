# Coding Agent — Usage Guide

An autonomous coding agent supporting conversational code editing, automatic bug fixing, and test execution. Compatible with Claude, DeepSeek, OpenAI, Groq, and Ollama.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Configuration](#2-configuration)
3. [Three Usage Modes](#3-three-usage-modes)
4. [chat Mode in Depth](#4-chat-mode-in-depth)
5. [run Mode in Depth](#5-run-mode-in-depth)
6. [GitHub Issue Mode](#6-github-issue-mode)
7. [Viewing Run Logs](#7-viewing-run-logs)
8. [Safety Mechanisms](#8-safety-mechanisms)
9. [Docker Sandbox](#9-docker-sandbox)
10. [Tips for Writing Good Task Descriptions](#10-tips-for-writing-good-task-descriptions)
11. [Frequently Asked Questions](#11-frequently-asked-questions)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. Installation

**Requirements:** Python 3.11+, pip

```bash
# Clone the project
git clone <repo-url>
cd coding-agent

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install
pip install -e ".[dev]"

# Verify installation
agent --help
```

**Optional: install additional language parsing support** (for more precise repo-map analysis across more languages)

```bash
pip install \
    tree-sitter-javascript \
    tree-sitter-typescript \
    tree-sitter-go \
    tree-sitter-rust \
    tree-sitter-java \
    tree-sitter-cpp \
    tree-sitter-c \
    tree-sitter-ruby
```

**Optional: install tiktoken** (accurate token counting; recommended when internet is available)

```bash
pip install tiktoken
```

---

## 2. Configuration

### 2.1 Choose a model provider

Edit `config/default.yaml` based on your provider:

**DeepSeek (recommended — cost-effective)**

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-flash        # fast version, suitable for everyday tasks
  # model: deepseek-v4-pro        # flagship version, suitable for complex tasks
  api_key: ${DEEPSEEK_API_KEY}
  base_url: https://api.deepseek.com
```

**Anthropic Claude**

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5
  api_key: ${ANTHROPIC_API_KEY}
  base_url:                        # leave blank
```

**OpenAI**

```yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  base_url:                        # leave blank
```

**Groq (extremely fast, good for debugging)**

```yaml
llm:
  provider: groq
  model: llama3-70b-8192
  api_key: ${GROQ_API_KEY}
  base_url: https://api.groq.com/openai/v1
```

**Ollama (runs locally, free)**

```yaml
llm:
  provider: ollama
  model: llama3               # name of the model you have pulled locally
  api_key:                    # leave blank
  base_url: http://localhost:11434/v1
```

### 2.2 Set your API key

Set the API key as an environment variable (**do not** write the key in plain text in the yaml file):

```bash
# Add to ~/.bashrc or ~/.zshrc for persistence
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# Reload (or open a new terminal)
source ~/.bashrc
```

### 2.3 Verify configuration

```bash
python smoke_test.py
```

Seeing `✅ COMPLETE` means the API is connected and tool execution is working — you're ready to use the agent.

---

## 3. Three Usage Modes

| Mode | Command | Best for |
|------|------|---------|
| **chat** | `agent chat` | Continuous conversation, iterative editing — most common |
| **run** | `agent run --task "..."` | One-shot well-defined tasks, batch processing |
| **GitHub Issue** | `python -m entry.github_issue` | Auto-fix an issue and open a PR |

---

## 4. chat Mode in Depth

### Basic usage

```bash
# Start on the project in the current directory
cd /path/to/your/project
agent chat

# Specify a project directory
agent chat --repo /path/to/project

# Switch model (without changing the config file)
agent chat --model deepseek-v4-pro
agent chat --model gpt-4o --provider openai
```

### Interactive interface

After starting, the interactive interface appears:

```
🤖 Coding Agent — Chat Mode
  Provider : deepseek
  Model    : deepseek-v4-flash
  Repo     : /your/project
  Type your task. Commands: /exit /stats /clear /help

you >
```

Type a task description and press Enter to send. Supports:
- **Backspace** to delete characters
- **↑↓ arrow keys** to navigate input history
- **Ctrl+A** to jump to the beginning of the line, **Ctrl+E** to jump to the end

### Built-in commands

| Command | Description |
|------|------|
| `/exit` or `/quit` | Quit |
| `/stats` | Show session statistics (rounds, steps, token usage) |
| `/clear` | Clear conversation history and start fresh (without exiting) |
| `/help` | Show command help |

### Multi-round conversation example

```
you > Show me what modules are in this project

  Agent working...
  (agent explores file structure; analysis streams out token by token)

  ─── Round 1 · 2 steps · 1,234 tokens · 5.2s ───

you > The parse_date function in utils.py can't handle empty strings — fix it

  Agent working...
  (agent reads the file, modifies the code, runs tests)

  ─── Round 2 · 4 steps · 3,421 tokens · 12.1s ───

you > Add a unit test for that fix

  ─── Round 3 · 3 steps · 2,890 tokens · 9.3s ───

you > /stats

  Session stats:
    Rounds  : 3
    Steps   : 9
    Tokens  : 7,545
```

**Key feature: history is retained after each round.** The agent can see what it did previously in the next round — no need to re-describe context.

### Understanding the output

```
  Agent working...
  (model thinking content streams in real time)    ← real-time token-by-token output

  [1] shell  ls -la                 ← step 1: shell tool call
  ✓                                 ← executed successfully
    main.py utils.py parser.py      ← first few lines of output

  [2] file_read  src/parser.py      ← step 2: file read
  ✓

  [3] file_write  src/parser.py     ← step 3: write changes
  ✓  Written 42 lines

  [4] test  tests/                  ← step 4: run tests
  ✓  5 passed in 0.12s

  ⟳ Reflection (test_failed)        ← automatic reflection on test failure

  ─── Round 2 · 4 steps · 3,421 tokens · 12.1s ───
```

---

## 5. run Mode in Depth

For well-defined tasks that don't require back-and-forth interaction.

### Basic usage

```bash
# Simplest: run in the current directory
agent run --task "Fix all failing tests"

# Specify a repo
agent run --repo /path/to/project --task "Refactor api.py into smaller functions"

# Write the task description in a file (recommended for complex tasks)
agent run --task-file task.txt
```

### All options

```
-r, --repo TEXT       Target repo path (default: current directory)
-t, --task TEXT       Task description (natural language)
-f, --task-file TEXT  Read task description from a file
-m, --model TEXT      Override model name
-p, --provider TEXT   Override provider
    --max-steps INT   Max steps (default 40)
-s, --stream          Streaming output (on by default)
    --confirm         Require user confirmation for dangerous commands
    --sandbox         Run commands in a Docker sandbox
-v, --verbose         Show debug logs
```

### Typical use cases

```bash
# Fix a specific test
agent run --task "tests/test_api.py::test_auth raises KeyError — fix it"

# Add a feature
agent run --task "Add a /health endpoint to src/api.py returning {status: ok, version: 1.0}"

# Refactor
agent run --task "Split functions longer than 50 lines in utils.py into smaller ones, keeping tests passing"

# Safe execution (confirm before dangerous commands)
agent run --task "Clean up the project: delete all .pyc files and __pycache__ directories" --confirm

# Docker sandbox (commands run in the container, host environment unaffected)
agent run --task "Install dependencies and run tests" --sandbox
```

---

## 6. GitHub Issue Mode

Automatically fetches a task description from a GitHub Issue, runs the agent, and creates a PR when done.

### Prerequisites

```bash
# Set your GitHub Token (needs repo permission)
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxx
```

Create one at GitHub → Settings → Developer settings → Personal access tokens, checking the `repo` scope.

### Usage

```bash
python -m entry.github_issue \
    --repo owner/repo-name \
    --issue 42 \
    --local-path /tmp/myrepo
```

**Options:**

```
-r, --repo TEXT         GitHub repository (format: owner/repo)
-i, --issue INTEGER     Issue number
-l, --local-path TEXT   Local path (cloned automatically; used directly if it already exists)
-c, --config TEXT       Config file path
    --no-pr             Fix the code only; do not create a PR
    --base-branch TEXT  PR target branch (default: main)
-v, --verbose           Show detailed logs
```

**Workflow:**

1. Fetch the title and description of Issue #42 as the task
2. Clone the repo to `/tmp/myrepo` (skipped if it already exists)
3. Create a new branch `agent/fix-issue-42-xxxxxxxx`
4. Run the agent on the new branch to complete the task
5. Push the branch to the remote
6. Automatically create a PR with a generated title and description

---

## 7. Viewing Run Logs

Each run generates a JSONL event log in `./logs/` recording the complete run.

### List log files

```bash
agent log list
agent log list --dir ./logs    # specify log directory
```

Example output:

```
Log files in ./logs:

  abc12345_20250525_143022.jsonl  (12.3 KB)
  def67890_20250524_091534.jsonl  (8.7 KB)
```

### View a single run in detail

```bash
agent log show logs/abc12345_20250525_143022.jsonl
```

Example output:

```
Event Log: abc12345_20250525_143022.jsonl
  Total events : 18
  Actions      : 6
  Reflections  : 1
  Tool calls   : {'shell': 2, 'file_read': 1, 'file_write': 1, 'test': 2}
  Final status : task_complete

Events:
  14:30:22  task_start
  14:30:25  action          tool=shell
  14:30:25  observation     status=success
  14:30:28  action          tool=file_read
  ...
  14:30:51  task_complete
```

Log files are standard JSON Lines format (one event per line), analyzable with any tool:

```bash
# View all action thoughts with jq
cat logs/abc12345_*.jsonl | jq 'select(.event_type=="action") | .payload.action.thought'

# Count tool call frequency
cat logs/abc12345_*.jsonl | jq 'select(.event_type=="action") | .payload.action.tool_call.name' | sort | uniq -c
```

---

## 8. Safety Mechanisms

The agent has three protection layers to prevent unintended operations:

### Layer 1: Hard blacklist (never executed, never asks the user)

The following commands are rejected unconditionally:

- `rm -rf /`, `rm -rf ~`
- `mkfs` (disk formatting)
- `dd if=` (disk write)
- `:(){:|:&};:` (fork bomb)
- `chmod -R 777 /`
- `> /dev/sda`

### Layer 2: Read-only whitelist (executed directly, no confirmation needed)

The following commands are considered safe read-only operations and execute immediately:

`ls`, `cat`, `grep`, `find`, `git status`, `git diff`, `git log`,
`pytest`, `python -m pytest`, `echo`, `pwd`, `diff`, `tree`, etc.

### Layer 3: Write confirmation (only in `--confirm` mode)

The following commands require user confirmation:

`rm`, `mv`, `pip install`, `git commit`, `git push`, `curl`, `wget`,
`chmod`, `sudo`, `docker`, output redirection (`>`), etc.

**Default behavior (without `--confirm`)**: Layer 3 is skipped; commands execute directly. Suitable for automation.

**With confirmation (`--confirm`)**: On write operations, you see:

```
  ⚠  Agent wants to run:
     $ git commit -m "fix parser bug"
  Allow? [y/N]
```

**chat mode enables confirmation by default** — every dangerous command asks for approval.

---

## 9. Docker Sandbox

Add `--sandbox` to run all shell commands, tests, and git operations inside a Docker container, fully isolated from the host.

### Prerequisites

Make sure Docker Desktop is installed and running:

```bash
docker --version
docker info    # should produce normal output
```

### Usage

```bash
# Enable sandbox in run mode
agent run --task "Install dependencies and run all tests" --sandbox

# Enable sandbox in chat mode
agent chat --sandbox
```

The first use pulls the `python:3.11-slim` image (~150 MB); subsequent uses reuse it.

### Sandbox properties

- The container is **network-isolated** by default (`--network none`), preventing the agent from making arbitrary network requests
- The repo directory is bind-mounted into the container — **file changes are visible in both directions**
  - Files written on the host are readable in the container
  - Files modified in the container are immediately visible on the host
- The container is **automatically cleaned up** when the session ends

---

## 10. Tips for Writing Good Task Descriptions

The quality of the task description directly determines how effective the agent is.

### Core principle: specific > vague

```bash
# ❌ Too vague — the agent doesn't know where to start
agent run --task "fix bug"

# ✅ Specify the file, symptom, and expected behavior
agent run --task "The parse() function in src/parser.py raises ValueError on empty string input.
It should return None instead. Fix it and add a test case for this in tests/test_parser.py."
```

### Description template

```
The [function/class] in [file/module] [what happens] when [what condition].
It should [expected behavior].
[Optional: which tests to run to verify the fix]
```

### Common task patterns

**Bug fix:**
```
tests/test_api.py::test_auth_token fails with KeyError: 'user_id'.
The issue is likely in the verify_token() function in src/auth.py.
Fix it so the test passes.
```

**Add a feature:**
```
Add a GET /api/v1/health endpoint to src/api.py
that returns JSON: {"status": "ok", "version": "1.0.0", "timestamp": <current UTC time>}.
Also add a test for this endpoint in tests/test_api.py.
```

**Refactor:**
```
The process_data() function in src/utils.py is 200 lines — too long.
Split it into several smaller single-responsibility functions.
Keep all existing tests passing.
Do not change the function's external interface.
```

**Code review style:**
```
Check all Python files under src/ and find:
1. Public functions without type annotations
2. Functions longer than 50 lines
3. Duplicate code blocks
Just produce a list — no modifications needed.
```

### Use a file for complex tasks

```bash
cat > task.txt << 'EOF'
Refactor the src/database.py module:

1. Current problems:
   - DatabaseManager class has 15 methods with unclear responsibilities
   - Connection pool logic and query logic are interleaved
   - No error handling

2. Goals:
   - Split into two classes: ConnectionPool and QueryExecutor
   - Each class should have no more than 8 methods
   - Add appropriate exception handling (using custom exception classes)
   - Keep all existing tests passing

3. Do not change:
   - The external API interface (parts imported by other modules)
   - Anything under the tests/ directory
EOF

agent run --task-file task.txt
```

---

## 11. Frequently Asked Questions

**Q: The agent produces no output and seems stuck**

Run `python smoke_test.py` first to check API connectivity. If the network is fine but it's still stuck, the model may be responding slowly. Add `--verbose` to see detailed logs:
```bash
agent chat --verbose
```

**Q: The agent is looping and repeating the same action**

The built-in loop detection handles this automatically (3 consecutive identical actions triggers a GIVE_UP). To interrupt early, press `Ctrl+C`, then `/clear` to reset history and re-describe the task.

**Q: How does the agent handle test failures?**

The built-in Reflection mechanism kicks in: when tests fail, the agent automatically re-analyzes the error and tries different fix strategies, continuing until the `max_steps` limit is reached.

**Q: The agent modified files but I'm not happy with the changes — how do I undo them?**

The agent does not auto-commit; all changes stay in the working tree. Use git to revert:
```bash
git checkout -- .          # revert all uncommitted changes
git checkout -- src/foo.py # revert a specific file
```

**Q: Token usage is too high**

A few ways to reduce token consumption:
```bash
# Use the flash version instead of pro
agent chat --model deepseek-v4-flash

# Reduce repo-map budget (inject less context)
# Edit config/default.yaml:
context:
  repo_map_budget: 4000    # down from 8000

# Reduce history window
context:
  history_window: 10       # down from 20
```

**Q: The sandbox doesn't have the project's dependencies**

Tell the agent to install them first in the task description:
```bash
agent run --task "First run pip install -r requirements.txt, then run all tests" --sandbox
```
Or use `setup_cmds` to pre-install in the container at startup (requires code-level configuration).

**Q: GitHub Issue mode fails to create a PR**

Check that the `GITHUB_TOKEN` has the `repo` scope. If you only want to fix the code without creating a PR:
```bash
python -m entry.github_issue --repo owner/repo --issue 42 \
    --local-path /tmp/myrepo --no-pr
```

---

## 12. Configuration Reference

Full explanation of `config/default.yaml`:

```yaml
llm:
  provider: deepseek          # model provider
  model: deepseek-v4-flash    # model name
  api_key: ${DEEPSEEK_API_KEY}  # environment variable reference
  base_url: https://api.deepseek.com  # fill in for OpenAI-compatible providers
  max_tokens: 4096            # maximum output token count

agent:
  max_steps: 40               # max steps per round (stops when exceeded)
  budget_tokens: 80000        # token budget per round
  log_dir: ./logs             # log directory

tools:
  shell:
    timeout: 30               # shell command timeout in seconds
    max_output_tokens: 8000   # output truncation length (prevents long output from flooding context)
  file:
    max_view_lines: 100       # max lines shown per file_view call

context:
  repo_map_budget: 8000       # max tokens for repo-map injected into the system prompt
  history_window: 20          # number of most recent conversation rounds to retain
```

### Multiple environment configs

You can maintain multiple config files and specify them with `-c`:

```bash
# Daily development: use flash (fast and cheap)
agent chat -c config/dev.yaml

# Complex tasks: use pro
agent run --task "..." -c config/pro.yaml
```

Example `config/dev.yaml`:

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-flash
  api_key: ${DEEPSEEK_API_KEY}
  base_url: https://api.deepseek.com

agent:
  max_steps: 20               # fewer steps during development to save time
  budget_tokens: 40000

context:
  repo_map_budget: 4000
  history_window: 10
```

---

## Quick Reference Card

```bash
# Install
pip install -e ".[dev]"

# Set key
export DEEPSEEK_API_KEY=sk-xxx

# Verify
python smoke_test.py

# Daily use
cd your-project
agent chat                          # start a conversation
agent chat --model deepseek-v4-pro  # switch model

# One-shot tasks
agent run --task "fix the failing tests"
agent run --task-file task.txt

# Safety options
agent run --task "..." --confirm    # confirm before dangerous commands
agent run --task "..." --sandbox    # Docker sandbox

# GitHub Issue
export GITHUB_TOKEN=ghp_xxx
python -m entry.github_issue -r owner/repo -i 42 -l /tmp/repo

# View logs
agent log list
agent log show logs/xxx.jsonl

# In-session commands
# /exit   quit
# /stats  view statistics
# /clear  clear history
# /help   show help
```
