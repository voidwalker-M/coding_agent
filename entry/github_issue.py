"""
entry/github_issue.py

GitHub Issue auto-fix entry point.

Workflow:
1. Fetch the Issue title + body as the task description
2. Clone or use an existing local repo
3. Run the agent on a new branch
4. After the agent completes, create a PR (optional)

Usage:
    python -m entry.github_issue \
        --repo owner/repo \
        --issue 42 \
        --local-path /tmp/myrepo

Dependencies:
    pip install PyGithub gitpython
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GitHub operations
# ---------------------------------------------------------------------------

def _get_github_client():
    """Initialize a PyGithub client, reading the token from the environment variable."""
    try:
        from github import Github
    except ImportError:
        raise ImportError("PyGithub not installed. Run: pip install PyGithub")

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN environment variable is not set.\n"
            "Create a token at https://github.com/settings/tokens"
        )
    return Github(token)


def fetch_issue(repo_name: str, issue_number: int) -> tuple[str, str, str]:
    """
    Fetch the content of a GitHub Issue.

    Returns:
        (title, body, html_url)
    """
    gh = _get_github_client()
    repo = gh.get_repo(repo_name)
    issue = repo.get_issue(issue_number)
    return issue.title, issue.body or "", issue.html_url


def create_pull_request(
    repo_name: str,
    branch: str,
    title: str,
    body: str,
    base: str = "main",
) -> str:
    """
    Create a PR and return the PR URL.

    Args:
        repo_name: "owner/repo" format
        branch:    source branch (the branch where the agent made changes)
        title:     PR title
        body:      PR description
        base:      target branch; default is main
    """
    gh = _get_github_client()
    repo = gh.get_repo(repo_name)

    # Check whether the base branch exists; fall back to master if not
    try:
        repo.get_branch(base)
    except Exception:
        base = "master"

    pr = repo.create_pull(
        title=title,
        body=body,
        head=branch,
        base=base,
    )
    return pr.html_url


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: str) -> tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd,
        )
        output = (proc.stdout + proc.stderr).strip()
        return proc.returncode == 0, output
    except Exception as e:
        return False, str(e)


def clone_repo(repo_name: str, local_path: str) -> None:
    """Clone the repo to a local path (skips if it already exists)."""
    path = Path(local_path)
    if path.exists() and (path / ".git").exists():
        logger.info("Repo already exists at %s, skipping clone", local_path)
        return

    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        url = f"https://{token}@github.com/{repo_name}.git"
    else:
        url = f"https://github.com/{repo_name}.git"

    click.echo(f"Cloning {repo_name} → {local_path} ...")
    ok, out = _run_git(["clone", url, local_path], cwd="/tmp")
    if not ok:
        raise RuntimeError(f"git clone failed: {out}")


def create_branch(local_path: str, branch: str) -> None:
    """Create and switch to a new branch."""
    ok, out = _run_git(["checkout", "-b", branch], cwd=local_path)
    if not ok:
        # Branch already exists; switch to it
        _run_git(["checkout", branch], cwd=local_path)


def push_branch(local_path: str, branch: str) -> None:
    """Push the branch to the remote."""
    ok, out = _run_git(
        ["push", "--set-upstream", "origin", branch],
        cwd=local_path,
    )
    if not ok:
        raise RuntimeError(f"git push failed: {out}")


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------

def run_on_issue(
    repo_name: str,
    issue_number: int,
    local_path: str,
    config_path: str | None = None,
    create_pr: bool = True,
    base_branch: str = "main",
) -> int:
    """
    Fetch the Issue, run the agent, and create a PR.

    Returns:
        0 if success, 1 if failed
    """
    from config.schema import load_config
    from agent.core import Agent, AgentConfig
    from agent.event_log import EventLog
    from agent.task import Task
    from llm.router import create_backend_from_config

    config = load_config(config_path)

    # 1. Fetch the Issue
    click.echo(f"\nFetching issue #{issue_number} from {repo_name} ...")
    try:
        title, body, issue_url = fetch_issue(repo_name, issue_number)
    except Exception as e:
        click.echo(f"Error fetching issue: {e}", err=True)
        return 1

    click.echo(f"  Title: {title}")
    description = f"Fix GitHub Issue #{issue_number}: {title}\n\n{body}"

    # 2. Clone if needed
    try:
        clone_repo(repo_name, local_path)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        return 1

    # 3. Create a working branch
    branch = f"agent/fix-issue-{issue_number}-{int(time.time())}"
    create_branch(local_path, branch)
    click.echo(f"  Branch: {branch}")

    # 4. Build the agent
    try:
        backend = create_backend_from_config({
            "provider": config.llm.provider,
            "model":    config.llm.model,
            "api_key":  config.llm.api_key or None,
            "base_url": config.llm.base_url or None,
            "max_tokens": config.llm.max_tokens,
        })
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return 1

    from entry.cli import _build_registry
    registry = _build_registry(config)

    agent_config = AgentConfig(
        max_steps=config.agent.max_steps,
        budget_tokens=config.agent.budget_tokens,
    )
    agent = Agent(backend, registry, agent_config)

    task = Task(
        description=description,
        repo_path=local_path,
        issue_url=issue_url,
        max_steps=config.agent.max_steps,
        budget_tokens=config.agent.budget_tokens,
    )

    # 5. Run the agent
    click.echo(f"\nRunning agent on issue #{issue_number} ...")
    t0 = time.time()
    with EventLog.create(task, log_dir=config.agent.log_dir) as log:
        result = agent.run(task, log)

    elapsed = time.time() - t0
    click.echo(f"  Status : {result.status.value}")
    click.echo(f"  Steps  : {result.steps_taken}")
    click.echo(f"  Tokens : {result.total_tokens:,}")
    click.echo(f"  Time   : {elapsed:.1f}s")

    if not result.is_success():
        click.echo(f"  Agent did not complete the task.", err=True)
        return 1

    # 6. Push the branch
    if create_pr:
        click.echo("\nPushing branch ...")
        try:
            push_branch(local_path, branch)
        except RuntimeError as e:
            click.echo(f"Warning: push failed: {e}", err=True)
            click.echo("Changes are committed locally. Push manually to create a PR.")
            return 0

        # 7. Create the PR
        pr_title = f"[Agent] Fix issue #{issue_number}: {title}"
        pr_body = (
            f"Fixes #{issue_number}\n\n"
            f"This PR was automatically generated by the coding agent.\n\n"
            f"## Summary\n{result.summary}\n\n"
            f"## Task\n{description[:500]}"
        )
        try:
            pr_url = create_pull_request(
                repo_name=repo_name,
                branch=branch,
                title=pr_title,
                body=pr_body,
                base=base_branch,
            )
            click.echo(f"\n✓ PR created: {pr_url}\n")
        except Exception as e:
            click.echo(f"Warning: PR creation failed: {e}", err=True)
            click.echo(f"Branch pushed. Create PR manually from branch: {branch}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--repo", "-r", required=True, help="GitHub repo (owner/repo)")
@click.option("--issue", "-i", required=True, type=int, help="Issue number")
@click.option(
    "--local-path", "-l", required=True,
    help="Local path to clone/use the repo",
)
@click.option("--config", "-c", default=None, help="Config YAML path")
@click.option("--no-pr", is_flag=True, help="Skip PR creation")
@click.option("--base-branch", default="main", help="Base branch for PR (default: main)")
@click.option("--verbose", "-v", is_flag=True)
def main(
    repo: str,
    issue: int,
    local_path: str,
    config: str | None,
    no_pr: bool,
    base_branch: str,
    verbose: bool,
) -> None:
    """Run the coding agent on a GitHub issue and create a PR."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )
    sys.exit(run_on_issue(
        repo_name=repo,
        issue_number=issue,
        local_path=local_path,
        config_path=config,
        create_pr=not no_pr,
        base_branch=base_branch,
    ))


if __name__ == "__main__":
    main()
