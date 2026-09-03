"""
benchmarks/run_swebench.py

Run the coding agent against SWE-bench (Lite) using the existing
eval.harness.EvalHarness (independent SWEbenchVerifier grading).

Two modes:
  --mock   : no LLM/API key needed. A scripted backend applies each instance's
             gold patch, proving the download -> clone -> edit -> test_patch ->
             grade pipeline end-to-end on real data.
  (real)   : builds the configured LLM backend exactly like `agent eval` and lets
             the agent read the issue, explore the repo, and fix it for itself.
             Requires the provider API key to be exported (e.g. GPT_OSS_API_KEY).

Grading mirrors the official SWE-bench "resolved" criterion (all FAIL_TO_PASS
pass AND all PASS_TO_PASS stay passing). No Docker: environments are built
best-effort per (repo, version) on the host, so some heavy repos may report
ENV_ERROR — see benchmarks/swebench.py for the full caveat.

Examples:
    # Validate the whole pipeline on light repos, no API key:
    python -m benchmarks.run_swebench --mock \
        --instances psf__requests-2317,pallets__flask-4045

    # Real run on a small slice (export your key first):
    export GPT_OSS_API_KEY=...
    python -m benchmarks.run_swebench --repos psf/requests --limit 3 \
        -o benchmarks/swebench_report.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.core import Agent, AgentConfig
from agent.task import Action, ActionType, ToolCall
from benchmarks.swebench import gold_patch, load_swebench, load_records
from eval.harness import EvalHarness


def _mock_factory(id_to_patch: dict[str, str]):
    """Factory whose agent applies the instance's gold patch, then finishes."""
    from tools.base import ToolRegistry
    from tools.file_tool import FileWriteTool
    from tools.shell_tool import ShellTool

    def factory(spec, repo_path):
        patch = id_to_patch[spec.id]
        script = [
            Action(ActionType.TOOL_CALL, "write gold patch to disk",
                   tool_call=ToolCall("file_write",
                                      {"path": ".gold.patch", "content": patch})),
            Action(ActionType.TOOL_CALL, "apply gold patch",
                   tool_call=ToolCall("shell",
                                      {"cmd": "git apply --whitespace=nowarn .gold.patch "
                                              "&& rm -f .gold.patch && echo GOLD_APPLIED"})),
            Action(ActionType.FINISH, "done", message="applied gold patch"),
        ]
        from llm.base import MockBackend
        registry = ToolRegistry().register(FileWriteTool()).register(ShellTool())
        return Agent(MockBackend(script), registry, AgentConfig(max_steps=spec.max_steps))

    return factory


def _real_factory(config, retriever_kind: str, engine: str, max_steps_override,
                  id_to_rec: dict | None = None):
    """Factory that builds the configured LLM-backed agent (mirrors `agent eval`)."""
    from entry.cli import _build_registry, _build_retriever
    from llm.router import create_backend_from_config

    backend = create_backend_from_config({
        "provider":   config.llm.provider,
        "model":      config.llm.model,
        "api_key":    config.llm.api_key or None,
        "base_url":   config.llm.base_url or None,
        "max_tokens": config.llm.max_tokens,
        # gpt-oss-120b frequently ends a turn after its reasoning channel with an
        # empty response; resample generously so one unlucky turn doesn't abort a
        # solvable task (see llm/openai_compat.py).
        "max_empty_retries": 6,
    })

    def factory(spec, repo_path):
        # Run the agent's test/shell tools inside THIS instance's virtualenv, so it
        # can actually import the project and run its suite to check its own fix.
        # Without this the agent edits blind: `pytest` would run under the agent's
        # own interpreter, which lacks the target repo's (often old) dependencies.
        runtime = None
        if id_to_rec and spec.id in id_to_rec:
            from benchmarks.swebench import venv_runtime_for
            runtime = venv_runtime_for(id_to_rec[spec.id])
        registry = _build_registry(
            config, runtime=runtime, repo_path=repo_path, block_test_edits=True,
        )
        rag = _build_retriever(repo_path, retriever_kind)
        agent_cfg = AgentConfig(
            max_steps=spec.max_steps if max_steps_override is None else max_steps_override,
            budget_tokens=config.agent.budget_tokens,
            retriever=rag,
            # SWE-bench tasks always run on a cloned git repo and always require a
            # code change, so refuse a "finish" that edited nothing — pushes the
            # model to actually apply its fix instead of just describing it.
            require_edit_before_finish=True,
            # Do NOT require a green test before the coder can finish.
            # That gate burned the step budget on untested patches (v2: 1/14).
            # Pair/pipeline still enforce tests on the reviewer role.
            require_test_before_finish=False,
            # Force search_text/find_symbol before the first edit so the model
            # does not patch a nearby file that merely looks related.
            require_locate_before_edit=True,
        )
        if engine in ("pair", "pipeline"):
            from agent.orchestrator import Orchestrator, OrchestratorAgent
            orch = Orchestrator(
                backend, registry, agent_cfg,
                max_iterations=2, topology=engine,
            )
            return OrchestratorAgent(orch)
        if engine == "langgraph":
            from agent.langgraph_loop import LangGraphAgent
            return LangGraphAgent(backend, registry, agent_cfg)
        return Agent(backend, registry, agent_cfg)

    return factory


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SWE-bench Lite through the agent eval harness.")
    ap.add_argument("--mock", action="store_true", help="Apply gold patches (no API key needed)")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--limit", type=int, default=None, help="Number of instances")
    ap.add_argument("--start", type=int, default=0, help="Start index into the (filtered) dataset")
    ap.add_argument("--instances", default=None,
                    help="Comma-separated instance ids (overrides --repos/--limit)")
    ap.add_argument("--repos", default=None,
                    help="Comma-separated repo filter, e.g. 'psf/requests,pallets/flask'")
    ap.add_argument("-k", "--attempts", type=int, default=1, help="Attempts per instance (pass@k)")
    ap.add_argument("--max-steps", type=int, default=None, help="Override per-task max steps")
    ap.add_argument("--test-timeout", type=int, default=1200, help="Per test-run timeout (s)")
    ap.add_argument("-m", "--model", default=None, help="Override model")
    ap.add_argument("-p", "--provider", default=None, help="Override provider")
    ap.add_argument("-e", "--engine",
                    choices=["native", "langgraph", "pair", "pipeline"],
                    default="native",
                    help="native ReAct, LangGraph, or multi-agent (pair=coder⟲reviewer)")
    ap.add_argument("-R", "--retriever", choices=["none", "rag"], default="none")
    ap.add_argument("-o", "--output", default=None, help="Save JSON report here")
    ap.add_argument("--results-dir", default="./eval_runs/swebench", help="Workdir/log root")
    ap.add_argument("--keep", action="store_true", help="Keep per-task workdirs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s", datefmt="%H:%M:%S",
    )

    instances = [s.strip() for s in args.instances.split(",")] if args.instances else None
    repos = [s.strip() for s in args.repos.split(",")] if args.repos else None

    specs, recs = load_swebench(
        split=args.split, limit=args.limit, start=args.start,
        instances=instances, repos=repos,
        max_steps=(args.max_steps or 40), test_timeout=args.test_timeout,
    )
    if not specs:
        print("No instances selected. Check --instances/--repos/--split.", file=sys.stderr)
        return 2

    if args.mock:
        id_to_patch = {r["instance_id"]: gold_patch(r) for r in recs}
        factory = _mock_factory(id_to_patch)
        model_name = "mock-gold-patch"
        header = "SWE-bench Lite — MOCK (gold patches, pipeline validation)"
    else:
        from config.schema import load_config, merge_cli_overrides
        config = load_config(None)
        config = merge_cli_overrides(config, provider=args.provider, model=args.model,
                                     max_steps=args.max_steps)
        if not (config.llm.api_key or "").strip():
            print("ERROR: no API key resolved for provider "
                  f"'{config.llm.provider}'. Export it first, e.g.:\n"
                  "  export GPT_OSS_API_KEY=your-key", file=sys.stderr)
            return 2
        factory = _real_factory(config, args.retriever, args.engine, args.max_steps,
                                id_to_rec={r["instance_id"]: r for r in recs})
        model_name = config.llm.model
        header = f"SWE-bench Lite — {config.llm.provider}/{config.llm.model}"

    print(f"\n🧪 {header}")
    print(f"   instances={len(specs)}  split={args.split}  attempts={args.attempts}  "
          f"engine={args.engine}  retriever={args.retriever}\n")

    def _progress(r) -> None:
        detail = r.detail or ""
        if r.passed:
            tag = "RESOLVED"
        elif "UNSUPPORTED" in detail:
            tag = "UNSUPPORTED"
        elif "ENV_ERROR" in detail:
            tag = "ENV_ERR"
        elif "PATCH_ERROR" in detail:
            tag = "PATCH_ERR"
        else:
            tag = "FAIL"
        print(f"  [{tag:<9}] {r.task_id:<28} "
              f"status={r.agent_status:<10} steps={r.steps:>2} tokens={r.tokens:>7} "
              f"{r.elapsed:>6.1f}s — {r.detail}")

    harness = EvalHarness(
        agent_factory=factory,
        results_dir=args.results_dir,
        keep_workdirs=args.keep,
        on_result=_progress,
        model_name=model_name,
    )
    report = harness.run_suite(specs, attempts=args.attempts)

    print("\n" + report.format_table())
    env_errors = sum(1 for r in report.results if "ENV_ERROR" in (r.detail or ""))
    patch_errors = sum(1 for r in report.results if "PATCH_ERROR" in (r.detail or ""))
    unsupported = sum(1 for r in report.results if "UNSUPPORTED" in (r.detail or ""))
    gradable = report.total - env_errors - patch_errors - unsupported
    if env_errors or patch_errors or unsupported:
        print(f"  note (no Docker): {unsupported} unsupported repo(s) (django/sympy/sphinx use "
              f"non-pytest runners), {env_errors} host-env-setup failure(s), "
              f"{patch_errors} test_patch apply issue(s).")
        if gradable:
            resolved_gradable = sum(
                1 for r in report.results if r.passed)
            print(f"  among {gradable} locally-gradable instance(s): "
                  f"{resolved_gradable} resolved = {resolved_gradable / gradable:.0%}")
    print()
    if args.output:
        report.save_json(args.output)
        print(f"  Report saved to {args.output}\n")

    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
