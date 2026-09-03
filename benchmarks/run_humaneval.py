"""
benchmarks/run_humaneval.py

Run the coding agent against the HumanEval benchmark using the existing
eval.harness.EvalHarness (independent PytestVerifier grading).

Two modes:
  --mock   : no LLM/API key needed. A scripted backend writes each problem's
             canonical solution, proving the download -> adapter -> harness ->
             verifier pipeline end-to-end on real data.
  (real)   : builds the configured LLM backend exactly like `agent eval` and
             lets the agent solve each problem for itself. Requires the provider
             API key to be exported (e.g. GPT_OSS_API_KEY for the UF proxy).

Examples:
    python -m benchmarks.run_humaneval --mock --limit 10
    python -m benchmarks.run_humaneval --limit 20 -o benchmarks/humaneval_report.json
    python -m benchmarks.run_humaneval --limit 20 -k 1 --model gpt-oss-120b
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
from benchmarks.humaneval import canonical_solution_file, load_humaneval, load_records
from eval.harness import EvalHarness
from llm.base import MockBackend


def _mock_factory(id_to_solution: dict[str, str]):
    """Factory whose agent writes the canonical solution for its task, then finishes."""
    from tools.base import ToolRegistry
    from tools.file_tool import FileWriteTool
    from tools.test_tool import PytestTool

    def factory(spec, repo_path):
        content = id_to_solution[spec.id]
        script = [
            Action(ActionType.TOOL_CALL, "write canonical solution",
                   tool_call=ToolCall("file_write", {"path": "solution.py", "content": content})),
            Action(ActionType.FINISH, "done", message="done"),
        ]
        registry = ToolRegistry().register(FileWriteTool()).register(PytestTool())
        return Agent(MockBackend(script), registry, AgentConfig(max_steps=spec.max_steps))

    return factory


def _real_factory(config, retriever_kind: str, engine: str, max_steps_override):
    """Factory that builds the configured LLM-backed agent (mirrors `agent eval`)."""
    from entry.cli import _build_registry, _build_retriever
    from llm.router import create_backend_from_config

    backend = create_backend_from_config({
        "provider":   config.llm.provider,
        "model":      config.llm.model,
        "api_key":    config.llm.api_key or None,
        "base_url":   config.llm.base_url or None,
        "max_tokens": config.llm.max_tokens,
    })

    def factory(spec, repo_path):
        registry = _build_registry(config, repo_path=repo_path)
        rag = _build_retriever(repo_path, retriever_kind)
        agent_cfg = AgentConfig(
            max_steps=spec.max_steps if max_steps_override is None else max_steps_override,
            budget_tokens=config.agent.budget_tokens,
            retriever=rag,
        )
        if engine == "langgraph":
            from agent.langgraph_loop import LangGraphAgent
            return LangGraphAgent(backend, registry, agent_cfg)
        return Agent(backend, registry, agent_cfg)

    return factory


def main() -> int:
    ap = argparse.ArgumentParser(description="Run HumanEval through the agent eval harness.")
    ap.add_argument("--mock", action="store_true", help="Use canonical solutions (no API key needed)")
    ap.add_argument("--limit", type=int, default=None, help="Number of problems (default: all 164)")
    ap.add_argument("--start", type=int, default=0, help="Start index into the dataset")
    ap.add_argument("-k", "--attempts", type=int, default=1, help="Attempts per problem (pass@1 / pass@k)")
    ap.add_argument("--max-steps", type=int, default=None, help="Override per-task max steps")
    ap.add_argument("-m", "--model", default=None, help="Override model")
    ap.add_argument("-p", "--provider", default=None, help="Override provider")
    ap.add_argument("-e", "--engine", choices=["native", "langgraph"], default="native")
    ap.add_argument("-R", "--retriever", choices=["none", "rag"], default="none")
    ap.add_argument("-o", "--output", default=None, help="Save JSON report here")
    ap.add_argument("--results-dir", default="./eval_runs/humaneval", help="Workdir/log root")
    ap.add_argument("--keep", action="store_true", help="Keep per-task workdirs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s", datefmt="%H:%M:%S",
    )

    specs = load_humaneval(limit=args.limit, start=args.start,
                           max_steps=(args.max_steps or 12))

    if args.mock:
        recs = load_records()[args.start:(args.start + args.limit) if args.limit else None]
        id_to_solution = {
            spec.id: canonical_solution_file(rec) for spec, rec in zip(specs, recs)
        }
        factory = _mock_factory(id_to_solution)
        model_name = "mock-model"
        header = "HumanEval — MOCK (canonical solutions, pipeline validation)"
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
        factory = _real_factory(config, args.retriever, args.engine, args.max_steps)
        model_name = config.llm.model
        header = f"HumanEval — {config.llm.provider}/{config.llm.model}"

    print(f"\n🧪 {header}")
    print(f"   problems={len(specs)}  attempts={args.attempts}  engine={args.engine}  "
          f"retriever={args.retriever}\n")

    def _progress(r) -> None:
        verdict = "PASS" if r.passed else "FAIL"
        print(f"  [{verdict}] {r.task_id:<34} "
              f"status={r.agent_status:<10} steps={r.steps:>2} tokens={r.tokens:>6} "
              f"{r.elapsed:>5.1f}s — {r.detail}")

    harness = EvalHarness(
        agent_factory=factory,
        results_dir=args.results_dir,
        keep_workdirs=args.keep,
        on_result=_progress,
        model_name=model_name,
    )
    report = harness.run_suite(specs, attempts=args.attempts)

    print("\n" + report.format_table() + "\n")
    if args.output:
        report.save_json(args.output)
        print(f"  Report saved to {args.output}\n")

    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
