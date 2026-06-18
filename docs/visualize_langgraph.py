#!/usr/bin/env python3
"""
docs/visualize_langgraph.py

Render the LangGraphAgent state graph using LangGraph's own drawing utilities.

This rebuilds the *same* node/edge topology that agent/langgraph_loop.py:run()
compiles (agent ⇄ tools, with a conditional exit to END), then dumps:
  - ASCII art to stdout
  - Mermaid text  -> docs/langgraph_graph.mmd
  - PNG (if the optional mermaid renderer is reachable) -> docs/langgraph_graph.png

Run:
    pip install -e ".[langgraph]"
    python docs/visualize_langgraph.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph


class _State(TypedDict, total=False):
    step: int
    status: str


def build_graph():
    """Same topology as LangGraphAgent.run() — bodies are stubs, only edges matter."""
    def agent_node(state: _State) -> dict:
        return {}

    def tool_node(state: _State) -> dict:
        return {}

    def route_after_agent(state: _State) -> str:
        # real impl: end on status!=running / max_steps / non-tool action; else "tools"
        return "tools"

    g = StateGraph(_State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile()


def main() -> None:
    out_dir = Path(__file__).parent
    graph = build_graph().get_graph()

    # 1) ASCII to terminal
    try:
        print(graph.draw_ascii())
    except Exception as exc:  # draw_ascii needs `grandalf`
        print(f"(ascii skipped: {exc})")

    # 2) Mermaid text (always works, no extra deps)
    mmd = graph.draw_mermaid()
    (out_dir / "langgraph_graph.mmd").write_text(mmd, encoding="utf-8")
    print(f"\nMermaid written to {out_dir / 'langgraph_graph.mmd'}\n")
    print(mmd)

    # 3) PNG (needs network for mermaid.ink, or a local renderer)
    try:
        png = graph.draw_mermaid_png()
        (out_dir / "langgraph_graph.png").write_bytes(png)
        print(f"PNG written to {out_dir / 'langgraph_graph.png'}")
    except Exception as exc:
        print(f"(png skipped: {exc})")


if __name__ == "__main__":
    main()
