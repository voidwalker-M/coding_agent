"""Eval report comparison helper."""

from eval.compare import compare_summaries


def test_compare_summaries_aligns_columns():
    reports = {
        "native": {"summary": {"success_rate": 0.5, "avg_steps": 4.0, "total_tool_errors": 1}},
        "langgraph": {"summary": {"success_rate": 0.75, "avg_steps": 5.0, "total_tool_errors": 0}},
    }
    text = compare_summaries(reports)
    assert "success_rate" in text
    assert "native" in text and "langgraph" in text
    assert "0.5" in text and "0.75" in text
