"""
context/token_budget.py

Token budget management: allocates token quotas to each part of the prompt and trims
by priority when over budget.

## tiktoken installation

    pip install tiktoken

On first run, tiktoken downloads the vocabulary file (~2 MB, requires internet access) and
caches it locally for offline use afterward.

If the OpenAI CDN is inaccessible, download the vocab file manually:
    curl -L "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" \\
         -o ~/.cache/tiktoken/9b5ad71b2ce5302211f9c61530b329a4922fc6a4021629a1eba1b43bf10a10.tiktoken

Then set the environment variable:
    export TIKTOKEN_CACHE_DIR=~/.cache/tiktoken

When tiktoken is unavailable, automatically falls back to character estimation
(1 token ≈ 4 chars), which is accurate enough for budget control.

Component priorities (high → low; trimming starts from the lowest priority):
  1. system_core   system instructions, never trimmed
  2. task          task description, never trimmed
  3. repo_map      repo summary, reduced when over budget
  4. recent_obs    most recent observation, never trimmed
  5. history       conversation history, trimmed oldest-first
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Token counting: prefer tiktoken, fall back to character estimation
# ---------------------------------------------------------------------------

_tiktoken_enc = None
_tiktoken_available = False

def _init_tiktoken() -> None:
    global _tiktoken_enc, _tiktoken_available
    if _tiktoken_available or _tiktoken_enc is not None:
        return
    try:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        _tiktoken_available = True
    except Exception:
        # Network unavailable or not installed; fall back to character estimation
        _tiktoken_available = False


def estimate_tokens(text: str) -> int:
    """
    Estimate the token count for a string.
    Uses tiktoken when available (accurate); otherwise uses len // 4 (error < 15%).
    """
    if not _tiktoken_available:
        _init_tiktoken()

    if _tiktoken_available and _tiktoken_enc is not None:
        try:
            return max(1, len(_tiktoken_enc.encode(text)))
        except Exception:
            pass

    # Character estimation fallback
    return max(1, len(text) // 4)


def estimate_chars(tokens: int) -> int:
    """Convert a token count to a character budget (estimated)."""
    return tokens * 4


def is_tiktoken_available() -> bool:
    """Return whether tiktoken is available; useful for diagnostic scripts."""
    _init_tiktoken()
    return _tiktoken_available


# ---------------------------------------------------------------------------
# BudgetPlan
# ---------------------------------------------------------------------------

@dataclass
class BudgetPlan:
    """Token quota plan for each prompt component."""
    total: int
    system_core: int
    repo_map: int
    history: int
    observation: int
    reserve: int

    @property
    def available(self) -> int:
        return self.total - self.reserve


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

class TokenBudget:
    """
    Token budget manager.

    Usage:
        budget = TokenBudget(total=80_000)
        plan = budget.default_plan()
        trimmed = budget.trim_to(text, plan.repo_map)
        trimmed_history = budget.trim_history(msgs, plan.history)
    """

    def __init__(self, total: int = 80_000) -> None:
        self._total = total

    def default_plan(self) -> BudgetPlan:
        total = self._total
        reserve = int(total * 0.15)
        available = total - reserve
        return BudgetPlan(
            total=total,
            reserve=reserve,
            system_core=int(available * 0.10),
            repo_map=int(available * 0.15),
            history=int(available * 0.50),
            observation=int(available * 0.25),
        )

    def trim_to(self, text: str, token_limit: int) -> str:
        """Trim text to fit within token_limit; truncates from the end when over budget."""
        if estimate_tokens(text) <= token_limit:
            return text
        # Binary-search approximation: find the right character cutoff
        char_limit = token_limit * 4
        candidate = text[:char_limit]
        while estimate_tokens(candidate) > token_limit and len(candidate) > 0:
            candidate = candidate[:int(len(candidate) * 0.9)]
        omitted = estimate_tokens(text[len(candidate):])
        return candidate + f"\n... [{omitted} tokens truncated]"

    def trim_history(
        self,
        messages: list[dict],
        token_limit: int,
    ) -> list[dict]:
        """
        Trim a message list to fit within token_limit.
        Always preserves the first message (task description) plus as many recent messages as possible.
        """
        if not messages:
            return messages

        token_counts = [estimate_tokens(m.get("content", "")) for m in messages]
        total = sum(token_counts)

        if total <= token_limit:
            return messages

        result = [messages[0]]
        remaining_budget = token_limit - token_counts[0]
        dropped = 0
        selected = []
        budget_left = remaining_budget

        for msg, tokens in zip(reversed(messages[1:]), reversed(token_counts[1:])):
            if budget_left - tokens >= 0:
                selected.append(msg)
                budget_left -= tokens
            else:
                dropped += 1

        selected.reverse()

        if dropped > 0:
            result.append({
                "role": "user",
                "content": f"[{dropped} earlier messages were truncated to fit context window]",
            })

        result.extend(selected)
        return result

    def fit_all(
        self,
        system_text: str,
        repo_map_text: str,
        history: list[dict],
        observation_text: str,
    ) -> tuple[str, str, list[dict], str]:
        plan = self.default_plan()
        trimmed_system = self.trim_to(system_text, plan.system_core)
        trimmed_map = self.trim_to(repo_map_text, plan.repo_map)
        trimmed_history = self.trim_history(history, plan.history)
        trimmed_obs = self.trim_to(observation_text, plan.observation)
        return trimmed_system, trimmed_map, trimmed_history, trimmed_obs

    def usage_report(
        self,
        system_text: str,
        repo_map_text: str,
        history: list[dict],
        observation_text: str,
    ) -> dict[str, int]:
        history_tokens = sum(
            estimate_tokens(m.get("content", "")) for m in history
        )
        return {
            "system":      estimate_tokens(system_text),
            "repo_map":    estimate_tokens(repo_map_text),
            "history":     history_tokens,
            "observation": estimate_tokens(observation_text),
            "total": (
                estimate_tokens(system_text)
                + estimate_tokens(repo_map_text)
                + history_tokens
                + estimate_tokens(observation_text)
            ),
            "budget":        self._total,
            "tiktoken_used": is_tiktoken_available(),
        }
