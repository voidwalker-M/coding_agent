"""
llm/openai_compat.py

OpenAI-compatible backend. Covers:
- OpenAI (api.openai.com)
- DeepSeek (api.deepseek.com) — deepseek-chat supports function calling; R1 does not
- Groq (api.groq.com)
- Ollama (localhost:11434/v1)

All providers use the openai SDK; switching only requires changing base_url + api_key.

When function calling is unsupported (e.g. DeepSeek R1), falls back to text parsing:
extracts a JSON-format tool call from the LLM's raw text output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.task import Action, ActionType, ToolCall
from llm.base import LLMBackend, LLMMessage, LLMResponse, LLMToolSchema

logger = logging.getLogger(__name__)

# Models that do not support function calling (prefix match)
_NO_FUNCTION_CALLING: tuple[str, ...] = (
    "deepseek-reasoner",    # DeepSeek R1
    "deepseek-r1",
)


class OpenAICompatBackend(LLMBackend):
    """
    OpenAI-compatible API backend.

    Args:
        model:      model name, e.g. "gpt-4o", "deepseek-chat", "llama3-70b-8192"
        api_key:    API key
        base_url:   API base URL; None uses the official OpenAI endpoint
        max_tokens: maximum output token count
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_tokens: int = 4096,
        max_empty_retries: int = 2,
        request_logprobs: bool = False,
    ) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._model = model
        self._max_tokens = max_tokens
        # Reasoning models (e.g. gpt-oss-120b) intermittently end their turn after
        # the analysis/reasoning channel without emitting any content or tool call
        # — a non-deterministic empty response. Observed empirically to happen on
        # a large fraction of turns via the LiteLLM/UF proxy, so a low retry count
        # lets a single unlucky turn abort an otherwise-solvable task (agent
        # gives up mid-run). A resample almost always returns a proper action, so
        # we retry generously before surfacing a give-up.
        self._max_empty_retries = max_empty_retries
        self._use_function_calling = not any(
            model.lower().startswith(prefix) for prefix in _NO_FUNCTION_CALLING
        )
        # Opt-in: ask the provider for token logprobs so the confidence/uncertainty
        # router (llm/model_router.py) can use a real signal instead of a heuristic.
        # Off by default because not every OpenAI-compatible proxy supports it.
        self._request_logprobs = request_logprobs

    def _logprob_kwargs(self) -> dict:
        return {"logprobs": True} if self._request_logprobs else {}

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_function_calling(self) -> bool:
        return self._use_function_calling

    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSchema],
    ) -> LLMResponse:
        api_messages = _to_openai_messages(messages)

        logger.debug(
            "OpenAI-compat request: model=%s messages=%d tools=%d fc=%s",
            self._model, len(api_messages), len(tools), self._use_function_calling,
        )

        if self._use_function_calling:
            response = self._complete_with_tools(api_messages, tools)
        else:
            response = self._complete_text_only(api_messages, tools)

        return response

    # ------------------------------------------------------------------
    # Function calling path
    # ------------------------------------------------------------------

    def _complete_with_tools(
        self,
        api_messages: list[dict],
        tools: list[LLMToolSchema],
    ) -> LLMResponse:
        api_tools = [_to_openai_tool(t) for t in tools]

        response = None
        for attempt in range(self._max_empty_retries + 1):
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=api_messages,
                tools=api_tools,
                tool_choice="auto",
                **self._logprob_kwargs(),
            )
            if not _is_empty_response(response.choices[0]):
                break
            if attempt < self._max_empty_retries:
                logger.warning(
                    "Empty response (finish_reason=%s, no content or tool_call); "
                    "resampling (attempt %d/%d)",
                    response.choices[0].finish_reason,
                    attempt + 1, self._max_empty_retries,
                )

        choice = response.choices[0]
        message = choice.message
        thought = message.content or "(no thought)"

        logger.debug(
            "OpenAI-compat response: finish_reason=%s input=%d output=%d",
            choice.finish_reason,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

        action = _parse_openai_response(choice, thought)

        return LLMResponse(
            action=action,
            raw_content=thought,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            logprob_avg=_mean_logprob(choice),
        )

    # ------------------------------------------------------------------
    # Text-parsing fallback (for R1 and other models without function calling)
    # ------------------------------------------------------------------

    def _complete_text_only(
        self,
        api_messages: list[dict],
        tools: list[LLMToolSchema],
    ) -> LLMResponse:
        # Inject tool descriptions into the system prompt; ask the model to output JSON
        tool_desc = _build_tool_description_for_text(tools)
        # Insert tool descriptions after the first system message
        augmented = list(api_messages)
        if augmented and augmented[0]["role"] == "system":
            augmented[0] = {
                "role": "system",
                "content": augmented[0]["content"] + "\n\n" + tool_desc,
            }

        response = None
        for attempt in range(self._max_empty_retries + 1):
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=augmented,
                **self._logprob_kwargs(),
            )
            if (response.choices[0].message.content or "").strip():
                break
            if attempt < self._max_empty_retries:
                logger.warning(
                    "Empty text response (finish_reason=%s); resampling (attempt %d/%d)",
                    response.choices[0].finish_reason,
                    attempt + 1, self._max_empty_retries,
                )

        choice = response.choices[0]
        raw_text = choice.message.content or ""

        action = _parse_text_response(raw_text)

        return LLMResponse(
            action=action,
            raw_content=raw_text,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            logprob_avg=_mean_logprob(choice),
        )


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def _to_openai_messages(messages: list[LLMMessage]) -> list[dict]:
    """Convert a list of LLMMessages into OpenAI messages format."""
    result = []
    for msg in messages:
        if msg.tool_call_id:
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })
        else:
            result.append({"role": msg.role, "content": msg.content})
    return result


def _to_openai_tool(schema: LLMToolSchema) -> dict:
    """Convert to OpenAI tool schema format."""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


def _mean_logprob(choice: Any) -> float | None:
    """Mean per-token logprob of the sampled answer, or None if unavailable.

    Fully defensive: providers that don't return logprobs (or return them in a
    different shape) simply yield None, and the confidence router falls back to
    its heuristic. Never raises.
    """
    try:
        content = choice.logprobs.content            # list of per-token entries
        vals = [tok.logprob for tok in content if getattr(tok, "logprob", None) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)
    except (AttributeError, TypeError, ZeroDivisionError):
        return None


def _is_empty_response(choice: Any) -> bool:
    """True when the model ended its turn with neither content nor a tool call.

    Reasoning models (gpt-oss and similar) intermittently stop after the
    reasoning/analysis channel without transitioning to a final answer or a tool
    call, yielding an empty response. This is non-deterministic, so the caller
    resamples instead of treating it as a genuine give-up.
    """
    message = choice.message
    has_content = bool((getattr(message, "content", None) or "").strip())
    has_tool_calls = bool(getattr(message, "tool_calls", None))
    return not has_content and not has_tool_calls


def _parse_openai_response(choice: Any, thought: str) -> Action:
    """Parse an OpenAI API choice and return an Action."""
    finish_reason = choice.finish_reason
    message = choice.message

    if finish_reason == "tool_calls" and message.tool_calls:
        # Take the first tool call (the agent calls one tool per turn)
        tc = message.tool_calls[0]
        try:
            params = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            params = {"raw": tc.function.arguments}

        return Action(
            action_type=ActionType.TOOL_CALL,
            thought=thought,
            tool_call=ToolCall(name=tc.function.name, params=params),
        )

    if finish_reason == "stop":
        if thought and thought != "(no thought)":
            return Action(
                action_type=ActionType.FINISH,
                thought="",       # standard chat models have no separate reasoning chain
                message=thought,  # the model's output is the final answer
            )
        return Action(
            action_type=ActionType.GIVE_UP,
            thought=thought,
            message="Model stopped with no content",
        )

    # length (token limit exceeded) or other finish reasons
    return Action(
        action_type=ActionType.GIVE_UP,
        thought=thought,
        message=f"Unexpected finish_reason: {finish_reason}",
    )


# ---------------------------------------------------------------------------
# Text-parsing fallback
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_INLINE_JSON_RE = re.compile(r"\{[^{}]+\}", re.DOTALL)

_FINISH_KEYWORDS = ("task complete", "task is complete", "i have finished", "all done")
_GIVE_UP_KEYWORDS = ("cannot solve", "give up", "unable to", "i cannot")


def _build_tool_description_for_text(tools: list[LLMToolSchema]) -> str:
    """
    Inject tool descriptions for models that don't support function calling.
    Asks the model to output a specific JSON format:
    {"tool": "tool_name", "params": {...}}
    or to output FINISH / GIVE_UP keywords.
    """
    if not tools:
        return ""

    lines = [
        "## Available tools",
        "To call a tool, output ONLY a JSON block in this exact format:",
        '```json\n{"tool": "<tool_name>", "params": {<params>}}\n```',
        "",
        "To finish the task, output: TASK_COMPLETE: <summary>",
        "To give up, output: GIVE_UP: <reason>",
        "",
        "Tools:",
    ]
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)


def _parse_text_response(text: str) -> Action:
    """
    Parse an Action from plain text.
    Priority: JSON block match → keyword match → fallback GIVE_UP.
    """
    text_stripped = text.strip()

    # Check for TASK_COMPLETE
    if text_stripped.upper().startswith("TASK_COMPLETE:"):
        summary = text_stripped[len("TASK_COMPLETE:"):].strip()
        return Action(
            action_type=ActionType.FINISH,
            thought=text_stripped,
            message=summary or "Task complete",
        )

    # Check for GIVE_UP
    if text_stripped.upper().startswith("GIVE_UP:"):
        reason = text_stripped[len("GIVE_UP:"):].strip()
        return Action(
            action_type=ActionType.GIVE_UP,
            thought=text_stripped,
            message=reason or "Agent gave up",
        )

    # Try to extract a JSON block (```json ... ```)
    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        return _try_parse_tool_json(block_match.group(1), thought=text_stripped)

    # Try to extract inline JSON
    for m in _INLINE_JSON_RE.finditer(text):
        action = _try_parse_tool_json(m.group(0), thought=text_stripped)
        if action is not None:
            return action

    # Keyword-based fallback
    text_lower = text.lower()
    if any(kw in text_lower for kw in _FINISH_KEYWORDS):
        return Action(
            action_type=ActionType.FINISH,
            thought=text_stripped,
            message=text_stripped,
        )
    if any(kw in text_lower for kw in _GIVE_UP_KEYWORDS):
        return Action(
            action_type=ActionType.GIVE_UP,
            thought=text_stripped,
            message=text_stripped,
        )

    # Could not parse; give up
    logger.warning("Could not parse action from text: %s", text_stripped[:100])
    return Action(
        action_type=ActionType.GIVE_UP,
        thought=text_stripped,
        message="Could not parse a valid action from model output",
    )


def _try_parse_tool_json(json_str: str, thought: str) -> Action | None:
    """Try to parse a JSON string into a TOOL_CALL Action; return None on failure."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    tool_name = data.get("tool") or data.get("name") or data.get("function")
    params = data.get("params") or data.get("arguments") or data.get("input") or {}

    if not tool_name or not isinstance(tool_name, str):
        return None

    return Action(
        action_type=ActionType.TOOL_CALL,
        thought=thought,
        tool_call=ToolCall(name=tool_name, params=params if isinstance(params, dict) else {}),
    )


# ---------------------------------------------------------------------------
# Streaming support
# ---------------------------------------------------------------------------

from llm.base import StreamCallback


def _openai_stream(
    self: "OpenAICompatBackend",
    messages: list,
    tools: list,
    on_text: StreamCallback | None = None,
    on_thought: StreamCallback | None = None,
) -> "LLMResponse":
    """
    OpenAI-compatible streaming implementation.
    on_text:    callback for each chunk of the final answer
    on_thought: callback for each chunk of the reasoning process (reasoning_content); reasoning models only
    """
    api_messages = _to_openai_messages(messages)

    if self._use_function_calling:
        return _stream_with_tools(self, api_messages, tools, on_text, on_thought)
    else:
        return _stream_text_only(self, api_messages, tools, on_text)


def _stream_with_tools(self, api_messages, tools, on_text, on_thought=None):
    api_tools = [_to_openai_tool(t) for t in tools] if tools else None

    kwargs = dict(
        model=self._model,
        max_tokens=self._max_tokens,
        messages=api_messages,
        stream=True,
    )
    if api_tools:
        kwargs["tools"] = api_tools
        kwargs["tool_choice"] = "auto"

    # Collect streaming chunks
    full_text = ""
    full_reasoning = ""  # reasoning_content (exclusive to reasoning models)
    finish_reason = None
    tool_calls_raw = []      # accumulate tool call deltas

    stream = self._client.chat.completions.create(**kwargs)
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            continue

        delta = choice.delta
        finish_reason = choice.finish_reason or finish_reason

        # reasoning_content delta (DeepSeek R1 / Claude thinking)
        reasoning_delta = getattr(delta, "reasoning_content", None)
        if reasoning_delta:
            full_reasoning += reasoning_delta
            if on_thought:
                on_thought(reasoning_delta)

        # text delta (final answer)
        if delta.content:
            full_text += delta.content
            if on_text:
                on_text(delta.content)

        # Accumulate tool call deltas
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                while len(tool_calls_raw) <= idx:
                    tool_calls_raw.append({"name": "", "arguments": ""})
                if tc_delta.function.name:
                    tool_calls_raw[idx]["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    tool_calls_raw[idx]["arguments"] += tc_delta.function.arguments

    # Build a mock choice object for reuse by _parse_openai_response
    import json as _json
    from types import SimpleNamespace

    if tool_calls_raw and finish_reason == "tool_calls":
        tcs = []
        for tc in tool_calls_raw:
            try:
                params = _json.loads(tc["arguments"])
            except Exception:
                params = {"raw": tc["arguments"]}
            fn = SimpleNamespace(name=tc["name"], arguments=tc["arguments"])
            tcs.append(SimpleNamespace(function=fn))
        mock_message = SimpleNamespace(content=full_text or None, tool_calls=tcs)
    else:
        mock_message = SimpleNamespace(content=full_text or None, tool_calls=None)

    mock_choice = SimpleNamespace(finish_reason=finish_reason or "stop", message=mock_message)
    # With reasoning_content: thought = reasoning, message = final answer
    # Without it (standard chat model): thought is empty, message = model output
    thought_for_parse = full_text or "(no thought)"
    action = _parse_openai_response(mock_choice, thought_for_parse)
    # If there is reasoning content, override action.thought
    if full_reasoning and action.action_type.value == "finish":
        action = action.__class__(
            action_type=action.action_type,
            thought=full_reasoning,
            tool_call=action.tool_call,
            message=action.message,
        )

    # Streaming mode can't get exact token counts; estimate
    from context.token_budget import estimate_tokens
    input_tokens = sum(estimate_tokens(m.get("content", "")) for m in api_messages)
    output_tokens = estimate_tokens(full_text)

    return LLMResponse(
        action=action,
        raw_content=full_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _stream_text_only(self, api_messages, tools, on_text):
    """Streaming path for models like R1 that don't support function calling."""
    tool_desc = _build_tool_description_for_text(tools)
    augmented = list(api_messages)
    if augmented and augmented[0]["role"] == "system":
        augmented[0] = {
            "role": "system",
            "content": augmented[0]["content"] + "\n\n" + tool_desc,
        }

    full_text = ""
    stream = self._client.chat.completions.create(
        model=self._model,
        max_tokens=self._max_tokens,
        messages=augmented,
        stream=True,
    )
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            continue
        delta = choice.delta
        if delta.content:
            full_text += delta.content
            if on_text:
                on_text(delta.content)

    action = _parse_text_response(full_text)

    from context.token_budget import estimate_tokens
    return LLMResponse(
        action=action,
        raw_content=full_text,
        input_tokens=sum(estimate_tokens(m.get("content", "")) for m in augmented),
        output_tokens=estimate_tokens(full_text),
    )


# Bind stream() method onto OpenAICompatBackend
OpenAICompatBackend.stream = _openai_stream
