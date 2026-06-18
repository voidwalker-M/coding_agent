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
    ) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._model = model
        self._max_tokens = max_tokens
        self._use_function_calling = not any(
            model.lower().startswith(prefix) for prefix in _NO_FUNCTION_CALLING
        )

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

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=api_messages,
            tools=api_tools,
            tool_choice="auto",
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

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=augmented,
        )

        choice = response.choices[0]
        raw_text = choice.message.content or ""

        action = _parse_text_response(raw_text)

        return LLMResponse(
            action=action,
            raw_content=raw_text,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
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
