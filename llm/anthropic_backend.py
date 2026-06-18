"""
llm/anthropic_backend.py

Anthropic Claude native backend.

Message format differences vs. OpenAI:
- system prompt is passed separately, not mixed into the messages array
- tool call results use role="user" with content type "tool_result"
- responses can contain both tool_use blocks and text blocks simultaneously
- stop_reason: "tool_use" | "end_turn" | "max_tokens"
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.task import Action, ActionType, ToolCall
from llm.base import LLMBackend, LLMMessage, LLMResponse, LLMToolSchema

logger = logging.getLogger(__name__)


class AnthropicBackend(LLMBackend):
    """
    Call Claude models via the anthropic SDK.

    Supports:
    - tool_use (function calling)
    - streaming (stream=True; current implementation is non-streaming, extensible in v2)
    - extended thinking (supported by claude-3-7-sonnet and similar models)
    """

    def __init__(self, model: str, api_key: str, max_tokens: int = 4096) -> None:
        try:
            import anthropic as _anthropic
            self._client = _anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self._model = model
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSchema],
    ) -> LLMResponse:
        # Extract the system prompt (Anthropic requires it as a separate parameter)
        system_content = ""
        non_system: list[LLMMessage] = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                non_system.append(msg)

        # Convert message format
        api_messages = _to_anthropic_messages(non_system)

        # Convert tool format
        api_tools = [_to_anthropic_tool(t) for t in tools]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_content:
            kwargs["system"] = system_content
        if api_tools:
            kwargs["tools"] = api_tools

        logger.debug(
            "Anthropic request: model=%s messages=%d tools=%d",
            self._model, len(api_messages), len(api_tools),
        )

        response = self._client.messages.create(**kwargs)

        logger.debug(
            "Anthropic response: stop_reason=%s input_tokens=%d output_tokens=%d",
            response.stop_reason,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        action = _parse_anthropic_response(response)

        return LLMResponse(
            action=action,
            raw_content=_extract_text(response),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def _to_anthropic_messages(messages: list[LLMMessage]) -> list[dict]:
    """
    Convert a list of LLMMessages into Anthropic API messages format.

    Tool result messages (tool execution results) require special handling:
    Anthropic requires role=user with a list of tool_result blocks as content.
    Convention: when tool_call_id is set, this message is a tool_result.
    """
    result = []
    for msg in messages:
        if msg.tool_call_id:
            # Tool execution result
            result.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }],
            })
        else:
            result.append({"role": msg.role, "content": msg.content})
    return result


def _to_anthropic_tool(schema: LLMToolSchema) -> dict:
    """Convert to Anthropic tool schema format."""
    return {
        "name": schema.name,
        "description": schema.description,
        "input_schema": schema.parameters,
    }


def _extract_text(response: Any) -> str:
    """Extract all text from the response's content blocks."""
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str):
            parts.append(block.text)
    return "\n".join(parts)


def _parse_anthropic_response(response: Any) -> Action:
    """
    Parse an Anthropic API response into an Action.

    Priority:
    1. stop_reason == "tool_use" → find the tool_use block → TOOL_CALL
    2. stop_reason == "end_turn" → FINISH
    3. Other (max_tokens, etc.) → GIVE_UP
    """
    # Extract thought (text block content)
    thought = _extract_text(response).strip() or "(no thought)"

    if response.stop_reason == "tool_use":
        # Find the first tool_use block
        for block in response.content:
            if block.type == "tool_use":
                return Action(
                    action_type=ActionType.TOOL_CALL,
                    thought=thought,
                    tool_call=ToolCall(
                        name=block.name,
                        params=dict(block.input),
                    ),
                )
        # stop_reason is tool_use but no block found (should not happen in practice)
        return Action(
            action_type=ActionType.GIVE_UP,
            thought=thought,
            message="stop_reason=tool_use but no tool_use block found",
        )

    if response.stop_reason == "end_turn":
        # Check whether the text content implies task completion.
        # Simple heuristic: if there is text content, treat it as FINISH.
        if thought and thought != "(no thought)":
            return Action(
                action_type=ActionType.FINISH,
                thought=thought,
                message=thought,
            )
        return Action(
            action_type=ActionType.GIVE_UP,
            thought=thought,
            message="Model ended turn with no content",
        )

    # max_tokens or other stop_reason
    return Action(
        action_type=ActionType.GIVE_UP,
        thought=thought,
        message=f"Unexpected stop_reason: {response.stop_reason}",
    )


# ---------------------------------------------------------------------------
# Streaming support (override StreamingMixin.stream())
# ---------------------------------------------------------------------------

from llm.base import StreamingMixin, StreamCallback

# AnthropicBackend would need to inherit StreamingMixin to override stream().
# Python does not allow modifying inheritance after the fact, so we monkey-patch
# the stream() method directly onto the class.

def _anthropic_stream(
    self: "AnthropicBackend",
    messages: list,
    tools: list,
    on_text: StreamCallback | None = None,
) -> LLMResponse:
    """
    Anthropic streaming implementation.
    Uses the anthropic SDK's stream() context manager to invoke on_text
    for each text_delta chunk, enabling real-time output.
    """
    # Extract system prompt
    system_content = ""
    non_system = []
    for msg in messages:
        if msg.role == "system":
            system_content = msg.content
        else:
            non_system.append(msg)

    api_messages = _to_anthropic_messages(non_system)
    api_tools = [_to_anthropic_tool(t) for t in tools]

    kwargs: dict = {
        "model": self._model,
        "max_tokens": self._max_tokens,
        "messages": api_messages,
    }
    if system_content:
        kwargs["system"] = system_content
    if api_tools:
        kwargs["tools"] = api_tools

    # Use the stream() context manager
    with self._client.messages.stream(**kwargs) as stream:
        for text_chunk in stream.text_stream:
            if on_text and text_chunk:
                on_text(text_chunk)

        # Retrieve the complete final response after the stream ends
        final = stream.get_final_message()

    action = _parse_anthropic_response(final)
    return LLMResponse(
        action=action,
        raw_content=_extract_text(final),
        input_tokens=final.usage.input_tokens,
        output_tokens=final.usage.output_tokens,
    )

# Bind stream() method onto AnthropicBackend
AnthropicBackend.stream = _anthropic_stream
