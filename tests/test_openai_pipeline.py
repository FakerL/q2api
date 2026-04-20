"""Tests for the OpenAI chat/completions → Kiro pipeline."""
import json
import sys
import os
import asyncio
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from claude_converter import (
    convert_openai_to_amazonq_request,
    convert_claude_to_amazonq_request,
    convert_openai_tool,
    _merge_adjacent_openai_messages,
    _convert_openai_messages_to_kiro,
    _extract_openai_tool_choice_hint,
    _build_response_format_hint,
    map_model_name,
    DEFAULT_USER_CONTENT,
    DEFAULT_USER_CONTENT_WITH_TOOL_RESULTS,
    DEFAULT_ASSISTANT_CONTENT,
    DEFAULT_ASSISTANT_CONTENT_WITH_TOOLS,
)
from claude_types import ClaudeRequest
from claude_stream import (
    claude_sse_to_openai_sse,
    collect_openai_response,
    _parse_sse_frames,
    _STOP_REASON_MAP,
)


# --- Helpers ---

class _Msg:
    """Minimal message object matching ChatMessage fields."""
    def __init__(self, role, content=None, tool_calls=None, tool_call_id=None, name=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name


class _Req:
    """Minimal request object matching ChatCompletionRequest fields."""
    def __init__(self, messages, model="auto", tools=None, tool_choice=None,
                 response_format=None, temperature=None, top_p=None,
                 max_tokens=None, max_completion_tokens=None, reasoning_effort=None):
        self.messages = messages
        self.model = model
        self.tools = tools
        self.tool_choice = tool_choice
        self.response_format = response_format
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort


# --- SSE helpers ---

def _make_claude_sse(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _async_iter(items):
    for item in items:
        yield item


# ============================================================
# 1. Basic text chat (converter)
# ============================================================

def test_basic_text_chat():
    req = _Req(messages=[_Msg("user", "hello")])
    result = convert_openai_to_amazonq_request(req)
    cs = result["conversationState"]
    assert cs["chatTriggerType"] == "MANUAL"
    content = cs["currentMessage"]["userInputMessage"]["content"]
    assert "hello" in content
    assert "--- SYSTEM PROMPT ---" in content
    assert cs["history"] == []


def test_system_prompt_extraction():
    req = _Req(messages=[_Msg("system", "You are helpful"), _Msg("user", "hi")])
    result = convert_openai_to_amazonq_request(req)
    content = result["conversationState"]["currentMessage"]["userInputMessage"]["content"]
    assert "You are helpful" in content
    assert "hi" in content


# ============================================================
# 2. Tool calls + tool results
# ============================================================

def test_tool_calls_and_results():
    msgs = [
        _Msg("user", "what time?"),
        _Msg("assistant", "", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "get_time", "arguments": "{}"}
        }]),
        _Msg("tool", "14:30", tool_call_id="call_1"),
        _Msg("user", "thanks"),
    ]
    req = _Req(messages=msgs)
    result = convert_openai_to_amazonq_request(req)
    history = result["conversationState"]["history"]
    current = result["conversationState"]["currentMessage"]["userInputMessage"]
    # History: user("what time?"), assistant(toolUses)
    # Current: user("thanks") with toolResults attached (carried from tool msg)
    assert len(history) == 2
    asst = next(h for h in history if "assistantResponseMessage" in h)
    assert "toolUses" in asst["assistantResponseMessage"]
    assert asst["assistantResponseMessage"]["toolUses"][0]["name"] == "get_time"
    # Tool results attached to current user turn
    tr = current.get("userInputMessageContext", {}).get("toolResults", [])
    assert len(tr) == 1
    assert tr[0]["toolUseId"] == "call_1"


# ============================================================
# 3. Multiple tool calls in one turn
# ============================================================

def test_multiple_tool_calls():
    msgs = [
        _Msg("user", "get both"),
        _Msg("assistant", "", tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
        ]),
        _Msg("tool", "result_a", tool_call_id="c1"),
        _Msg("tool", "result_b", tool_call_id="c2"),
        _Msg("user", "ok"),
    ]
    req = _Req(messages=msgs)
    result = convert_openai_to_amazonq_request(req)
    history = result["conversationState"]["history"]
    asst = next(h for h in history if "assistantResponseMessage" in h)
    assert len(asst["assistantResponseMessage"]["toolUses"]) == 2


# ============================================================
# 4. Conversation ending in tool results (synthetic user turn)
# ============================================================

def test_conversation_ending_in_tool_results():
    msgs = [
        _Msg("user", "do it"),
        _Msg("assistant", "", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "run", "arguments": "{}"}
        }]),
        _Msg("tool", "done", tool_call_id="c1"),
    ]
    req = _Req(messages=msgs)
    result = convert_openai_to_amazonq_request(req)
    current = result["conversationState"]["currentMessage"]["userInputMessage"]
    # Content includes system prompt wrapper
    assert DEFAULT_USER_CONTENT_WITH_TOOL_RESULTS in current["content"]
    tr = current["userInputMessageContext"].get("toolResults", [])
    assert len(tr) == 1
    assert tr[0]["toolUseId"] == "c1"


# ============================================================
# 5. Streaming reasoning deltas
# ============================================================

@pytest.mark.asyncio
async def test_streaming_reasoning_deltas():
    sse_items = [
        _make_claude_sse("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        _make_claude_sse("content_block_start", {"index": 0, "content_block": {"type": "thinking"}}),
        _make_claude_sse("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
        _make_claude_sse("content_block_stop", {"index": 0}),
        _make_claude_sse("content_block_start", {"index": 1, "content_block": {"type": "text"}}),
        _make_claude_sse("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "Hello"}}),
        _make_claude_sse("content_block_stop", {"index": 1}),
        _make_claude_sse("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
        _make_claude_sse("message_stop", {}),
    ]
    chunks = []
    async for chunk in claude_sse_to_openai_sse(_async_iter(sse_items), "test", "id1", 1000):
        chunks.append(chunk)

    all_text = "".join(chunks)
    assert "reasoning_content" in all_text
    assert "hmm" in all_text
    assert "Hello" in all_text
    assert "[DONE]" in all_text


# ============================================================
# 6. Streaming tool-call argument deltas
# ============================================================

@pytest.mark.asyncio
async def test_streaming_tool_call_deltas():
    sse_items = [
        _make_claude_sse("message_start", {"message": {"usage": {"input_tokens": 5}}}),
        _make_claude_sse("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "id": "tc1", "name": "get_time"}}),
        _make_claude_sse("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"tz":'}}),
        _make_claude_sse("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": '"UTC"}'}}),
        _make_claude_sse("content_block_stop", {"index": 0}),
        _make_claude_sse("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}}),
        _make_claude_sse("message_stop", {}),
    ]
    chunks = []
    async for chunk in claude_sse_to_openai_sse(_async_iter(sse_items), "test", "id2", 1000):
        chunks.append(chunk)

    # Parse all data chunks
    parsed = []
    for c in chunks:
        if c.startswith("data: ") and not c.startswith("data: [DONE]"):
            parsed.append(json.loads(c[6:]))

    # Find tool_calls chunks
    tc_chunks = [p for p in parsed if p.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
    assert len(tc_chunks) >= 2  # start + 2 argument deltas
    # First should have id and name
    first_tc = tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert first_tc["id"] == "tc1"
    assert first_tc["function"]["name"] == "get_time"
    # Check finish_reason
    last_data = [p for p in parsed if p["choices"][0].get("finish_reason")]
    assert last_data[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ============================================================
# 7. Finish-reason mapping
# ============================================================

def test_finish_reason_mapping():
    assert _STOP_REASON_MAP["end_turn"] == "stop"
    assert _STOP_REASON_MAP["tool_use"] == "tool_calls"
    assert _STOP_REASON_MAP["max_tokens"] == "length"
    assert _STOP_REASON_MAP["stop_sequence"] == "stop"


# ============================================================
# 8. response_format hint injection
# ============================================================

def test_response_format_json_object():
    req = _Req(
        messages=[_Msg("user", "give json")],
        response_format={"type": "json_object"},
    )
    result = convert_openai_to_amazonq_request(req)
    content = result["conversationState"]["currentMessage"]["userInputMessage"]["content"]
    assert "valid JSON only" in content


def test_response_format_json_schema():
    req = _Req(
        messages=[_Msg("user", "give json")],
        response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}},
    )
    result = convert_openai_to_amazonq_request(req)
    content = result["conversationState"]["currentMessage"]["userInputMessage"]["content"]
    assert "valid JSON only" in content
    assert '"x"' in content


# ============================================================
# 9. max_completion_tokens aliasing
# ============================================================

def test_max_completion_tokens_alias():
    req = _Req(messages=[_Msg("user", "hi")], max_completion_tokens=500)
    result = convert_openai_to_amazonq_request(req)
    assert result["inferenceConfig"]["maxTokens"] == 500


def test_max_tokens_minus_one():
    req = _Req(messages=[_Msg("user", "hi")], max_tokens=-1)
    result = convert_openai_to_amazonq_request(req)
    assert result["inferenceConfig"]["maxTokens"] == 32000


def test_max_completion_tokens_precedence_over_max_tokens():
    """max_completion_tokens=0 should win over max_tokens=4096."""
    req = _Req(messages=[_Msg("user", "hi")], max_completion_tokens=0, max_tokens=4096)
    result = convert_openai_to_amazonq_request(req)
    # max_completion_tokens=0 is falsey but explicitly set, so no inferenceConfig.maxTokens
    assert "inferenceConfig" not in result or "maxTokens" not in result.get("inferenceConfig", {})


# ============================================================
# 10. Adjacent assistant messages merge preserving tool_calls
# ============================================================

def test_adjacent_assistant_merge():
    msgs = [
        _Msg("assistant", "part1"),
        _Msg("assistant", "part2", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]),
    ]
    merged = _merge_adjacent_openai_messages(msgs)
    assert len(merged) == 1
    assert "part1" in merged[0].content
    assert "part2" in merged[0].content
    assert merged[0].tool_calls is not None
    assert len(merged[0].tool_calls) == 1


# ============================================================
# 11. Adjacent tool messages do NOT merge
# ============================================================

def test_adjacent_tool_messages_no_merge():
    msgs = [
        _Msg("tool", "result1", tool_call_id="c1"),
        _Msg("tool", "result2", tool_call_id="c2"),
    ]
    merged = _merge_adjacent_openai_messages(msgs)
    assert len(merged) == 2


# ============================================================
# Non-streaming accumulation
# ============================================================

@pytest.mark.asyncio
async def test_collect_openai_response():
    sse_items = [
        _make_claude_sse("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        _make_claude_sse("content_block_start", {"index": 0, "content_block": {"type": "text"}}),
        _make_claude_sse("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hi there"}}),
        _make_claude_sse("content_block_stop", {"index": 0}),
        _make_claude_sse("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
        _make_claude_sse("message_stop", {}),
    ]
    result = await collect_openai_response(_async_iter(sse_items), "test", "id3", 1000)
    assert result["choices"][0]["message"]["content"] == "Hi there"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 10


# ============================================================
# P1 fix: list-valued content handled correctly
# ============================================================

def test_list_content_user_message():
    """Content-parts array should be extracted, not dropped."""
    req = _Req(messages=[_Msg("user", [{"type": "text", "text": "hello world"}])])
    result = convert_openai_to_amazonq_request(req)
    content = result["conversationState"]["currentMessage"]["userInputMessage"]["content"]
    assert "hello world" in content


# ============================================================
# P1 fix: chat-only models don't get tool_choice hint
# ============================================================

def test_chat_only_no_tool_choice_hint():
    req = _Req(
        messages=[_Msg("user", "hi")],
        model="claude-sonnet-4-chat",
        tools=[{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}],
        tool_choice="required",
    )
    result = convert_openai_to_amazonq_request(req)
    content = result["conversationState"]["currentMessage"]["userInputMessage"]["content"]
    assert "MUST use" not in content
    # Tools should be stripped
    assert result["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"].get("tools", []) == []


# ============================================================
# P1 fix: tool_use finish_reason from streaming
# ============================================================

@pytest.mark.asyncio
async def test_tool_use_finish_reason():
    sse_items = [
        _make_claude_sse("message_start", {"message": {"usage": {"input_tokens": 5}}}),
        _make_claude_sse("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "id": "tc1", "name": "fn"}}),
        _make_claude_sse("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        _make_claude_sse("content_block_stop", {"index": 0}),
        _make_claude_sse("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}}),
        _make_claude_sse("message_stop", {}),
    ]
    chunks = []
    async for chunk in claude_sse_to_openai_sse(_async_iter(sse_items), "test", "id4", 1000):
        chunks.append(chunk)
    # Last data chunk before [DONE] should have finish_reason=tool_calls
    parsed = [json.loads(c[6:]) for c in chunks if c.startswith("data: ") and not c.startswith("data: [DONE]")]
    last = parsed[-1]
    assert last["choices"][0]["finish_reason"] == "tool_calls"


# ============================================================
# P1 fix: merged user messages preserve images
# ============================================================

def test_merged_user_messages_preserve_images():
    """Adjacent user messages with image content should preserve images after merge."""
    from claude_converter import _merge_adjacent_openai_messages, _convert_openai_messages_to_kiro
    msgs = [
        _Msg("user", "describe this"),
        _Msg("user", [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
        ]),
    ]
    merged = _merge_adjacent_openai_messages(msgs)
    assert len(merged) == 1
    assert "describe this" in merged[0].content
    assert "what is in this image" in merged[0].content
    # _raw_contents should be preserved
    assert hasattr(merged[0], '_raw_contents')
    assert len(merged[0]._raw_contents) == 2


# ============================================================
# P2 fix: web_search renamed in tool_choice hint
# ============================================================

def test_tool_choice_web_search_renamed():
    from claude_converter import _extract_openai_tool_choice_hint
    hint = _extract_openai_tool_choice_hint(
        {"type": "function", "function": {"name": "web_search"}},
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert "remote_web_search" in hint
    assert "web_search" not in hint.replace("remote_web_search", "")


# ============================================================
# Regression: tool results flushed before next assistant turn
# ============================================================

def test_tool_result_flushed_before_second_assistant():
    """user -> assistant(tool_calls) -> tool -> assistant(final) -> user
    must synthesize a user turn with tool results before the second assistant."""
    msgs = [
        _Msg("user", "what time is it?"),
        _Msg("assistant", "", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "get_time", "arguments": "{}"}
        }]),
        _Msg("tool", "14:30", tool_call_id="c1"),
        _Msg("assistant", "It is 14:30."),
        _Msg("user", "thanks"),
    ]
    req = _Req(messages=msgs)
    result = convert_openai_to_amazonq_request(req)
    history = result["conversationState"]["history"]

    # Expected history: user, assistant(toolUses), user(toolResults), assistant("It is 14:30.")
    assert len(history) == 4

    # 1st: user
    assert "userInputMessage" in history[0]
    # 2nd: assistant with toolUses
    assert "toolUses" in history[1]["assistantResponseMessage"]
    # 3rd: synthetic user with toolResults
    assert "userInputMessage" in history[2]
    tr = history[2]["userInputMessage"]["userInputMessageContext"].get("toolResults", [])
    assert len(tr) == 1
    assert tr[0]["toolUseId"] == "c1"
    # 4th: assistant final answer
    assert history[3]["assistantResponseMessage"]["content"] == "It is 14:30."

    # Current message should be "thanks" with no tool results
    current = result["conversationState"]["currentMessage"]["userInputMessage"]
    assert "thanks" in current["content"]
    assert not current.get("userInputMessageContext", {}).get("toolResults")


# ============================================================
# P1 fix: tool_choice="none" strips tools from payload
# ============================================================

def test_tool_choice_none_strips_tools():
    req = _Req(
        messages=[_Msg("user", "hi")],
        tools=[{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}],
        tool_choice="none",
    )
    result = convert_openai_to_amazonq_request(req)
    ctx = result["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]
    assert ctx.get("tools", []) == []
    # No tool_choice hint either
    content = result["conversationState"]["currentMessage"]["userInputMessage"]["content"]
    assert "MUST use" not in content


# ============================================================
# P1 fix: malformed tool-call arguments raise ValueError
# ============================================================

def test_malformed_tool_call_arguments_raises():
    msgs = [
        _Msg("user", "go"),
        _Msg("assistant", "", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "run", "arguments": "{bad json"}
        }]),
        _Msg("user", "next"),
    ]
    req = _Req(messages=msgs)
    with pytest.raises(ValueError, match="Malformed tool_call arguments"):
        convert_openai_to_amazonq_request(req)


# ============================================================
# P2 fix: large json_schema truncates at safe boundary
# ============================================================

def test_large_json_schema_truncates_safely():
    from claude_converter import _build_response_format_hint
    big_schema = {"type": "object", "properties": {f"field_{i}": {"type": "string"} for i in range(200)}}
    hint = _build_response_format_hint({"type": "json_schema", "json_schema": {"schema": big_schema}})
    assert "valid JSON only matching this schema" in hint
    # Should contain truncated schema (500 chars + "...")
    assert "field_" in hint
    assert hint.count("...") >= 1
    # Schema portion should be roughly 503 chars (500 + "...")
    schema_line = [l for l in hint.split("\n") if l.startswith("{")][0]
    assert len(schema_line) == 503


# ============================================================
# Model alias support for deepseek / minimax / glm
# ============================================================

def test_non_claude_model_aliases_resolve_to_upstream_ids():
    assert map_model_name("deepseek-3.2") == "deepseek-3.2"
    assert map_model_name("deepseek-3-2") == "deepseek-3.2"
    assert map_model_name("kiro-deepseek-3-2-agentic") == "deepseek-3.2"

    assert map_model_name("minimax-m2.1") == "minimax-m2.1"
    assert map_model_name("minimax-m2-1") == "minimax-m2.1"
    assert map_model_name("kiro-minimax-m2-1-chat") == "minimax-m2.1"

    assert map_model_name("minimax-m2.5") == "minimax-m2.5"
    assert map_model_name("minimax-m2-5") == "minimax-m2.5"
    assert map_model_name("kiro-minimax-m2-5") == "minimax-m2.5"

    assert map_model_name("glm-5") == "glm-5"
    assert map_model_name("kiro-glm-5-agentic") == "glm-5"


def test_deepseek_image_input_still_allowed():
    req = _Req(messages=[_Msg("user", [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ])], model="deepseek-3.2")
    result = convert_openai_to_amazonq_request(req)
    current = result["conversationState"]["currentMessage"]["userInputMessage"]
    assert current["modelId"] == "deepseek-3.2"
    assert current.get("images")


def test_openai_text_only_model_rejects_image_input():
    req = _Req(messages=[_Msg("user", [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ])], model="glm-5")
    with pytest.raises(ValueError, match="text-only"):
        convert_openai_to_amazonq_request(req)


def test_claude_text_only_model_rejects_image_input():
    req = ClaudeRequest(
        model="minimax-m2.5",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    },
                },
            ],
        }],
    )
    with pytest.raises(ValueError, match="text-only"):
        convert_claude_to_amazonq_request(req)
