# q2api vs CLIProxyAPIPlus Alignment

## ✅ Aligned Features

### Request Translation
| Feature | Status | Notes |
|---------|--------|-------|
| Origin normalization | ✅ | `KIRO_CLI`→`CLI`, `KIRO_AI_EDITOR`→`AI_EDITOR`, etc. |
| Thinking mode (5 methods) | ✅ | Header, Claude API, OpenAI, AMP/Cursor, model name |
| Thinking injection | ✅ | `<thinking_mode>enabled</thinking_mode>`, 16000 tokens |
| Empty content handling | ✅ | `.` for assistant, `Continue` for user |
| Tool description limit | ✅ | 10237 chars with UTF-8 safe truncation |
| Tool compression | ✅ | 20KB threshold, 2-step (schema + description) |
| Web search filtering | ✅ | Filter + Fetch tool hint injection |
| Agentic mode | ✅ | `-agentic` suffix triggers chunked write prompt |
| InferenceConfig | ✅ | `maxTokens`, `temperature`, `topP` |
| Tool choice hint | ✅ | System prompt injection for `any`/`tool` |
| MCP tool name shortening | ✅ | 64 char limit |

### Models Supported
```
claude-sonnet-4, claude-sonnet-4.5, claude-sonnet-4.6, claude-haiku-4.5, claude-opus-4.5, claude-opus-4.6
+ Agentic variants: claude-opus-4.5-agentic, etc.
```

## ❌ Not Implemented (CLIProxyAPIPlus only)

| Feature | Reason |
|---------|--------|
| `profileArn` | AWS-specific, not needed for our use case |
| `contextUsageEvent` | Response parsing in executor, not converter |
| Truncation detection | Response-side feature (4 types + soft failure) |
| Retry with exponential backoff | Executor-level, uses simple retry in replicate.py |
| Real-time usage updates | Executor streaming feature |
| Message merging (adjacent same-role) | Partially implemented in `process_history` |

## System Prompt Structure

Order (matches CLIProxyAPIPlus):
1. `<thinking_mode>enabled</thinking_mode>` (if enabled, not already present)
2. `<max_thinking_length>16000</max_thinking_length>` (if enabled)
3. `[Context: Current time is ...]`
4. Original system prompt
5. Agentic chunked write prompt (if `-agentic` model)
6. Tool choice hint (if specified)
7. Web search alternative hint (if web_search filtered)

## Request Payload Structure

```json
{
  "conversationState": {
    "chatTriggerType": "MANUAL",
    "conversationId": "uuid",
    "currentMessage": {
      "userInputMessage": {
        "content": "--- SYSTEM PROMPT ---\n...\n--- END SYSTEM PROMPT ---\n\n{user_content}",
        "modelId": "claude-opus-4.5",
        "origin": "CLI",
        "userInputMessageContext": { "tools": [...], "toolResults": [...] }
      }
    },
    "history": [...]
  },
  "inferenceConfig": { "maxTokens": 4096, "temperature": 0.7, "topP": 0.9 }
}
```

## Key Constants

```python
DEFAULT_ASSISTANT_CONTENT_WITH_TOOLS = "."
DEFAULT_ASSISTANT_CONTENT = "."
DEFAULT_USER_CONTENT_WITH_TOOL_RESULTS = "Tool results provided."
DEFAULT_USER_CONTENT = "Continue"
TOOL_COMPRESSION_TARGET_SIZE = 20 * 1024  # 20KB
MIN_TOOL_DESCRIPTION_LENGTH = 50
KIRO_MAX_TOOL_DESC_LEN = 10237
```

## Changelog

### 2026-02-08
- Added agentic mode (`-agentic` model variants with chunked write prompt)
- Added InferenceConfig support (`maxTokens`, `temperature`, `topP`)
- Added web_search tool filtering with Fetch alternative hint
- Added Claude Opus 4.6 model support
- Fixed Anthropic-Beta header passthrough for thinking detection
- Aligned `chatTriggerType` field position (first in conversationState)
- Updated all empty content placeholders to minimal strings
