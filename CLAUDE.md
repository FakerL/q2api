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
| web_search → remote_web_search | ✅ | Rename tool + fallback description |
| Agentic mode | ✅ | `-agentic` suffix triggers chunked write prompt |
| Chat-only mode | ✅ | `-chat` suffix strips all tools |
| InferenceConfig | ✅ | `maxTokens`, `temperature`, `topP` |
| Tool choice hint | ✅ | System prompt injection for `any`/`tool` |
| MCP tool name shortening | ✅ | 64 char limit |
| System prompt re-injection | ✅ | Skipped when `len(history) > 0` |
| Tool result deduplication | ✅ | By `toolUseId` on currentMessage |
| `ensureKiroInputSchema` | ✅ | Defaults to `{"type":"object","properties":{}}` |
| Model suffix stripping | ✅ | `-agentic`, `-chat` stripped for model resolution |

### Models Supported
```
auto, claude-sonnet-4, claude-sonnet-4.5, claude-sonnet-4.6, claude-haiku-4.5, claude-opus-4.5, claude-opus-4.6
+ Agentic variants: claude-opus-4.5-agentic, etc.
+ Chat-only variants: claude-opus-4.5-chat, etc.
```

## ❌ Not Implemented (CLIProxyAPIPlus only)

| Feature | Reason |
|---------|--------|
| `profileArn` | AWS-specific, not needed for our use case |
| `contextUsageEvent` | Response parsing in executor, not converter |
| Truncation detection | Response-side feature (4 types + soft failure) |
| Retry with exponential backoff | Executor-level, uses simple retry in replicate.py |
| Real-time usage updates | Executor streaming feature |
| Dynamic web_search description cache | Fetched from MCP tools/list at runtime; we use static fallback |

## System Prompt Structure

Order (matches CLIProxyAPIPlus):
1. `<thinking_mode>enabled</thinking_mode>` (if enabled, not already present)
2. `<max_thinking_length>16000</max_thinking_length>` (if enabled)
3. `[Context: Current time is ...]`
4. Original system prompt
5. Agentic chunked write prompt (if `-agentic` model)
6. Tool choice hint (if specified)

Note: System prompt is only injected on first turn (`len(history) == 0`).

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
REMOTE_WEB_SEARCH_DESCRIPTION = "WebSearch looks up information..."
```
