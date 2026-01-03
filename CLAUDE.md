# q2api vs CLIProxyAPIPlus 差异

## Origin 字段

| 项目 | origin 值 |
|------|-----------|
| CLIProxyAPIPlus | `CLI` (通过 `normalizeOrigin` 将 `KIRO_CLI` 转为 `CLI`) |
| q2api | `KIRO_CLI` |

原因：直接调用 Kiro 后端时，使用 `CLI` 会报 `INVALID_MODEL_ID` 错误，必须使用 `KIRO_CLI`。

## Thinking Mode 处理

检测逻辑相同：
1. 检查 `Anthropic-Beta` header 是否包含 `interleaved-thinking`
2. 检查 Claude API 格式 `thinking.type = "enabled"`
3. 检查 OpenAI 格式 `reasoning_effort` 参数
4. 检查 AMP/Cursor 格式 `<thinking_mode>interleaved</thinking_mode>`
5. 检查模型名是否包含 `thinking` 或 `reason`

注入逻辑差异：
| 项目 | 注入值 |
|------|--------|
| CLIProxyAPIPlus | `<thinking_mode>enabled</thinking_mode>` |
| q2api | `<thinking_mode>interleaved</thinking_mode>` |

两者都：
- 检测请求中是否已有 `<thinking_mode>` 标签，避免重复注入
- 使用 `<max_thinking_length>200000</max_thinking_length>`

## 模型名称映射

CLIProxyAPIPlus 支持多种前缀：
- `kiro-claude-opus-4-5` → `claude-opus-4.5`
- `amazonq-claude-opus-4-5` → `claude-opus-4.5`
- `claude-opus-4-5` → `claude-opus-4.5`

q2api 映射：
- `claude-opus-4-5-20251101` → `claude-opus-4.5`（Anthropic 规范名）
- `claude-opus-4-5` → `claude-opus-4.5`
- `claude-opus-4.5` → `claude-opus-4.5`

## CLIProxyAPIPlus 独有功能

1. **InferenceConfig**: 支持 `maxTokens`, `temperature`, `topP` 参数
2. **ProfileArn**: 支持 `profileArn` 字段
3. **Agentic 模式**: 为 `-agentic` 模型变体注入分块写入优化提示（防止大文件写入超时）

## 工具描述长度限制

| 项目 | 限制 |
|------|------|
| CLIProxyAPIPlus | 10237 字符 |
| q2api | 10240 字符 |

## 请求结构

两者相同：
```json
{
  "conversationState": {
    "chatTriggerType": "MANUAL",
    "conversationId": "uuid",
    "currentMessage": {
      "userInputMessage": {
        "content": "--- SYSTEM PROMPT ---\n...\n--- END SYSTEM PROMPT ---\n\n{user_content}",
        "modelId": "claude-opus-4.5",
        "origin": "KIRO_CLI",
        "userInputMessageContext": {
          "tools": [...],
          "toolResults": [...]
        }
      }
    },
    "history": [...]
  }
}
```

## 系统提示结构

两者顺序相同：
1. Thinking hint（如果启用且未存在）
2. 时间戳上下文 `[Context: Current time is ...]`
3. 原始系统提示内容
4. Tool choice hint（如果指定）
