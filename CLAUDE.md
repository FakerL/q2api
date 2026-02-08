# q2api vs CLIProxyAPIPlus 差异

## ✅ 已对齐的功能 (2026-02-08)

以下功能已与 CLIProxyAPIPlus 最新实现对齐：

### Origin 字段规范化
- **实现**: `normalize_origin()` 函数
- **映射规则**:
  - `KIRO_CLI` → `CLI`
  - `KIRO_AI_EDITOR` → `AI_EDITOR`
  - `AMAZON_Q` → `CLI`
  - `KIRO_IDE` → `AI_EDITOR`

### Thinking Mode 处理
检测逻辑相同：
1. 检查 `Anthropic-Beta` header 是否包含 `interleaved-thinking`
2. 检查 Claude API 格式 `thinking.type = "enabled"`
3. 检查 OpenAI 格式 `reasoning_effort` 参数
4. 检查 AMP/Cursor 格式 `<thinking_mode>interleaved</thinking_mode>`
5. 检查模型名是否包含 `thinking` 或 `reason`

注入逻辑已对齐：
- **注入值**: `<thinking_mode>enabled</thinking_mode>`
- **Token 预算**: `<max_thinking_length>16000</max_thinking_length>` (降低以为工具输出预留空间)
- **避免重复注入**: 检测请求中是否已有 `<thinking_mode>` 标签

### 空内容处理 (CRITICAL FIX)
**问题**: Kiro API 要求所有消息内容非空，否则返回 "Improperly formed request" 错误

**解决方案** (对齐 CLIProxyAPIPlus commits 98edcad3, 4e3bad39, 88872baf):

1. **占位符文本优化** - 防止模型模仿
   - ❌ 旧值: `"I'll help you with that."`, `"I understand."`
   - ✅ 新值: `"."` (最小中性字符串)
   - **原因**: 在多工具调用的 agentic 会话中，模型会学习并模仿这些短语

2. **常量定义**:
   ```python
   DEFAULT_ASSISTANT_CONTENT_WITH_TOOLS = "."
   DEFAULT_ASSISTANT_CONTENT = "."
   DEFAULT_USER_CONTENT_WITH_TOOL_RESULTS = "Tool results provided."
   DEFAULT_USER_CONTENT = "Continue"
   ```

3. **应用位置**:
   - ✅ 历史用户消息 (history user messages)
   - ✅ 当前用户消息 (current user message) - **关键修复**
   - ✅ 助手消息 (assistant messages)
   - **注意**: 空内容检查必须在 `isLastMessage` 分支之前执行

### 工具描述长度限制
- **限制**: 10237 字符 (Kiro API 限制 10240，预留 "..." 后缀空间)
- **UTF-8 安全截断**: 避免破坏多字节字符

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
4. **contextUsageEvent**: 处理上下文使用百分比事件

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
        "origin": "CLI",
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

## 更新日志

### 2026-02-08
- ✅ 对齐 origin 字段规范化 (normalize_origin)
- ✅ 更新 thinking mode 注入值为 `enabled` 模式
- ✅ 降低 max_thinking_length 从 200000 到 16000
- ✅ 修复空内容处理，使用最小占位符 "." 防止模型模仿
- ✅ 更新工具描述长度限制为 10237
- ✅ 确保当前用户消息也应用空内容检查
