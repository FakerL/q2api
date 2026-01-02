import json
import uuid
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

try:
    from .claude_types import ClaudeRequest, ClaudeMessage, ClaudeTool
except ImportError:
    # Fallback for dynamic loading where relative import might fail
    # We assume claude_types is available in sys.modules or we can import it directly if in same dir
    import sys
    if "v2.claude_types" in sys.modules:
        from v2.claude_types import ClaudeRequest, ClaudeMessage, ClaudeTool
    else:
        # Try absolute import assuming v2 is in path or current dir
        try:
            from claude_types import ClaudeRequest, ClaudeMessage, ClaudeTool
        except ImportError:
             # Last resort: if loaded via importlib in app.py, we might need to rely on app.py injecting it
             # But app.py loads this module.
             pass

logger = logging.getLogger(__name__)

# Thinking mode hint - matches CLIProxyAPIPlus format
THINKING_HINT = """<thinking_mode>enabled</thinking_mode>
<max_thinking_length>200000</max_thinking_length>"""


def is_thinking_enabled(req) -> bool:
    """Check if thinking mode is enabled in request."""
    thinking = getattr(req, 'thinking', None)
    if thinking is None:
        return False
    if isinstance(thinking, dict):
        return thinking.get('type') == 'enabled' or thinking.get('budget_tokens', 0) > 0
    return False

def get_current_timestamp() -> str:
    """Get current timestamp in CLIProxyAPIPlus format."""
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

def map_model_name(claude_model: str) -> str:
    """Map Claude model name to Amazon Q model ID.

    Accepts both short names (e.g., claude-sonnet-4) and canonical names
    (e.g., claude-sonnet-4-20250514).
    """
    DEFAULT_MODEL = "auto"

    # Available models in the service
    VALID_MODELS = {"auto", "claude-sonnet-4", "claude-sonnet-4.5", "claude-haiku-4.5", "claude-opus-4.5"}

    # Mapping from canonical names to AWS model IDs
    CANONICAL_TO_SHORT = {
        # Anthropic canonical names
        "claude-sonnet-4-20250514": "claude-sonnet-4",
        "claude-sonnet-4-5-20250929": "claude-sonnet-4.5",
        "claude-haiku-4-5-20251001": "claude-haiku-4.5",
        "claude-opus-4-5-20251101": "claude-opus-4.5",
    }

    model_lower = claude_model.lower()

    # Check if it's a valid short name
    if model_lower in VALID_MODELS:
        return model_lower

    # Check if it's a canonical name
    if model_lower in CANONICAL_TO_SHORT:
        return CANONICAL_TO_SHORT[model_lower]

    # Unknown model - log warning and return default
    logger.warning(f"Unknown model '{claude_model}', falling back to default model '{DEFAULT_MODEL}'")
    return DEFAULT_MODEL

def extract_text_from_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract text from Claude content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""

def extract_images_from_content(content: Union[str, List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Extract images from Claude content and convert to Amazon Q format."""
    if not isinstance(content, list):
        return None
    
    images = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                fmt = media_type.split("/")[-1] if "/" in media_type else "png"
                images.append({
                    "format": fmt,
                    "source": {
                        "bytes": source.get("data", "")
                    }
                })
    return images if images else None

def convert_tool(tool: ClaudeTool) -> Dict[str, Any]:
    """Convert Claude tool to Amazon Q tool."""
    desc = tool.description or ""
    # CLIProxyAPIPlus: Ensure non-empty description
    if not desc.strip():
        desc = f"Tool: {tool.name}"
    elif len(desc) > 10240:
        desc = desc[:10100] + "\n\n...(truncated)"

    return {
        "toolSpecification": {
            "name": tool.name,
            "description": desc,
            "inputSchema": {"json": tool.input_schema}
        }
    }

def merge_user_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge consecutive user messages, keeping only the last 2 messages' images."""
    if not messages:
        return {}
    
    all_contents = []
    base_context = None
    base_origin = None
    base_model = None
    all_images = []
    
    for msg in messages:
        content = msg.get("content", "")
        if base_context is None:
            base_context = msg.get("userInputMessageContext", {})
        if base_origin is None:
            base_origin = msg.get("origin", "CLI")
        if base_model is None:
            base_model = msg.get("modelId")
        
        if content:
            all_contents.append(content)
        
        # Collect images from each message
        msg_images = msg.get("images")
        if msg_images:
            all_images.append(msg_images)
    
    result = {
        "content": "\n\n".join(all_contents),
        "userInputMessageContext": base_context or {},
        "origin": base_origin or "KIRO_CLI",
        "modelId": base_model
    }
    
    # Only keep images from the last 2 messages that have images
    if all_images:
        kept_images = []
        for img_list in all_images[-2:]:  # Take last 2 messages' images
            kept_images.extend(img_list)
        if kept_images:
            result["images"] = kept_images
    
    return result

def process_history(messages: List[ClaudeMessage]) -> List[Dict[str, Any]]:
    """Process history messages to match Amazon Q format (alternating user/assistant)."""
    history = []
    seen_tool_use_ids = set()
    
    raw_history = []
    
    # First pass: convert individual messages
    for msg in messages:
        if msg.role == "user":
            content = msg.content
            text_content = ""
            tool_results = None
            images = extract_images_from_content(content)
            
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "text":
                            text_parts.append(block.get("text", ""))
                        elif btype == "tool_result":
                            if tool_results is None:
                                tool_results = []
                            
                            tool_use_id = block.get("tool_use_id")
                            raw_c = block.get("content", [])
                            
                            aq_content = []
                            if isinstance(raw_c, str):
                                aq_content = [{"text": raw_c}]
                            elif isinstance(raw_c, list):
                                for item in raw_c:
                                    if isinstance(item, dict):
                                        if item.get("type") == "text":
                                            aq_content.append({"text": item.get("text", "")})
                                        elif "text" in item:
                                            aq_content.append({"text": item["text"]})
                                    elif isinstance(item, str):
                                        aq_content.append({"text": item})
                            
                            if not any(i.get("text", "").strip() for i in aq_content):
                                aq_content = [{"text": "Tool use was cancelled by the user"}]
                                
                            # Merge if exists
                            existing = next((r for r in tool_results if r["toolUseId"] == tool_use_id), None)
                            if existing:
                                existing["content"].extend(aq_content)
                            else:
                                tool_results.append({
                                    "toolUseId": tool_use_id,
                                    "content": aq_content,
                                    "status": block.get("status", "success")
                                })
                text_content = "\n".join(text_parts)
            else:
                text_content = extract_text_from_content(content)
            
            user_ctx = {}
            if tool_results:
                user_ctx["toolResults"] = tool_results

            u_msg = {
                "content": text_content,
                "userInputMessageContext": user_ctx,
                "origin": "KIRO_CLI"
            }

            # CLIProxyAPIPlus: Ensure non-empty content
            if not u_msg["content"].strip():
                if tool_results:
                    u_msg["content"] = "Tool results provided."
                else:
                    u_msg["content"] = "Continue"

            if images:
                u_msg["images"] = images
                
            raw_history.append({"userInputMessage": u_msg})
            
        elif msg.role == "assistant":
            content = msg.content
            text_content = extract_text_from_content(content)

            # CLIProxyAPIPlus: No messageId field
            entry = {
                "assistantResponseMessage": {
                    "content": text_content
                }
            }
            
            if isinstance(content, list):
                tool_uses = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tid = block.get("id")
                        if tid and tid not in seen_tool_use_ids:
                            seen_tool_use_ids.add(tid)
                            tool_uses.append({
                                "toolUseId": tid,
                                "name": block.get("name"),
                                "input": block.get("input", {})
                            })
                if tool_uses:
                    entry["assistantResponseMessage"]["toolUses"] = tool_uses
            
            raw_history.append(entry)

    # Second pass: merge consecutive user messages
    pending_user_msgs = []
    for item in raw_history:
        if "userInputMessage" in item:
            pending_user_msgs.append(item["userInputMessage"])
        elif "assistantResponseMessage" in item:
            if pending_user_msgs:
                merged = merge_user_messages(pending_user_msgs)
                history.append({"userInputMessage": merged})
                pending_user_msgs = []
            history.append(item)
            
    if pending_user_msgs:
        merged = merge_user_messages(pending_user_msgs)
        history.append({"userInputMessage": merged})
        
    return history

def convert_claude_to_amazonq_request(req: ClaudeRequest, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert ClaudeRequest to Amazon Q request body."""
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
        
    # 1. Tools
    aq_tools = []
    long_desc_tools = []
    if req.tools:
        for t in req.tools:
            if t.description and len(t.description) > 10240:
                long_desc_tools.append({"name": t.name, "full_description": t.description})
            aq_tools.append(convert_tool(t))
            
    # 2. Current Message (last user message)
    last_msg = req.messages[-1] if req.messages else None
    prompt_content = ""
    tool_results = None
    has_tool_result = False
    images = None
    
    if last_msg and last_msg.role == "user":
        content = last_msg.content
        images = extract_images_from_content(content)
        
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_result":
                        has_tool_result = True
                        if tool_results is None:
                            tool_results = []
                        
                        tid = block.get("tool_use_id")
                        raw_c = block.get("content", [])
                        
                        aq_content = []
                        if isinstance(raw_c, str):
                            aq_content = [{"text": raw_c}]
                        elif isinstance(raw_c, list):
                            for item in raw_c:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        aq_content.append({"text": item.get("text", "")})
                                    elif "text" in item:
                                        aq_content.append({"text": item["text"]})
                                elif isinstance(item, str):
                                    aq_content.append({"text": item})
                                    
                        if not any(i.get("text", "").strip() for i in aq_content):
                            aq_content = [{"text": "Tool use was cancelled by the user"}]
                            
                        existing = next((r for r in tool_results if r["toolUseId"] == tid), None)
                        if existing:
                            existing["content"].extend(aq_content)
                        else:
                            tool_results.append({
                                "toolUseId": tid,
                                "content": aq_content,
                                "status": block.get("status", "success")
                            })
            prompt_content = "\n".join(text_parts)
        else:
            prompt_content = extract_text_from_content(content)
            
    # 3. Context - CLIProxyAPIPlus doesn't use envState
    user_ctx = {}
    if aq_tools:
        user_ctx["tools"] = aq_tools
    if tool_results:
        user_ctx["toolResults"] = tool_results

    # 4. Format Content - Match CLIProxyAPIPlus structure exactly
    # Structure:
    # --- SYSTEM PROMPT ---
    # <thinking_mode>...</thinking_mode> (if enabled)
    # <max_thinking_length>...</max_thinking_length> (if enabled)
    #
    # [Context: Current time is {timestamp}]
    #
    # {system prompt content}
    # --- END SYSTEM PROMPT ---
    #
    # {user content or "Tool results provided."}

    # Build system prompt inner content
    sys_parts = []

    # 1. Thinking hint (if enabled) - at the very beginning
    if is_thinking_enabled(req):
        sys_parts.append(THINKING_HINT)

    # 2. Timestamp context
    sys_parts.append(f"[Context: Current time is {get_current_timestamp()}]")

    # 3. System prompt content
    if req.system:
        sys_text = ""
        if isinstance(req.system, str):
            sys_text = req.system
        elif isinstance(req.system, list):
            parts = []
            for b in req.system:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
            sys_text = "\n".join(parts)
        if sys_text:
            sys_parts.append(sys_text)

    # Join system prompt parts and wrap in markers
    sys_inner = "\n\n".join(sys_parts)
    formatted_content = f"--- SYSTEM PROMPT ---\n{sys_inner}\n--- END SYSTEM PROMPT ---"

    # 4. Append user content AFTER the system prompt wrapper
    # CLIProxyAPIPlus: Never send empty content
    if has_tool_result and not prompt_content:
        formatted_content += "\n\nTool results provided."
    elif prompt_content:
        formatted_content += f"\n\n{prompt_content}"
    else:
        # Fallback for empty content
        formatted_content += "\n\nContinue"
            
    # 5. Model
    model_id = map_model_name(req.model)

    # 6. User Input Message
    user_input_msg = {
        "content": formatted_content,
        "userInputMessageContext": user_ctx,
        "origin": "KIRO_CLI",
        "modelId": model_id
    }
    if images:
        user_input_msg["images"] = images
        
    # 7. History
    history_msgs = req.messages[:-1] if len(req.messages) > 1 else []
    aq_history = process_history(history_msgs)
    
    # 8. Final Body
    return {
        "conversationState": {
            "conversationId": conversation_id,
            "history": aq_history,
            "currentMessage": {
                "userInputMessage": user_input_msg
            },
            "chatTriggerType": "MANUAL"
        }
    }
