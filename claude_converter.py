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

import re
import base64
import copy

logger = logging.getLogger(__name__)

# Tool compression constants (from CLIProxyAPIPlus)
TOOL_COMPRESSION_TARGET_SIZE = 20 * 1024  # 20KB
MIN_TOOL_DESCRIPTION_LENGTH = 50

# Constants for empty content handling (aligned with CLIProxyAPIPlus)
# IMPORTANT: Use minimal neutral strings that the model won't mimic in responses.
# Previously used conversational phrases like "I'll help you with that." which caused
# the model to parrot them back in agentic sessions with many tool calls.
DEFAULT_ASSISTANT_CONTENT_WITH_TOOLS = "."
DEFAULT_ASSISTANT_CONTENT = "."
DEFAULT_USER_CONTENT_WITH_TOOL_RESULTS = "Tool results provided."
DEFAULT_USER_CONTENT = "Continue"

# Kiro Agentic System Prompt - injected for -agentic model variants to prevent timeouts on large writes.
# AWS Kiro API has a 2-3 minute timeout for large file write operations.
KIRO_AGENTIC_SYSTEM_PROMPT = """
# CRITICAL: CHUNKED WRITE PROTOCOL (MANDATORY)

You MUST follow these rules for ALL file operations. Violation causes server timeouts and task failure.

## ABSOLUTE LIMITS
- **MAXIMUM 350 LINES** per single write/edit operation - NO EXCEPTIONS
- **RECOMMENDED 300 LINES** or less for optimal performance
- **NEVER** write entire files in one operation if >300 lines

## MANDATORY CHUNKED WRITE STRATEGY

### For NEW FILES (>300 lines total):
1. FIRST: Write initial chunk (first 250-300 lines) using write_to_file/fsWrite
2. THEN: Append remaining content in 250-300 line chunks using file append operations
3. REPEAT: Continue appending until complete

### For EDITING EXISTING FILES:
1. Use surgical edits (apply_diff/targeted edits) - change ONLY what's needed
2. NEVER rewrite entire files - use incremental modifications
3. Split large refactors into multiple small, focused edits

### For LARGE CODE GENERATION:
1. Generate in logical sections (imports, types, functions separately)
2. Write each section as a separate operation
3. Use append operations for subsequent sections

## EXAMPLES OF CORRECT BEHAVIOR

✅ CORRECT: Writing a 600-line file
- Operation 1: Write lines 1-300 (initial file creation)
- Operation 2: Append lines 301-600

✅ CORRECT: Editing multiple functions
- Operation 1: Edit function A
- Operation 2: Edit function B
- Operation 3: Edit function C

❌ WRONG: Writing 500 lines in single operation → TIMEOUT
❌ WRONG: Rewriting entire file to change 5 lines → TIMEOUT
❌ WRONG: Generating massive code blocks without chunking → TIMEOUT

## WHY THIS MATTERS
- Server has 2-3 minute timeout for operations
- Large writes exceed timeout and FAIL completely
- Chunked writes are FASTER and more RELIABLE
- Failed writes waste time and require retry

REMEMBER: When in doubt, write LESS per operation. Multiple small operations > one large operation."""

# Web search tool renaming - Kiro API uses remote_web_search instead of web_search
REMOTE_WEB_SEARCH_DESCRIPTION = "WebSearch looks up information outside the model's training data. Supports multiple queries to gather comprehensive information."


def calculate_tools_size(tools: List[Dict[str, Any]]) -> int:
    """Calculate the JSON serialized size of the tools list."""
    if not tools:
        return 0
    return len(json.dumps(tools))


def simplify_input_schema(schema: Any) -> Any:
    """Simplify input_schema by keeping only essential fields: type, enum, required."""
    if schema is None or not isinstance(schema, dict):
        return schema

    simplified = {}
    for key in ("type", "enum", "required"):
        if key in schema:
            simplified[key] = schema[key]

    if "properties" in schema and isinstance(schema["properties"], dict):
        simplified["properties"] = {
            k: simplify_input_schema(v) for k, v in schema["properties"].items()
        }

    if "items" in schema:
        simplified["items"] = simplify_input_schema(schema["items"])

    if "additionalProperties" in schema:
        simplified["additionalProperties"] = simplify_input_schema(schema["additionalProperties"])

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and isinstance(schema[key], list):
            simplified[key] = [simplify_input_schema(item) for item in schema[key]]

    return simplified


def compress_tool_description(desc: str, target_len: int) -> str:
    """Compress description to target length with UTF-8 safe truncation."""
    if target_len < MIN_TOOL_DESCRIPTION_LENGTH:
        target_len = MIN_TOOL_DESCRIPTION_LENGTH
    if len(desc) <= target_len:
        return desc

    trunc_len = target_len - 3
    if trunc_len < MIN_TOOL_DESCRIPTION_LENGTH - 3:
        trunc_len = MIN_TOOL_DESCRIPTION_LENGTH - 3

    # UTF-8 safe truncation
    while trunc_len > 0 and (desc[trunc_len] & 0xC0) == 0x80 if isinstance(desc[trunc_len], int) else ord(desc[trunc_len]) >= 0x80 and ord(desc[trunc_len]) < 0xC0:
        trunc_len -= 1

    return desc[:trunc_len] + "..." if trunc_len > 0 else desc[:MIN_TOOL_DESCRIPTION_LENGTH]


def compress_tools_if_needed(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compress tools if total size exceeds threshold."""
    if not tools:
        return tools

    original_size = calculate_tools_size(tools)
    if original_size <= TOOL_COMPRESSION_TARGET_SIZE:
        return tools

    logger.info(f"Tools size {original_size} bytes exceeds {TOOL_COMPRESSION_TARGET_SIZE}, compressing")

    # Deep copy to avoid modifying original
    compressed = copy.deepcopy(tools)

    # Step 1: Simplify input_schema
    for tool in compressed:
        spec = tool.get("toolSpecification", {})
        if "inputSchema" in spec and "json" in spec["inputSchema"]:
            spec["inputSchema"]["json"] = simplify_input_schema(spec["inputSchema"]["json"])

    size_after_schema = calculate_tools_size(compressed)
    if size_after_schema <= TOOL_COMPRESSION_TARGET_SIZE:
        logger.info(f"Compression complete after schema simplification: {size_after_schema} bytes")
        return compressed

    # Step 2: Compress descriptions proportionally
    ratio = TOOL_COMPRESSION_TARGET_SIZE / size_after_schema * 0.8
    for tool in compressed:
        spec = tool.get("toolSpecification", {})
        desc = spec.get("description", "")
        if desc:
            target_len = max(MIN_TOOL_DESCRIPTION_LENGTH, int(len(desc) * ratio))
            spec["description"] = compress_tool_description(desc, target_len)

    final_size = calculate_tools_size(compressed)
    logger.info(f"Compression complete: {original_size} -> {final_size} bytes ({100*(original_size-final_size)/original_size:.1f}% reduction)")
    return compressed


# Image support constants
SUPPORTED_IMAGE_FORMATS = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB

# Data URL pattern: data:[<mediatype>][;base64],<data>
DATA_URL_PATTERN = re.compile(r'^data:([^;,]+)(;base64)?,(.+)$', re.DOTALL)


def parse_data_url(data_url: str) -> tuple:
    """Parse a data URL and extract media type and base64 data.

    Args:
        data_url: Data URL in format data:[<mediatype>][;base64],<data>

    Returns:
        Tuple of (media_type, base64_data)

    Raises:
        ValueError: If the data URL is invalid or not base64 encoded
    """
    match = DATA_URL_PATTERN.match(data_url)
    if not match:
        raise ValueError("Invalid data URL format")

    media_type = match.group(1)
    is_base64 = match.group(2) == ";base64"
    data = match.group(3)

    if not is_base64:
        raise ValueError("Only base64 encoded data URLs are supported")

    if media_type not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {media_type}")

    # Validate base64 encoding
    try:
        decoded = base64.b64decode(data)
        if len(decoded) > MAX_IMAGE_SIZE:
            raise ValueError(f"Image too large: {len(decoded)} bytes (max {MAX_IMAGE_SIZE})")
    except Exception as e:
        if "Image too large" in str(e):
            raise
        raise ValueError(f"Invalid base64 encoding: {e}")

    return media_type, data


def convert_image_url_to_image_source(image_url_block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert OpenAI-style image_url block to Anthropic-style image source.

    Args:
        image_url_block: Dict with "image_url" key containing {"url": "data:..."}

    Returns:
        Dict with type, media_type, and data for Anthropic format, or None on error
    """
    image_url = image_url_block.get("image_url", {})
    url = image_url.get("url", "") if isinstance(image_url, dict) else ""

    if not url:
        logger.warning("image_url block missing url field")
        return None

    if not url.startswith("data:"):
        logger.warning("Only data URLs are supported for image_url, got: %s...", url[:50])
        return None

    try:
        media_type, base64_data = parse_data_url(url)
        return {
            "type": "base64",
            "media_type": media_type,
            "data": base64_data
        }
    except ValueError as e:
        logger.warning("Failed to parse data URL: %s", e)
        return None


# Thinking mode hint - matches CLIProxyAPIPlus format
# Use "enabled" mode with 16000 token budget to reserve space for tool outputs and prevent truncation
THINKING_HINT = "<thinking_mode>enabled</thinking_mode>\n<max_thinking_length>16000</max_thinking_length>"

# Pattern for detecting AMP/Cursor format thinking tags
THINKING_MODE_PATTERN = re.compile(r"<thinking_mode>(.*?)</thinking_mode>")


def is_thinking_enabled(req, headers=None) -> bool:
    """Check if thinking mode is enabled in request or headers.

    Supports multiple formats:
    - Anthropic-Beta header with interleaved-thinking
    - Claude API format: thinking.type = "enabled"
    - OpenAI format: reasoning_effort parameter
    - AMP/Cursor format: <thinking_mode>interleaved</thinking_mode> in system prompt
    - Model name hints: model contains "thinking" or "reason"
    """
    # Check Anthropic-Beta header first (Claude Code uses this)
    if headers and is_thinking_enabled_from_header(headers):
        return True

    # Check Claude API format
    thinking = getattr(req, 'thinking', None)
    if thinking and isinstance(thinking, dict):
        if thinking.get('type') == 'enabled' or thinking.get('budget_tokens', 0) > 0:
            return True

    # Check OpenAI reasoning_effort format
    reasoning_effort = getattr(req, 'reasoning_effort', None)
    if reasoning_effort and reasoning_effort not in ['', 'none']:
        return True

    # Check AMP/Cursor format: <thinking_mode>interleaved</thinking_mode> in system prompt
    body_str = ""
    if hasattr(req, 'system') and req.system:
        if isinstance(req.system, str):
            body_str = req.system
        elif isinstance(req.system, list):
            body_str = " ".join(b.get("text", "") for b in req.system if isinstance(b, dict))

    if body_str and "<thinking_mode>" in body_str:
        match = THINKING_MODE_PATTERN.search(body_str)
        if match and match.group(1) in ("interleaved", "enabled"):
            logger.debug(f"Thinking mode enabled via AMP/Cursor format: {match.group(1)}")
            return True

    # Check model name hints
    model = getattr(req, 'model', '')
    if model and ('thinking' in model.lower() or 'reason' in model.lower()):
        logger.debug(f"Thinking mode enabled via model name hint: {model}")
        return True

    return False

def is_thinking_enabled_from_header(headers) -> bool:
    """Check if thinking mode is enabled via Anthropic-Beta header."""
    if not headers:
        return False
    beta_header = headers.get('Anthropic-Beta', '')
    return 'interleaved-thinking' in beta_header

def has_thinking_tag_in_body(req) -> bool:
    """Check if request already contains thinking configuration tags.

    This prevents duplicate injection when client (e.g., Claude Code) already includes thinking config.
    """
    if hasattr(req, 'system') and req.system:
        sys_text = ""
        if isinstance(req.system, str):
            sys_text = req.system
        elif isinstance(req.system, list):
            sys_text = " ".join(b.get("text", "") for b in req.system if isinstance(b, dict))
        if "<thinking_mode>" in sys_text or "<max_thinking_length>" in sys_text:
            return True
    return False


def normalize_origin(origin: str) -> str:
    """Normalize origin value for Kiro API compatibility.

    Maps various origin values to the canonical forms that Kiro API expects.
    This matches the behavior in CLIProxyAPIPlus.
    """
    origin_map = {
        "KIRO_CLI": "CLI",
        "KIRO_AI_EDITOR": "AI_EDITOR",
        "AMAZON_Q": "CLI",
        "KIRO_IDE": "AI_EDITOR",
    }
    return origin_map.get(origin, origin)


def map_model_name(claude_model: str) -> str:
    """Map Claude model name to Amazon Q model ID.

    Accepts both short names (e.g., claude-sonnet-4) and canonical names
    (e.g., claude-sonnet-4-20250514).
    """
    DEFAULT_MODEL = "auto"

    # Available models in the service (aligned with CLIProxyAPIPlus)
    VALID_MODELS = {
        "auto",
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-sonnet-4.6",
        "claude-haiku-4.5",
        "claude-opus-4.5",
        "claude-opus-4.6",
    }

    # Mapping from canonical names to AWS model IDs
    CANONICAL_TO_SHORT = {
        # Anthropic canonical names
        "claude-sonnet-4-20250514": "claude-sonnet-4",
        "claude-sonnet-4-5-20250929": "claude-sonnet-4.5",
        "claude-sonnet-4-6-20260217": "claude-sonnet-4.6",
        "claude-haiku-4-5-20251001": "claude-haiku-4.5",
        "claude-opus-4-5-20251101": "claude-opus-4.5",
        "claude-opus-4-6-20260201": "claude-opus-4.6",
        # Hyphenated variants (kiro format)
        "claude-opus-4-5": "claude-opus-4.5",
        "claude-opus-4-6": "claude-opus-4.6",
        "claude-sonnet-4-5": "claude-sonnet-4.5",
        "claude-sonnet-4-6": "claude-sonnet-4.6",
        "claude-haiku-4-5": "claude-haiku-4.5",
    }

    model_lower = claude_model.lower()

    # Strip -agentic/-chat suffix for model resolution (handled separately)
    for suffix in ("-agentic", "-chat"):
        if suffix in model_lower:
            model_lower = model_lower.replace(suffix, "")

    # Check if it's a valid short name
    if model_lower in VALID_MODELS:
        return model_lower

    # Check if it's a canonical name
    if model_lower in CANONICAL_TO_SHORT:
        return CANONICAL_TO_SHORT[model_lower]

    # Unknown model - pass through as-is (upstream supports many models)
    logger.info(f"Passing through unknown model '{claude_model}' as-is")
    return model_lower

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
    """Extract images from Claude content and convert to Amazon Q format.

    Supports both Anthropic format (type: "image") and OpenAI format (type: "image_url").
    """
    if not isinstance(content, list):
        return None

    images = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        # Anthropic format: type: "image" with source.type: "base64"
        if block_type == "image":
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

        # OpenAI format: type: "image_url" with image_url.url: "data:..."
        elif block_type == "image_url":
            image_source = convert_image_url_to_image_source(block)
            if image_source:
                media_type = image_source.get("media_type", "image/png")
                fmt = SUPPORTED_IMAGE_FORMATS.get(media_type, "png")
                images.append({
                    "format": fmt,
                    "source": {
                        "bytes": image_source.get("data", "")
                    }
                })

    return images if images else None

def extract_tool_choice_hint(req: ClaudeRequest) -> str:
    """Extract tool_choice from Claude request and return system prompt hint."""
    if not hasattr(req, 'tool_choice') or not req.tool_choice:
        return ""

    tool_choice = req.tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get('type', '')
        if choice_type == 'any':
            return "[INSTRUCTION: You MUST use at least one of the available tools to respond. Do not respond with text only - always make a tool call.]"
        elif choice_type == 'tool':
            tool_name = tool_choice.get('name', '')
            if tool_name:
                return f"[INSTRUCTION: You MUST use the tool named '{tool_name}' to respond. Do not use any other tool or respond with text only.]"

    return ""

def shorten_tool_name_if_needed(name: str) -> str:
    """Shorten tool names that exceed 64 characters for MCP compatibility."""
    limit = 64
    if len(name) <= limit:
        return name

    # For MCP tools, preserve prefix and last segment
    if name.startswith("mcp__"):
        idx = name.rfind("__")
        if idx > 0:
            candidate = "mcp__" + name[idx+2:]
            return candidate[:limit] if len(candidate) > limit else candidate

    return name[:limit]

def is_agentic_model(model: str) -> bool:
    """Check if model is an agentic variant (has -agentic suffix)."""
    if not model:
        return False
    return "-agentic" in model.lower()


def is_chat_only_model(model: str) -> bool:
    """Check if model is a chat-only variant (has -chat suffix).

    Chat-only mode strips all tools for pure conversation mode.
    """
    if not model:
        return False
    return "-chat" in model.lower()


def ensure_kiro_input_schema(schema: Any) -> Any:
    """Ensure tool input_schema is never None.

    Kiro API requires a valid input_schema. If None, return a default empty object schema.
    """
    if schema is not None:
        return schema
    return {"type": "object", "properties": {}}


def convert_tool(tool: ClaudeTool) -> Dict[str, Any]:
    """Convert Claude tool to Amazon Q tool."""
    # Shorten tool name if needed
    name = shorten_tool_name_if_needed(tool.name)

    desc = tool.description or ""

    # Rename web_search → remote_web_search for Kiro API compatibility
    if name == "web_search":
        name = "remote_web_search"
        desc = desc or REMOTE_WEB_SEARCH_DESCRIPTION
        logger.debug("Renamed tool web_search → remote_web_search")

    # Ensure non-empty description
    if not desc.strip():
        desc = f"Tool: {name}"
        logger.debug(f"Tool '{name}' has empty description, using default")

    # Enhanced truncation with UTF-8 safety
    # Kiro API limit is 10240 bytes, leave room for "..." suffix
    max_desc_len = 10237
    if len(desc) > max_desc_len:
        # Find safe truncation point to avoid breaking UTF-8 characters
        trunc_len = max_desc_len - 30
        while trunc_len > 0 and not desc[trunc_len:trunc_len+1].encode('utf-8', errors='ignore'):
            trunc_len -= 1
        desc = desc[:trunc_len] + "... (description truncated)"
        logger.debug(f"Tool '{name}' description truncated from {len(tool.description)} to {len(desc)} chars")

    return {
        "toolSpecification": {
            "name": name,
            "description": desc,
            "inputSchema": {"json": ensure_kiro_input_schema(tool.input_schema)}
        }
    }

def merge_user_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge consecutive user messages into one."""
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
            base_origin = msg.get("origin", "CLI")  # Use normalized origin
        if base_model is None:
            base_model = msg.get("modelId")

        if content:
            all_contents.append(content)

        # Collect images from each message
        msg_images = msg.get("images")
        if msg_images:
            all_images.extend(msg_images)

    result = {
        "content": "\n\n".join(all_contents),
        "userInputMessageContext": base_context or {},
        "origin": base_origin or "CLI",  # Use normalized origin
        "modelId": base_model
    }

    # Keep all images
    if all_images:
        result["images"] = all_images

    return result

def process_history(messages: List[ClaudeMessage], thinking_enabled: bool = False) -> List[Dict[str, Any]]:
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
                "origin": "CLI"  # Use normalized origin
            }

            # CRITICAL FIX (CLIProxyAPIPlus): Ensure non-empty content for ALL user messages
            # This must happen BEFORE the isLastMessage check to fix compaction requests
            if not u_msg["content"].strip():
                if tool_results:
                    u_msg["content"] = DEFAULT_USER_CONTENT_WITH_TOOL_RESULTS
                else:
                    u_msg["content"] = DEFAULT_USER_CONTENT
                logger.debug(f"User content was empty, using default: {u_msg['content']}")

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
                            tool_name = block.get("name")
                            # Rename web_search → remote_web_search to match convertClaudeToolsToKiro
                            if tool_name == "web_search":
                                tool_name = "remote_web_search"
                            tool_uses.append({
                                "toolUseId": tid,
                                "name": tool_name,
                                "input": block.get("input", {})
                            })
                if tool_uses:
                    entry["assistantResponseMessage"]["toolUses"] = tool_uses

            # CRITICAL FIX (CLIProxyAPIPlus): Kiro API requires non-empty content for assistant messages
            # Use minimal neutral string to prevent model parroting
            if not entry["assistantResponseMessage"]["content"].strip():
                if entry["assistantResponseMessage"].get("toolUses"):
                    entry["assistantResponseMessage"]["content"] = DEFAULT_ASSISTANT_CONTENT_WITH_TOOLS
                else:
                    entry["assistantResponseMessage"]["content"] = DEFAULT_ASSISTANT_CONTENT
                logger.debug(f"Assistant content was empty, using default: {entry['assistantResponseMessage']['content']}")

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

def deduplicate_tool_results(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate tool results by toolUseId.

    If multiple tool results share the same toolUseId, keep only the first one.
    This prevents duplicate results from being sent to the Kiro API.
    """
    if not tool_results:
        return tool_results
    seen = set()
    deduped = []
    for tr in tool_results:
        tid = tr.get("toolUseId", "")
        if tid not in seen:
            seen.add(tid)
            deduped.append(tr)
        else:
            logger.debug(f"Deduplicated tool result with toolUseId: {tid}")
    return deduped

def convert_claude_to_amazonq_request(req: ClaudeRequest, conversation_id: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Convert ClaudeRequest to Amazon Q request body.

    Aligned with CLIProxyAPIPlus implementation. Supports:
    - Tool calling with web_search → remote_web_search renaming
    - Agentic mode with chunked write optimization
    - Chat-only mode (strips tools for -chat model variants)
    - InferenceConfig (max_tokens, temperature, top_p)
    - Thinking mode detection (5 methods)
    - Empty content handling
    """
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    # Check model variants
    is_agentic = is_agentic_model(req.model)
    is_chat_only = is_chat_only_model(req.model)

    # Extract inference parameters
    max_tokens = getattr(req, 'max_tokens', None)
    temperature = getattr(req, 'temperature', None)
    top_p = getattr(req, 'top_p', None)

    # Handle max_tokens = -1 as "use maximum" (Kiro max output is ~32000 tokens)
    if max_tokens == -1:
        max_tokens = 32000
        logger.debug("max_tokens=-1 converted to 32000")

    # 1. Tools - chat-only mode strips all tools
    aq_tools = []
    if req.tools and not is_chat_only:
        for t in req.tools:
            aq_tools.append(convert_tool(t))
        # Apply dynamic compression if total tools size exceeds threshold
        aq_tools = compress_tools_if_needed(aq_tools)

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
    # Deduplicate tool results before building context
    if tool_results:
        tool_results = deduplicate_tool_results(tool_results)
    user_ctx = {}
    if aq_tools:
        user_ctx["tools"] = aq_tools
    if tool_results:
        user_ctx["toolResults"] = tool_results

    # 4. Format Content - Match CLIProxyAPIPlus structure exactly
    # Structure:
    # --- SYSTEM PROMPT ---
    # <thinking_mode>enabled</thinking_mode> (if enabled)
    # <max_thinking_length>16000</max_thinking_length> (if enabled)
    #
    # [Context: Current time is {timestamp}]
    #
    # {system prompt content}
    #
    # {agentic prompt} (if -agentic model)
    # {tool_choice hint}
    # --- END SYSTEM PROMPT ---
    #
    # {user content or fallback}

    # Build system prompt inner content
    sys_parts = []

    # 1. Thinking hint - only inject if not already present in system prompt
    # CLIProxyAPIPlus: Skip injection if client (e.g., Claude Code) already includes thinking config
    if is_thinking_enabled(req, headers) and not has_thinking_tag_in_body(req):
        sys_parts.append(THINKING_HINT)

    # 2. Timestamp context
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    sys_parts.append(f"[Context: Current time is {timestamp}]")

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

    # 4. Agentic mode optimization prompt (for -agentic model variants)
    if is_agentic:
        sys_parts.append(KIRO_AGENTIC_SYSTEM_PROMPT)
        logger.debug("Injected agentic mode optimization prompt")

    # 5. Tool choice hint (if specified)
    tool_choice_hint = extract_tool_choice_hint(req)
    if tool_choice_hint:
        sys_parts.append(tool_choice_hint)

    # Join system prompt parts and wrap in markers
    sys_inner = "\n\n".join(sys_parts)

    # 10. History
    history_msgs = req.messages[:-1] if len(req.messages) > 1 else []
    aq_history = process_history(history_msgs, thinking_enabled=is_thinking_enabled(req, headers))

    # CLIProxyAPIPlus: Only inject system prompt on first turn to avoid re-injection
    effective_system_prompt = sys_inner
    if len(aq_history) > 0:
        effective_system_prompt = ""

    if effective_system_prompt:
        formatted_content = f"--- SYSTEM PROMPT ---\n{effective_system_prompt}\n--- END SYSTEM PROMPT ---"
    else:
        formatted_content = ""

    # 7. Append user content
    # CRITICAL FIX (CLIProxyAPIPlus): Never send empty content
    if has_tool_result and not prompt_content.strip():
        user_text = DEFAULT_USER_CONTENT_WITH_TOOL_RESULTS
    elif prompt_content.strip():
        user_text = prompt_content
    else:
        user_text = DEFAULT_USER_CONTENT

    if formatted_content:
        formatted_content += f"\n\n{user_text}"
    else:
        formatted_content = user_text

    logger.debug(f"Final content length: {len(formatted_content)}, has_tool_result: {has_tool_result}, prompt_content_len: {len(prompt_content)}")

    # 8. Model
    model_id = map_model_name(req.model)

    # 9. User Input Message - use normalized origin
    user_input_msg = {
        "content": formatted_content,
        "userInputMessageContext": user_ctx,
        "origin": normalize_origin("KIRO_CLI"),  # Normalize origin: KIRO_CLI -> CLI
        "modelId": model_id
    }
    if images:
        user_input_msg["images"] = images

    # 11. Build inferenceConfig if we have any inference parameters
    inference_config = None
    if max_tokens or temperature is not None or top_p is not None:
        inference_config = {}
        if max_tokens:
            inference_config["maxTokens"] = max_tokens
        if temperature is not None:
            inference_config["temperature"] = temperature
        if top_p is not None:
            inference_config["topP"] = top_p

    # 12. Final Body
    result = {
        "conversationState": {
            "chatTriggerType": "MANUAL",  # Must be first field per CLIProxyAPIPlus
            "conversationId": conversation_id,
            "currentMessage": {
                "userInputMessage": user_input_msg
            },
            "history": aq_history
        }
    }

    # Add inferenceConfig if present (Kiro API supports this)
    if inference_config:
        result["inferenceConfig"] = inference_config
        logger.debug(f"Added inferenceConfig: {inference_config}")

    return result
