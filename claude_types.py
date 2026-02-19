from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field

class ClaudeMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ClaudeTool(BaseModel):
    name: str
    description: Optional[str] = ""
    input_schema: Optional[Dict[str, Any]] = None

class ClaudeRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    max_tokens: int = 4096
    temperature: Optional[float] = None
    top_p: Optional[float] = None  # Added for CLIProxyAPIPlus alignment
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # Added for tool_choice hint
    stream: bool = False
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    thinking: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None  # OpenAI-style thinking mode ("low", "medium", "high")