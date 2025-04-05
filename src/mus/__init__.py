from .main import Mus
from .llm.types import File, Query, ToolUse, ToolResult, Assistant
from .llm.llm import IterableResult
import sys

if not "extism" in sys.modules:
    # If extism is already imported, we're probably in a WASM env already
    try:
        import extism
    except ImportError:
        pass
    else:
        from ._sandbox import sandbox

try:
    import anthropic
except ImportError:
    pass
else:
    from .llm.anthropic import AnthropicLLM

try:
    import boto3
except ImportError:
    pass
else:
    from .llm.bedrock import BedrockLLM

try:
    import openai
except ImportError:
    pass
else:
    from .llm.openai import OpenAILLM