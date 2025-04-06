from .main import Mus
from .llm.types import File, Query, ToolUse, ToolResult, Assistant, System
from .llm.llm import IterableResult
import sys

# If extism is already imported, we're probably in a WASM env already
# so we can't import it again
# and we can't nest sandboxes
if not "extism" in sys.modules:
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