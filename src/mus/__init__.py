from .llm.types import File, Query, ToolUse, ToolResult, Assistant, System
from .llm.llm import IterableResult, Bot
from .llm.mock_client import StubLLM
from .state import State, StateReference
import sys

# Try to import wasmtime, if it fails, we are probably in a sandbox
# so we can't import it again
# and we can't nest sandboxes
if not "wasmtime" in sys.modules:
    try:
        import wasmtime
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

try:
    from google import genai
except ImportError:
    pass
else:
    from .llm.google import GoogleGenAILLM