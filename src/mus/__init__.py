from .llm.types import File as File, Query as Query, ToolUse as ToolUse, ToolResult as ToolResult, Assistant as Assistant, System as System
from .llm.llm import IterableResult as IterableResult, Bot as Bot, LLM as LLM
from .llm.mock_client import StubLLM as StubLLM
from .state import State as State, StateReference as StateReference
import sys
import importlib.util

# Try to import wasmtime, if it fails, we are probably in a sandbox
# so we can't import it again
# and we can't nest sandboxes
if "wasmtime" not in sys.modules:
    if importlib.util.find_spec("wasmtime"):
        from ._sandbox import sandbox as sandbox

if importlib.util.find_spec("anthropic"):
    from .llm.anthropic import AnthropicLLM as AnthropicLLM

if importlib.util.find_spec("boto3"):
    from .llm.bedrock import BedrockLLM as BedrockLLM

if importlib.util.find_spec("openai"):
    from .llm.openai import OpenAILLM as OpenAILLM

if importlib.util.find_spec("google.genai"):
    from .llm.google import GoogleGenAILLM as GoogleGenAILLM