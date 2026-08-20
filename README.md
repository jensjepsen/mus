# MUS

A small for fun library to play around with tool use in LLMs

Currently supports the Bedrock, Anthropic and OpenAI API's.

## Installation
```bash
python -m pip install "mus[all] @ https://github.com/jensjepsen/mus/releases/download/vX.X.X/mus-X.X.X-py3-none-any.whl"
```

## Usage
```python
# import stuff and make a model
import asyncio
from mus import AnthropicLLM, File, System, Bot

model = AnthropicLLM(model="claude-3.5-sonnet")
```

<!-- invisible-code-block: python
# Setup the mock model for the examples
from mus import ToolUse, ToolResult
import datetime
model.put_text("hello", "Hello")
model.put_tool_use("What is seven times three?", ToolUse(id="calc", name="calculate", input={"expression": "7 * 3"}) )
-->

```python
async def main():
    # Configuring a bot
    bot = Bot("You are a nice bot", model=model)

    # The response from the bot is a generator of deltas from the bot, so we can stream them as they come in
    async for msg in bot("hello"):
        print(msg, end="")

    # Or we can collect them all at once, by converting the response to a string
    full_response = await bot("What is 10 + 7?").string()
    assert type(full_response) == str


    # Sending images to a bot

    async for msg in bot(
            "Could you describe this image? "
            + File.image("tests/fixtures/cat.png")
            + " Do it as a poem <3"
        ):
        print(msg, end="")


    # Making a bot that can call functions

    # We use types to annotations to tell the bot the types of the arguments
    # and add a docstring to the function to tell the bot what it does
    import typing as t
    async def sum(a: t.Annotated[float, "The first operand"], b: t.Annotated[float, "The second operand"]):
        """
        Sum two numbers
        """
        return str(a + b)

    math_bot = Bot(functions=[sum], model=model)

    async for msg in math_bot("What is 10 + 7?"):
        print(msg, end="")


    # Making a bot using a decorator
    @Bot(model=model)
    def haiku_bot(topic: str):
        # The return value of the function will be the query for the bot
        # we can use the System class to add a system prompt to the bot, to make it dynamic
        return (
            System(f"You're really good at writing haikus. Current date is {datetime.datetime.now().isoformat()}")
            + f"Write a nice haiku about this topic: {topic}"
        )

    async for msg in haiku_bot("dogs"):
        print(msg, end="")


    # Making a natural language function
    @Bot(model=model).fun
    async def calculate(expression: str):
        """
        Calculate a mathematical expression
        """
        return eval(expression) # bad idea IRL, but nice for demo

    # The input to the function is now a natural language query
    result = await calculate("What is seven times three?")

    # While the return value is the result of the function
    print(result)
    assert result == 21 # and the return type of the function is preserved


    # Sandboxing a bot
    from mus import sandbox
    @sandbox(llms={"model": model})
    async def sandbot():
        """
        All the code in this function will be sandboxed,
        and run in a WASM interpreter.
        """
        import mus

        async def run_some_code(code: str):
            """
            Runs python untrusted python code, which would be a pretty bad idea without sandboxing
            """
            return exec(code)


        @mus.Bot(model=model, functions=[run_some_code])
        def danger_bot(task: str):
            return "Generate python code to solve this task: " + task
        
        async for msg in danger_bot("Generate a function that returns the sum of two numbers"):
            print(msg, end="")
        
    await sandbot()

asyncio.run(main())
```

### Cache points

A `CachePoint` marks a prompt-cache breakpoint *inside* a message. Everything before it becomes a cacheable prefix that's reused on later calls, so a large shared document or context only has to be processed once, only the content after the cache point is reprocessed.

```python
from mus import CachePoint

document = "...a very large document..."

# Compose a cache point into the query with `+`. Everything up to the
# CachePoint is cached; only the trailing question varies between calls.
query = (
    "Here is a document:\n"
    + document
    + CachePoint()  # cache everything up to here
    + "\n\nSummarise it in one sentence."
)

# Pass `query` to a bot like any other query.
```

`CachePoint` applies to the Anthropic and Bedrock backends. Providers that cache automatically (OpenAI, Google, Mistral) ignore the marker, so the same query stays portable across providers. Pass `CachePoint(ttl="1h")` to request the longer cache TTL where the provider supports it (Anthropic).


### Stop reasons

Every provider reports why generation ended. mus normalises those into a common `StopReason`, keeping the provider's own value in `raw`.

| `kind` | planned? | Anthropic | OpenAI | Bedrock | Google | Mistral |
|---|---|---|---|---|---|---|
| `end_turn` | yes | `end_turn` | `stop` | `end_turn` | `STOP` | `stop` |
| `stop_sequence` | yes | `stop_sequence` | `stop` | `stop_sequence` | `STOP` | `stop` |
| `tool_use` | yes | `tool_use` | `tool_calls` | `tool_use` | `STOP` + call | `tool_calls` |
| `max_tokens` | no | `max_tokens` | `length` | `max_tokens` | `MAX_TOKENS` | `length` |
| `content_filter` | no | `refusal` | `content_filter` | `content_filtered` | `SAFETY` | |
| `malformed_tool_call` | no | derived | derived | derived | `MALFORMED_FUNCTION_CALL` | derived |
| `pause_turn`, `error`, `unknown` | no | | | | | |

Planned stops are the ones you asked for, and are reported on the result. Only some providers distinguish a natural stop from a stop-sequence hit; where they don't, both arrive as `end_turn`.

<!-- invisible-code-block: python
# Stub responses for the examples below
model.put_text("Say hi", "Hi there!")
model.put_stop_reason("Say hi", "end_turn")
-->

```python
async def planned_stop():
    bot = Bot("You are a nice bot", model=model)

    result = bot("Say hi")
    await result.string()

    assert result.stop_reason is not None
    assert result.stop_reason.kind == "end_turn"

asyncio.run(planned_stop())
```

Every other stop raises `LLMStoppedException`, so a truncated or filtered response isn't mistaken for a complete one. The exception carries the state needed to carry on:

<!-- invisible-code-block: python
model.put_text("Write an essay", "Cartography, the art and")
model.put_stop_reason("Write an essay", "max_tokens", "length")
-->

```python
from mus import IterableResult, Query
from mus.llm.exceptions import LLMStoppedException

async def truncated():
    bot = Bot("You are a nice bot", model=model)

    try:
        await bot("Write an essay", max_tokens=64).string()
    except LLMStoppedException as e:
        assert e.stop_reason.kind == "max_tokens"
        assert e.stop_reason.raw == "length"        # the provider's own value
        assert e.partial_text == "Cartography, the art and"
        assert e.pending_tool_call is False         # was a tool call mid-flight?
        assert e.history                            # the whole turn, tool calls included

asyncio.run(truncated())
```

Recovery happens at the call site, and the continuation prompt is yours to write, since the model has no way of knowing it was cut off unless you tell it.

<!-- invisible-code-block: python
model.put_text("Continue where you stopped.", " science of map-making.")
model.put_stop_reason("Continue where you stopped.", "end_turn")
-->

```python
async def recover():
    bot = Bot("You are a nice bot", model=model)

    try:
        text = await bot("Write an essay", max_tokens=64).string()
    except LLMStoppedException as e:
        if e.stop_reason.kind == "max_tokens" and not e.pending_tool_call:
            text = e.partial_text + await IterableResult(
                bot.query(history=e.history + [Query("Continue where you stopped.")])
            ).string()

    assert text == "Cartography, the art and science of map-making."

asyncio.run(recover())
```

`partial_text` is exactly what the provider emitted, so a cut mid-word joins without a separator. Prompt the continuation to restart the final sentence to trade a few repeated tokens for a clean join.

### Recovering from a stop in-loop

Catching at the call site unwinds the whole turn, which is fine for a single call but throws away work in a long tool-calling flow: a truncation three calls deep loses the calls that already succeeded. Pass a `stop_recovery_hook` to recover in place instead, and the flow carries on.

The hook returns `StopRecoveryContinue` to keep this turn's partial output and generate onward from it, `StopRecoveryReset` to discard it and re-issue the turn, or `None` to give up (which raises, exactly as with no hook). `append` is where you tell the model what happened.

<!-- invisible-code-block: python
model.put_text("Tell me a long story", "Once upon a")
model.put_stop_reason("Tell me a long story", "max_tokens", "length")
model.put_text("You were cut off; continue.", " time, all was well.")
model.put_stop_reason("You were cut off; continue.", "end_turn")
-->

```python
from mus import StopRecoveryContinue, StopRecoveryReset

async def on_stop(error, attempt):
    if error.pending_tool_call:
        # A half-emitted tool call can't be continued -- the turn holds a
        # malformed tool block that providers reject on the next request.
        return StopRecoveryReset()
    if error.stop_reason.kind == "max_tokens":
        return StopRecoveryContinue(append=[Query("You were cut off; continue.")])
    return None  # anything else: give up and let it raise

async def long_flow():
    bot = Bot("You are a nice bot", model=model, stop_recovery_hook=on_stop)
    result = bot("Tell me a long story")
    assert await result.string() == "Once upon a time, all was well."

asyncio.run(long_flow())
```

A reset rewinds to the last *committed* point rather than the top of the turn — tools run as their deltas arrive, so rewinding past a completed tool call would fire its side effects a second time. It also yields a `DeltaStreamReset` so consumers drop the discarded output; a continue yields none, since what they have already rendered is still valid.

Returning `StopRecoveryContinue` for a stop with `pending_tool_call` is coerced to a reset, with a warning. `RetryPolicy(max_stop_recovery_attempts=...)` caps how many rounds the hook gets, separately from the pre-stream `max_recovery_attempts`.

This is a different hook from `error_recovery_hook`, which handles calls that fail *before* the stream starts (a context-window overflow, say). That one still never sees a stop.

`pending_tool_call` says whether the turn can be continued at all: `False` to append and carry on, `True` to re-issue it. It is derived from tool blocks left in flight at the stop, so it depends on what the provider exposes. Anthropic, Bedrock, OpenAI and Mistral report it; Google sets it only for `MALFORMED_FUNCTION_CALL`. A tool is never invoked with truncated arguments either way.

Values mus doesn't recognise normalise to `unknown`, which also raises, with `raw` preserved so a new provider value can be handled without waiting for a mus release.

OpenAI-compatible gateways such as OpenRouter normalise the upstream reason, and will report a planned `tool_calls` for a call the upstream actually cut off at the token limit. Where a gateway also sends the upstream value as `native_finish_reason`, mus reads it and takes the unplanned reading, since normalising can hide a truncation but never invent one. Point `OpenAILLM` at a gateway by passing your own client:

```python
from openai import AsyncClient
from mus import OpenAILLM

gateway_model = OpenAILLM(
    "openai/gpt-4o-mini",
    AsyncClient(base_url="https://openrouter.ai/api/v1", api_key="sk-or-v1-..."),
)
```


## Contributing
We use uv.

See uv docs on installation here: [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

### Install project:
```bash
uv sync --all-extras
```

### Linting & Testing
```bash
uv run pyright
uv run ruff
uv run pytest
```

### Building
```bash
uv build
```

## TODO
- [ ] BUG: Sandbox external functions fail silently, when using positional args - should either work with pos args, or enforce kwargs
- [ ] BUG: Sandbox breaks with unhelpful error when trying to use uknown classes (i.e. forgetting to import mus, and doing mus.Delta)
- [ ] Figure out mistral prefilling (w. prefix=True)
- [ ] Test merging and pruning deltas 
- [ ] Fill retry on bad LLM output
- [ ] BUG: tools are intercepted before usage is yielded, which means that usage is yielded in wrong order
- [ ] Return usage for fill operations
- [ ] Prefill examples should fail when encountering unknown type
    - [ ] Example generation should be optional and be simpler (i.e. no special chars etc)
- [ ] Allow for trimming historic messages
- [ ] Error handling
    - [X] Surface stop reasons via a common API, raising on unplanned stops
    - [ ] Handle errors from underlying sdks
    - [ ] Define possible retry strategies
        - [ ] How do we recover from wrong function input from llm?
- [ ] Add debug mode
- [ ] Add pre-commit
    - [ ] Bandit
    - [ ] commitzen
    - [ ] pyright
- [X] Add code coverage
- [ ] Pyodide patching
    - [ ] Anthropic
    - [ ] OpenAI
    - [ ] Bedrock
    - [ ] A single method to patch everything
    - [ ] Add tests that actually use pyodide?
- [ ] Add pyodide example page
- [ ] Add streaming tool inputs for:
    - [ ] Google
    - [ ] Mistral
    - [ ] Anthropic
    - [X] Bedrock
