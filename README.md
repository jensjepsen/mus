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

There is no recovery hook, because a half-emitted tool call can't be continued: the assistant turn holds a malformed tool block that providers reject on the next request. Recovery happens at the call site instead, and the continuation prompt is yours to write, since the model has no way of knowing it was cut off unless you tell it.

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


### Durable runs

A mus run is normally ephemeral: if the process dies mid-conversation the turn is lost, completed tool calls are forgotten, and a reconnecting client has nothing to reattach to. `mus.dbos` runs a bot inside a [DBOS](https://docs.dbos.dev/) workflow so a run survives a crash or a deploy, without re-billing completed provider calls or re-firing completed tools, and can be tailed from anywhere by workflow id.

Install the extra with `pip install "mus[dbos]"`.

<!-- invisible-code-block: python
import importlib.util
have_dbos = importlib.util.find_spec("dbos") is not None
-->
<!-- skip: start if(not have_dbos, "needs the dbos extra") -->
<!-- invisible-code-block: python
# Set up DBOS on a throwaway SQLite database, and a stubbed model, so the
# examples below actually run.
import tempfile
from dbos import DBOS, DBOSConfig, SetWorkflowID
from mus import ToolUse
from mus import dbos as mus_dbos

_db = f"sqlite:///{tempfile.mkdtemp()}/mus.sqlite"
DBOS.destroy()
DBOS(config=DBOSConfig(name="mus-readme", database_url=_db, system_database_url=_db))
DBOS.launch()

async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return {"Paris": "17C, rain"}.get(city, "unknown")

QUESTION = "What is the weather in Paris?"
model.put_tool_use(QUESTION, ToolUse(id="t1", name="get_weather", input={"city": "Paris"}))
-->

The bot is built *inside* the workflow, and `bot.query` runs in the workflow body. Each provider call and each tool call becomes a checkpointed step, so a recovered run picks up where it stopped instead of replaying from the top.

```python
@DBOS.workflow()
async def weather_agent(question: str) -> str:
    bot = mus_dbos.durable(Bot(
        prompt="Use the tool for every city mentioned.",
        model=model,
        functions=[get_weather],
    ))
    result = bot(question)
    text = await result.string()
    await bot.close()
    return text
```

Start a run and hand out its id, then read it from anywhere -- another process, a web handler -- while it is still running. Keying the workflow id on your request id also makes the run idempotent: a retried request re-attaches to the existing run rather than starting a second conversation.

```python
async def demo():
    with SetWorkflowID("support-42"):
        handle = await DBOS.start_workflow_async(weather_agent, QUESTION)
    workflow_id = handle.workflow_id
    answer = await handle.get_result()

    # Tail the run's deltas by id -- this works from any process.
    streamed = [delta async for delta in mus_dbos.read(workflow_id)]
    assert len(streamed) > 0

    # Or reattach as the result object you already know, which behaves exactly
    # as it does in-process.
    result = mus_dbos.attach(workflow_id)
    assert await result.string() == answer
    return result.usage, result.stop_reason

usage, stop_reason = asyncio.run(demo())
```

Pass `offset=` to resume mid-stream, so a client that reconnects after a dropped socket or a page refresh continues where it left off rather than replaying the transcript.

**What is guaranteed.** A tool that *completed* before a crash never runs again. A tool interrupted *mid-execution* does run again, because DBOS cannot know whether its side effect landed -- so side-effecting tools still need to be idempotent for that window. What durability buys is that the window is one tool call, not the whole run.

**Failures reach the reader.** If the run raises, a terminal delta carrying `metadata["mus.error"]` is written before the error propagates, so a failed run is distinguishable from a truncated one.

Tools may be defined anywhere -- module level, closures over local state, callable objects, or built dynamically per run. Only the wrapper is registered with DBOS; your tool is reached through mus's own invocation path, so schema validation and the fallback function still apply.

Without the extra installed `mus.dbos` still imports, but `durable()`, `read()` and `attach()` raise. A `durable()` that quietly wasn't durable would be worse than an error -- run the bot unwrapped if you don't need durability.

<!-- invisible-code-block: python
DBOS.destroy()
-->
<!-- skip: end -->


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
