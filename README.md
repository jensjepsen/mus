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
