# MUS

A small for fun library to play around with tool use in LLMs

Currently supports the Bedrock, Anthropic and OpenAI API's.

## Installation
```bash
python -m pip install "mus[all] @ https://github.com/jensjepsen/mus/releases/download/vX.X.X/mus-X.X.X-py3-none-any.whl"
```

## Usage
```python
# import stuff and make a client
import asyncio
from mus import Mus, AnthropicLLM, File
from anthropic import AsyncAnthropicBedrock
m = Mus()
client = AnthropicLLM(AsyncAnthropicBedrock(
    aws_region="us-west-2",
))
```

<!-- invisible-code-block: python
# Setup the mock client for the examples
from mus import ToolUse, ToolResult
client.put_text("hello", "Hello")
client.put_tool_use("What is seven times three?", ToolUse(id="calc", name="calculate", input={"expression": "7 * 3"}) )
client.put_tool_result("What is seven times three?", ToolResult(id="calc", content="21"))
-->

```python
async def main():
    # Configuring a bot
    bot = m.llm("You are a nice bot", client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0")

    # The response from the bot is a generator of deltas from the bot, so we can stream them as they come in
    async for msg in bot("hello"):
        m.print(msg)

    # Or we can collect them all at once, by converting the response to a string
    full_response = await bot("What is 10 + 7?").string()
    assert type(full_response) == str


    # Sending images to a bot

    async for msg in bot(
            "Could you describe this image? "
            + File.image("tests/fixtures/cat.png")
            + " Do it as a poem <3"
        ):
        m.print(msg)


    # Making a bot that can call functions

    # We use types to annotations to tell the bot the types of the arguments
    # and add a docstring to the function to tell the bot what it does
    import typing as t
    async def sum(a: t.Annotated[float, "The first operand"], b: t.Annotated[float, "The second operand"]):
        """
        Sum two numbers
        """
        return str(a + b)

    math_bot = m.llm(functions=[sum], client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0")

    async for msg in math_bot("What is 10 + 7?"):
        m.print(msg)


    # Making a bot using a decorator
    @m.llm("You write nice haikus", client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0")
    def haiku_bot(topic: str):
        # The return value of the function will be the query for the bot
        return f"""
            Write a nice haiku about this topic: {topic}
        """

    async for msg in haiku_bot("dogs"):
        print(msg)


    # Making a natural language function
    @m.llm(client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0").fun
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
asyncio.run(main())
```


## Contributing
We use pipx, poetry and poethepoet.
```bash
python -m pip install pipx
poetry install --all-extras
```

### Testing
```bash
poetry poe test
```

### Building
```bash
poetry build
```

## TODO
- [X] Pass additional arguments to stream calls in underlying sdk, such as num tokens, headers etc
    - [X] Max tokens
    - [X] Model
    - [X] Top k
    - [X] Top p
    - [X] Temperature
    - [X] Headers
- [X] Document .fun decorator
- [X] Make LLM.__call__ function as primary decorator to avoid having to do .bot
- [X] Abstract away underlying api message structure
- [ ] Allow for trimming historic messages
- [X] Return usage stats, such as tokens generated etc in `IterableResult`
- [X] OpenAI client
- [X] Bedrock Converse client
- [ ] Error handling
    - [ ] Handle errors from underlying sdks
    - [ ] Define possible retry strategies
        - [ ] How do we recover from wrong function input from llm?
- [ ] Add debug mode
- [ ] Add pre-commit
    - [ ] Bandit
    - [ ] commitzen
    - [ ] pyright
- [X] Allow passing stop sequences to llm
- [X] Add code coverage
- [ ] Pyodide patching
    - [ ] Anthropic
    - [ ] OpenAI
    - [ ] Bedrock
    - [ ] A single method to patch everything
    - [ ] Add tests that actually use pyodide?
- [ ] Add pyodide example page
- [X] Default client init from client wrapper, to avoid having to pass the low level client explicitly
- [ ] Remove all interpreter code
- [ ] Extism first class support w. tests

