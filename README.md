# MUS

A small for fun library to play around with tool use in LLMs

Currently supports the Anthropic Claude family of models on AWS Bedrock, through the `anthropic[bedrock]` package.

## Installation
```bash
python -m pip install git+https://github.com/jensjepsen/mus.git
```

## Usage
```python
# import stuff and make a client
from mus import Mus, AnthropicLLM, File
from anthropic import AnthropicBedrock
m = Mus()
client = AnthropicLLM(AnthropicBedrock(
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
# Configuring a bot
bot = m.llm("You are a nice bot", client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0")

# The response from the bot is a generator of deltas from the bot, so we can stream them as they come in
for msg in bot("hello"):
    m.print(msg)

# Or we can collect them all at once, by converting the response to a string
full_response = str(bot("What is 10 + 7?"))
assert type(full_response) == str


# Sending images to a bot

for msg in bot(
        "Could you describe this image? "
        + File.image("tests/fixtures/cat.png")
        + " Do it as a poem <3"
    ):
    m.print(msg)


# Making a bot that can call functions

# We use types to annotations to tell the bot the types of the arguments
# and add a docstring to the function to tell the bot what it does
import typing as t
def sum(a: t.Annotated[float, "The first operand"], b: t.Annotated[float, "The second operand"]):
    """
    Sum two numbers
    """
    return str(a + b)

math_bot = m.llm(functions=[sum], client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0")

for msg in math_bot("What is 10 + 7?"):
    m.print(msg)


# Making a bot using a decorator
@m.llm("You write nice haikus", client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0")
def haiku_bot(topic: str):
    # The return value of the function will be the query for the bot
    return f"""
        Write a nice haiku about this topic: {topic}
    """

for msg in haiku_bot("dogs"):
    print(msg)


# Making a natural language function
@m.llm(client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0").fun
def calculate(expression: str):
    """
    Calculate a mathematical expression
    """
    return eval(expression) # bad idea IRL, but nice for demo

# The input to the function is now a natural language query
result = calculate("What is seven times three?")

# While the return value is the result of the function
print(result)
assert result == 21 # and the return type of the function is preserved
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
- [ ] Abstract away underlying api message structure
- [ ] Allow for trimming historic messages
- [ ] Return usage stats, such as tokens generated etc in `IterableResult`
- [ ] OpenAI client
- [ ] Bedrock Converse client
