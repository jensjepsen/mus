# MUS

A small for fun library to play around with tool use in LLMs

Currently supports the Anthropic Claude family of models on AWS Bedrock.

## Installation
python -m pip install git+https://github.com/jensjepsen/mus.git

## Usage
```python
from mus import Mus
from anthropic import AnthropicBedrock
m = Mus()
client = AnthropicBedrock(
    aws_region="us-west-2",
)

# Configuring a bot
bot = m.llm("You are a nice bot")

for msg in bot("hello"):
    m.print(msg)

# Making a bot that can call functions
import typing as t
def sum(a: t.Annotated[float, "The first operand"], b: t.Annotated[float, "The second operand"]):
    return str(a + b)

math_bot = m.llm(functions=[sum])

for msg in math_bot("What is 10 + 7?"):
    m.print(msg)


# Making a bot using a decorator
@m.llm("You write nice haikus").bot
def haiku_bot(topic: str):
    return f"""
        Write a nice haiku about this topic: {topic}
    """

for msg in haiku_bot("dogs"):
    print(msg)
```


## TODO
- [] Pass additional arguments to stream calls in underlying sdk, such as num tokens, headers etc
- [] Document .func decorator
- [] Make LLM.__call__ function as primary decorator to avoid having to do .bot or .func
