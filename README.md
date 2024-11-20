# MUS

A small for fun library to play around with tool use in LLMs

Currently supports the Anthropic Claude family of models on AWS Bedrock, through the `anthropic[bedrock]` package.

## Installation
```bash
python -m pip install git+https://github.com/jensjepsen/mus.git
```

## Usage
<!-- invisible-code-block: python
# This could be some state setup needed to demonstrate things
-->
```python
from mus import Mus, AnthropicLLM
from anthropic import AnthropicBedrock
m = Mus()
client = AnthropicLLM(AnthropicBedrock(
    aws_region="us-west-2",
))

# Configuring a bot
bot = m.llm("You are a nice bot", client=client, model="anthropic.claude-3-5-sonnet-20241022-v2:0")
client.put_text("hello")
for msg in bot("hello"):
    assert str(msg) == "hello", msg
    m.print(msg)

# Making a bot that can call functions
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
    return f"""
        Write a nice haiku about this topic: {topic}
    """

for msg in haiku_bot("dogs"):
    print(msg)
```


## TODO
- [X] Pass additional arguments to stream calls in underlying sdk, such as num tokens, headers etc
    - [X] Max tokens
    - [X] Model
    - [X] Top k
    - [X] Top p
    - [X] Temperature
    - [X] Headers
- [ ] Document .func decorator
- [X] Make LLM.__call__ function as primary decorator to avoid having to do .bot
- [ ] Abstract away underlying api message structure into four message types, system, user, assistant, tool, with functions to convert to and from
- [ ] Allow for trimming historic messages
- [ ] Return usage stats, such as tokens generated etc in `IterableResult`
- [ ] OpenAI client
- [ ] Bedrock Converse client
