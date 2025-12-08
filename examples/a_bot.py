
import contextlib
import httpx
import os

import typing as t
import typer
import pathlib
import asyncio
import openai
from mus import BedrockLLM, GoogleGenAILLM, OpenAILLM
async def run_bot(state: t.Optional[pathlib.Path]=None):
        
        import boto3
        import aiobotocore.session
        from mus import State, Bot
        #session = aiobotocore.session.get_session()
        #async with session.create_client("bedrock-runtime") as client:
        #    nova = BedrockLLM("eu.anthropic.claude-3-7-sonnet-20250219-v1:0", client)
        gem = GoogleGenAILLM("gemini-2.5-flash-lite")

        #openai_llm = OpenAILLM("gpt-5-mini")
        
        states = State()

        if state:
            if state.exists():
                states.load(state)
        
        async def math(op: str, a: float, b: float):
            """
            Do simple math operations.
            add, sub, mul, div
            """
            if op == "add":
                result = a + b
            elif op == "sub":
                result = a - b
            elif op == "mul":
                result = a * b
            elif op == "div":
                result = a / b
            else:
                result = "Invalid operation. Please use add, sub, mul, or div."
            
            return str(result)
        async def num(a: int):
                """
                get the secret number
                """
                return str(42)
        
        async def poem(a: str):
            """
            Return a nice poem.
            """
            return "Roses are red,\nViolets are blue,\nSugar is sweet,\nAnd so are you.".replace("Roses", a)

        prompt = """
    You are a helpful assistant. You can perform simple math operations and return the secret number.
    You can use the tools provided to answer questions.
    You can also answer questions directly without using tools.
    You will be asked to perform operations like addition, subtraction, multiplication, and division.
    You can also return the secret number when asked.
    You will be provided with a question and you should respond with the answer.
    """

        bot = Bot(prompt, functions=[math, num, poem], model=gem, cache={
            "cache_system_prompt": True,
            "cache_tools": True
        })

        class ToFill(t.TypedDict):
            """Example class to fill with random data."""
            a: str
            b: int

        #print(await bot.fill("Call ToFill with something random", ToFill))

        response = None
        h = states("history", [])
        while True:
            try:
                q = input("User: ")
                
                async for msg in (response := bot(q, previous=response)):
                    print(msg, end="")
                    if msg.usage:
                        print(f" ----- (Input: {msg.usage.input_tokens}, Output: {msg.usage.output_tokens}, Cache read: {msg.usage.cache_read_input_tokens}, Cache written: {msg.usage.cache_written_input_tokens})")
                print()
                h(h() + response.history)
            except KeyboardInterrupt:
                break
        
        if state:
            states.dump(state, indent=2)

def main(state: t.Optional[pathlib.Path]=None):
    asyncio.run(run_bot(state))

if __name__ == "__main__":
    typer.run(main)
