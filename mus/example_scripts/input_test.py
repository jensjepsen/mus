from .. import run

from ..stub import *


def something(asd: str):
    """prints something"""
    print(asd)

@run(something=something, input=input)
def setup():
    """Prints 'Hello, world!'"""
    print("Hello, world!")
    def inner():
        """Prints 'Inner function'"""
        print("Inner function")
    inner()
    something("\n----else----\n")
    hello = llm("You're a poet, always answer in haiku")

    class Hello:
        poem: Annotated[str, "The topic of the poem"]

    c = hello("What's up", Hello)
    print(c)

    for b in hello("poem"):
        print(b)

setup()