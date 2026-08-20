"""Helper process: crash *inside* a provider turn, mid-token.

The provider step writes raw deltas as they arrive. If the process dies partway
through, that partial prefix stays in the raw key -- streams are append-only --
and on recovery the step re-runs from scratch and appends its deltas again.

_DurableLLM finds its turn by counting offsets, so the question is whether it
then reads the abandoned prefix as though it were the new turn.

    python dbos_offset_helper.py <db> <mode:crash|recover>
prints the text mus actually received.
"""

import asyncio
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import Delta, DeltaText
from mus.llm.llm import Bot
from mus.llm.types import LLM
import mus.dbos as mus_dbos

DB, MODE = sys.argv[1], sys.argv[2]
WF_ID = "offset-crash-1"
TOKENS = ["Once", " upon", " a", " time"]
CRASH_AFTER = 2  # tokens written before the process dies

DBOS(config=DBOSConfig(
    name="mus-offset-test",
    system_database_url=f"sqlite:///{DB}",
    database_url=f"sqlite:///{DB}",
))


class TokenLLM(LLM):
    """Yields tokens one at a time; dies mid-turn in crash mode."""

    provider = "tokens"

    def __init__(self):
        pass

    async def stream(self, **kwargs):
        for i, tok in enumerate(TOKENS):
            yield Delta(content=DeltaText(data=tok))
            # The step has written token i by the time we resume here.
            if MODE == "crash" and i == CRASH_AFTER - 1:
                os._exit(9)


@DBOS.workflow()
async def run() -> str:
    bot = mus_dbos.durable(Bot(prompt="t", model=TokenLLM()))
    result = bot("go")
    text = await result.string()
    await bot.close()
    return text


async def main():
    if MODE == "crash":
        with SetWorkflowID(WF_ID):
            handle = await DBOS.start_workflow_async(run)
        print(await handle.get_result())
    else:
        handle = await DBOS.retrieve_workflow_async(WF_ID)
        print(await handle.get_result())


if __name__ == "__main__":
    DBOS.launch()
    try:
        asyncio.run(main())
    finally:
        DBOS.destroy()
