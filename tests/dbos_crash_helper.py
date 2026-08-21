"""Helper process for the closure-tool crash test.

Run as a subprocess so the crash is a real process death, leaving the workflow
PENDING for the next launch to recover. A completed workflow is never replayed,
so an in-process "run it twice" shortcut proves nothing.

    python dbos_crash_helper.py <db_path> <effects_path> crash|recover
"""

import asyncio
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import StubLLM
from mus.llm.llm import Bot
from mus.llm.types import ToolUse
import mus.dbos as mus_dbos

DB, EFFECTS, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
WF_ID = "closure-crash-1"

DBOS(config=DBOSConfig(
    name="mus-crash-test",
    system_database_url=f"sqlite:///{DB}",
    database_url=f"sqlite:///{DB}",
))


@DBOS.workflow()
async def run() -> list:
    # Defined inside the workflow: rebuilt on every replay, which is the risk.
    async def record(city: str) -> str:
        """Record a city."""
        with open(EFFECTS, "a") as f:
            f.write(city + "\n")
        # Die *after* the tool's side effect but before its step is recorded on
        # the second call, so the first call is checkpointed and the second is not.
        if MODE == "crash" and city == "Tokyo":
            os._exit(9)
        return "recorded:" + city

    model = StubLLM()
    # One turn asking for both, so the first tool completes and checkpoints
    # before the second one dies.
    model.put_tool_use("go", ToolUse(id="t1", name="record", input={"city": "Paris"}))
    model.put_tool_use("go", ToolUse(id="t2", name="record", input={"city": "Tokyo"}))
    bot = mus_dbos.durable(Bot(prompt="t", model=model, functions=[record]))
    out = [
        d.content.data.content.val
        async for d in bot.query("go")
        if type(d.content).__name__ == "DeltaToolResult"
    ]
    await bot.close()
    return out


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
