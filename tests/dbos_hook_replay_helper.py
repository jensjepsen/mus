"""Helper: does on_delta re-fire for a turn served from its checkpoint?

The hook runs inside ``mus.provider_turn``. A replayed turn does not execute
its step, so the hook must not fire for it -- otherwise a consumer counting
tokens, billing, or streaming to a client double-counts every recovery.

The kill lands inside the tool, after the first turn has checkpointed. On
recovery that turn is served from its record (hook silent) while the
continuation runs live (hook fires).

    python dbos_hook_replay_helper.py <db> <marker> <hooklog> crash|recover
"""

import asyncio
import json
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import StubLLM
from mus.llm.llm import Bot
from mus.llm.types import DeltaText, ToolUse
import mus.dbos as mus_dbos

DB, MARKER, HOOKLOG, MODE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
WF_ID = "hook-replay-1"

DBOS(config=DBOSConfig(
    name="mus-hook-replay",
    database_url=f"sqlite:///{DB}",
    system_database_url=f"sqlite:///{DB}",
))


def already_crashed() -> bool:
    return os.path.exists(MARKER)


RUN = "recovery" if already_crashed() else "first"


async def on_delta(delta):
    text = delta.content.data if isinstance(delta.content, DeltaText) else ""
    with open(HOOKLOG, "a") as f:
        f.write(f"{RUN}\t{type(delta.content).__name__}\t{text}\n")


async def record(city: str) -> str:
    """Record a city."""
    # Die inside the tool: the provider turn above it has already checkpointed,
    # so recovery replays that turn rather than re-running it.
    if not already_crashed():
        with open(MARKER, "w") as f:
            f.write(city)
        print("    [crash] inside the tool", flush=True)
        os._exit(9)
    return "ok:" + city


@DBOS.workflow()
async def run() -> int:
    model = StubLLM()
    model.put_text("go", "PROLOGUE ")
    model.put_tool_use("go", ToolUse(id="t1", name="record", input={"city": "Paris"}))
    bot = mus_dbos.durable(
        Bot(prompt="t", model=model, functions=[record]), on_delta=on_delta
    )
    seen = 0
    async for _ in bot.query("go"):
        seen += 1
    await bot.close()
    return seen


async def main():
    if MODE == "crash":
        with SetWorkflowID(WF_ID):
            handle = await DBOS.start_workflow_async(run)
        await handle.get_result()
    else:
        handle = await DBOS.retrieve_workflow_async(WF_ID)
        seen = await handle.get_result()
        rows = [l.split("\t") for l in open(HOOKLOG).read().splitlines()]
        print("RESULT " + json.dumps({
            "seen": seen,
            "first": [f"{k}:{t}" for r, k, t in rows if r == "first"],
            "recovery": [f"{k}:{t}" for r, k, t in rows if r == "recovery"],
        }))


if __name__ == "__main__":
    DBOS.launch()
    try:
        asyncio.run(main())
    finally:
        DBOS.destroy()
