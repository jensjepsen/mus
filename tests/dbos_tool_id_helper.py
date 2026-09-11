"""Helper: a re-run that asks for FEWER tools than the killed run did.

Each ``tool_invocation_id`` is minted by a ``mus.id`` step, and a step is a
positional operation -- so the NUMBER of tool ids a turn mints is part of the
workflow's determinism contract. A turn killed mid-stream is not checkpointed,
so recovery re-runs it against a live model, which may ask for fewer tools the
second time. The body then reaches a different operation at a position where
``mus.id`` was recorded, and DBOS aborts the run.

Fewer is the failing direction: extra ids land on fresh function_ids past the
recorded tail and go unnoticed, which makes this intermittent in production.

    python dbos_tool_id_helper.py <db> <marker> crash|recover
"""

import asyncio
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import StubLLM
from mus.llm.llm import Bot
from mus.llm.types import ToolUse
import mus.dbos as mus_dbos

DB, MARKER, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
WF_ID = "tool-id-determinism-1"

DBOS(config=DBOSConfig(
    name="mus-tool-ids",
    database_url=f"sqlite:///{DB}",
    system_database_url=f"sqlite:///{DB}",
))


def already_crashed() -> bool:
    return os.path.exists(MARKER)


class SlowStub(StubLLM):
    """Paced so the kill lands while the provider step is still running."""

    async def stream(self, **kwargs):
        async for delta in super().stream(**kwargs):
            await asyncio.sleep(0.05)
            yield delta


async def record(city: str) -> str:
    """Record a city."""
    return "ok:" + city


@DBOS.workflow()
async def run() -> int:
    model = SlowStub()
    # The killed run asks for three tools; the re-run asks for one, as a live
    # model legitimately might.
    cities = ["Paris"] if already_crashed() else ["Paris", "Tokyo", "Cairo"]
    for i, city in enumerate(cities):
        model.put_tool_use(
            "go", ToolUse(id=f"t{i}", name="record", input={"city": city})
        )
    for i in range(8):
        model.put_text("go", f"tail{i} ")

    bot = mus_dbos.durable(Bot(prompt="t", model=model, functions=[record]))
    seen = 0
    async for _ in bot.query("go"):
        seen += 1
        # Kill once all three tool ids are minted, with the turn still streaming
        # so its step never checkpoints.
        if not already_crashed() and seen >= 4:
            with open(MARKER, "w") as f:
                f.write(str(seen))
            print(f"    [crash] after {seen} deltas", flush=True)
            os._exit(9)
    await bot.close()
    return seen


async def main():
    if MODE == "crash":
        with SetWorkflowID(WF_ID):
            handle = await DBOS.start_workflow_async(run)
        await handle.get_result()
    else:
        handle = await DBOS.retrieve_workflow_async(WF_ID)
        print(f"RESULT recovered seen={await handle.get_result()}")


if __name__ == "__main__":
    DBOS.launch()
    try:
        asyncio.run(main())
    finally:
        DBOS.destroy()
