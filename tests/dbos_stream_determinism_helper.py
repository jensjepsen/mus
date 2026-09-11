"""Helper for the mid-provider-turn crash test.

A crash while the provider is still streaming leaves ``mus.provider_turn``
uncheckpointed, so recovery re-runs it live rather than replaying it. A real
model emits a different number of deltas the second time; the stub here stands
in for that by returning a shorter response on the re-run.

That difference is the whole point. A stub that replays identically cannot
reproduce this, which is why every other crash test in this suite misses it:
they all crash *after* the provider turn has checkpointed.

    python dbos_stream_determinism_helper.py <db> <marker> crash|recover
"""

import asyncio
import json
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import StubLLM
from mus.llm.llm import Bot
import mus.dbos as mus_dbos

DB, MARKER, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
WF_ID = "stream-determinism-1"
CRASH_AT = 8

DBOS(config=DBOSConfig(
    name="mus-stream-determinism",
    system_database_url=f"sqlite:///{DB}",
    database_url=f"sqlite:///{DB}",
))


def already_crashed() -> bool:
    return os.path.exists(MARKER)


class SlowStub(StubLLM):
    """Paced so the kill lands while the provider step is still running."""

    async def stream(self, **kwargs):
        async for delta in super().stream(**kwargs):
            await asyncio.sleep(0.05)
            yield delta


@DBOS.workflow()
async def run() -> int:
    model = SlowStub()
    # The provider answers differently the second time, as a live one would.
    # Distinct wording per attempt, so a stream that splices the abandoned
    # attempt onto the re-run is obvious rather than plausible-looking.
    if already_crashed():
        for i in range(3):
            model.put_text("go", f"RE{i} ")
    else:
        for i in range(20):
            model.put_text("go", f"first{i} ")

    bot = mus_dbos.durable(Bot(prompt="t", model=model))
    result = bot("go")
    seen = 0
    async for _ in result:
        seen += 1
        if not already_crashed() and seen >= CRASH_AT:
            with open(MARKER, "w") as f:
                f.write(str(seen))
            print(f"    [crash] mid provider turn after {seen} deltas", flush=True)
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
        seen = await handle.get_result()
        streamed = [d async for d in mus_dbos.read(WF_ID)]
        # What a client reconnecting to this run actually reads.
        text = await mus_dbos.attach(WF_ID).string()
        print("RESULT " + json.dumps({
            "seen": seen,
            "records": len(streamed),
            "attached": text,
        }))


if __name__ == "__main__":
    DBOS.launch()
    try:
        asyncio.run(main())
    finally:
        DBOS.destroy()
