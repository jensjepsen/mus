"""Helper: a run that survives a transient provider error, then is killed.

mus retries a transient failure, and DBOS checkpoints that failed step by
serialising the exception. Recovering the workflow in a fresh process re-raises
it from that record -- which means reconstructing it. A mus exception cannot be
reconstructed, because ``provider`` is keyword-only and therefore absent from
``args``, so the recovery dies inside the deserialiser before any mus code
runs.

Not a determinism failure: the op sequence lines up, since the checkpointed
failure replays and the retry repeats identically. It fails one step earlier
than that, on reading the record back.

    python dbos_retry_helper.py <db> <marker> <attempts> crash|recover
"""

import asyncio
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import StubLLM
from mus.llm.llm import Bot
from mus.llm.exceptions import LLMServerException
import mus.dbos as mus_dbos

DB, MARKER, ATTEMPTS, MODE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
WF_ID = "retry-then-crash-1"

DBOS(config=DBOSConfig(
    name="mus-retry-crash",
    database_url=f"sqlite:///{DB}",
    system_database_url=f"sqlite:///{DB}",
))


def already_crashed() -> bool:
    return os.path.exists(MARKER)


class FlakyStub(StubLLM):
    """Fails the first provider call of the first process, then behaves."""

    async def stream(self, **kwargs):
        if not already_crashed():
            seen = int(open(ATTEMPTS).read() or 0) if os.path.exists(ATTEMPTS) else 0
            with open(ATTEMPTS, "w") as f:
                f.write(str(seen + 1))
            if seen == 0:
                raise LLMServerException("transient", provider="stub")
        async for delta in super().stream(**kwargs):
            await asyncio.sleep(0.05)
            yield delta


@DBOS.workflow()
async def run() -> int:
    model = FlakyStub()
    for i in range(20):
        model.put_text("go", f"tok{i} ")
    bot = mus_dbos.durable(Bot(prompt="t", model=model))
    seen = 0
    async for _ in bot.query("go"):
        seen += 1
        # Kill after the retry has been checkpointed, with the turn still
        # streaming so its step never completes.
        if not already_crashed() and seen >= 6:
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
