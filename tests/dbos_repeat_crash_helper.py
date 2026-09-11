"""Helper: a run interrupted several times before it finally completes.

Each interrupted attempt leaves its deltas in the stream -- it is append-only --
and each re-run opens with a DeltaStreamReset. A reader must end up with only
the attempt that finished, however many came before it. A bad deploy loop
restarting the same run repeatedly is the realistic version of this.

    python dbos_repeat_crash_helper.py <db> <attempts> <mode>
"""

import asyncio
import json
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import StubLLM
from mus.llm.llm import Bot
import mus.dbos as mus_dbos

DB, ATTEMPTS, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
WF_ID = "repeat-crash-1"

DBOS(config=DBOSConfig(
    name="mus-repeat-crash",
    database_url=f"sqlite:///{DB}",
    system_database_url=f"sqlite:///{DB}",
))


def _claim_attempt() -> int:
    """This process's attempt number, claimed once at import.

    Read inside the workflow instead and it would advance on every call, so the
    run would not keep a stable identity for the length of the process.
    """
    n = int(open(ATTEMPTS).read() or 0) if os.path.exists(ATTEMPTS) else 0
    with open(ATTEMPTS, "w") as f:
        f.write(str(n + 1))
    return n


ATTEMPT = _claim_attempt()


class SlowStub(StubLLM):
    async def stream(self, **kwargs):
        async for delta in super().stream(**kwargs):
            await asyncio.sleep(0.05)
            yield delta


@DBOS.workflow()
async def run() -> int:
    n = ATTEMPT
    # A distinct alphabet per attempt, so a stream that keeps an abandoned one
    # is obvious. The last attempt is short and finishes.
    label = "ABC"[min(n, 2)]
    model = SlowStub()
    for i in range(3 if n >= 2 else 20):
        model.put_text("go", f"{label}{i} ")

    bot = mus_dbos.durable(Bot(prompt="t", model=model))
    seen = 0
    async for _ in bot.query("go"):
        seen += 1
        if n < 2 and seen >= 4:
            print(f"    [crash {n}] after {seen} deltas", flush=True)
            os._exit(9)
    await bot.close()
    return seen


async def main():
    n = ATTEMPT
    if MODE == "start":
        with SetWorkflowID(WF_ID):
            handle = await DBOS.start_workflow_async(run)
        await handle.get_result()
    else:
        handle = await DBOS.retrieve_workflow_async(WF_ID)
        seen = await handle.get_result()
        records = [d async for d in mus_dbos.read(WF_ID)]
        text = await mus_dbos.attach(WF_ID).string()
        print("RESULT " + json.dumps({
            "attempts": n + 1, "seen": seen,
            "records": len(records), "attached": text,
        }))


if __name__ == "__main__":
    DBOS.launch()
    try:
        asyncio.run(main())
    finally:
        DBOS.destroy()
