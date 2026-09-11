"""Helper: a turn with tools, re-run live with different content.

The killed attempt and the re-run ask for the same NUMBER of tools -- so the
op sequence lines up -- but say different things and call the tool with a
different argument. That covers the combination the other crash tests miss:
tools present, and the recovered turn genuinely differing from the recorded one.

    python dbos_tools_divergence_helper.py <db> <marker> <tools> crash|recover
"""

import asyncio
import json
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import StubLLM
from mus.llm.llm import Bot
from mus.llm.types import DeltaToolResult, DeltaToolUse, ToolUse
import mus.dbos as mus_dbos

DB, MARKER, TOOLLOG, MODE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
WF_ID = "tools-divergence-1"

DBOS(config=DBOSConfig(
    name="mus-tools-divergence",
    database_url=f"sqlite:///{DB}",
    system_database_url=f"sqlite:///{DB}",
))


def already_crashed() -> bool:
    return os.path.exists(MARKER)


class SlowStub(StubLLM):
    async def stream(self, **kwargs):
        async for delta in super().stream(**kwargs):
            await asyncio.sleep(0.05)
            yield delta


async def record(city: str) -> str:
    """Record a city."""
    with open(TOOLLOG, "a") as f:
        f.write(city + "\n")
    return "ok:" + city


@DBOS.workflow()
async def run() -> int:
    model = SlowStub()
    city = "Tokyo" if already_crashed() else "Paris"
    label = "second" if already_crashed() else "first"
    model.put_text("go", f"{label} ")
    model.put_tool_use("go", ToolUse(id="t1", name="record", input={"city": city}))
    for i in range(6):
        model.put_text("go", f"{label}{i} ")

    bot = mus_dbos.durable(Bot(prompt="t", model=model, functions=[record]))
    seen = 0
    async for _ in bot.query("go"):
        seen += 1
        # Mid-stream, after the tool call has been announced but before the
        # stream drains -- so no tool has run and the turn is not checkpointed.
        if not already_crashed() and seen >= 3:
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
        seen = await handle.get_result()
        records = [d async for d in mus_dbos.read(WF_ID)]
        # Raw records include the abandoned attempt -- the stream is append-only.
        # What a reader ends up with is the interesting part, so report both.
        result = mus_dbos.attach(WF_ID)
        text = await result.string()
        tools = [l.strip() for l in open(TOOLLOG)] if os.path.exists(TOOLLOG) else []

        def count(items, kind):
            return sum(1 for d in items if isinstance(getattr(d, "content", None), kind))

        print("RESULT " + json.dumps({
            "seen": seen,
            "attached": text,
            "raw_uses": count(records, DeltaToolUse),
            "reader_uses": count(result.history, DeltaToolUse),
            "reader_results": count(result.history, DeltaToolResult),
            "tools_fired": tools,
        }))


if __name__ == "__main__":
    DBOS.launch()
    try:
        asyncio.run(main())
    finally:
        DBOS.destroy()
