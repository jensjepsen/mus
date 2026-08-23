"""Helper process for the correlation-id determinism test.

``bot.query`` runs in the workflow body, so any id it mints is regenerated on
replay -- while the deltas already written to the durable stream keep the
originals. A real process death is required: a *completed* workflow is never
replayed, so running the same workflow id twice in-process proves nothing.

Prints the durable stream with ids normalised to first-seen indices, so a clean
run and a crash-recovered run can be compared for structure rather than for
literal uuids (which differ between workflow ids by design).

    python dbos_ids_helper.py <db_path> clean|crash|recover
"""

import asyncio
import json
import os
import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID

from mus import StubLLM
from mus.llm.llm import Bot
from mus.llm.types import ToolUse
import mus.dbos as mus_dbos

DB, MODE = sys.argv[1], sys.argv[2]
WF_ID = "ids-clean" if MODE == "clean" else "ids-crash"

DBOS(config=DBOSConfig(
    name="mus-ids-test",
    system_database_url=f"sqlite:///{DB}",
    database_url=f"sqlite:///{DB}",
))


@DBOS.workflow()
async def run() -> int:
    async def record(city: str) -> str:
        """Record a city."""
        # Die after the first tool is checkpointed, inside the second, so the
        # stream carries a pre-crash prefix that replay must not contradict.
        if MODE == "crash" and city == "Tokyo":
            os._exit(9)
        return "recorded:" + city

    model = StubLLM()
    model.put_tool_use("go", ToolUse(id="t1", name="record", input={"city": "Paris"}))
    model.put_tool_use("go", ToolUse(id="t2", name="record", input={"city": "Tokyo"}))
    model.put_text("go", "thinking")
    bot = mus_dbos.durable(Bot(prompt="t", model=model, functions=[record]))
    n = 0
    async for _ in bot.query("go"):
        n += 1
    await bot.close()
    return n


def normalise(rows):
    """Ids -> first-seen indices, so structure can be compared across runs."""
    seen: dict = {}

    def idx(prefix, value):
        if value is None:
            return None
        if value not in seen:
            seen[value] = f"{prefix}{len([k for k in seen if seen[k][0] == prefix])}"
        return seen[value]

    return [
        [kind, idx("s", sid), idx("t", tid)] for kind, sid, tid in rows
    ]


async def main():
    if MODE == "recover":
        handle = await DBOS.retrieve_workflow_async(WF_ID)
        await handle.get_result()
    else:
        with SetWorkflowID(WF_ID):
            handle = await DBOS.start_workflow_async(run)
        await handle.get_result()

    rows = [
        (type(d.content).__name__, d.stream_id, d.tool_invocation_id)
        async for d in mus_dbos.read(WF_ID)
    ]
    uses = {t for k, _, t in rows if k == "DeltaToolUse"}
    results = {t for k, _, t in rows if k == "DeltaToolResult"}
    print("RESULT " + json.dumps({
        "normalised": normalise(rows),
        "orphaned_results": sorted(str(t) for t in results - uses),
        "distinct_stream_ids": len({s for _, s, _ in rows if s}),
    }))


if __name__ == "__main__":
    DBOS.launch()
    try:
        asyncio.run(main())
    finally:
        DBOS.destroy()
