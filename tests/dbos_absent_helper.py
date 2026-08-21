"""Helper process: run mus with `dbos` genuinely unimportable.

Monkeypatching HAS_DBOS only exercises the branch; it does not prove the module
imports at all without the package. This blocks the import for real, in a
subprocess so the blocker cannot leak into other tests.

Prints one result line per check, then OK or FAIL.
"""

import asyncio
import sys


class _BlockDbos:
    def find_spec(self, name, path=None, target=None):
        if name == "dbos" or name.startswith("dbos."):
            raise ImportError(f"blocked for testing: {name}")
        return None


sys.meta_path.insert(0, _BlockDbos())

failures = []


def check(label, cond):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        failures.append(label)


import mus.dbos as mus_dbos  # noqa: E402
from mus import StubLLM  # noqa: E402
from mus.llm.llm import Bot  # noqa: E402
from mus.llm.types import ToolUse  # noqa: E402

check("mus.dbos imports without dbos", True)
check("HAS_DBOS is False", mus_dbos.HAS_DBOS is False)


async def main():
    calls = []

    async def peek(city: str) -> str:
        """Look a city up."""
        calls.append(city)
        return "weather:" + city

    model = StubLLM()
    model.put_tool_use("go", ToolUse(id="t1", name="peek", input={"city": "Paris"}))

    # durable() must REFUSE rather than quietly hand back a bot that isn't
    # durable -- a caller would otherwise believe completed tools never re-fire.
    try:
        mus_dbos.durable(Bot(prompt="t", model=model, functions=[peek]))
        check("durable() refuses without dbos", False)
    except RuntimeError as e:
        check("durable() names the extra", "mus[dbos]" in str(e))

    # ...and the plain bot still works, which is what to use instead.
    plain = Bot(prompt="t", model=model, functions=[peek])
    deltas = [d async for d in plain.query("go")]
    results = [
        d.content.data.content.val
        for d in deltas
        if type(d.content).__name__ == "DeltaToolResult"
    ]
    check("an unwrapped bot still runs the tool", calls == ["Paris"])
    check("tool result flows through", results == ["weather:Paris"])

    # sleep is a utility, not a guarantee, so it degrades rather than raising.
    await mus_dbos.sleep(0.001)
    check("sleep() falls back", True)

    try:
        mus_dbos.attach("id")
        check("attach() raises", False)
    except RuntimeError as e:
        check("attach() names the extra", "mus[dbos]" in str(e))

    try:
        async for _ in mus_dbos.read("id"):
            pass
        check("read() raises", False)
    except RuntimeError as e:
        check("read() names the extra", "mus[dbos]" in str(e))


asyncio.run(main())
print("FAILURES:" + ",".join(failures) if failures else "ALL OK")
sys.exit(1 if failures else 0)
