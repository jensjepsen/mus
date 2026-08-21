"""Tests for the DBOS durable-execution adapter.

Runs on SQLite, so no database service is needed. Follows the fixture order DBOS
documents: destroy, configure, reset_system_database(truncate=True), launch.

The focus here is the guarantee that tools may be defined *anywhere* -- as
closures over local state, as callable objects, or built dynamically per run.
mus accepts any callable carrying ``__name__``, ``__doc__`` and annotations, and
wrapping a bot for durability must not narrow that: the tool itself is never
registered or pickled, only a generic per-name step wrapper is.
"""

import os
import typing as t
from dataclasses import dataclass

import pytest

pytest.importorskip("dbos")

from dbos import DBOS, DBOSConfig  # noqa: E402

from mus import Delta, DeltaText, DeltaToolResult, DeltaToolUse, StubLLM  # noqa: E402
from mus.llm.llm import Bot  # noqa: E402
from mus.llm.types import ToolUse  # noqa: E402
import mus.dbos as mus_dbos  # noqa: E402


@pytest.fixture()
def reset_dbos(tmp_path):
    DBOS.destroy()
    config: DBOSConfig = {
        "name": "mus-tests",
        "system_database_url": f"sqlite:///{tmp_path}/dbos.sqlite",
        "database_url": f"sqlite:///{tmp_path}/dbos.sqlite",
    }
    DBOS(config=config)
    DBOS.reset_system_database(truncate=True)
    DBOS.launch()
    mus_dbos._PROVIDER_TURN = None
    yield
    DBOS.destroy()


def _model_calling(tool_name: str, **args) -> StubLLM:
    """A stub that asks for one tool call, then answers."""
    model = StubLLM()
    model.put_tool_use("go", ToolUse(id="t1", name=tool_name, input=args))
    return model


def _kinds(deltas) -> list[str]:
    return [type(d.content).__name__ for d in deltas]


def _results(deltas) -> list:
    return [
        d.content.data.content.val
        for d in deltas
        if isinstance(d.content, DeltaToolResult)
    ]


# --- tools defined anywhere ------------------------------------------------


@pytest.mark.asyncio
async def test_closure_tool_over_local_state(reset_dbos):
    """A tool defined inside the workflow, capturing local state."""

    @DBOS.workflow()
    async def run() -> dict:
        secret = "captured-value"          # local to this run
        calls: list[str] = []

        async def peek(city: str) -> str:
            """Look something up."""
            calls.append(city)
            return f"{secret}:{city}"

        bot = mus_dbos.durable(
            Bot(prompt="t", model=_model_calling("peek", city="Paris"), functions=[peek])
        )
        deltas = [d async for d in bot.query("go")]
        await bot.close()
        return {"kinds": _kinds(deltas), "results": _results(deltas), "calls": calls}

    out = await (await DBOS.start_workflow_async(run)).get_result()
    assert out["calls"] == ["Paris"]
    assert out["results"] == ["captured-value:Paris"]
    assert "DeltaToolUse" in out["kinds"]
    assert "DeltaToolResult" in out["kinds"]


@pytest.mark.asyncio
async def test_callable_object_tool(reset_dbos):
    """A tool built at runtime as a callable object."""

    class Lookup:
        __name__ = "lookup"
        __doc__ = "Look a city up."

        def __init__(self, suffix: str):
            self.suffix = suffix
            self.seen: list[str] = []

        async def __call__(self, city: str) -> str:
            self.seen.append(city)
            return city + self.suffix

    @DBOS.workflow()
    async def run() -> dict:
        tool = Lookup("!")
        bot = mus_dbos.durable(
            Bot(
                prompt="t",
                model=_model_calling("lookup", city="Tokyo"),
                functions=[tool],
            )
        )
        deltas = [d async for d in bot.query("go")]
        await bot.close()
        return {"results": _results(deltas), "seen": tool.seen}

    out = await (await DBOS.start_workflow_async(run)).get_result()
    assert out["seen"] == ["Tokyo"]
    assert out["results"] == ["Tokyo!"]


@pytest.mark.asyncio
async def test_dynamically_built_tools_get_distinct_steps(reset_dbos):
    """Several tools made by a factory, each checkpointed under its own name."""

    def make_tool(name: str):
        async def _tool(value: str) -> str:
            return f"{name}({value})"

        _tool.__name__ = name
        _tool.__doc__ = f"Dynamic tool {name}."
        return _tool

    @DBOS.workflow()
    async def run() -> list:
        tools = [make_tool(f"dyn_{i}") for i in range(3)]
        model = StubLLM()
        # One turn asking for all three at once.
        for i in range(3):
            model.put_response(
                "go",
                Delta(
                    content=DeltaToolUse(
                        data=ToolUse(id=f"t{i}", name=f"dyn_{i}", input={"value": "x"})
                    )
                ),
            )
        bot = mus_dbos.durable(
            Bot(prompt="t", model=model, functions=tools)
        )
        deltas = [d async for d in bot.query("go")]
        await bot.close()
        return _results(deltas)

    handle = await DBOS.start_workflow_async(run)
    results = await handle.get_result()
    assert results == ["dyn_0(x)", "dyn_1(x)", "dyn_2(x)"]

    steps = await DBOS.list_workflow_steps_async(handle.workflow_id)
    names = [s.get("function_name") for s in steps]
    # Each dynamic tool is checkpointed under its own name, not one generic step.
    assert "mus.tool:dyn_0" in names
    assert "mus.tool:dyn_1" in names
    assert "mus.tool:dyn_2" in names


@pytest.mark.asyncio
async def test_tool_runner_preserves_validation(reset_dbos):
    """The step wraps mus's own invoke, so bad input is still rejected there."""

    @DBOS.workflow()
    async def run() -> list:
        async def strict(count: int) -> str:
            """Needs an int."""
            return str(count)

        model = StubLLM()
        model.put_tool_use(
            "go", ToolUse(id="t1", name="strict", input={"wrong": "arg"})
        )
        bot = mus_dbos.durable(Bot(prompt="t", model=model, functions=[strict]))
        deltas = [d async for d in bot.query("go")]
        await bot.close()
        return _results(deltas)

    results = await (await DBOS.start_workflow_async(run)).get_result()
    # mus's schema validation still runs inside the step.
    assert "error" in str(results[0])


# --- running without dbos installed ----------------------------------------


def test_everything_works_without_dbos_installed():
    """mus must not gain a hard dependency on dbos.

    Runs in a subprocess with the `dbos` import blocked outright. Monkeypatching
    HAS_DBOS only exercises the branch -- it leaves dbos imported, so it cannot
    show that the module loads at all without the package, which is the thing
    that actually breaks for a user who never installs the extra.

    The helper checks: mus.dbos imports, HAS_DBOS is False, durable() /
    read() / attach() all raise naming the missing extra, an unwrapped bot
    still runs normally, and sleep() degrades to asyncio.

    durable() refuses rather than passing through on purpose. A durable() that
    quietly wasn't durable would leave callers believing completed tools never
    re-fire -- the same silent-wrong-answer shape as an untruncated-looking
    truncated response. A bot that doesn't need durability doesn't need wrapping.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    helper = Path(__file__).parent / "dbos_absent_helper.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}
    proc = subprocess.run(
        [_sys.executable, str(helper)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr[-800:]
    assert "ALL OK" in proc.stdout, proc.stdout


# --- closure tools across a real crash -------------------------------------


@pytest.mark.asyncio
async def test_completed_closure_tool_is_not_re_fired_after_a_crash(tmp_path):
    """A crash replays the workflow body, rebuilding the closure tool.

    The completed tool's cached step must short-circuit *before* the rebuilt
    closure is reached, or a durable run would re-fire side effects a plain mus
    run never would.

    Needs a real process death: a *completed* workflow is never replayed, so
    running the same workflow id twice in-process proves nothing.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    helper = Path(__file__).parent / "dbos_crash_helper.py"
    db = tmp_path / "crash.sqlite"
    effects = tmp_path / "effects.log"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}

    crashed = subprocess.run(
        [_sys.executable, str(helper), str(db), str(effects), "crash"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert crashed.returncode == 9, f"expected a crash, got {crashed.returncode}"
    assert effects.read_text().split() == ["Paris", "Tokyo"]

    recovered = subprocess.run(
        [_sys.executable, str(helper), str(db), str(effects), "recover"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert recovered.returncode == 0, recovered.stderr[-800:]

    fired = effects.read_text().split()
    # Paris completed and was checkpointed, so it must not run again. Tokyo was
    # in flight when the process died, so it does -- DBOS cannot know whether
    # its side effect landed.
    assert fired.count("Paris") == 1, fired
    assert fired.count("Tokyo") == 2, fired


# --- the IterableResult surface --------------------------------------------


@pytest.mark.asyncio
async def test_iterable_result_behaves_normally_in_and_out_of_the_workflow(reset_dbos):
    """A durable run's result is an ordinary mus IterableResult.

    Both inside the workflow (`bot(query)`) and from anywhere afterwards
    (`attach(workflow_id)`), the usual accessors must work -- .string(),
    .usage, .stop_reason and .history -- since attach() rebuilds the result
    from the stream rather than from Bot.query.
    """
    from mus import Usage
    from mus.llm.types import StopReason

    @DBOS.workflow()
    async def run() -> dict:
        async def peek(city: str) -> str:
            """Look a city up."""
            return "weather:" + city

        model = StubLLM()
        model.put_tool_use("go", ToolUse(id="t1", name="peek", input={"city": "Paris"}))
        # The continuation turn: text, usage and a planned stop reason.
        model.put_response("go", Delta(content=DeltaText(data="done")))
        model.put_response(
            "go",
            Delta(
                content=DeltaText(data=""),
                usage=Usage(input_tokens=11, output_tokens=7),
                stop_reason=StopReason(kind="end_turn", raw="stop"),
            ),
        )

        bot = mus_dbos.durable(Bot(prompt="t", model=model, functions=[peek]))
        result = bot("go")
        text = await result.string()
        await bot.close()
        return {
            "text": text,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "stop_reason": result.stop_reason.kind if result.stop_reason else None,
            "history_len": len(result.history),
        }

    handle = await DBOS.start_workflow_async(run)
    inside = await handle.get_result()

    assert "done" in inside["text"]
    assert inside["input_tokens"] == 11
    assert inside["output_tokens"] == 7
    assert inside["stop_reason"] == "end_turn"
    assert inside["history_len"] > 0

    # ...and the same, reconstructed from the stream by a different caller.
    outside = mus_dbos.attach(handle.workflow_id)
    text = await outside.string()
    assert text == inside["text"]
    assert outside.usage.input_tokens == inside["input_tokens"]
    assert outside.usage.output_tokens == inside["output_tokens"]
    assert outside.stop_reason is not None
    assert outside.stop_reason.kind == inside["stop_reason"]
    assert len(outside.history) == inside["history_len"]


# --- crashing inside a provider turn ---------------------------------------


@pytest.mark.asyncio
async def test_crash_mid_provider_turn_does_not_replay_the_abandoned_prefix(tmp_path):
    """A partial turn left in the raw key must not be read as the new turn.

    The provider step writes deltas as they arrive, so a crash mid-token leaves
    an abandoned prefix behind -- streams are append-only, it cannot be removed.
    On recovery the step re-runs and appends its deltas again. If the wrapper
    finds its turn by counting offsets, it reads the stale prefix first and mus
    sees duplicated content.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    helper = Path(__file__).parent / "dbos_offset_helper.py"
    db = tmp_path / "offset.sqlite"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}

    crashed = subprocess.run(
        [_sys.executable, str(helper), str(db), "crash"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert crashed.returncode == 9, f"expected a crash, got {crashed.returncode}"

    recovered = subprocess.run(
        [_sys.executable, str(helper), str(db), "recover"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert recovered.returncode == 0, recovered.stderr[-1500:]

    text = recovered.stdout.strip().splitlines()[-1]
    assert text == "Once upon a time", f"got {text!r}"


# --- failures reaching the reader ------------------------------------------


@pytest.mark.asyncio
async def test_a_failing_run_writes_the_reason_into_the_stream(reset_dbos):
    """A failed run must not look like a truncated one.

    If the bot raises, the workflow goes to ERROR and the stream simply stops.
    A reader tailing it sees output end mid-conversation with no explanation --
    indistinguishable from a model that finished early. The failure has to be
    written into the stream before it propagates.
    """

    @DBOS.workflow()
    async def run() -> str:
        async def boom(x: int) -> str:
            """Always fails."""
            raise ValueError("tool exploded")

        model = StubLLM()
        model.put_tool_use("go", ToolUse(id="t1", name="boom", input={"x": 1}))
        bot = mus_dbos.durable(Bot(prompt="t", model=model, functions=[boom]))
        async for _ in bot.query("go"):
            pass
        return "unreachable"

    handle = await DBOS.start_workflow_async(run)
    with pytest.raises(Exception):
        await handle.get_result()

    deltas = [d async for d in mus_dbos.read(handle.workflow_id)]
    errors = [d for d in deltas if d.metadata and "mus.error" in d.metadata]
    assert len(errors) == 1, [d.metadata for d in deltas]

    err = (errors[0].metadata or {})["mus.error"]
    assert err["type"] == "ValueError"
    assert "tool exploded" in err["message"]
    # It is the last thing in the stream, and the stream is closed after it.
    assert deltas[-1] is errors[0]


# --- mus surfaces that never ran through the adapter -----------------------


@dataclass
class Person:
    """A person."""

    name: str
    age: int


@pytest.mark.asyncio
async def test_fill_works_through_a_durable_bot(reset_dbos):
    """Bot.fill uses no_stream=True and function_choice="any"."""

    @DBOS.workflow()
    async def run() -> dict:
        model = StubLLM()
        model.put_tool_use(
            "who", ToolUse(id="f1", name="Person", input={"name": "Ada", "age": 36})
        )
        bot = mus_dbos.durable(Bot(prompt="t", model=model))
        person = t.cast(Person, await bot.fill("who", Person))
        return {"name": person.name, "age": person.age}

    out = await (await DBOS.start_workflow_async(run)).get_result()
    assert out == {"name": "Ada", "age": 36}


@pytest.mark.asyncio
async def test_fun_works_through_a_durable_bot(reset_dbos):
    """Bot.fun turns a function into a natural-language callable."""

    @DBOS.workflow()
    async def run() -> int:
        model = StubLLM()
        model.put_tool_use(
            "add them", ToolUse(id="f1", name="add", input={"a": 2, "b": 3})
        )
        bot = mus_dbos.durable(Bot(prompt="t", model=model))

        @bot.fun
        async def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        return int(await add("add them"))

    assert await (await DBOS.start_workflow_async(run)).get_result() == 5


@pytest.mark.asyncio
async def test_no_stream_works_through_a_durable_bot(reset_dbos):
    """no_stream still goes through the provider step."""

    @DBOS.workflow()
    async def run() -> str:
        model = StubLLM()
        model.put_text("go", "not streamed")
        bot = mus_dbos.durable(Bot(prompt="t", model=model))
        result = bot("go", no_stream=True)
        text = await result.string()
        await bot.close()
        return text

    assert "not streamed" in await (await DBOS.start_workflow_async(run)).get_result()


@pytest.mark.asyncio
async def test_two_durable_bots_keep_separate_streams(reset_dbos):
    """Two bots in one workflow must not write into each other's stream."""

    @DBOS.workflow()
    async def run() -> None:
        first = StubLLM()
        first.put_text("go", "from-alpha")
        second = StubLLM()
        second.put_text("go", "from-beta")

        alpha = mus_dbos.durable(Bot(prompt="t", model=first), key="alpha")
        beta = mus_dbos.durable(Bot(prompt="t", model=second), key="beta")

        await alpha("go").string()
        await beta("go").string()
        await alpha.close()
        await beta.close()

    handle = await DBOS.start_workflow_async(run)
    await handle.get_result()

    a = await mus_dbos.attach(handle.workflow_id, key="alpha").string()
    b = await mus_dbos.attach(handle.workflow_id, key="beta").string()
    assert "from-alpha" in a and "from-beta" not in a
    assert "from-beta" in b and "from-alpha" not in b


# --- small guards ----------------------------------------------------------


@pytest.mark.asyncio
async def test_sleep_is_durable_inside_a_workflow(reset_dbos):
    """sleep() uses DBOS's durable sleep when it can, so replays skip it."""

    @DBOS.workflow()
    async def run() -> bool:
        await mus_dbos.sleep(0.01)
        return True

    assert await (await DBOS.start_workflow_async(run)).get_result() is True


@pytest.mark.asyncio
async def test_durable_bot_refuses_outside_a_workflow(reset_dbos):
    """The provider step needs a workflow context; say so rather than failing oddly."""
    model = StubLLM()
    model.put_text("go", "hi")
    bot = mus_dbos.durable(Bot(prompt="t", model=model))
    with pytest.raises(RuntimeError, match="inside a DBOS workflow"):
        await bot("go").string()


@pytest.mark.asyncio
async def test_close_is_idempotent(reset_dbos):
    """A failing run closes the stream itself; an explicit close must not double it."""

    @DBOS.workflow()
    async def run() -> bool:
        model = StubLLM()
        model.put_text("go", "hi")
        bot = mus_dbos.durable(Bot(prompt="t", model=model))
        await bot("go").string()
        await bot.close()
        await bot.close()
        return True

    assert await (await DBOS.start_workflow_async(run)).get_result() is True
