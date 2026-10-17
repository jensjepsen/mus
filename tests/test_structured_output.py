"""Every provider must put a JSON schema on the wire in its own spelling.

``Bot.fill(strategy="native")`` asks the provider to constrain its response to
a schema, rather than faking a tool call to get structure out. Each provider
spells that differently, and an adapter that drops it silently returns prose
that then fails to parse.

Adapters are enumerated from the package, so one added without support fails
here rather than shipping with a silent gap.
"""

import typing as t
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import pytest

from mus import Bot, StubLLM  # noqa: E402
from mus.llm.types import Delta, DeltaText, Usage  # noqa: E402

SCHEMA: t.Dict[str, t.Any] = {
    "name": "Weather",
    "description": "A weather report.",
    "schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}


async def _drain(llm, **extra):
    try:
        async for _ in llm.stream(prompt="p", model="m", history=[], **extra):
            pass
    except Exception:
        # The mocks return nothing usable; only the outgoing request matters.
        pass


def _anthropic():
    from mus.llm.anthropic import AnthropicLLM

    client = Mock()
    llm = AnthropicLLM("m", client)

    def sent():
        return client.messages.stream.call_args.kwargs.get("output_config")

    # {"format": {"type": "json_schema", "schema": {...}}}
    def ok(v):
        fmt = (v or {}).get("format") or {}
        return fmt.get("type") == "json_schema" and "properties" in (
            fmt.get("schema") or {}
        )

    return llm, sent, ok


def _bedrock():
    from mus.llm.bedrock import BedrockLLM

    client = AsyncMock()
    llm = BedrockLLM("m", client)

    def sent():
        return client.converse_stream.call_args.kwargs.get(
            "additionalModelRequestFields"
        )

    # Converse's typed outputConfig field is absent from botocore 1.40.x, so the
    # schema rides the documented escape hatch for model-specific parameters.
    def ok(v):
        fmt = ((v or {}).get("output_config") or {}).get("format") or {}
        return fmt.get("type") == "json_schema" and "properties" in (
            fmt.get("schema") or {}
        )

    return llm, sent, ok


def _google():
    from mus.llm.google import GoogleGenAILLM

    client = Mock()
    client.aio = Mock()
    client.aio.models = Mock()
    client.aio.models.generate_content_stream = AsyncMock(return_value=[])
    llm = GoogleGenAILLM("m", client)

    def sent():
        cfg = client.aio.models.generate_content_stream.call_args.kwargs["config"]
        return (
            getattr(cfg, "response_mime_type", None),
            getattr(cfg, "response_json_schema", None)
            or getattr(cfg, "response_schema", None),
        )

    def ok(v):
        mime, schema = v if v else (None, None)
        return mime == "application/json" and bool(schema)

    return llm, sent, ok


def _mistral():
    from mus.llm.mistral import MistralLLM

    client = Mock()
    client.chat = Mock()
    client.chat.stream_async = AsyncMock(return_value=[])
    llm = MistralLLM("m", client)

    def sent():
        return client.chat.stream_async.call_args.kwargs.get("response_format")

    def ok(v):
        # The adapter model_dump()s the request, so this arrives as a dict and
        # schema_definition serialises under its alias, "schema".
        js = (v or {}).get("json_schema") or {}
        return (v or {}).get("type") == "json_schema" and "properties" in (
            js.get("schema") or js.get("schema_definition") or {}
        )

    return llm, sent, ok


def _openai():
    import openai as sdk
    from mus.llm.openai import OpenAILLM

    client = AsyncMock(spec=sdk.AsyncClient)
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=[])
    llm = OpenAILLM("m", client)

    def sent():
        return client.chat.completions.create.call_args.kwargs.get("response_format")

    def ok(v):
        return (v or {}).get("type") == "json_schema" and "schema" in (
            (v or {}).get("json_schema") or {}
        )

    return llm, sent, ok


def _openai_responses():
    import openai as sdk
    from mus.llm.openai_responses import OpenAIResponsesLLM

    client = AsyncMock(spec=sdk.AsyncClient)
    client.responses = AsyncMock()
    client.responses.create = AsyncMock(return_value=[])
    llm = OpenAIResponsesLLM("m", client)

    def sent():
        return client.responses.create.call_args.kwargs.get("text")

    def ok(v):
        fmt = (v or {}).get("format") or {}
        return fmt.get("type") == "json_schema" and "schema" in fmt

    return llm, sent, ok


BUILDERS = {
    "anthropic": _anthropic,
    "bedrock": _bedrock,
    "google": _google,
    "mistral": _mistral,
    "openai": _openai,
    "openai_responses": _openai_responses,
}

SDKS = {
    "anthropic": "anthropic",
    "bedrock": "aioboto3",
    "google": "google.genai",
    "mistral": "mistralai",
    "openai": "openai",
    "openai_responses": "openai",
}


def _adapter_modules() -> t.List[str]:
    import pkgutil
    import sys

    # Via sys.modules because conftest stands a Mock in for the `mus` package.
    package = sys.modules["mus.llm"]
    skip = {"types", "llm", "exceptions", "mock_client"}
    return sorted(
        m.name
        for m in pkgutil.iter_modules(package.__path__)
        if not m.name.startswith("_") and m.name not in skip
    )


def test_every_adapter_is_covered_here():
    assert set(_adapter_modules()) == set(BUILDERS), (
        "adapters without a structured-output case: "
        f"{set(_adapter_modules()) - set(BUILDERS)}"
    )


@pytest.mark.parametrize("provider", sorted(BUILDERS))
@pytest.mark.asyncio
async def test_schema_reaches_the_wire(provider):
    pytest.importorskip(SDKS[provider])
    llm, sent, ok = BUILDERS[provider]()

    await _drain(llm, structured_output=SCHEMA)

    value = sent()
    assert ok(value), f"{provider} dropped the schema; it sent {value!r}"


@pytest.mark.parametrize("provider", sorted(BUILDERS))
@pytest.mark.asyncio
async def test_no_schema_sends_nothing(provider):
    pytest.importorskip(SDKS[provider])
    llm, sent, ok = BUILDERS[provider]()

    await _drain(llm)

    assert not ok(sent()), f"{provider} constrained the response unasked"


# --- Bot.fill(strategy="native") -------------------------------------------


@pytest.mark.asyncio
async def test_native_fill_parses_the_structured_response():
    """The provider returns JSON directly; no tool call is involved."""

    @dataclass
    class Weather:
        """A weather report."""

        city: str
        celsius: int

    model = StubLLM()
    model.put_text("go", '{"city": "Paris", ')
    model.put_text("go", '"celsius": 17}')
    model.put_response(
        "go",
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=9, output_tokens=4)),
    )

    filled = await Bot(prompt="p", model=model).fill("go", Weather, strategy="native")

    assert filled.value == Weather(city="Paris", celsius=17)
    assert filled.usage.input_tokens == 9


@pytest.mark.asyncio
async def test_native_fill_sends_a_strict_schema():
    """Derived schemas need adapting before a provider will take them.

    Bedrock rejects a schema without ``additionalProperties: false`` outright
    -- a live Converse call returns ValidationException -- and OpenAI's strict
    mode requires it too, along with every property being required.
    """

    @dataclass
    class Weather:
        """A weather report."""

        city: str
        celsius: int

    sent = {}

    class Spy(StubLLM):
        async def stream(self, **kwargs):
            sent.update(kwargs)
            async for d in super().stream(**kwargs):
                yield d

    model = Spy()
    model.put_text("go", '{"city": "Paris", "celsius": 17}')
    await Bot(prompt="p", model=model).fill("go", Weather, strategy="native")

    schema = sent["structured_output"]["schema"]
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["celsius", "city"]


@pytest.mark.asyncio
async def test_native_fill_schema_override_goes_on_the_wire():
    """A caller may hand-tune the schema a provider receives.

    Derived schemas are minimal -- no per-field descriptions, no enums, no
    formats -- and provider dialects differ, so the escape hatch matters. The
    type is still what the result is built from.
    """

    @dataclass
    class Weather:
        """A weather report."""

        city: str
        celsius: int

    override = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city named in the text."},
            "celsius": {"type": "integer", "description": "Degrees Celsius."},
        },
        "required": ["city", "celsius"],
        "additionalProperties": False,
    }

    sent = {}

    class Spy(StubLLM):
        async def stream(self, **kwargs):
            sent.update(kwargs)
            async for d in super().stream(**kwargs):
                yield d

    model = Spy()
    model.put_text("go", '{"city": "Paris", "celsius": 17}')
    filled = await Bot(prompt="p", model=model).fill(
        "go", Weather, strategy="native", json_schema=override
    )

    assert sent["structured_output"]["schema"] == override
    assert filled.value == Weather(city="Paris", celsius=17)


@pytest.mark.asyncio
async def test_native_fill_validates_against_the_schema_it_sent():
    """The override is the contract the model was given, so it is the one checked.

    Validating against the derived schema instead would reject a response that
    correctly obeyed the request.
    """

    @dataclass
    class Weather:
        """A weather report."""

        city: str
        celsius: int

    # Permits celsius to be absent, which the derived schema would not.
    lenient = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "celsius": {"type": "integer"}},
        "required": ["city"],
        "additionalProperties": False,
    }

    model = StubLLM()
    model.put_text("go", '{"city": "Paris"}')

    with pytest.raises(TypeError) as excinfo:
        await Bot(prompt="p", model=model).fill(
            "go", Weather, strategy="native", json_schema=lenient
        )

    # Schema validation passed -- the response obeyed the contract it was given
    # -- so the mismatch surfaces where it belongs: building the type.
    assert "Weather" in str(excinfo.value)
