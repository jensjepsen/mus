from mus.llm.types import Delta, DeltaText, ToolUse, DeltaToolUse
import pytest
from unittest.mock import MagicMock, patch
from mus import sandbox, LLM
from wasmtime import Trap
import io
import cattrs

class MockLLM(MagicMock):
    def set_response(self, responses):
        self.stream.return_value.__aiter__.return_value = iter(responses)

@pytest.fixture
def mock_client():

    return MockLLM()

@pytest.mark.asyncio
async def test_sandbox_simple(mock_client):
    code = """\
            bot = mus.Bot(model=model)
            async for delta in bot("Test query"):
                print(str(delta))
            """
    sb = sandbox(
        llms={
            "model": mock_client
        }
    )

    res = await sb(code)
    assert mock_client.stream.called

@pytest.mark.asyncio
async def test_sandbox_inputs(capsys):
    code = """\
            print("Got some kwargs", a, b, c, type(a), type(a["d"]), type(b), type(c), type(c[0]))
            """
    sb = sandbox(stdout=True)
    await sb(code, a=dict(d=5), b=10, c=[123, 456, 789])
    captured = capsys.readouterr()
    assert captured.out == "Got some kwargs {'d': 5} 10 [123, 456, 789] <class 'dict'> <class 'int'> <class 'int'> <class 'list'> <class 'int'>\n"

    @sandbox(stdout=True)
    async def decorated_func_without_model():
        """A simple bot that does not use the LLMClient."""
        print("This should also run without a model")
    
    await decorated_func_without_model(a=10)
    captured = capsys.readouterr()
    assert captured.out == "This should also run without a model\n"

@pytest.mark.asyncio
async def test_sandbox_no_model(capsys):
    code = """\
            print("This should run without a model")
            """
    sb = sandbox(stdout=True)
    await sb(code)
    captured = capsys.readouterr()
    assert captured.out == "This should run without a model\n"

    @sandbox(stdout=True)
    async def decorated_func_without_model():
        """A simple bot that does not use the LLMClient."""
        print("This should also run without a model")
    
    await decorated_func_without_model()
    captured = capsys.readouterr()
    assert captured.out == "This should also run without a model\n"

@pytest.mark.asyncio
async def test_sandbox_return_dict():
    code = """\
            result = {"key1": "value1", "key2": 42}
            print("Returning a dict")
            return result
            """
    sb = sandbox(stdout=True)
    result = await sb(code)
    assert result == {"key1": "value1", "key2": 42}

    @sandbox(stdout=True)
    async def decorated_func_returning_dict():
        """A simple bot that returns a dict."""
        print("Returning a dict from decorated function")
        return {"foo": "bar", "baz": 123}

    result = await decorated_func_returning_dict()
    assert result == {"foo": "bar", "baz": 123}


@pytest.mark.asyncio
async def test_sandbox_as_decorator(mock_client):
    
    @sandbox(llms={"mock_client": mock_client})
    async def decorated_func():
        """A simple bot that uses the LLMClient."""

        import mus
        bot = mus.Bot(model=mock_client)
        await (bot("Test query").string())
        async for delta in bot("Test query"):
            print(str(delta))

    await decorated_func()

    # Check that the function is decorated correctly
    assert decorated_func.__name__ == "decorated_func"
    assert decorated_func.__doc__ == "A simple bot that uses the LLMClient."
    
    assert mock_client.stream.called

@pytest.mark.asyncio
async def test_sandbox_with_tools(capsys, mock_client):
    # test tools defined in sandbox code

    mock_client.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaToolUse(data=ToolUse(name="test_function", input={"a": 2, "b": "world"}, id="tool1"))),
    ])
    code = """\
            async def test_function(a: int, b: str) -> str:
                \"\"\"Hello\"\"\"
                print(f"Function called with a={a}, b={b}")
                return f"Received a={a}, b={b}"
            bot = mus.Bot(model=model, functions=[test_function])
            async for delta in bot("Call test_function with a=2, b='world'"):
                print(str(delta))
            """
    await sandbox(stdout=True, llms={"model": mock_client})(code)
    captured = capsys.readouterr()
    assert "Function called with a=2, b=world" in captured.out
    assert "Received a=2, b=world" in captured.out
    assert mock_client.stream.called
    
@pytest.mark.asyncio
async def test_sandbox_with_fuel(mock_client):
    code = """\
            bot = mus.Bot(model=model)
            async for delta in bot("Test query"):
                print(str(delta))
            """
    
    # Test with insufficient fuel
    with pytest.raises(Trap):
        await sandbox(fuel=10, llms={"model": mock_client})(code)

    # Test with sufficient fuel
    await sandbox(fuel=100_000_000, llms={"model": mock_client})(code)
    assert mock_client.stream.called

    @sandbox(fuel=10, llms={"model": mock_client})
    async def decorated_func_with_fuel():
        import mus
        bot = mus.Bot(model=model)
        await (bot("Test query").string())
        async for delta in bot("Test query"):
            print(str(delta))
    
    with pytest.raises(Trap):
        await decorated_func_with_fuel()

    @sandbox(fuel=100_000_000, llms={"model": mock_client})
    async def decorated_func_with_sufficient_fuel():
        import mus
        bot = mus.Bot(model=model)
        await (bot("Test query").string())
        async for delta in bot("Test query"):
            print(str(delta))
    await decorated_func_with_sufficient_fuel()

@pytest.mark.asyncio
async def test_sandbox_stdout(capsys):
    code = """\
            print("Hello world!")
            """
    
    await sandbox(stdout=True)(code)
    captured = capsys.readouterr()
    assert captured.out == "Hello world!\n", "No output captured"

    @sandbox(stdout=True)
    async def decorated_func_with_stdout():
        print("Test query")
    
    await decorated_func_with_stdout()
    captured = capsys.readouterr()
    assert captured.out == "Test query\n", "No output captured"

@pytest.mark.asyncio
async def test_sandbox_no_stdout(capsys):
    code = """\
            print("This should not be printed")
            """
    
    await sandbox()(code)

    @sandbox()
    async def decorated_func_without_stdout():
        print("This should also not be printed")
    
    await decorated_func_without_stdout()

    # No output should be captured
    with pytest.raises(AssertionError):
        assert capsys.readouterr().out

@pytest.mark.asyncio
async def test_sandbox_multiple_models(mock_client):
    another_mock_client = MockLLM()
    code = """\
            bot1 = mus.Bot(model=model1)
            async for delta in bot1("Test query 1"):
                print(str(delta))
            bot2 = mus.Bot(model=model2)
            async for delta in bot2("Test query 2"):
                print(str(delta))
            """
    await sandbox(llms={"model1": mock_client, "model2": another_mock_client})(code)
    assert mock_client.stream.called
    assert another_mock_client.stream.called

@pytest.fixture
def mock_stdin(monkeypatch):
    def _mock_stdin(text):
        monkeypatch.setattr('sys.stdin', io.StringIO(text))
    return _mock_stdin

@pytest.mark.asyncio
async def test_sandbox_stdin(capsys, mock_stdin):
    code = """\
            import sys
            input_value = input()
            print(f"Input was: {input_value}")
            """
    mock_stdin("Test input")

    await sandbox(stdin=True, stdout=True)(code)
    
    
    captured = capsys.readouterr()
    assert captured.out == "Input was: Test input\n"

    @sandbox(stdin=True, stdout=True)
    async def decorated_func_with_stdin():
        input_value = input()
        print(f"Input was: {input_value}")
    
    mock_stdin("Decorated input")
    await decorated_func_with_stdin()
    
    captured = capsys.readouterr()
    assert captured.out == "Input was: Decorated input\n"

@pytest.mark.asyncio
async def test_sandbox_call_external_function(capsys):
    
    async def test_function(a: int, b: str) -> str:
        """Hello"""
        return f"Received a={a}, b={b}"
    
    @sandbox(stdout=True, functions=[test_function])
    async def decorated_func_with_external_function():
        result = await test_function(a=2, b="decorated")
        print(f"Function result: {result}")

    await decorated_func_with_external_function()
    captured = capsys.readouterr()
    assert 'Function result: Received a=2, b=decorated\n' in captured.out

@pytest.mark.asyncio
async def test_sandbox_call_external_function_in_code(capsys):
    async def test_function(a: int, b: str) -> str:
        """Hello"""
        return f"Received a={a}, b={b}"
    
    code = """\
            result = await test_function(a=3, b="code")
            print(f"Function result: {result}")
            """
    await sandbox(functions=[test_function], stdout=True)(code)
    captured = capsys.readouterr()
    
    assert captured.out == 'Function result: Received a=3, b=code\n'

@pytest.mark.asyncio
async def test_sandbox_call_nonexistent_function():
    code = """\
            result = nonexistent_function(a=3, b="code")
            print(f"Function result: {result}")
            """
    with pytest.raises(RuntimeError) as excinfo:
        await sandbox()(code)
    assert "name 'nonexistent_function' is not defined" in str(excinfo.value)

    @sandbox()
    async def decorated_func_with_nonexistent_function():
        result = nonexistent_function(a=2, b="decorated")
        print(f"Function result: {result}")
    
    with pytest.raises(RuntimeError) as excinfo:
        await decorated_func_with_nonexistent_function()
    
    assert "name 'nonexistent_function' is not defined" in str(excinfo.value)

@pytest.mark.asyncio
async def test_sandbox_use_external_function_as_tool(capsys, mock_client):

    async def tool(a: int, b: str) -> str:
        """Hello"""
        return f"Received a={a}, b={b}"
    
    mock_client.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaToolUse(data=ToolUse(name="tool", input={"a": 5, "b": "tool"}, id="tool1"))),
    ])

    @sandbox(stdout=True, tools=[tool], llms={"mock_client": mock_client})
    async def decorated_func_with_tool():
        await tool(a=1, b="test")  # direct call to tool to ensure it's in scope
        print("Starting bot with tool")
        import mus
        bot = mus.Bot(model=mock_client, functions=[tool])
        async for delta in bot("Call test_function with a=5, b='tool'"):
            print(str(delta))
        

    await decorated_func_with_tool()
    captured = capsys.readouterr()
    #assert "Function 'tool' not found in context" not in captured.out
    #assert "Received a=5, b=tool" in captured.out
    #assert mock_client.stream.called