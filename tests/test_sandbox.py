import pytest
from unittest.mock import MagicMock
from mus import sandbox
from mus.llm.types import LLM
from wasmtime import Trap
import io

class MockClient():
    def __init__(self):
        self.stream = MagicMock()
    
    def set_response(self, responses):
        mock_response = MagicMock()
        mock_response.__aiter__.return_value = iter(responses)
        self.stream.return_value.__aenter__.return_value = mock_response

@pytest.fixture
def mock_client():
    return MockClient()


@pytest.mark.asyncio
async def test_sandbox(mock_client):
    code = """\
            bot = mus.Bot(model=model)
            async for delta in bot("Test query"):
                print(str(delta))
            """

    await sandbox(model=mock_client, code=code)
    
    assert mock_client.stream.called

@pytest.mark.asyncio
async def test_sandbox_no_model(capsys):
    code = """\
            print("This should not run without a model")
            """
    await sandbox(code=code, stdout=True)
    captured = capsys.readouterr()
    assert captured.out == "This should not run without a model\n"

    @sandbox(stdout=True)
    async def decorated_func_without_model():
        """A simple bot that does not use the LLMClient."""
        print("This should also run without a model")
    
    await decorated_func_without_model()
    captured = capsys.readouterr()
    assert captured.out == "This should also run without a model\n"


@pytest.mark.asyncio
async def test_sandbox_as_decorator(mock_client):
    @sandbox
    async def decorated_func(model: LLM):
        """A simple bot that uses the LLMClient."""

        import mus
        bot = mus.Bot(model=model)
        await (bot("Test query").string())
        async for delta in bot("Test query"):
            print(str(delta))
    
    await decorated_func(mock_client)

    # Check that the function is decorated correctly
    assert decorated_func.__name__ == "decorated_func"
    assert decorated_func.__doc__ == "A simple bot that uses the LLMClient."
    
    assert mock_client.stream.called

    @sandbox()
    async def decorated_func_with_wrapper(model: LLM):
        """A simple bot that uses the LLMClient with a wrapper."""

        import mus
        bot = mus.Bot(model=model)
        await (bot("Test query").string())
        async for delta in bot("Test query with"):
            print(str(delta))
    
    await decorated_func_with_wrapper(mock_client)

    # Check that the function is decorated correctly
    assert decorated_func_with_wrapper.__name__ == "decorated_func_with_wrapper"
    assert decorated_func_with_wrapper.__doc__ == "A simple bot that uses the LLMClient with a wrapper."


@pytest.mark.asyncio
async def test_sandbox_with_fuel(mock_client):
    code = """\
            bot = mus.Bot(model=model)
            async for delta in bot("Test query"):
                print(str(delta))
            """
    # Test with insufficient fuel
    with pytest.raises(Trap):
        await sandbox(model=mock_client, code=code, fuel=10)

    # Test with sufficient fuel
    await sandbox(model=mock_client, code=code, fuel=100_000_000)
    assert mock_client.stream.called

    @sandbox(fuel=10)
    async def decorated_func_with_fuel(model: LLM):
        import mus
        bot = mus.Bot(model=model)
        await (bot("Test query").string())
        async for delta in bot("Test query"):
            print(str(delta))
    
    with pytest.raises(Trap):
        await decorated_func_with_fuel(mock_client)
    
    @sandbox(fuel=100_000_000)
    async def decorated_func_with_sufficient_fuel(model: LLM):
        import mus
        bot = mus.Bot(model=model)
        await (bot("Test query").string())
        async for delta in bot("Test query"):
            print(str(delta))
    await decorated_func_with_sufficient_fuel(mock_client)

@pytest.mark.asyncio
async def test_sandbox_stdout(capsys):
    code = """\
            print("Hello world!")
            """
    
    await sandbox(code=code, stdout=True)
    captured = capsys.readouterr()
    assert captured.out == "Hello world!\n"

    @sandbox(stdout=True)
    async def decorated_func_with_stdout():
        print("Test query")
    
    await decorated_func_with_stdout()
    captured = capsys.readouterr()
    assert captured.out == "Test query\n"

@pytest.mark.asyncio
async def test_sandbox_no_stdout(capsys):
    code = """\
            print("This should not be printed")
            """
    
    await sandbox(code=code)

    @sandbox
    async def decorated_func_without_stdout():
        print("This should also not be printed")
    
    await decorated_func_without_stdout()

    # No output should be captured
    with pytest.raises(AssertionError):
        assert capsys.readouterr().out

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

    await sandbox(code=code, stdin=True, stdout=True)
    
    
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