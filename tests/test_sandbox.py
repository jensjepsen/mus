import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from mus import sandbox
from mus.llm.types import LLMClient
from wasmtime import Trap


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
            bot = mus.LLM(model=model)
            async for delta in bot("Test query"):
                print(str(delta))
            """

    sandbox(model=mock_client, code=code)
    
    assert mock_client.stream.called

@pytest.mark.asyncio
async def test_sandbox_as_decorator(mock_client):
    @sandbox
    async def decorated_func(model: LLMClient):
        import mus
        bot = mus.LLM(model=model)
        await (bot("Test query").string())
        async for delta in bot("Test query"):
            print(str(delta))
    
    decorated_func(mock_client)

    assert mock_client.stream.called

@pytest.mark.asyncio
async def test_sandbox_with_fuel(mock_client):
    code = """\
            bot = mus.LLM(model=model)
            async for delta in bot("Test query"):
                print(str(delta))
            """
    # Test with insufficient fuel
    with pytest.raises(Trap):
        sandbox(model=mock_client, code=code, fuel=10)

    # Test with sufficient fuel
    sandbox(model=mock_client, code=code, fuel=100_000_000)
    assert mock_client.stream.called