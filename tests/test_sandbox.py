import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from mus.sandbox import sandbox


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
    code = """import mus
m = mus.Mus()
bot = m.llm(client=client, model="claude-3-5-sonnet-20241022")
await bot("Test query").string()
"""

    sandbox(mock_client, code)
    
    assert mock_client.stream.called
    
