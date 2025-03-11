import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import urllib3
import requests
import sys
from httpx import Headers, Response, Request, URL
import urllib3.contrib


from mus.pyodide.httpx import get_requests_transport, RequestsTransport, Stream

@pytest.fixture(autouse=True, scope="function")
def mock_wait_for_streaming_ready():
    """Mock the wait_for_streaming_ready function."""
    wait_for_streaming_ready = AsyncMock()
    wait_for_streaming_ready.return_value = None
    with patch("mus.pyodide.httpx.wait_for_streaming_ready", wait_for_streaming_ready):
        yield wait_for_streaming_ready


@pytest.fixture
def mock_request():
    """Create a mock HTTPX Request object."""
    request = Mock()
    request.url = URL("https://example.com/test")
    request.method = "GET"
    request.headers = Headers({"user-agent": "Test Agent", "accept": "application/json"})
    request.content = b"test content"
    return request


@pytest.fixture
def mock_requests_response():
    """Create a mock requests Response object."""
    response = Mock()
    response.status_code = 200
    response.headers = {"content-type": "application/json", "content-length": "42"}
    
    # Mock the iter_content method to return chunks
    chunks = [b'{"status":', b'"success",', b'"data":', b'"test"}']
    response.iter_content.return_value = chunks
    
    return response


@pytest.mark.asyncio
@patch("urllib3.disable_warnings")
async def test_get_requests_transport(mock_disable_warnings, mock_wait_for_streaming_ready):
    """Test the get_requests_transport function."""
    # Reset the mock to ensure we're testing the actual call
    mock_wait_for_streaming_ready.reset_mock()
    
    transport = await get_requests_transport()
    
    # Check that the necessary setup functions were called
    mock_disable_warnings.assert_called_once_with(urllib3.exceptions.InsecureRequestWarning)
    mock_wait_for_streaming_ready.assert_called_once()
    
    # Check that we got a RequestsTransport instance
    assert isinstance(transport, RequestsTransport)
    assert hasattr(transport, "session")
    assert isinstance(transport.session, requests.Session)


@pytest.mark.asyncio
async def test_stream_aiter(mock_requests_response):
    """Test the Stream.__aiter__ method."""
    stream = Stream(mock_requests_response)
    
    # Collect all chunks from the stream
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify that we got all the expected chunks
    expected_chunks = [b'{"status":', b'"success",', b'"data":', b'"test"}']
    assert chunks == expected_chunks
    mock_requests_response.iter_content.assert_called_once()


@pytest.mark.asyncio
@patch.object(requests.Session, "request")
async def test_handle_async_request(mock_request_method, mock_request, mock_requests_response):
    """Test the RequestsTransport.handle_async_request method."""
    # Set up the mock to return our mock response
    mock_request_method.return_value = mock_requests_response
    
    # Create the transport and call handle_async_request
    transport = RequestsTransport()
    response = await transport.handle_async_request(mock_request)
    
    # Check that the request was made with the correct parameters
    mock_request_method.assert_called_once_with(
        "GET",
        "https://example.com/test",
        headers={"user-agent": "Test Agent", "accept": "application/json"},
        data=b"test content",
        stream=True
    )
    
    # Check that the response was properly constructed
    assert response.status_code == 200
    assert dict(response.headers) == {"content-type": "application/json", "content-length": "42"}
    assert response.request == mock_request
    assert isinstance(response.stream, Stream)


@patch.object(requests.Session, "close")
def test_transport_close(mock_close):
    """Test the RequestsTransport.close method."""
    transport = RequestsTransport()
    transport.close()
    mock_close.assert_called_once()


@pytest.mark.asyncio
@patch("requests.Session")
@patch("urllib3.disable_warnings")
async def test_full_request_flow(mock_disable_warnings, mock_request, mock_requests_response, mock_wait_for_streaming_ready):
    """Test the full flow from creating a transport to getting a response."""
    # Reset the mock to ensure we're testing the actual call
    mock_wait_for_streaming_ready.reset_mock()
    
    
    # Create the transport and make a request
    transport = await get_requests_transport()
    transport.session.request.return_value = mock_requests_response

    # Verify the setup was done correctly
    mock_disable_warnings.assert_called_once_with(urllib3.exceptions.InsecureRequestWarning)
    mock_wait_for_streaming_ready.assert_awaited_once()
    
    response = await transport.handle_async_request(mock_request)
    # Check the response
    assert response.status_code == 200
    
    # Test reading the response body
    #body_chunks = []
    #async for chunk in response.stream:
    #    body_chunks.append(chunk)
    
   # assert body_chunks == [b'{"status":', b'"success",', b'"data":', b'"test"}']