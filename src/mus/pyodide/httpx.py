import urllib3
import requests
import urllib3.contrib.emscripten.fetch
from httpx import Headers, Response, Request, SyncByteStream

__all__ = ["get_requests_transport"]

async def get_requests_transport():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    await urllib3.contrib.emscripten.fetch.wait_for_streaming_ready()
    return RequestsTransport()

class Stream(SyncByteStream):
    def __init__(self, requests_response):
        self.requests_response = requests_response

    def __iter__(self):
        for chunk in self.requests_response.iter_content():
            if chunk:
                yield chunk

class RequestsTransport:
    def __init__(self):
        self.session = requests.Session()

    def handle_request(self, request: Request) -> Response:
        url = str(request.url)
        method = request.method
        headers = dict(request.headers)
        content = request.content
        
        requests_response = self.session.request(
            method,
            url,
            headers=headers,
            data=content,
            stream=True
        )

        return Response(
            status_code=requests_response.status_code,
            headers=Headers(requests_response.headers),
            stream=Stream(requests_response),
            request=request,
        )

    def close(self):
        self.session.close()