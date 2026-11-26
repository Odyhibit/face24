"""Local browser metadata and cache policy for the Gradio demo."""
from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse


class FrontendMetadataMiddleware:
    """Serve Gradio's manifest link without enabling PWA/offline caching."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        path = scope['path']
        if path == '/manifest.json' and scope['method'] == 'GET':
            response = JSONResponse(
                {'name': 'Face24 QR codec demo', 'short_name': 'Face24',
                 'start_url': '/', 'display': 'browser'},
                media_type='application/manifest+json', headers={'Cache-Control': 'no-store'})
            return await response(scope, receive, send)

        async def send_response(message):
            if message['type'] == 'http.response.start' and path in ('/', '/config', '/gradio_api/info'):
                MutableHeaders(scope=message)['Cache-Control'] = 'no-store'
            await send(message)

        # Streaming queue responses and versioned assets pass through unchanged.
        return await self.app(scope, receive, send_response)
