from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from demo.frontend import FrontendMetadataMiddleware


def test_manifest_and_browser_config_do_not_need_pwa_or_cached_pages():
    app = FastAPI()
    app.add_middleware(FrontendMetadataMiddleware)

    @app.get('/manifest.json')
    def disabled_manifest():
        return JSONResponse({'detail': 'PWA not enabled'}, status_code=404)

    @app.get('/')
    @app.get('/config')
    @app.get('/gradio_api/info')
    def page():
        return JSONResponse({'demo': True})

    with TestClient(app) as client:
        manifest = client.get('/manifest.json')
        assert manifest.status_code == 200
        assert manifest.headers['content-type'] == 'application/manifest+json'
        assert manifest.json()['display'] == 'browser'
        assert manifest.headers['cache-control'] == 'no-store'
        for path in ('/', '/config', '/gradio_api/info'):
            response = client.get(path)
            assert response.json() == {'demo': True}
            assert response.headers['cache-control'] == 'no-store'


def test_queue_stream_is_preserved_by_frontend_middleware():
    app = FastAPI()
    app.add_middleware(FrontendMetadataMiddleware)

    @app.get('/gradio_api/queue/data')
    def queue():
        return StreamingResponse(iter(['data: first\n\n', 'data: second\n\n']),
                                 media_type='text/event-stream')

    with TestClient(app) as client:
        response = client.get('/gradio_api/queue/data')
        assert response.text == 'data: first\n\ndata: second\n\n'
        assert response.headers['content-type'].startswith('text/event-stream')
        assert 'cache-control' not in response.headers
