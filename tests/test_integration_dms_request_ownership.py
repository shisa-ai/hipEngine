"""HTTP integration contracts for typed DMS configuration."""
from fastapi.testclient import TestClient

from hipengine import DMSConfig
from hipengine.server.api import ServerConfig, create_app
from tests.support.server_api_fakes import FakeLLM


def test_http_status_reports_the_typed_sidecar_path():
    config = ServerConfig(model='fake', dms=DMSConfig('typed.json'),
                          prefix_cache='off', speculative_mtp_serving='off', eager_load=False)
    with TestClient(create_app(config, llm=FakeLLM())) as client:
        response = client.get('/ready')
        assert response.status_code == 200, response.text
        assert response.json()['dms']['requested_metadata_path'] == 'typed.json'
