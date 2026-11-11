from types import SimpleNamespace

import google.auth

from pts.transformers.clinical_mira.moa import vertex


def _fake_credentials(monkeypatch):
    creds = SimpleNamespace(token=None, refresh=lambda request: setattr(creds, 'token', 'tok-1'))
    monkeypatch.setattr(google.auth, 'default', lambda scopes: (creds, 'proj'))
    return creds


def test_global_client_points_at_the_vertex_openapi_endpoint(monkeypatch):
    _fake_credentials(monkeypatch)

    client = vertex.vertex_client('open-targets-eu-dev')

    assert str(client.base_url).rstrip('/') == (
        'https://aiplatform.googleapis.com/v1/projects/open-targets-eu-dev/locations/global/endpoints/openapi'
    )
    assert client.api_key == 'tok-1'


def test_regional_client_uses_the_regional_host(monkeypatch):
    _fake_credentials(monkeypatch)

    client = vertex.vertex_client('p', location='us-central1')

    assert str(client.base_url).startswith(
        'https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/'
    )


def test_each_client_gets_a_fresh_token(monkeypatch):
    tokens = iter(['a', 'b'])
    creds = SimpleNamespace(token=None)
    creds.refresh = lambda request: setattr(creds, 'token', next(tokens))
    monkeypatch.setattr(google.auth, 'default', lambda scopes: (creds, 'proj'))

    assert [vertex.vertex_client('p').api_key, vertex.vertex_client('p').api_key] == ['a', 'b']
