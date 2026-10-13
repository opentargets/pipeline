"""Tests for the S3 connector configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pps.external.s3 import S3Config, parse_endpoint_url

S3_CONFIG_FILE = Path(__file__).parent / 'data_samples' / 'example_s3_config.json'

ENV = {
    'AWS_S3_BUCKET_NAME': 'my-bucket',
    'AWS_ENDPOINT_URL': 's3.my-domain.com:9000',
    'AWS_ACCESS_KEY_ID': 'my-access-key-id',
    'AWS_SECRET_ACCESS_KEY': 'my-secret-access-key',
}


@pytest.fixture
def s3_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Set the S3 environment variables."""
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_from_json() -> None:
    config = S3Config.from_json(str(S3_CONFIG_FILE))

    assert config.bucket_name == 'my_bucket'
    assert config.s3_host_url == 'https://some_host_url.com'
    assert config.s3_host_port == 1234
    assert config.access_key_id.get_secret_value() == 'my_access_key'
    assert config.secret_access_key.get_secret_value() == 'my_secret_access'


def test_secrets_are_hidden_from_repr() -> None:
    config = S3Config.from_json(str(S3_CONFIG_FILE))

    assert 'my_access_key' not in repr(config)
    assert 'my_secret_access' not in repr(config)


def test_from_json_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        S3Config.from_json(str(tmp_path / 'missing.json'))


@pytest.mark.parametrize(
    ('change', 'field'),
    [
        pytest.param({'bucket_name': None}, 'bucket_name', id='missing field'),
        pytest.param({'s3_host_port': 'not-a-port'}, 's3_host_port', id='invalid port'),
    ],
)
def test_from_json_invalid_content(tmp_path: Path, change: dict, field: str) -> None:
    content = json.loads(S3_CONFIG_FILE.read_text()) | change
    path = tmp_path / 's3.json'
    path.write_text(json.dumps({k: v for k, v in content.items() if v is not None}))

    with pytest.raises(ValidationError, match=field):
        S3Config.from_json(str(path))


@pytest.mark.usefixtures('s3_env')
def test_from_env() -> None:
    config = S3Config.from_env()

    assert config.bucket_name == 'my-bucket'
    assert config.s3_host_url == 's3.my-domain.com'
    assert config.s3_host_port == 9000
    assert config.access_key_id.get_secret_value() == 'my-access-key-id'
    assert config.secret_access_key.get_secret_value() == 'my-secret-access-key'


@pytest.mark.parametrize('name', ['AWS_S3_BUCKET_NAME', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'])
def test_from_env_missing_variable(s3_env: pytest.MonkeyPatch, name: str) -> None:
    s3_env.delenv(name)

    with pytest.raises(ValueError, match=f'{name} environment variable is not set'):
        S3Config.from_env()


@pytest.mark.parametrize(
    ('endpoint', 'host', 'port'),
    [
        pytest.param('s3.my-domain.com:9000', 's3.my-domain.com', 9000, id='host and port'),
        pytest.param('s3.my-domain.com', 's3.my-domain.com', 443, id='host'),
        pytest.param('https://s3.my-domain.com:9000', 's3.my-domain.com', 9000, id='https, host and port'),
        pytest.param('https://s3.amazonaws.com', 's3.amazonaws.com', 443, id='https and host'),
        pytest.param('https://s3.amazonaws.com/', 's3.amazonaws.com', 443, id='trailing slash'),
    ],
)
def test_parse_endpoint_url(endpoint: str, host: str, port: int) -> None:
    assert parse_endpoint_url(endpoint) == (host, port)


@pytest.mark.parametrize(
    ('endpoint', 'message'),
    [
        pytest.param('http://s3.my-domain.com:9000', 'only connects over https', id='http'),
        pytest.param('s3.my-domain.com:9000:1', 'the port must be an integer', id='too many parts'),
        pytest.param('s3.my-domain.com:http', 'the port must be an integer', id='non numeric port'),
        pytest.param('https://s3.my-domain.com/bucket', "must be 'host\\[:port\\]'", id='path'),
        pytest.param('https://', "must be 'host\\[:port\\]'", id='no host'),
    ],
)
def test_parse_endpoint_url_rejects_invalid_endpoints(endpoint: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_endpoint_url(endpoint)


def test_from_env_rejects_an_invalid_endpoint(s3_env: pytest.MonkeyPatch) -> None:
    s3_env.setenv('AWS_ENDPOINT_URL', 'http://s3.my-domain.com:9000')

    with pytest.raises(ValueError, match='only connects over https'):
        S3Config.from_env()


def test_from_env_rejects_an_empty_endpoint(s3_env: pytest.MonkeyPatch) -> None:
    s3_env.setenv('AWS_ENDPOINT_URL', '')

    with pytest.raises(ValueError, match='AWS_ENDPOINT_URL environment variable is not set'):
        S3Config.from_env()


def test_from_env_defaults_to_aws_over_https(s3_env: pytest.MonkeyPatch) -> None:
    s3_env.delenv('AWS_ENDPOINT_URL')

    config = S3Config.from_env()

    assert config.s3_host_url == 's3.amazonaws.com'
    assert config.s3_host_port == 443


def test_read_prefers_the_file(s3_env: pytest.MonkeyPatch) -> None:
    assert S3Config.read(str(S3_CONFIG_FILE)).bucket_name == 'my_bucket'


@pytest.mark.usefixtures('s3_env')
def test_read_falls_back_to_the_environment() -> None:
    assert S3Config.read().bucket_name == 'my-bucket'
