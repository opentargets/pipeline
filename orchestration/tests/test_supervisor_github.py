"""Tests for the supervisor's GitHub App authentication and issue commenting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from orchestration.supervisor.github import GitHubApp, app_jwt, installation_token, read_app_key

APP_ID = '4699938'
INSTALLATION_ID = 156145657
REPO = 'opentargets/pipeline'


def _rsa_pem_pair() -> tuple[str, str]:
    """A throwaway RSA key pair, PEM-encoded, for signing and verifying test JWTs.

    Generated fresh per call rather than hardcoded: a hardcoded key committed to the
    repo reads exactly like a real leaked credential to anyone scanning history for
    one, which is the opposite of what a credential-hygiene test should produce.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def _decode(token: str, public_pem: str) -> dict[str, Any]:
    """Decode a test JWT, skipping expiry checks since tests pin an arbitrary `now`."""
    return jwt.decode(token, public_pem, algorithms=['RS256'], options={'verify_exp': False})


class TestAppJwt:
    def test_carries_the_app_id_as_issuer(self) -> None:
        private_pem, public_pem = _rsa_pem_pair()
        token = app_jwt(APP_ID, private_pem, datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC))
        assert _decode(token, public_pem)['iss'] == APP_ID

    def test_iat_and_exp_bracket_now(self) -> None:
        private_pem, public_pem = _rsa_pem_pair()
        now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
        claims = _decode(app_jwt(APP_ID, private_pem, now), public_pem)
        now_ts = int(now.timestamp())
        assert claims['iat'] <= now_ts <= claims['exp']

    def test_exp_is_at_most_ten_minutes_beyond_now(self) -> None:
        """GitHub rejects a JWT whose exp is more than ten minutes into the future."""
        private_pem, public_pem = _rsa_pem_pair()
        now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
        claims = _decode(app_jwt(APP_ID, private_pem, now), public_pem)
        assert claims['exp'] - int(now.timestamp()) <= 600

    def test_a_malformed_key_raises_without_echoing_it(self) -> None:
        garbage = 'not-a-real-private-key-SECRETVALUE12345'
        with pytest.raises(RuntimeError) as excinfo:
            app_jwt(APP_ID, garbage, datetime.now(UTC))
        assert garbage not in str(excinfo.value)


class TestInstallationToken:
    def test_a_201_response_returns_the_token(self) -> None:
        private_pem, _ = _rsa_pem_pair()
        session = MagicMock()
        session.post.return_value = MagicMock(status_code=201, json=lambda: {'token': 'ghs_abc'})

        result = installation_token(session, APP_ID, private_pem, INSTALLATION_ID, datetime.now(UTC))

        assert result == 'ghs_abc'
        url = session.post.call_args.args[0]
        assert url == f'https://api.github.com/app/installations/{INSTALLATION_ID}/access_tokens'

    def test_a_non_201_response_raises_with_status_and_body_not_a_keyerror(self) -> None:
        private_pem, _ = _rsa_pem_pair()
        session = MagicMock()
        session.post.return_value = MagicMock(status_code=401, text='{"message": "Bad credentials"}')

        with pytest.raises(RuntimeError, match='401') as excinfo:
            installation_token(session, APP_ID, private_pem, INSTALLATION_ID, datetime.now(UTC))
        assert 'Bad credentials' in str(excinfo.value)

    def test_the_private_key_never_appears_in_the_raised_message(self) -> None:
        """The failure that would turn a cron log into a credential leak.

        A cron's stderr goes to a log file that outlives the process and is readable
        by anyone on the VM, so neither the private key nor its content may ever
        surface through an exception raised here.
        """
        private_pem, _ = _rsa_pem_pair()
        session = MagicMock()
        session.post.return_value = MagicMock(status_code=401, text='{"message": "Bad credentials"}')

        with pytest.raises(RuntimeError) as excinfo:
            installation_token(session, APP_ID, private_pem, INSTALLATION_ID, datetime.now(UTC))

        assert private_pem not in str(excinfo.value)
        assert private_pem not in repr(excinfo.value)


class TestReadAppKey:
    def test_reads_the_latest_version_by_default(self) -> None:
        client = MagicMock()
        client.access_secret_version.return_value = MagicMock(
            payload=MagicMock(data=b'-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n')
        )

        result = read_app_key(client, 'open-targets-eu-dev', 'supervisor-github-app-key')

        client.access_secret_version.assert_called_once_with(
            request={'name': 'projects/open-targets-eu-dev/secrets/supervisor-github-app-key/versions/latest'}
        )
        assert result.startswith('-----BEGIN RSA PRIVATE KEY-----')

    def test_reads_a_pinned_version_when_given_one(self) -> None:
        client = MagicMock()
        client.access_secret_version.return_value = MagicMock(payload=MagicMock(data=b'key-material'))

        read_app_key(client, 'open-targets-eu-dev', 'supervisor-github-app-key', version='3')

        client.access_secret_version.assert_called_once_with(
            request={'name': 'projects/open-targets-eu-dev/secrets/supervisor-github-app-key/versions/3'}
        )


class TestGitHubAppComment:
    def test_posts_to_the_right_url_with_the_right_body(self) -> None:
        private_pem, _ = _rsa_pem_pair()
        session = MagicMock()
        session.post.side_effect = [
            MagicMock(status_code=201, json=lambda: {'token': 'ghs_abc'}),
            MagicMock(status_code=201, json=lambda: {'id': 1}),
        ]
        app = GitHubApp(session, APP_ID, private_pem, INSTALLATION_ID, REPO)

        app.comment(42, 'hello world')

        comment_call = session.post.call_args_list[1]
        assert comment_call.args[0] == f'https://api.github.com/repos/{REPO}/issues/42/comments'
        assert comment_call.kwargs['json'] == {'body': 'hello world'}
        assert comment_call.kwargs['headers']['Authorization'] == 'token ghs_abc'

    def test_mints_a_fresh_token_for_every_comment(self) -> None:
        """Installation tokens expire in one hour; nothing here may cache one across calls."""
        private_pem, _ = _rsa_pem_pair()
        session = MagicMock()
        session.post.side_effect = [
            MagicMock(status_code=201, json=lambda: {'token': 'ghs_first'}),
            MagicMock(status_code=201, json=lambda: {'id': 1}),
            MagicMock(status_code=201, json=lambda: {'token': 'ghs_second'}),
            MagicMock(status_code=201, json=lambda: {'id': 2}),
        ]
        app = GitHubApp(session, APP_ID, private_pem, INSTALLATION_ID, REPO)

        app.comment(1, 'first')
        app.comment(2, 'second')

        token_calls = [c for c in session.post.call_args_list if 'access_tokens' in c.args[0]]
        assert len(token_calls) == 2
        second_comment_call = session.post.call_args_list[3]
        assert second_comment_call.kwargs['headers']['Authorization'] == 'token ghs_second'

    def test_a_non_201_comment_response_raises(self) -> None:
        private_pem, _ = _rsa_pem_pair()
        session = MagicMock()
        session.post.side_effect = [
            MagicMock(status_code=201, json=lambda: {'token': 'ghs_abc'}),
            MagicMock(status_code=422, text='{"message": "Validation Failed"}'),
        ]
        app = GitHubApp(session, APP_ID, private_pem, INSTALLATION_ID, REPO)

        with pytest.raises(RuntimeError, match='422'):
            app.comment(1, 'hello')

    def test_repr_never_includes_the_private_key(self) -> None:
        """Checks a single-line fragment too, not just the whole multi-line PEM.

        A leaky `repr` built with `!r` on the key (as a dataclass-generated one would
        be) escapes the PEM's newlines, which would defeat a containment check on the
        whole string: the escaped form is no longer the same contiguous text. A
        fragment from one line survives that escaping unchanged, so it still catches
        the leak.
        """
        private_pem, _ = _rsa_pem_pair()
        app = GitHubApp(MagicMock(), APP_ID, private_pem, INSTALLATION_ID, REPO)
        rendered = repr(app)
        assert private_pem not in rendered
        assert private_pem.splitlines()[1] not in rendered
