"""Live checks against the real GitHub App and Secret Manager.

Skipped unless RUN_GITHUB_TESTS is set, because these need credentials and network.
Run with: RUN_GITHUB_TESTS=1 uv run --frozen pytest tests/test_supervisor_github_live.py -rxs

Deliberately stops short of `GitHubApp.comment`: posting a comment is a write to a
shared repo's issue tracker, and a test that leaves debris behind on every run is
worse than no live test at all. `GET /app` returning this App's own slug, and minting
a real installation token, prove the auth chain end to end without writing anything.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import requests
from google.cloud import secretmanager

from orchestration.supervisor.github import GITHUB_API, app_jwt, installation_token, read_app_key

pytestmark = pytest.mark.skipif(
    not os.environ.get('RUN_GITHUB_TESTS'),
    reason='needs GCP and GitHub credentials, set RUN_GITHUB_TESTS=1 to run',
)

PROJECT = os.environ.get('GITHUB_TEST_PROJECT', 'open-targets-eu-dev')
SECRET = os.environ.get('GITHUB_TEST_SECRET', 'supervisor-github-app-key')
APP_ID = os.environ.get('GITHUB_TEST_APP_ID', '4699938')
APP_SLUG = os.environ.get('GITHUB_TEST_APP_SLUG', 'opentargets-pipeline-supervisor')
INSTALLATION_ID = int(os.environ.get('GITHUB_TEST_INSTALLATION_ID', '156145657'))


@pytest.fixture
def private_key() -> str:
    client = secretmanager.SecretManagerServiceClient()
    return read_app_key(client, PROJECT, SECRET)


class TestLiveGitHubAuth:
    def test_the_app_jwt_authenticates_as_this_app(self, private_key: str) -> None:
        """GET /app is authenticated with the bare App JWT, before any installation token exists.

        Its slug matching the known one proves the key, the claims and the signature
        are all correct.
        """
        token = app_jwt(APP_ID, private_key, datetime.now(UTC))
        response = requests.get(
            f'{GITHUB_API}/app',
            headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
            timeout=30,
        )
        assert response.status_code == 200
        assert response.json()['slug'] == APP_SLUG

    def test_an_installation_token_can_be_minted(self, private_key: str) -> None:
        session = requests.Session()
        token = installation_token(session, APP_ID, private_key, INSTALLATION_ID, datetime.now(UTC))
        assert token.startswith('ghs_')
