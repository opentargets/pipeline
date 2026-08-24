"""Authenticate as the pipeline supervisor GitHub App and comment on the run's issue.

This is the only module that talks to GitHub; `observer.py` and `report.py` (later
phases) decide *what* to say and hand it to `GitHubApp.comment`.

Authentication is GitHub's two-step App flow: a JWT signed with the App's private key
proves the App's identity, then that JWT is exchanged for an installation access
token scoped to the repos the App is installed on. The JWT construction is pure and
signs nothing over the network; the exchange and the comment post are the two places
this module touches HTTP, and both take an injected session so no unit test needs
credentials.

**Nothing here is cached.** Installation tokens expire in one hour, so a supervisor
that ran unattended for longer than that with a cached token would start failing
silently between wakeups. Minting a fresh token on every call is simpler than
tracking an expiry and refreshing, and it is correct by construction. Callers must
also never pass the private key as a CLI argument: a flag's value is visible in `ps`
output to every user on the machine and lands in shell history, the same reason
`cli.py`'s `_airflow_credentials` reads Airflow's password from the environment
rather than a flag. This runs from cron, whose stderr goes to a log file that
outlives the process and is readable by anyone on the VM, so a private key or an
installation token must never appear in an exception message, a `repr`, or a log
line either — see `app_jwt` and `installation_token` for how that is enforced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import jwt
from jwt.exceptions import PyJWTError

GITHUB_API = 'https://api.github.com'
"""Base URL for both the JWT- and the installation-token-authenticated calls."""

_JWT_BACKDATE_SECONDS = 60
"""How far before `now` to backdate `iat`, tolerating clock skew with GitHub's host."""

_JWT_LIFETIME_SECONDS = 540
"""How far past `now` to set `exp`.

GitHub rejects a JWT whose `exp` is more than ten minutes into the future *from now*,
not from `iat`. Backdating `iat` by `_JWT_BACKDATE_SECONDS` and setting `exp` this far
past `now` keeps `exp - iat` at exactly six hundred seconds while leaving a minute of
margin under that ceiling.
"""


def app_jwt(app_id: str, private_key: str, now: datetime) -> str:
    """Sign a JSON Web Token asserting this App's identity.

    Pure: given the same inputs it produces the same signature, and it never touches
    the network — only `installation_token`, which exchanges the result, does. See
    GitHub's docs on "Authenticating as a GitHub App".

    Args:
        app_id: The GitHub App's numeric id, carried as the `iss` claim.
        private_key: The App's RSA private key, PEM-encoded.
        now: The current time, injected so tests can pin `iat`/`exp` without waiting
            on the clock.

    Returns:
        A compact RS256-signed JWS, with `iat`/`exp` bracketing `now` per
        `_JWT_BACKDATE_SECONDS`/`_JWT_LIFETIME_SECONDS` above.

    Raises:
        RuntimeError: If signing fails, e.g. because `private_key` is not a valid PEM
            key. The underlying `PyJWTError` is re-raised as this type with a fixed
            message and `from None`, so a malformed key can never be echoed back
            through it or through a chained traceback.
    """
    issued_at = int(now.timestamp()) - _JWT_BACKDATE_SECONDS
    claims = {'iat': issued_at, 'exp': issued_at + _JWT_BACKDATE_SECONDS + _JWT_LIFETIME_SECONDS, 'iss': app_id}
    try:
        return jwt.encode(claims, private_key, algorithm='RS256')
    except PyJWTError:
        raise RuntimeError('failed to sign the app JWT: private_key is not a valid PEM-encoded key') from None


def installation_token(session: Any, app_id: str, private_key: str, installation_id: int, now: datetime) -> str:
    """Exchange a signed app JWT for an installation access token.

    Mints a fresh token on every call — see this module's docstring for why nothing
    here caches one.

    Args:
        session: An HTTP session (`requests`-shaped), injected so no unit test needs
            network access.
        app_id: The GitHub App's numeric id.
        private_key: The App's RSA private key, PEM-encoded.
        installation_id: The installation to mint a token for.
        now: The current time, threaded through to `app_jwt`.

    Returns:
        The installation access token.

    Raises:
        RuntimeError: If the exchange does not return HTTP 201, or if signing the JWT
            fails. The response body is included for diagnosis, but never the private
            key or the JWT itself — GitHub's own error responses do not echo either
            back, and this function never builds a message from them.
    """
    assertion = app_jwt(app_id, private_key, now)
    response = session.post(
        f'{GITHUB_API}/app/installations/{installation_id}/access_tokens',
        headers={'Authorization': f'Bearer {assertion}', 'Accept': 'application/vnd.github+json'},
    )
    if response.status_code != 201:
        raise RuntimeError(
            f'installation token exchange failed with HTTP {response.status_code}: {response.text}'
        )
    return response.json()['token']


def read_app_key(client: Any, project: str, secret: str, version: str = 'latest') -> str:
    """Read the App's private key from Secret Manager.

    Args:
        client: A `secretmanager.SecretManagerServiceClient`, injected so no unit
            test needs credentials.
        project: GCP project holding the secret.
        secret: Secret id.
        version: Secret version, defaulting to the latest enabled version.

    Returns:
        The PEM-encoded private key.
    """
    name = f'projects/{project}/secrets/{secret}/versions/{version}'
    return client.access_secret_version(request={'name': name}).payload.data.decode()


class GitHubApp:
    """Comments on issues in one repo, authenticated as the pipeline supervisor App.

    Args:
        session: An HTTP session, injected so no unit test needs network access.
        app_id: The GitHub App's numeric id.
        private_key: The App's RSA private key, PEM-encoded. Held only in memory —
            never written to disk, and never allowed into an exception message or
            this object's `repr` (the default `object.__repr__` this class inherits
            prints only the type and address, never instance attributes).
        installation_id: The installation covering `repo`.
        repo: The `owner/name` repository this App comments on.
    """

    def __init__(self, session: Any, app_id: str, private_key: str, installation_id: int, repo: str) -> None:
        self.session = session
        self.app_id = app_id
        self.private_key = private_key
        self.installation_id = installation_id
        self.repo = repo

    def comment(self, issue: int, body: str) -> None:
        """Post a comment on one issue.

        Mints a fresh installation token for this call — see this module's
        docstring for why one is never reused across calls.

        Args:
            issue: The issue (or pull request) number within `repo`.
            body: The comment's markdown body.

        Raises:
            RuntimeError: If minting a token fails, or the comment post does not
                return HTTP 201.
        """
        token = installation_token(
            self.session, self.app_id, self.private_key, self.installation_id, datetime.now(tz=UTC)
        )
        response = self.session.post(
            f'{GITHUB_API}/repos/{self.repo}/issues/{issue}/comments',
            headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github+json'},
            json={'body': body},
        )
        if response.status_code != 201:
            raise RuntimeError(f'comment post failed with HTTP {response.status_code}: {response.text}')
