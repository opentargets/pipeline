"""Clients for models served by Vertex AI's OpenAI-compatible endpoint (such as gpt-oss).

Authentication is the Google Application Default Credentials the pipeline already uses.
The access token lasts about an hour, so callers build a new client for every chunk.
"""

from __future__ import annotations

import google.auth
import google.auth.transport.requests
from openai import AsyncOpenAI

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
DEFAULT_PROJECT = 'open-targets-eu-dev'
DEFAULT_LOCATION = 'global'
GPT_OSS_120B = 'openai/gpt-oss-120b-maas'
TIMEOUT_SECONDS = 600


def vertex_client(project: str = DEFAULT_PROJECT, location: str = DEFAULT_LOCATION) -> AsyncOpenAI:
    """Return an OpenAI client for Vertex AI's chat-completions endpoint with a fresh token."""
    credentials, _ = google.auth.default(scopes=SCOPES)
    credentials.refresh(google.auth.transport.requests.Request())
    host = 'aiplatform.googleapis.com' if location == 'global' else f'{location}-aiplatform.googleapis.com'
    return AsyncOpenAI(
        base_url=f'https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi',
        api_key=credentials.token,
        timeout=TIMEOUT_SECONDS,
    )
