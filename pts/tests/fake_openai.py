"""A stand-in for ``openai.AsyncOpenAI`` that scripts what ``responses.create`` does per input."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import httpx2
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
    UnprocessableEntityError,
)

Outcome = str | BaseException
"""An ``output_text`` to return, or an exception to raise."""


class FakeClient:
    """Answer ``responses.create`` from scripted outcomes keyed by its ``input``.

    A single outcome answers every call for its input; a sequence answers one
    call each, in order. Inputs absent from ``outcomes`` go to ``default``.
    """

    def __init__(
        self,
        outcomes: Mapping[str, Outcome | Sequence[Outcome]] | None = None,
        default: Callable[[str], Outcome] | None = None,
    ) -> None:
        self._outcomes = {
            text: outcome if isinstance(outcome, str | BaseException) else list(outcome)
            for text, outcome in (outcomes or {}).items()
        }
        self._default = default
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.closed = False
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            outcome = self._next(kwargs['input'])
        finally:
            self.in_flight -= 1
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(output_text=outcome)

    def _next(self, text: str) -> Outcome:
        scripted = self._outcomes.get(text)
        if isinstance(scripted, list):
            if not scripted:
                msg = f'no outcome left for input {text!r}'
                raise AssertionError(msg)
            return scripted.pop(0)
        if scripted is not None:
            return scripted
        if self._default is None:
            msg = f'no outcome scripted for input {text!r}'
            raise AssertionError(msg)
        return self._default(text)

    def calls_for(self, text: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call['input'] == text]

    async def close(self) -> None:
        self.closed = True


_SDK = AsyncOpenAI(api_key='not-a-key')


def _request() -> httpx2.Request:
    return httpx2.Request('POST', 'https://api.openai.com/v1/responses')


def status_error(status: int, code: str | None, message: str) -> APIStatusError:
    """The error the SDK raises for a response of ``status`` with this error object."""
    error = {'message': message, 'type': 'invalid_request_error', 'param': None, 'code': code}
    response = httpx2.Response(status, json={'error': error}, request=_request())
    return _SDK._make_status_error_from_response(response)


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=_request())


def flex_unavailable() -> RateLimitError:
    return cast(RateLimitError, status_error(429, 'resource_unavailable', 'Flex capacity unavailable'))


def rate_limited() -> RateLimitError:
    return cast(RateLimitError, status_error(429, 'rate_limit_exceeded', 'Rate limit reached'))


def bad_request() -> BadRequestError:
    return cast(BadRequestError, status_error(400, 'invalid_prompt', 'Invalid prompt'))


def unprocessable() -> UnprocessableEntityError:
    return cast(UnprocessableEntityError, status_error(422, None, 'Unprocessable'))
