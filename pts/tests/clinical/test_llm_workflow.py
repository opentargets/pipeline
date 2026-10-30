import asyncio
from collections.abc import Iterator, Sequence

import pytest
from loguru import logger
from openai import APIConnectionError, BadRequestError, UnprocessableEntityError
from pydantic import BaseModel

from pts.transformers.clinical.llm_workflow import (
    SYSTEMIC_ERROR_TYPES,
    Extracted,
    Failed,
    _patch_schema,
    client_timeout,
    extract_prompts,
)
from tests.fake_openai import (
    FakeClient,
    bad_request,
    connection_error,
    flex_unavailable,
    rate_limited,
    unprocessable,
)


class Trial(BaseModel):
    id: str
    drug: str | None = None


def _answer(drug: str = 'aspirin', trial_id: str = 'from-model') -> str:
    return Trial(id=trial_id, drug=drug).model_dump_json()


def _extract(
    client: FakeClient,
    prompts: Sequence[tuple[str, str]],
    service_tier: str = 'auto',
    concurrency: int = 4,
) -> dict[str, Extracted | Failed]:
    return asyncio.run(
        extract_prompts(
            client,
            Trial,
            'system prompt',
            prompts,
            model='test-model',
            service_tier=service_tier,
            concurrency=concurrency,
        )
    )


@pytest.fixture
def logged() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m).strip()), level='INFO', format='{message}')
    yield messages
    logger.remove(sink)


def test_success_is_extracted_with_the_prompt_id() -> None:
    client = FakeClient({'prompt 1': _answer('ibuprofen')})

    result = _extract(client, [('nct1', 'prompt 1')])

    assert result == {'nct1': Extracted(Trial(id='nct1', drug='ibuprofen'))}


def test_results_follow_prompt_order() -> None:
    client = FakeClient(default=lambda text: _answer(text))
    prompts = [(f'nct{i}', f'prompt {i}') for i in (3, 1, 2)]

    result = _extract(client, prompts)

    assert list(result) == ['nct3', 'nct1', 'nct2']
    assert all(isinstance(outcome, Extracted) for outcome in result.values())


def test_request_parameters() -> None:
    client = FakeClient({'prompt 1': _answer()})

    _extract(client, [('nct1', 'prompt 1')], service_tier='default')

    assert client.calls == [
        {
            'model': 'test-model',
            'instructions': 'system prompt',
            'input': 'prompt 1',
            'text': {
                'format': {
                    'type': 'json_schema',
                    'name': 'Trial',
                    'schema': _patch_schema(Trial.model_json_schema(by_alias=True)),
                    'strict': True,
                }
            },
            'service_tier': 'default',
            'store': False,
        }
    ]
    schema = client.calls[0]['text']['format']['schema']
    assert schema['additionalProperties'] is False
    assert schema['required'] == ['id', 'drug']


def test_invalid_output_fails_validation() -> None:
    client = FakeClient({'prompt 1': '{"drug": 1}'})

    result = _extract(client, [('nct1', 'prompt 1')])

    failure = result['nct1']
    assert isinstance(failure, Failed)
    assert failure.error_type == 'ValidationError'
    assert failure.message.startswith('2 validation errors for Trial')


@pytest.mark.parametrize(
    ('error', 'error_type'),
    [
        (bad_request(), 'BadRequestError'),
        (unprocessable(), 'UnprocessableEntityError'),
        (connection_error(), 'APIConnectionError'),
    ],
)
def test_api_errors_fail_with_their_class_and_message(error: Exception, error_type: str) -> None:
    client = FakeClient({'prompt 1': error})

    result = _extract(client, [('nct1', 'prompt 1')])

    assert result == {'nct1': Failed(error_type, str(error))}


def test_error_without_text_fails_with_its_repr() -> None:
    error = KeyError()
    client = FakeClient({'prompt 1': error})

    result = _extract(client, [('nct1', 'prompt 1')])

    assert str(error) == ''
    assert result == {'nct1': Failed('KeyError', repr(error))}


def test_flex_unavailable_falls_back_to_auto_once(logged: list[str]) -> None:
    client = FakeClient({'prompt 1': [flex_unavailable(), _answer()]})

    result = _extract(client, [('nct1', 'prompt 1')], service_tier='flex')

    assert result == {'nct1': Extracted(Trial(id='nct1', drug='aspirin'))}
    assert [call['service_tier'] for call in client.calls] == ['flex', 'auto']
    assert 'Flex capacity unavailable for id=nct1 — retrying with standard processing.' in logged


def test_flex_unavailable_on_auto_fails() -> None:
    error = flex_unavailable()
    client = FakeClient({'prompt 1': error})

    result = _extract(client, [('nct1', 'prompt 1')], service_tier='auto')

    assert result == {'nct1': Failed('RateLimitError', str(error))}
    assert len(client.calls) == 1


def test_flex_unavailable_twice_fails_after_auto() -> None:
    error = flex_unavailable()
    client = FakeClient({'prompt 1': error})

    result = _extract(client, [('nct1', 'prompt 1')], service_tier='flex')

    assert result == {'nct1': Failed('RateLimitError', str(error))}
    assert [call['service_tier'] for call in client.calls] == ['flex', 'auto']


def test_plain_rate_limit_on_flex_does_not_fall_back() -> None:
    error = rate_limited()
    client = FakeClient({'prompt 1': error})

    result = _extract(client, [('nct1', 'prompt 1')], service_tier='flex')

    assert result == {'nct1': Failed('RateLimitError', str(error))}
    assert len(client.calls) == 1


def test_calls_stay_within_concurrency() -> None:
    prompts = [(f'nct{i}', f'prompt {i}') for i in range(12)]
    client = FakeClient(default=lambda _: _answer())

    _extract(client, prompts, concurrency=3)

    assert len(client.calls) == 12
    assert client.peak_in_flight == 3


def test_fallbacks_stay_within_concurrency() -> None:
    prompts = [(f'nct{i}', f'prompt {i}') for i in range(12)]
    client = FakeClient({text: [flex_unavailable(), _answer()] for _, text in prompts})

    result = _extract(client, prompts, service_tier='flex', concurrency=3)

    assert all(isinstance(outcome, Extracted) for outcome in result.values())
    assert len(client.calls) == 24
    assert client.peak_in_flight == 3


def test_one_prompt_makes_one_call() -> None:
    client = FakeClient({'prompt 1': _answer()})

    result = _extract(client, [('nct1', 'prompt 1')])

    assert len(client.calls) == 1
    assert list(result) == ['nct1']


def test_client_is_left_open() -> None:
    client = FakeClient({'prompt 1': _answer(), 'prompt 2': connection_error()})

    _extract(client, [('nct1', 'prompt 1'), ('nct2', 'prompt 2')])

    assert not client.closed


def test_run_and_failures_are_logged(logged: list[str]) -> None:
    prompts = [(f'nct{i}', f'prompt {i}') for i in range(8)]
    client = FakeClient({text: _answer() for _, text in prompts[:1]}, default=lambda _: RuntimeError('boom'))

    _extract(client, prompts, service_tier='default', concurrency=3)

    assert 'Running extraction: model=test-model  concurrency=3  service_tier=default' in logged
    assert 'Extraction complete: 1 succeeded, 7 failed.' in logged
    failed = [m for m in logged if m.startswith('Failed  id=')]
    assert failed == [f'Failed  id={f"nct{i}":<30}  type=RuntimeError  message=boom' for i in range(1, 6)]
    assert '... plus 2 more errors' in logged


def test_no_failures_logs_no_failure_lines(logged: list[str]) -> None:
    client = FakeClient(default=lambda _: _answer())

    _extract(client, [('nct1', 'prompt 1'), ('nct2', 'prompt 2')])

    assert 'Extraction complete: 2 succeeded, 0 failed.' in logged
    assert not [m for m in logged if m.startswith(('Failed  id=', '... plus'))]


def test_systemic_error_types() -> None:
    assert {'APIConnectionError', 'RateLimitError', 'ServiceTierExhausted'} <= SYSTEMIC_ERROR_TYPES
    assert APIConnectionError.__name__ in SYSTEMIC_ERROR_TYPES
    assert not {BadRequestError.__name__, UnprocessableEntityError.__name__} & SYSTEMIC_ERROR_TYPES
    assert 'ValidationError' not in SYSTEMIC_ERROR_TYPES


@pytest.mark.parametrize(('service_tier', 'timeout'), [('flex', 900), ('auto', 600), ('default', 600)])
def test_client_timeout(service_tier: str, timeout: int) -> None:
    assert client_timeout(service_tier) == timeout
