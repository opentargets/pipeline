from pathlib import Path

import clinical_mining.workflows.llm as llm_workflow
import pytest

from pts.tasks.llm_extract import _run_extraction_in_thread


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_run_extraction_closes_async_client_before_loop_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeClient()

    def create_client(**kwargs: object) -> _FakeClient:
        return client

    async def fake_run_async(
        *args: object, **kwargs: object
    ) -> tuple[list[object], list[object]]:
        return [], []

    monkeypatch.setattr(llm_workflow, 'AsyncOpenAI', create_client)
    monkeypatch.setattr(llm_workflow, '_run_async', fake_run_async)
    prompt_path = tmp_path / 'system_prompt.txt'
    prompt_path.write_text('system prompt')

    result = _run_extraction_in_thread(
        prompts=[{'id': 'a', 'prompt': 'prompt'}, {'id': 'b', 'prompt': 'prompt'}],
        model_class='clinical_mining.schemas.ClinicalReportExtractionSchema',
        system_prompt_path=str(prompt_path),
        model='test-model',
        openai_key='test-key',
        service_tier='auto',
        concurrency=1,
        max_retries=0,
    )

    assert result is not None and result.is_empty()
    assert client.closed
