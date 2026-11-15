"""The local, append-only result cache of structured LLM calls.

A result is keyed on everything that can change it: the input text, the system prompt,
the response schema and the model. :py:func:`.claude.run_batch_cached` writes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pts.transformers.clinical_mira.llm_schema import _read_cache, result_key

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

__all__ = ['cached_results', 'result_key']


def cached_results(
    requests: list[tuple[str, str]],
    *,
    system_prompt: str,
    response_model: type[BaseModel],
    model: str,
    cache_path: Path,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    """Return the cached result of each ``(request_id, input_text)``, calling no model.

    Returns:
        The validated response per cached request id, and no errors.
    """
    cached = _read_cache(cache_path)
    results = {}
    for request_id, text in requests:
        key = result_key(text, system_prompt, response_model, model)
        if key in cached:
            results[request_id] = response_model.model_validate(cached[key])
    return results, {}
