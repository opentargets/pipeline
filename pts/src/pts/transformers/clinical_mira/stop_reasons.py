"""Clinical trial stop reason classification with a HuggingFace sequence classifier."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

STOP_REASONS_MODEL = 'opentargets/clinical_trial_stop_reasons'
STOP_REASONS_THRESHOLD = 0.3


def get_device() -> torch.device:
    """Return the best available torch device (MPS > CUDA > CPU)."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@lru_cache(maxsize=1)
def _load_model_assets(model_name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, dict[int, str]]:
    """Load the tokenizer, model, and label mapping — only once, the model is cached."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer is None:
        raise ValueError(f'{model_name} has no tokenizer')

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(get_device())
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    return tokenizer, model, id2label


def _predict_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    id2label: dict[int, str],
    threshold: float,
) -> list[list[str]]:
    """Run inference on a single batch and return predicted labels per text."""
    import torch

    device = next(model.parameters()).device

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = model(**encoded).logits
    probs = torch.sigmoid(logits).cpu()

    results = []
    for prob_row in probs.tolist():
        labels = [id2label[i] for i, prob in enumerate(prob_row) if prob >= threshold]
        results.append(labels or ['Uncategorised'])

    return results


def predict_trial_stop_reasons(
    texts: list[str],
    *,
    model_name: str = STOP_REASONS_MODEL,
    threshold: float = STOP_REASONS_THRESHOLD,
    batch_size: int = 32,
) -> list[list[str]]:
    """Classify clinical trial stop reasons for a list of texts.

    Args:
        texts:      Input texts to classify.
        model_name: Stop reasons classifier in HuggingFace (default 'opentargets/clinical_trial_stop_reasons').
        threshold:  Minimum sigmoid probability to assign a label (default 0.3).
        batch_size: Number of texts processed per forward pass (default 32).

    Returns:
        A list of predicted label lists, one per input text.
        Texts with no label above the threshold are assigned ['Uncategorised'].
    """
    import torch

    tokenizer, model, id2label = _load_model_assets(model_name)
    results = []

    with torch.inference_mode():
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            batch_results = _predict_batch(batch_texts, tokenizer, model, id2label, threshold)
            results.extend(batch_results)

    return results


def stop_reasons_with_status(texts: list[str], batch_size: int = 32) -> list[tuple[list[str] | None, str | None]]:
    """Stop reason labels per text, or ``(None, error)`` for the texts of a batch that raised."""
    if not texts:
        return []
    _load_model_assets(STOP_REASONS_MODEL)
    results: list[tuple[list[str] | None, str | None]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            labels = predict_trial_stop_reasons(batch, batch_size=batch_size)
        except Exception as e:
            logger.warning(f'stop reasons batch of {len(batch)} texts failed: {e}')
            results.extend([(None, f'stop reasons batch failed: {e}')] * len(batch))
        else:
            results.extend((found, None) for found in labels)
    return results
