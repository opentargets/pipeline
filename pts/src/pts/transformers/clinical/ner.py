"""Disease and drug entity extraction with HuggingFace NER pipelines.

Ported from OnToma's ``ontoma.ner`` (Open Targets), without Spark.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

DISEASE_MODEL = 'alvaroalon2/biobert_diseases_ner'
DISEASE_NER_MODEL = DISEASE_MODEL
CHEMICAL_MODEL = 'alvaroalon2/biobert_chemical_ner'
DRUGTEMIST_MODEL = 'BSC-NLP4BIA/bsc-bio-ehr-es-carmen-drugtemist'

DRUG_NER_MODEL = f'{CHEMICAL_MODEL}+{DRUGTEMIST_MODEL}+regex'

DISEASE_LABELS = ['DISEASE']
DISEASE_MIN_LENGTH = 2
DISEASE_DROP_DIGITS = True
BIOBERT_LABELS = ['CHEMICAL', 'DRUG', 'MEDICATION', 'CHEM']
DRUGTEMIST_LABELS = ['CHEMICAL', 'DRUG', 'MEDICATION', 'CHEM', 'FARMACO', 'NORMALIZABLES']

DRUG_NAME_PATTERNS = [
    r'\w*mab\b',
    r'\w*cept\b',
    r'\w*tinib\b',
    r'\w*nib\b',
    r'\w*ciclib\b',
    r'\w*tidine\b',
    r'\w*prazole\b',
    r'\w*statin\b',
    r'\w*olol\b',
    r'\w*pril\b',
    r'\w*sartan\b',
    r'\w*floxacin\b',
    r'\w*cillin\b',
    r'\w*mycin\b',
    r'\w*cycline\b',
]


def _create_ner_pipeline(
    model_name: str,
    tokenizer_name: str | None = None,
    aggregation_strategy: str = 'max',
) -> Callable[..., Any]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    device: str | int = 'mps' if torch.backends.mps.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    if device == 'mps':
        model = model.to('mps')
    ner = pipeline(
        'token-classification',
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=aggregation_strategy,
        device=device,
        ignore_labels=[],
    )

    def run(texts: Any) -> Any:
        with torch.no_grad():
            return ner(texts)

    return run


def _disease_mentions(biobert: Callable[..., Any], text: str) -> list[str]:
    found: set[str] = set()
    for ent in biobert(text):
        label = ent.get('entity_group', '').upper()
        if any(wanted in label for wanted in DISEASE_LABELS):
            word = ent['word'].strip()
            if len(word) >= DISEASE_MIN_LENGTH and not (DISEASE_DROP_DIGITS and word.isdigit()):
                found.add(word.lower())
    return sorted(found)


def disease_entities_with_status(texts: list[str]) -> list[tuple[list[str] | None, str | None]]:
    """Lower-cased, sorted disease mentions per text, or ``(None, error)`` where one failed; blank texts get ``[]``."""
    if not any(text.strip() for text in texts):
        return [([], None) for _ in texts]
    biobert = _create_ner_pipeline(DISEASE_MODEL, DISEASE_MODEL, 'max')
    results: list[tuple[list[str] | None, str | None]] = []
    for text in texts:
        if not text.strip():
            results.append(([], None))
            continue
        try:
            results.append((_disease_mentions(biobert, text), None))
        except Exception as e:
            logger.debug(f'disease ner failed for {text!r}: {e}')
            results.append((None, f'disease ner failed: {e}'))
    return results


def _extract_with_regex(text: str, patterns: list[str]) -> set[str]:
    drugs: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            clean = match.strip().lower()
            if len(clean) > 3 and not clean[0].isdigit():
                drugs.add(clean)
    return drugs


def _extract_entities_from_batch(
    texts: list[str], ner_pipeline: Callable[..., Any], label_filters: list[str]
) -> tuple[list[set[str]], str | None]:
    """Entities per text, and the error if the batch failed (its texts then have no entities)."""
    if not texts:
        return [], None
    try:
        batch_entities = ner_pipeline(texts)
        if not isinstance(batch_entities[0], list):
            batch_entities = [batch_entities]
        if len(batch_entities) != len(texts):
            raise ValueError(f'expected {len(texts)} results, got {len(batch_entities)}')
        results = []
        for entities in batch_entities:
            drugs: set[str] = set()
            for ent in entities:
                label = ent.get('entity_group', '').upper()
                if any(wanted in label for wanted in label_filters):
                    word = ent['word'].strip().replace(' ##', '').replace('##', '').strip()
                    if len(word) > 1 and not word.isdigit():
                        drugs.add(word.lower())
            results.append(drugs)
    except Exception as e:
        logger.debug(f'ner batch failed: {e}')
        return [set() for _ in texts], str(e)
    return results, None


def _process_batch_tiered(
    texts: list[str],
    biobert_pipeline: Callable[..., Any],
    drugtemist_pipeline: Callable[..., Any],
    batch_size: int,
) -> tuple[list[list[str]], list[str | None]]:
    """Drug names per text, and the error of each text whose model batch failed (else None)."""
    all_results: list[list[str] | None] = []
    errors: list[str | None] = [None] * len(texts)
    fallback_texts: list[str] = []
    fallback_indices: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        regex_results = [_extract_with_regex(text, DRUG_NAME_PATTERNS) if text else set() for text in batch]
        biobert_results, error = _extract_entities_from_batch(batch, biobert_pipeline, BIOBERT_LABELS)
        if error is not None:
            errors[start : start + len(batch)] = [error] * len(batch)
        for offset, (regex_drugs, biobert_drugs, text) in enumerate(
            zip(regex_results, biobert_results, batch, strict=True)
        ):
            combined = regex_drugs | biobert_drugs
            if not combined and text and text.strip():
                fallback_texts.append(text)
                fallback_indices.append(start + offset)
                all_results.append(None)
            else:
                all_results.append(sorted(combined))
        logger.debug(f'processed {min(start + batch_size, len(texts))}/{len(texts)} with BioBERT')

    if fallback_texts:
        logger.info(f'running DrugTEMIST on {len(fallback_texts)} texts that BioBERT missed')
        for start in range(0, len(fallback_texts), batch_size):
            batch = fallback_texts[start : start + batch_size]
            indices = fallback_indices[start : start + batch_size]
            drugtemist_results, error = _extract_entities_from_batch(batch, drugtemist_pipeline, DRUGTEMIST_LABELS)
            if error is not None:
                for index in indices:
                    errors[index] = errors[index] or error
            for index, drugs in zip(indices, drugtemist_results, strict=True):
                all_results[index] = sorted(drugs)

    results = [r if r is not None else [] for r in all_results]
    return results, errors


def drug_entities_with_status(texts: list[str], batch_size: int) -> list[tuple[list[str] | None, str | None]]:
    """Sorted drug names per text, or ``(None, error)`` where a model batch failed.

    Regex and BioBERT hits are unioned; DrugTEMIST runs only on texts where both found nothing.
    """
    if not texts:
        return []
    biobert = _create_ner_pipeline(CHEMICAL_MODEL, None, 'max')
    drugtemist = _create_ner_pipeline(DRUGTEMIST_MODEL, None, 'max')
    extracted, errors = _process_batch_tiered(texts, biobert, drugtemist, batch_size)
    return [
        (None, f'ner batch failed: {error}') if error is not None else (drugs, None)
        for drugs, error in zip(extracted, errors, strict=True)
    ]
