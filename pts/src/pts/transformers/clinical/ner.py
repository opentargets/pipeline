"""Disease and drug entity extraction with HuggingFace NER pipelines.

Ported from OnToma's ``ontoma.ner`` (Open Targets), without Spark.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

DISEASE_MODEL = 'alvaroalon2/biobert_diseases_ner'
CHEMICAL_MODEL = 'alvaroalon2/biobert_chemical_ner'
DRUGTEMIST_MODEL = 'BSC-NLP4BIA/bsc-bio-ehr-es-carmen-drugtemist'

DISEASE_LABELS = ['DISEASE']
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


def extract_disease_entities(texts: pl.Series) -> pl.Series:
    """Extract lower-cased, sorted disease mentions from each text; runs the model once per distinct text."""
    distinct = [t for t in texts.drop_nulls().unique().sort().to_list() if t.strip()]
    results: dict[str, list[str]] = {}
    failures = 0
    if distinct:
        logger.info(f'extracting disease entities from {len(distinct)} distinct texts')
        biobert = _create_ner_pipeline(DISEASE_MODEL, DISEASE_MODEL, 'max')
        for text in distinct:
            found: set[str] = set()
            try:
                for ent in biobert(text):
                    label = ent.get('entity_group', '').upper()
                    if any(wanted in label for wanted in DISEASE_LABELS):
                        word = ent['word'].strip()
                        if len(word) > 1 and not word.isdigit():
                            found.add(word.lower())
            except Exception as e:
                failures += 1
                logger.debug(f'disease ner failed for {text!r}: {e}')
                found = set()
            results[text] = sorted(found)
    if failures:
        logger.warning(f'disease ner failed for {failures} of {len(distinct)} distinct texts; their results are empty')
    return pl.Series(
        texts.name,
        [results.get(t, []) if t is not None else [] for t in texts.to_list()],
        dtype=pl.List(pl.String),
    )


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
) -> tuple[list[set[str]], int]:
    """Entities per text, and the number of texts in a batch that failed."""
    if not texts:
        return [], 0
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
        return [set() for _ in texts], len(texts)
    return results, 0


def _process_batch_tiered(
    texts: list[str],
    biobert_pipeline: Callable[..., Any],
    drugtemist_pipeline: Callable[..., Any],
    batch_size: int,
) -> tuple[list[list[str]], int]:
    all_results: list[list[str] | None] = []
    fallback_texts: list[str] = []
    fallback_indices: list[int] = []
    failures = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        regex_results = [_extract_with_regex(text, DRUG_NAME_PATTERNS) if text else set() for text in batch]
        biobert_results, failed = _extract_entities_from_batch(batch, biobert_pipeline, BIOBERT_LABELS)
        failures += failed
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
            drugtemist_results, failed = _extract_entities_from_batch(batch, drugtemist_pipeline, DRUGTEMIST_LABELS)
            failures += failed
            for index, drugs in zip(indices, drugtemist_results, strict=True):
                all_results[index] = sorted(drugs)

    return [r if r is not None else [] for r in all_results], failures


def extract_drug_entities(labels: pl.Series, batch_size: int) -> pl.Series:
    """Extract sorted drug names from each label: regex and BioBERT, then DrugTEMIST where both found nothing."""
    distinct = labels.drop_nulls().unique().sort().to_list()
    results: dict[str, list[str]] = {}
    if distinct:
        logger.info(f'extracting drug entities from {len(distinct)} distinct texts (batch_size={batch_size})')
        biobert = _create_ner_pipeline(CHEMICAL_MODEL, None, 'max')
        drugtemist = _create_ner_pipeline(DRUGTEMIST_MODEL, None, 'max')
        extracted, failures = _process_batch_tiered(distinct, biobert, drugtemist, batch_size)
        results = dict(zip(distinct, extracted, strict=True))
        if failures:
            logger.warning(f'drug ner failed for {failures} texts; their results are empty')
    return pl.Series(
        labels.name,
        [results.get(t, []) if t is not None else [] for t in labels.to_list()],
        dtype=pl.List(pl.String),
    )
