"""Run one datasource's evidence through the post-processing recipe and write it.

A datasource module generates its raw evidence as a `LazyFrame`, declares its settings as an
`EvidencePostprocessor`, and hands both to `run_evidence`. A datasource that adds steps after the
recipe, such as directions of effect, composes `build_luts`, `spec.run` and `write_evidence`
itself. Nothing is written between generation and post-processing.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence.core import Evidence
from pts.transformers.evidence.postprocess import EvidencePostprocessor, ValidationLuts
from pts.transformers.utils.dataset import write_dataset
from pts.transformers.utils.validation_lut import build_disease_lut, build_publication_lut, build_target_lut

#: Columns every raw evidence frame must carry.
REQUIRED_COLUMNS = ['datasourceId', 'targetFromSourceId', 'diseaseFromSourceMappedId']

#: Columns the recipe adds, so a datasource may name them in `unique_fields` without producing them.
DERIVED_COLUMNS = {'targetId', 'diseaseId'}


def check_raw_evidence(lf: pl.LazyFrame, spec: EvidencePostprocessor) -> None:
    """Check a raw evidence frame's columns before post-processing it.

    Columns outside `evidence_schema` pass through uncast, with a warning.

    Args:
        lf: raw evidence.
        spec: the datasource's post-processing settings.

    Raises:
        ValueError: if a required column or a `unique_fields` column is missing.
    """
    names = lf.collect_schema().names()
    missing = [c for c in REQUIRED_COLUMNS if c not in names]
    missing += [c for c in dict.fromkeys(spec.unique_fields) if c not in names and c not in DERIVED_COLUMNS]
    if missing:
        msg = f'{spec.datasource_id} raw evidence is missing {missing}'
        raise ValueError(msg)
    unknown = [c for c in names if c not in evidence_schema]
    if unknown:
        logger.warning(f'{spec.datasource_id} raw evidence has columns outside evidence_schema: {unknown}')


def build_luts(raw: pl.LazyFrame, *, disease: str, target: str, literature: str | None = None) -> ValidationLuts:
    """Build the look-up tables for one datasource's raw evidence.

    The publication table is built only when the evidence carries `literature`.

    Args:
        raw: raw evidence.
        disease: location of the disease index.
        target: location of the target index.
        literature: location of the literature export, needed when `raw` carries `literature`.

    Returns:
        The look-up tables.

    Raises:
        ValueError: if `raw` carries `literature` and no `literature` location is given.
    """
    has_literature = 'literature' in raw.collect_schema().names()
    if has_literature and literature is None:
        msg = 'evidence carries literature but no literature export was given'
        raise ValueError(msg)
    return ValidationLuts(
        disease=build_disease_lut(disease),
        target=build_target_lut(target),
        publication=build_publication_lut(literature) if has_literature and literature else None,
    )


def write_evidence(evidence: Evidence, destination: dict[str, str]) -> None:
    """Write post-processed evidence, split into valid and failed.

    Args:
        evidence: post-processed evidence.
        destination: `evidence` and `failed_evidence` output directories.
    """
    logger.info(f'writing valid evidence to {destination["evidence"]}')
    write_dataset(evidence.valid(), destination['evidence'])
    logger.info(f'writing failed evidence to {destination["failed_evidence"]}')
    write_dataset(evidence.invalid(), destination['failed_evidence'])


def run_evidence(
    raw: pl.LazyFrame,
    spec: EvidencePostprocessor,
    *,
    disease: str,
    target: str,
    destination: dict[str, str],
    literature: str | None = None,
) -> None:
    """Check, post-process and write one datasource's raw evidence.

    For datasources with nothing to add after the recipe. Those that do compose `build_luts`,
    `spec.run` and `write_evidence` themselves.

    Args:
        raw: raw evidence.
        spec: the datasource's post-processing settings.
        disease: location of the disease index.
        target: location of the target index.
        destination: `evidence` and `failed_evidence` output directories.
        literature: location of the literature export, needed when `raw` carries `literature`.
    """
    check_raw_evidence(raw, spec)
    luts = build_luts(raw, disease=disease, target=target, literature=literature)
    write_evidence(spec.run(raw, luts), destination)
