"""Write post-processed evidence, and run a datasource end to end."""

from __future__ import annotations

import polars as pl
from loguru import logger
from ot_datamodels.base import DatasetModel

from pts.schemas.dataset.conform import conform
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.evidence import Evidence
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.utils.dataset import sink_dataset


def write_evidence(
    evidence: Evidence,
    destination: dict[str, str],
    *,
    model: type[DatasetModel],
    alongside: list[pl.LazyFrame] | None = None,
) -> None:
    """Write evidence split into `evidence` (valid, conformed to `model`) and `failed_evidence`, computing it once.

    `alongside` are further sinks planned and collected with them, so a refused destination stops all writes.
    """
    logger.info(f'writing valid evidence to {destination["evidence"]}')
    logger.info(f'writing failed evidence to {destination["failed_evidence"]}')
    pl.collect_all([
        sink_dataset(conform(evidence.valid(), model), destination['evidence']),
        sink_dataset(evidence.invalid(), destination['failed_evidence']),
        *(alongside or []),
    ])


def run_evidence(
    raw: pl.LazyFrame,
    datasource: EvidenceDatasource,
    *,
    disease: str,
    target: str,
    destination: dict[str, str],
    model: type[DatasetModel],
    literature: str | None = None,
) -> None:
    """Check, post-process and write raw evidence for a datasource with no steps after the recipe."""
    luts = build_luts(raw, disease=disease, target=target, literature=literature)
    write_evidence(datasource.postprocess(raw, luts), destination, model=model)
