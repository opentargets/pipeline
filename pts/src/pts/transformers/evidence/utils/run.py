"""Write post-processed evidence, and run a datasource end to end."""

from __future__ import annotations

import polars as pl
from loguru import logger

from pts.schemas.dataset import DatasetModel
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.evidence import Evidence
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.utils.dataset import check_destinations_absent, sink_dataset, validate_dataset


def write_evidence(
    evidence: Evidence,
    destination: dict[str, str],
    *,
    alongside: list[pl.LazyFrame] | None = None,
    schema: type[DatasetModel] | None = None,
) -> None:
    """Write evidence split into `evidence` (valid) and `failed_evidence`, computing it once.

    `alongside` are further sinks planned and collected with them, so a refused destination stops all writes.

    With a `schema`, the evidence is collected and the valid part validated against it before anything is
    written, so a violation writes neither output.

    Raises:
        pandera.errors.SchemaErrors: if `schema` is given and the valid evidence violates it.
    """
    valid = evidence.valid()
    if schema is not None:
        # checked first, so an occupied destination fails before the evidence is computed
        check_destinations_absent([destination['evidence'], destination['failed_evidence']])
        evidence = Evidence(evidence.lf.collect().lazy())
        valid = validate_dataset(evidence.valid(), schema).lazy()
    logger.info(f'writing valid evidence to {destination["evidence"]}')
    logger.info(f'writing failed evidence to {destination["failed_evidence"]}')
    pl.collect_all([
        sink_dataset(valid, destination['evidence']),
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
    literature: str | None = None,
) -> None:
    """Check, post-process and write raw evidence for a datasource with no steps after the recipe."""
    luts = build_luts(raw, disease=disease, target=target, literature=literature)
    write_evidence(datasource.postprocess(raw, luts), destination, schema=datasource.schema)
