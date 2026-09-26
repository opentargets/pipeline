"""Generate and validate ENCORE genetic-interaction evidence.

Polars port of the `encore` datasource of the generic PySpark `evidence_postprocess` step
(`pts.pyspark.evidence_postprocess`), replacing `evidence_encore`'s `pyspark:` task in
`pts/config.yaml` with a native `transformer:` one. Reuses the same `pts.transformers.evidence.utils`
building blocks `pts.transformers.evidence.gwas_evidence` introduced (disease/target lookup tables,
validation, identifier assignment, dating, scoring), plus two pieces ENCORE needed that GWAS
evidence didn't exercise: `validate_datasource` (a hard filter, not a flag) and a real `score_expr`
(a linear rescale, not a straight column copy).

`score_expression: ABS(geneticInteractionScore) / 16.0` here matches the fix landed upstream (see
`pts/config.yaml` at commit 568b9ceb -- https://github.com/opentargets/pipeline/blob/568b9ceb1578062b6282410fa11a63993eaf42f2/pts/config.yaml),
confirmed against real data with zero error across all 112,169 joinable rows between a real
input/output pair (`work/input/evidence/encore`, a backed-up `work/output/evidence_encore_bak`).
The generic PySpark step's earlier `score_expression` (still what's on `main` as of this writing)
referenced `geneticInteractionPValue`, a column the current raw ENCORE export no longer provides
(only `geneticInteractionScore`, a signed strength/direction statistic, is present) -- running that
expression as written would raise an unresolved-column error, not silently produce nulls, since
`harmonise_to_schema` only casts types for columns the source dataframe already has, it does not
add missing target-schema columns. No clamping is applied here, matching
`calculate_evidence_score`'s own semantics: a score outside `[0, 1]` (e.g. a `geneticInteractionScore`
whose magnitude exceeds 16 -- none exist in the local sample, but the field's own description
allows cooperative/positive values a plain negation wouldn't have scored correctly) is flagged
`NO_VALID_SCORE` and excluded from `evidence`, not silently clamped into validity.

The raw ENCORE export format also changed upstream at that same commit: from `input/evidence/encore.json.gz`
(gzipped JSON) to `input/evidence/encore` (parquet) -- matched here by reading it as `scan_dataset`'s
default `parquet` format instead of `ndjson`.
"""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.encore_evidence import EncoreEvidenceSchema
from pts.transformers.evidence.utils import (
    assign_evidence_identifier,
    build_disease_lut,
    build_target_lut,
    calculate_evidence_score,
    resolve_evidence_date,
    validate_datasource,
    validate_diseases,
    validate_target,
    validate_uniqueness,
)
from pts.transformers.utils.dataset import scan_dataset, write_dataset

#: `score = ABS(geneticInteractionScore) / 16.0` -- see module docstring for provenance.
_SCORE_EXPR = pl.col('geneticInteractionScore').abs() / 16.0


def encore_evidence(
    source: dict[str, Path],
    destination: dict[str, Path],
    settings: dict[str, Any],
    _config: Config,
) -> None:
    """Build, validate and write ENCORE genetic-interaction evidence.

    Args:
        source: paths to `evidence` (raw ENCORE export) and `disease`, `target` (PTS outputs, for
            validation).
        destination: `evidence` (valid records) and `failed_evidence` (flagged records) paths.
        settings: `unique_fields` (list[str], the evidence-identifier key columns).
        _config: otter Config object, unused.
    """
    unique_fields = settings['unique_fields']

    logger.info(f'Reading raw ENCORE evidence from {source["evidence"]}')
    raw_evidence = scan_dataset(str(source['evidence'])).collect()

    logger.info(f'Reading disease index from {source["disease"]}')
    disease_lut = build_disease_lut(scan_dataset(str(source['disease'])).select('id', 'obsoleteTerms').collect())
    logger.info(f'Reading target index from {source["target"]}')
    target_lut = build_target_lut(
        scan_dataset(str(source['target'])).select('id', 'biotype', 'proteinIds', 'approvedSymbol').collect()
    )

    logger.info('Validating and dating ENCORE evidence')
    processed = _process_encore_evidence(raw_evidence, disease_lut, target_lut, unique_fields)

    valid = processed.filter(pl.col('qualityControls').list.len() == 0)
    invalid = processed.filter(pl.col('qualityControls').list.len() > 0)

    logger.info(f'Writing {valid.height} valid and {invalid.height} invalid ENCORE evidence records')
    write_dataset(valid, str(destination['evidence']), schema=EncoreEvidenceSchema)
    write_dataset(invalid, str(destination['failed_evidence']))


def _process_encore_evidence(
    raw_evidence: pl.DataFrame,
    disease_lut: pl.DataFrame,
    target_lut: pl.DataFrame,
    unique_fields: list[str],
) -> pl.DataFrame:
    """Validate, identify, date and score raw ENCORE evidence.

    Pure (no I/O), so it can be exercised directly against an already-loaded frame -- e.g. a real
    ENCORE export snapshot -- without needing production's parquet input format.

    Args:
        raw_evidence: raw ENCORE evidence, as read from the source export.
        disease_lut: output of `luts.build_disease_lut`.
        target_lut: output of `luts.build_target_lut`.
        unique_fields: column names that define evidence uniqueness.

    Returns:
        Evidence with `diseaseId`, `targetId`, `id`, `evidenceDate`, `score` and a fully resolved
        `qualityControls`.
    """
    return (
        raw_evidence
        .pipe(validate_datasource, 'encore')
        .pipe(validate_diseases, disease_lut)
        .pipe(validate_target, target_lut)
        .pipe(assign_evidence_identifier, unique_fields)
        .pipe(validate_uniqueness)
        .pipe(resolve_evidence_date)
        .pipe(calculate_evidence_score, _SCORE_EXPR)
    )
