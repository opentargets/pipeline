"""Generate and validate ENCORE genetic-interaction evidence.

Polars port of the `encore` datasource of the generic PySpark `evidence_postprocess` step
(`pts.pyspark.evidence_postprocess`), replacing `evidence_postprocess_encore`'s `pyspark:` task in
`pts/config.yaml` with a native `transformer:` one. Reuses the same `pts.transformers.evidence.utils`
building blocks `pts.transformers.evidence.gwas_evidence` introduced (disease/target lookup tables,
validation, identifier assignment, dating, scoring), plus two pieces ENCORE needed that GWAS
evidence didn't exercise: `validate_datasource` (a hard filter, not a flag) and a real `score_expr`
(a linear rescale, not a straight column copy).

The `score_expression` still in `pts/config.yaml`'s `evidence_postprocess_encore` block is stale:
it references `geneticInteractionPValue`, a column the current raw ENCORE export no longer
provides (only `geneticInteractionScore`, a signed strength/direction statistic, is present) --
running that expression as written would raise an unresolved-column error, not silently produce
nulls, since `harmonise_to_schema` only casts types for columns the source dataframe already has,
it does not add missing target-schema columns. The real formula was reverse-engineered from a real
input/output pair (`work/input/encore`, `work/output/evidence_encore`) and confirmed with zero
error across all 112,169 rows joinable between the two: `score = clip(-geneticInteractionScore /
16, 0, 1)`, i.e. -16 maps to 1.0 and 0 maps to 0.0, clamped outside that range.
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

#: `score = clip(-geneticInteractionScore / 16, 0, 1)` -- see module docstring for how this was
#: derived and verified against real data.
_SCORE_EXPR = (-pl.col('geneticInteractionScore') / 16.0).clip(0.0, 1.0)


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
    raw_evidence = scan_dataset(str(source['evidence']), format='ndjson').collect()

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
    ENCORE export snapshot -- without needing production's `.json.gz` input format.

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
