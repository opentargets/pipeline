"""Ingest curated loss-of-function assessments from OTAR2075."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict

from pts.schemas.variant_annotation import variant_annotation_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.variant_identifier import hash_long_variant_ids

_DESCRIPTIONS = {
    'lof': 'Assessed to cause LoF',
    'likely_lof': 'Suspected to cause LoF',
    'uncertain': 'Uncertain LoF assessment',
    'likely_not_lof': 'Suspected not to cause LoF',
    'not_lof': 'Assessed not to cause LoF',
}


class LofCurationSettings(BaseModel):
    """LoF ingestion has no configurable transformation options."""

    model_config = ConfigDict(extra='forbid')


def normalise_lof_score(assessment: pl.Expr) -> pl.Expr:
    """Map the five curated verdicts to ordinal scores; unknowns remain null."""
    return assessment.replace_strict(
        {'lof': 1.0, 'likely_lof': 0.5, 'uncertain': 0.0, 'likely_not_lof': -0.5, 'not_lof': -1.0},
        default=None,
        return_dtype=pl.Float64,
    )


def _validated_variants(frame: pl.LazyFrame, reference: pl.LazyFrame) -> pl.DataFrame:
    """Keep GRCh38 IDs whose entire REF span matches the base-level reference.

    Only the small curation table and requested reference bases are materialized.
    The genome table is scanned with a streaming semi-join against requested
    positions. A left join then ensures absent bases fail rather than shortening
    a deletion's comparison. Row identities preserve duplicate curation records.
    """
    raw = frame.select('Variant ID GRCh38', pl.col('Verdict').alias('_verdict')).collect()
    valid = (
        raw
        .with_row_index('_row')
        .filter(pl.col('Variant ID GRCh38').str.contains(r'^[^-\s]+-[1-9]\d*-[ACGTN]+-[ACGTN]+$'))
        .with_columns(pl.col('Variant ID GRCh38').str.split_exact('-', 3).alias('_h38'))
        .select(
            '_row',
            '_verdict',
            pl.col('_h38').struct.field('field_0').str.strip_prefix('chr').replace('M', 'MT').alias('chromosome'),
            pl.col('_h38').struct.field('field_1').cast(pl.Int32, strict=False).alias('position'),
            pl.col('_h38').struct.field('field_2').alias('referenceAllele'),
            pl.col('_h38').struct.field('field_3').alias('alternateAllele'),
        )
        .filter(
            pl.col('position').is_not_null(),
            pl.col('chromosome') != '',
            pl.col('referenceAllele') != pl.col('alternateAllele'),
        )
    )
    if valid.is_empty():
        logger.info(
            'LoF reference validation: input={}, kept=0, invalid_or_missing_id={}, ref_failed=0', raw.height, raw.height
        )
        return valid

    expected = valid.select(
        '_row',
        pl.col('chromosome').alias('chr'),
        pl.int_ranges(
            pl.col('position').cast(pl.Int64),
            pl.col('position').cast(pl.Int64) + pl.col('referenceAllele').str.len_bytes(),
        ).alias('pos'),
        pl.col('referenceAllele').str.split('').alias('_expected'),
    ).explode('pos', '_expected')
    keys = expected.select('chr', 'pos').unique()
    bases = (
        reference
        .select('chr', pl.col('pos').cast(pl.Int64), pl.col('ref').alias('_reference'))
        .filter(
            pl.col('chr').is_in(keys['chr'].unique().implode()),
            pl.col('pos').is_between(keys['pos'].min(), keys['pos'].max()),
        )
        .join(keys.lazy(), on=['chr', 'pos'], how='semi')
        .collect(engine='streaming')
    )
    matching_rows = (
        expected
        .join(bases, on=['chr', 'pos'], how='left', validate='m:1')
        .group_by('_row')
        .agg((pl.col('_expected') == pl.col('_reference')).fill_null(False).all().alias('_matches'))
        .filter('_matches')
    )
    result = valid.join(matching_rows.select('_row'), on='_row', how='semi', maintain_order='left')
    logger.info(
        'LoF reference validation: input={}, kept={}, invalid_or_missing_id={}, ref_failed={}',
        raw.height,
        result.height,
        raw.height - valid.height,
        valid.height - result.height,
    )
    return result


def transform_lof_curation(frame: pl.LazyFrame, reference: pl.LazyFrame) -> pl.LazyFrame:
    """Use all GRCh38 ID components and drop records without a full REF match.

    Coordinates are 1-based. Validate every REF base, including deletion spans,
    against the GRCh38 chr/pos/ref dataset before constructing output IDs.
    Missing/malformed IDs, missing contigs/positions, and REF mismatches are
    dropped. Matching duplicates and their individual verdicts are preserved.

    REF agreement establishes sequence compatibility, not ALT identity or the
    correctness of the original liftover. Distinct source variants may share a
    GRCh38 ID; their curation rows remain separate.
    """
    parsed = _validated_variants(frame, reference).lazy()
    verdict = pl.col('_verdict')
    return (
        parsed
        .select(
            pl.concat_str('chromosome', 'position', 'referenceAllele', 'alternateAllele', separator='_').alias(
                'variantId'
            ),
            'chromosome',
            'position',
            'referenceAllele',
            'alternateAllele',
            pl.concat_list(
                pl.struct(
                    pl.lit('LossOfFunctionCuration').alias('method'),
                    verdict.alias('assessment'),
                    pl.lit(None, dtype=pl.Float32).alias('score'),
                    pl.lit(None, dtype=pl.String).alias('assessmentFlag'),
                    pl.lit(None, dtype=pl.String).alias('targetId'),
                    normalise_lof_score(verdict).alias('normalisedScore'),
                )
            ).alias('variantEffect'),
            pl.concat_str(
                verdict.replace_strict(_DESCRIPTIONS, default=None),
                pl.lit(' by OTAR2075 variant curation effort.'),
            ).alias('variantDescription'),
        )
        .select(variant_annotation_schema.names())
        .pipe(hash_long_variant_ids)
    )


def lof_curation(source: dict[str, str], destination: str, settings: dict[str, Any], config: Config) -> None:
    """Read the multiline curation CSV and write variant annotation Parquet.

    Args:
        source: Paths keyed by ``curation`` (CSV) and ``reference`` (GRCh38 Parquet).
        destination: Output annotation dataset directory.
        settings: No settings are currently supported.
        config: Pipeline runtime configuration.
    """
    LofCurationSettings.model_validate(settings)
    result = transform_lof_curation(
        scan_dataset(source['curation'], format='csv', infer_schema=False, null_values=''),
        scan_dataset(source['reference']),
    )
    logger.info('Writing LoF curation annotations to {}', destination)
    write_dataset(result, destination)
