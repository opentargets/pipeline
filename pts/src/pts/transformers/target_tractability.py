"""Target tractability dataset generation.

Processes the Open Targets tractability TSV and produces one row per
tractability bucket assessment (flat schema) across small molecule,
antibody, and other modalities. Only targets present in output/target are
retained.
"""

from __future__ import annotations

import re
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.target_tractability import tractability_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset

BUCKET_COLUMN = re.compile(r'.*_B\d+_.*')
IS_ONE = r'^[\x00-\x20\x7f]*\+?0*1(\.[0-9]*)?[\x00-\x20\x7f]*$'


def build_tractability(raw: pl.DataFrame, target_ids: pl.DataFrame) -> pl.DataFrame:
    """Build tractability assessments from the tractability TSV.

    Columns matching the pattern ``*_B{N}_*`` are tractability bucket columns.
    Rows whose ``ensembl_gene_id`` is not present in the target universe are
    dropped.

    Args:
        raw: Raw tractability TSV, every column read as a string.
        target_ids: Frame with a single ``id`` column from output/target.

    Returns:
        One row per tractability assessment with columns ``targetId``,
        ``modality``, ``category`` and ``value``.
    """
    bucket_cols = [c for c in raw.columns if BUCKET_COLUMN.match(c)]
    if not bucket_cols:
        return pl.DataFrame(schema=tractability_schema)

    labels = pl.DataFrame({
        'column': bucket_cols,
        'position': range(len(bucket_cols)),
        'modality': [c.split('_')[0] for c in bucket_cols],
        'category': [c.split('_')[-1] for c in bucket_cols],
    })
    return (
        raw
        .select(pl.col('ensembl_gene_id').alias('targetId'), *bucket_cols)
        .with_row_index('row')
        .join(target_ids.select(pl.col('id').alias('targetId')), on='targetId', how='inner')
        .unpivot(bucket_cols, index=['targetId', 'row'], variable_name='column', value_name='cell')
        .join(labels, on='column', how='left')
        .sort('targetId', 'row', 'position')
        .select(
            'targetId',
            'modality',
            'category',
            pl.col('cell').cast(pl.String).str.contains(IS_ONE).fill_null(False).alias('value'),
        )
    )


def target_tractability(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build tractability assessments and write the output dataset.

    Args:
        source: Mapping with keys ``tractability`` (TSV) and ``target``
            (output/target parquet used for ID validation).
        destination: Output path for the tractability parquet.
        settings: Unused.
        config: Unused.
    """
    logger.info('reading tractability TSV')
    raw = scan_dataset(source['tractability'], format='tsv', infer_schema=False).collect()

    logger.info('reading target ids')
    target_ids = scan_dataset(source['target']).select('id').collect()

    logger.info('building tractability assessments')
    result = build_tractability(raw, target_ids)

    logger.info(f'writing {result.height} tractability assessments to {destination}')
    write_dataset(result, destination)
