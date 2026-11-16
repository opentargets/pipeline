"""What the credible set stages read: a staged source, and the study of a credible set.

A staged source is one source's credible sets or credible set variants, as PIS stages them.
`read_source_dataset` reads one under the contract every stage relies on: the stage's schema,
nothing null that keys a row, and quality flags as lists. A source that breaks it stops the
stage before any check runs, rather than having its rows flagged for what is a staging bug.

The annotation rules join a credible set to its study, and a QTL study to its gene:
`study_of_set` and `qtl_genes` are those two joins, so every rule makes them the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pts.schemas.study import StudyType
from pts.transformers.utils.dataset import scan_dataset


def read_source_dataset(path: str, schema: pl.Schema, required: Sequence[str]) -> pl.LazyFrame:
    """Read one source's dataset in `schema`, with no null in the `required` columns and `qualityControls` a list.

    A column of `schema` absent from the dataset raises in `match_to_schema`. A null in a `required`
    column means the source breaks its contract, so the stage stops before any check runs; a null
    `qualityControls` becomes an empty list, so every check can rely on it.

    Raises:
        ValueError: if a `required` column has nulls, naming each such column and its count.
    """
    frame = (
        scan_dataset(path)
        .match_to_schema(schema)
        .with_columns(pl.col('qualityControls').fill_null(pl.lit([], dtype=schema['qualityControls'])))
    )
    null_count = frame.select(pl.col(*required).null_count()).collect().row(0, named=True)
    if missing := {column: count for column, count in null_count.items() if count}:
        msg = f'{path}: required columns have nulls: {missing}'
        raise ValueError(msg)
    return frame


def study_of_set(credible_set: pl.LazyFrame) -> pl.LazyFrame:
    """The `studyId` of each credible set, keyed by `credibleSetId` for a join to its variants."""
    return credible_set.select(pl.col('id').alias('credibleSetId'), 'studyId')


def qtl_genes(study: pl.LazyFrame) -> pl.LazyFrame:
    """The `geneId` of each QTL study that has one."""
    qtl = (pl.col('studyType') != StudyType.GWAS.value) & pl.col('geneId').is_not_null()
    return study.filter(qtl).select('studyId', 'geneId')
