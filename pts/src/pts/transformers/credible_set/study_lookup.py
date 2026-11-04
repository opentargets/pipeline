"""Study lookups shared by the credible set annotations."""

import polars as pl

from pts.schemas.study import StudyType


def study_of_set(sets: pl.LazyFrame) -> pl.LazyFrame:
    """The `studyId` of each credible set, keyed by `credibleSetId` for a join to its variants."""
    return sets.select(pl.col('id').alias('credibleSetId'), 'studyId')


def qtl_genes(studies: pl.LazyFrame) -> pl.LazyFrame:
    """The `geneId` of each QTL study that has one."""
    qtl = (pl.col('studyType') != StudyType.GWAS.value) & pl.col('geneId').is_not_null()
    return studies.filter(qtl).select('studyId', 'geneId')
