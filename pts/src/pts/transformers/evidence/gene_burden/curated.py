"""Manually curated gene burden evidence."""

from __future__ import annotations

import polars as pl
from loguru import logger

from pts.schemas.gene_burden import CURATED_INPUT_COLUMNS
from pts.transformers.evidence.gene_burden.common import DATASOURCE_ID, DATATYPE_ID, spark_distinct, to_union
from pts.transformers.utils.spark_csv import read_spark_csv, spark_csv_double, spark_csv_int

#: Case counts the curation writes as `N.0`.
CASE_COUNT_COLUMNS = ['studyCases', 'studyCasesWithQualifyingVariants']

_ZERO_FRACTION = r'^([+-]?[0-9]+)\.0+$'


def _case_count(column: str) -> pl.Expr:
    """An integer field that may also be written with a zero fraction."""
    field = pl.col(column)
    return pl.coalesce(spark_csv_int(field), spark_csv_int(field.str.extract(_ZERO_FRACTION, 1)))


def _log_case_counts(curated: pl.DataFrame) -> None:
    for column in CASE_COUNT_COLUMNS:
        gained = curated.select((spark_csv_int(pl.col(column)).is_null() & _case_count(column).is_not_null()).sum())
        if count := gained.item():
            logger.info(f'{count} curated evidence have {column} written as N.0, read as N.')


def parse_curated(path: str) -> pl.LazyFrame:
    """Curated evidence from the tab-separated curation file, read by column position.

    Numbers parse as Spark's CSV reader parses them, except the case counts, which may be written
    `N.0`. The p-value mantissa and exponent are taken as written, and the curated disease id is
    carried in `diseaseFromSourceMappedId`.

    Args:
        path: the curation file, with a header line.

    Returns:
        The evidence in `GENE_BURDEN_UNION`.
    """
    curated = read_spark_csv(path, separator='\t', new_columns=CURATED_INPUT_COLUMNS)
    _log_case_counts(curated)
    odds_ratio = spark_csv_double(pl.col('oddsRatio'))
    beta = spark_csv_double(pl.col('beta'))
    lower = spark_csv_double(pl.col('ConfidenceIntervalLower'))
    upper = spark_csv_double(pl.col('ConfidenceIntervalUpper'))
    literature = pl.col('literature')
    return (
        curated
        .select(
            datasourceId=pl.lit(DATASOURCE_ID),
            datatypeId=pl.lit(DATATYPE_ID),
            projectId=pl.col('projectId'),
            targetFromSourceId=pl.col('targetFromSourceId'),
            diseaseFromSource=pl.col('diseaseFromSource'),
            diseaseFromSourceMappedId=pl.col('diseaseFromSourceMappedId'),
            resourceScore=spark_csv_double(pl.col('resourceScore')),
            pValueMantissa=spark_csv_double(pl.col('pValueMantissa')),
            pValueExponent=spark_csv_int(pl.col('pValueExponent')),
            oddsRatio=odds_ratio,
            oddsRatioConfidenceIntervalLower=pl.when(odds_ratio.is_not_null()).then(lower),
            oddsRatioConfidenceIntervalUpper=pl.when(odds_ratio.is_not_null()).then(upper),
            beta=beta,
            betaConfidenceIntervalLower=pl.when(beta.is_not_null()).then(lower),
            betaConfidenceIntervalUpper=pl.when(beta.is_not_null()).then(upper),
            sex=pl.col('sex').str.split(', '),
            ancestry=pl.col('ancestry'),
            ancestryId=pl.col('ancestryId'),
            cohortId=pl.col('cohortId'),
            studySampleSize=spark_csv_int(pl.col('studySampleSize')),
            studyCases=_case_count('studyCases'),
            studyCasesWithQualifyingVariants=_case_count('studyCasesWithQualifyingVariants'),
            allelicRequirements=pl.when(pl.col('allelicRequirements').is_not_null()).then(
                pl.concat_list('allelicRequirements')
            ),
            statisticalMethod=pl.col('statisticalMethod'),
            statisticalMethodOverview=pl.col('statisticalMethodOverview'),
            # spark's array of a null is a list holding null
            literature=pl
            .when(literature.is_not_null())
            .then(pl.concat_list(literature))
            .otherwise(pl.lit([None], dtype=pl.List(pl.String))),
        )
        .pipe(to_union)
        .pipe(spark_distinct)
        .lazy()
    )
