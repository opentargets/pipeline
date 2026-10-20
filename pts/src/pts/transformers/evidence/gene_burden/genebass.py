"""Genebass gene burden evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.gene_burden.common import DATASOURCE_ID, DATATYPE_ID, to_parsed
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_frames import spark_distinct
from pts.transformers.utils.spark_math import pvalue_exponent, pvalue_mantissa, substitute_zero_pvalues

P_VALUE_CUTOFF = 6.7e-7

METHOD_DESCRIPTIONS = {
    'pLoF': 'Burden test carried out with rare pLOF variants.',
    'missense|LC': (
        'Burden test carried out with rare missense variants including low-confidence pLOF and in-frame indels.'
    ),
    'synonymous': 'Burden test carried out with rare synonymous variants.',
    'pLoF|missense|LC': 'Burden test carried out with pLOF or missense variants.',
}

_COLUMNS = [
    'gene_id',
    'annotation',
    'n_cases',
    'n_controls',
    'trait_type',
    'phenocode',
    'description',
    'Pvalue_Burden',
    'BETA_Burden',
    'SE_Burden',
]


def parse_genebass(path: str) -> pl.LazyFrame:
    """Genebass evidence: non-categorical trait associations with a burden p ≤ 6.7e-7.

    A p-value of 0.0 takes the smallest one above it in the whole input, categorical traits
    included. Binary and quantitative effects are both reported as beta ± SE.

    Args:
        path: the Genebass results, a directory of parquet files.

    Returns:
        The evidence in `GENE_BURDEN_PARSED`.
    """
    associations = scan_dataset(path).select(_COLUMNS)
    pvalue = pl.col('Pvalue_Burden')
    candidates = associations.filter((pvalue <= P_VALUE_CUTOFF) | (pvalue == 0.0))
    significant = (
        substitute_zero_pvalues(candidates, 'Pvalue_Burden', label='Genebass', scope=associations)
        .filter(pvalue <= P_VALUE_CUTOFF)
        .filter(pl.col('trait_type') != 'categorical')
        .collect(engine='streaming')
        .pipe(spark_distinct)
    )
    beta = pl.col('BETA_Burden')
    return (
        significant
        .select(
            datasourceId=pl.lit(DATASOURCE_ID),
            datatypeId=pl.lit(DATATYPE_ID),
            targetFromSourceId=pl.col('gene_id'),
            diseaseFromSource=pl.col('description'),
            diseaseFromSourceId=pl.col('phenocode'),
            pValueMantissa=pvalue_mantissa(pvalue),
            pValueExponent=pvalue_exponent(pvalue),
            beta=beta,
            betaConfidenceIntervalLower=beta - pl.col('SE_Burden'),
            betaConfidenceIntervalUpper=beta + pl.col('SE_Burden'),
            resourceScore=pvalue,
            ancestry=pl.lit('EUR'),
            ancestryId=pl.lit('HANCESTRO_0005'),
            projectId=pl.lit('Genebass'),
            cohortId=pl.lit('UK Biobank 450k'),
            studySampleSize=pl.col('n_cases') + pl.col('n_controls').fill_null(0),
            studyCases=pl.col('n_cases'),
            statisticalMethod=pl.col('annotation'),
            statisticalMethodOverview=pl.col('annotation').replace(METHOD_DESCRIPTIONS),
            literature=pl.lit(['36778668']),
        )
        .pipe(to_parsed)
        .pipe(spark_distinct)
        .lazy()
    )
