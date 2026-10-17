"""FinnGen gene burden evidence."""

from __future__ import annotations

import io

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.evidence.gene_burden.common import DATASOURCE_ID, DATATYPE_ID, to_parsed
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_casts import cast_double
from pts.transformers.utils.spark_frames import spark_distinct
from pts.transformers.utils.spark_math import pvalue_exponent, pvalue_mantissa, spark_pow

COHORT_ID = 'FinnGen R12'
STUDY_SAMPLE_SIZE = 500348


def _phenotypes(path: str) -> pl.DataFrame:
    """The phenocode and name of every endpoint in the manifest, a JSON array."""
    data, _ = StorageHandle(path).read()
    manifest = pl.read_json(io.BytesIO(data), infer_schema_length=None)
    return manifest.select(
        PHENO=pl.col('phenocode').cast(pl.String), diseaseFromSource=pl.col('phenostring').cast(pl.String)
    )


def parse_finngen(path: str, phenotypes: str, release: str) -> pl.LazyFrame:
    """FinnGen LoF burden evidence: associations past a Bonferroni correction over the genes tested.

    The p-value is `10 ** -LOG10P`; a LOG10P too large for a double gives a score of 0.0, with no
    exponent or mantissa. The correction counts a missing gene as one more.

    Args:
        path: the burden results, a gzipped TSV with a header.
        phenotypes: the endpoint manifest, a JSON array.
        release: the FinnGen release, as `releaseVersion`.

    Returns:
        The evidence in `GENE_BURDEN_PARSED`.
    """
    results = scan_dataset(path, format='tsv', infer_schema=False, quote_char=None)
    associations = (
        results
        .select('PHENO', 'ID', cast_double('BETA'), cast_double('SE'), cast_double('LOG10P'))
        .join(_phenotypes(phenotypes).lazy(), on='PHENO', how='left', maintain_order='left_right')
        .select(
            targetFromSourceId=pl.col('ID').str.split('.').list.first(),
            diseaseFromSource=pl.col('diseaseFromSource'),
            diseaseFromSourceId=pl.col('PHENO'),
            beta=pl.col('BETA'),
            betaConfidenceIntervalLower=pl.col('BETA') - pl.col('SE'),
            betaConfidenceIntervalUpper=pl.col('BETA') + pl.col('SE'),
            resourceScore=spark_pow(10.0, -pl.col('LOG10P')),
        )
        .collect(engine='streaming')
    )
    genes = associations['targetFromSourceId'].n_unique()
    significant = associations.filter(pl.col('resourceScore') <= 0.05 / genes) if genes else associations
    score = pl.col('resourceScore')
    evidence = (
        significant
        .select(
            datasourceId=pl.lit(DATASOURCE_ID),
            datatypeId=pl.lit(DATATYPE_ID),
            targetFromSourceId=pl.col('targetFromSourceId'),
            diseaseFromSource=pl.col('diseaseFromSource'),
            diseaseFromSourceId=pl.col('diseaseFromSourceId'),
            pValueMantissa=pvalue_mantissa(score),
            pValueExponent=pvalue_exponent(score),
            beta=pl.col('beta'),
            betaConfidenceIntervalLower=pl.col('betaConfidenceIntervalLower'),
            betaConfidenceIntervalUpper=pl.col('betaConfidenceIntervalUpper'),
            resourceScore=score,
            ancestry=pl.lit('finnish'),
            ancestryId=pl.lit('HANCESTRO_0321'),
            literature=pl.lit(['36653562']),
            projectId=pl.lit('FinnGen'),
            cohortId=pl.lit(COHORT_ID),
            releaseVersion=pl.lit(release, dtype=pl.String),
            studySampleSize=pl.lit(STUDY_SAMPLE_SIZE, dtype=pl.Int32),
            statisticalMethod=pl.lit('LoF burden'),
            statisticalMethodOverview=pl.lit('Burden test carried out with LoF variants with MAF smaller than 1%.'),
        )
        .pipe(to_parsed)
        .pipe(spark_distinct)
    )
    underflows = evidence.filter(pl.col('resourceScore') == 0.0).height
    if underflows:
        logger.warning(f'{underflows} FinnGen evidence have a LOG10P too large for a p-value; their score is 0.0.')
    return evidence.lazy()
