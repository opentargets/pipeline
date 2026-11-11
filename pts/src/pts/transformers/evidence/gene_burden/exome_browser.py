"""Shared reading of the Broad exome results browsers' gene results (SCHEMA, BipEx)."""

from __future__ import annotations

import gzip
from dataclasses import dataclass

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.evidence.gene_burden.common import DATASOURCE_ID, DATATYPE_ID, to_parsed
from pts.transformers.utils.spark_math import pvalue_exponent, pvalue_mantissa

#: The tests reported, in the order a tie is broken.
TESTS = ['ptv', 'ptv_mis']

METHOD_DESCRIPTIONS = {
    'ptv': 'Case-control burden test carried out with rare protein-truncating variants.',
    'ptv_mis': 'Case-control burden test carried out with rare protein-truncating and damaging missense variants.',
}

#: One test's statistics: p-value, odds ratio and its interval bounds.
TEST_COLUMNS = ['pValue', 'oddsRatio', 'oddsRatioConfidenceIntervalLower', 'oddsRatioConfidenceIntervalUpper']


@dataclass(frozen=True)
class Study:
    """What a browser's evidence carries besides the gene results."""

    project_id: str
    cohort_id: str
    disease: str
    disease_id: str
    browser: str
    literature: list[str] | None = None


def read_gene_results(path: str) -> pl.DataFrame:
    """The meta-analysis rows of a browser's `gene_results.tsv.bgz`, every column a string, `NA` null."""
    data, _ = StorageHandle(path).read()
    results = pl.read_csv(gzip.decompress(data), separator='\t', infer_schema=False, null_values='NA', quote_char=None)
    return results.filter(pl.col('group') == 'meta')


def most_significant(results: pl.DataFrame, tests: dict[str, list[pl.Expr]]) -> pl.DataFrame:
    """Each gene's most significant test, ties to the earlier test in `TESTS`.

    Args:
        results: the gene results.
        tests: per test in `TESTS`, its `TEST_COLUMNS` and case carriers, as expressions over `results`.

    Returns:
        `gene_id`, `n_cases`, `n_controls`, `statisticalMethod`, the `TEST_COLUMNS` and
        `studyCasesWithQualifyingVariants`, as doubles and integers; genes without a p-value are dropped.
    """
    names = [*TEST_COLUMNS, 'studyCasesWithQualifyingVariants']
    per_test = pl.concat([
        results.select(
            'gene_id',
            pl.col('n_cases').cast(pl.Int32),
            pl.col('n_controls').cast(pl.Int32),
            statisticalMethod=pl.lit(test),
            rank=pl.lit(rank),
            **dict(zip(names, tests[test], strict=True)),
        )
        for rank, test in enumerate(TESTS)
    ])
    return (
        per_test
        .with_columns(pl.col(TEST_COLUMNS).cast(pl.Float64), pl.col('studyCasesWithQualifyingVariants').cast(pl.Int32))
        .filter(pl.col('pValue').is_not_null() & ~pl.col('pValue').is_nan())
        .sort('gene_id', 'pValue', 'rank')
        .unique('gene_id', keep='first', maintain_order=True)
        .drop('rank')
    )


def to_evidence(significant: pl.DataFrame, study: Study, release: str) -> pl.LazyFrame:
    """The significant genes' evidence in `GENE_BURDEN_PARSED`."""
    pvalue = pl.col('pValue')
    url = pl.concat_str(pl.lit(f'{study.browser}/gene/'), pl.col('gene_id'))
    return (
        significant
        .select(
            datasourceId=pl.lit(DATASOURCE_ID),
            datatypeId=pl.lit(DATATYPE_ID),
            targetFromSourceId=pl.col('gene_id'),
            diseaseFromSource=pl.lit(study.disease),
            curatedDiseaseFromSourceMappedId=pl.lit(study.disease_id),
            pValueMantissa=pvalue_mantissa(pvalue),
            pValueExponent=pvalue_exponent(pvalue),
            oddsRatio=pl.col('oddsRatio'),
            oddsRatioConfidenceIntervalLower=pl.col('oddsRatioConfidenceIntervalLower'),
            oddsRatioConfidenceIntervalUpper=pl.col('oddsRatioConfidenceIntervalUpper'),
            resourceScore=pvalue,
            projectId=pl.lit(study.project_id),
            cohortId=pl.lit(study.cohort_id),
            releaseVersion=pl.lit(release, dtype=pl.String),
            studySampleSize=pl.col('n_cases') + pl.col('n_controls'),
            studyCases=pl.col('n_cases'),
            studyCasesWithQualifyingVariants=pl.col('studyCasesWithQualifyingVariants'),
            statisticalMethod=pl.col('statisticalMethod'),
            statisticalMethodOverview=pl.col('statisticalMethod').replace_strict(METHOD_DESCRIPTIONS),
            literature=pl.lit(study.literature, dtype=pl.List(pl.String)),
            urls=pl.concat_list(pl.struct(url=url, niceName=pl.lit(f'{study.project_id} browser'))),
        )
        .pipe(to_parsed)
        .lazy()
    )
