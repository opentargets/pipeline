"""AstraZeneca PheWAS Portal gene burden evidence."""

from __future__ import annotations

import polars as pl
from loguru import logger

from pts.transformers.evidence.gene_burden.common import DATASOURCE_ID, DATATYPE_ID, to_parsed
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_casts import cast_double, to_float, to_int
from pts.transformers.utils.spark_csv import read_spark_csv
from pts.transformers.utils.spark_frames import spark_distinct
from pts.transformers.utils.spark_math import pvalue_exponent, pvalue_mantissa, substitute_zero_pvalues
from pts.transformers.utils.spark_strings import regexp_extract

PROJECT_ID = 'AstraZeneca PheWAS Portal'

AZ_METHOD_DESCRIPTIONS = {
    'ptv': 'Burden test carried out with PTVs with a MAF smaller than 0.1%.',
    'ptv5pcnt': 'Burden test carried out with PTVs with a MAF smaller than 5%.',
    'UR': 'Burden test carried out with ultra rare damaging variants (MAF ≈ 0%).',
    'URmtr': 'Burden test carried out with MTR-informed ultra rare damaging variants (MAF ≈ 0%).',
    'raredmg': 'Burden test carried out with rare missense variants with a MAF smaller than 0.005%.',
    'raredmgmtr': 'Burden test carried out with MTR-informed rare missense variants with a MAF smaller than 0.005%.',
    'flexdmg': 'Burden test carried out with damaging variants with a MAF smaller than 0.01%.',
    'flexnonsyn': 'Burden test carried out with non synonymous variants with a MAF smaller than 0.01%.',
    'flexnonsynmtr': 'Burden test carried out with MTR-informed non synonymous variants with a MAF smaller than 0.01%.',
    'ptvraredmg': 'Burden test carried out with PTV or rare missense variants.',
    'rec': 'Burden test carried out with non-synonymous recessive variants with a MAF smaller than 1%.',
    'syn': 'Burden test carried out with synonymous variants.',
}

P_VALUE_CUTOFF = 1e-7

_RELEASE_PATTERN = r'https://azphewas.com/geneView/([^/]+)/'


def _release_version(genes: str) -> str | None:
    """The portal release hash in the gene links file's first link."""
    links = read_spark_csv(genes, has_header=False, new_columns=['gene', 'link'])
    if links.is_empty():
        raise ValueError(f'no gene links in {genes}')
    return links.select(regexp_extract(pl.col('link'), _RELEASE_PATTERN, 1)).item(0, 0)


def _effect(trait_type: str, column: str) -> pl.Expr:
    """The column as float32 on rows of the trait type, widened to Float64."""
    return to_float(pl.when(pl.col('Type') == trait_type).then(pl.col(column))).cast(pl.Float64)


def parse_az(binary: str, quantitative: str, genes: str, phenotypes: str) -> pl.LazyFrame:
    """AZ PheWAS evidence: binary and quantitative associations with p ≤ 1e-7.

    Every input column is text. A p-value of 0.0 takes the smallest one above it among the
    associations kept; effects are cast through float32. A phenotype without a link gets null
    `urls`, and one linked more than once gets a row per link.

    Args:
        binary: binary trait associations, parquet.
        quantitative: quantitative trait associations, parquet.
        genes: gene links CSV without a header; its first link names the release.
        phenotypes: phenotype name and link CSV without a header.

    Returns:
        The evidence in `GENE_BURDEN_PARSED`.
    """
    associations = pl.concat(
        [
            scan_dataset(binary).rename({
                'BinOddsRatioLCI': 'LCI',
                'BinOddsRatioUCI': 'UCI',
                'BinNcases': 'nCases',
                'BinQVcases': 'nCasesQV',
                'BinNcontrols': 'nControls',
            }),
            scan_dataset(quantitative).with_columns(nCases=pl.col('nSamples')).rename({'YesQV': 'nCasesQV'}),
        ],
        how='diagonal',
    )
    significant = (
        associations
        .with_columns(cast_double('pValue'))
        .filter(pl.col('pValue') <= P_VALUE_CUTOFF)
        .pipe(spark_distinct)
        .collect()
    )
    significant = substitute_zero_pvalues(significant, 'pValue', label='AZ')

    links = read_spark_csv(phenotypes, has_header=False, new_columns=['diseaseFromSource', 'url'])
    pvalue = pl.col('pValue')
    method = pl.col('CollapsingModel')
    evidence = (
        significant
        .select(
            datasourceId=pl.lit(DATASOURCE_ID),
            datatypeId=pl.lit(DATATYPE_ID),
            allelicRequirements=pl.when(method == 'rec').then(pl.lit(['recessive'])).otherwise(pl.lit(['dominant'])),
            targetFromSourceId=pl.col('Gene'),
            diseaseFromSource=pl.col('Phenotype'),
            pValueMantissa=pvalue_mantissa(pvalue),
            pValueExponent=pvalue_exponent(pvalue),
            beta=_effect('Quantitative', 'beta'),
            betaConfidenceIntervalLower=_effect('Quantitative', 'LCI'),
            betaConfidenceIntervalUpper=_effect('Quantitative', 'UCI'),
            oddsRatio=_effect('Binary', 'BinOddsRatio'),
            oddsRatioConfidenceIntervalLower=_effect('Binary', 'LCI'),
            oddsRatioConfidenceIntervalUpper=_effect('Binary', 'UCI'),
            resourceScore=pvalue,
            ancestry=pl.lit('EUR'),
            ancestryId=pl.lit('HANCESTRO_0005'),
            literature=pl.lit(['34375979']),
            projectId=pl.lit(PROJECT_ID),
            cohortId=pl.lit('UK Biobank 470k'),
            releaseVersion=pl.lit(_release_version(genes), dtype=pl.String),
            studySampleSize=to_int(pl.col('nSamples')),
            studyCases=to_int(pl.col('nCases')),
            studyCasesWithQualifyingVariants=to_int(pl.col('nCasesQV')),
            statisticalMethod=method,
            statisticalMethodOverview=method.replace(AZ_METHOD_DESCRIPTIONS),
        )
        .join(links, on='diseaseFromSource', how='left', maintain_order='left_right')
        .with_columns(
            urls=pl.when(pl.col('url').is_not_null()).then(
                pl.concat_list(pl.struct('url', niceName=pl.lit(PROJECT_ID)))
            )
        )
        .pipe(to_parsed)
        .pipe(spark_distinct)
    )
    unlinked = evidence['urls'].null_count()
    if unlinked:
        logger.warning(f'{unlinked} AZ evidence have no phenotype link; their urls are null.')
    return evidence.lazy()
