"""Genes & Health gene burden evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.gene_burden.common import DATASOURCE_ID, DATATYPE_ID, to_parsed
from pts.transformers.utils.dataset import read_spreadsheet
from pts.transformers.utils.spark_casts import cast_double, to_int
from pts.transformers.utils.spark_frames import spark_distinct
from pts.transformers.utils.spark_math import (
    pvalue_exponent,
    pvalue_mantissa,
    spark_exp,
    spark_pow,
    substitute_zero_pvalues,
)

PROJECT_ID = 'Genes & Health'

GNH_VARIANT_MASK_DESC = {
    'A': 'high-confidence LoF variants',
    'B': 'deleterious missense variants',
    'C': 'missense variants',
    'D': 'synonymous variants',
}
GNH_TEST_DESC = {
    'ADD': 'Burden',
    'ADD-SKAT': 'SKAT',
    'ADD-SKATO': 'SKAT-O',
}
GNH_MAF_DESC = {
    '0.01': 'with a MAF smaller than 1%',
    '0.001': 'with a MAF smaller than 0.1%',
    '0.0001': 'with a MAF smaller than 0.01%',
    'singleton': 'restricted to singletons',
}

_SOUTH_ASIAN = ('Pakistani and Bangladeshi', 'HANCESTRO_0006')


def _lookup(expr: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    return expr.replace_strict(mapping, default=None, return_dtype=pl.String)


def _is_quantitative() -> pl.Expr:
    return pl.col('QT').str.to_lowercase().is_in(['true', '1', '1.0']).fill_null(value=False)


def _method_overview(test: pl.Expr, suffix: str) -> pl.Expr:
    return pl.concat_str(
        test,
        pl.lit(' test carried out with '),
        _lookup(pl.col('Variant Mask'), GNH_VARIANT_MASK_DESC),
        pl.lit(' '),
        _lookup(pl.col('Freq').str.to_lowercase(), GNH_MAF_DESC),
        pl.lit(suffix),
    )


def _study(ancestry: tuple[str | None, str | None], allelic_requirement: str, cohort: str) -> dict[str, pl.Expr]:
    return {
        'targetFromSourceId': pl.col('Gene ID'),
        'diseaseFromSource': pl.col('Phenotype'),
        'diseaseFromSourceId': pl.col('EFO ID'),
        'curatedDiseaseFromSourceMappedId': pl.col('EFO ID'),
        'isQuantitative': _is_quantitative(),
        'ancestry': pl.lit(ancestry[0], dtype=pl.String),
        'ancestryId': pl.lit(ancestry[1], dtype=pl.String),
        'allelicRequirements': pl.lit(allelic_requirement),
        'cohortId': pl.lit(cohort),
    }


def _additive(st9: pl.DataFrame) -> pl.DataFrame:
    return st9.select(
        **_study(_SOUTH_ASIAN, 'dominant', PROJECT_ID),
        pValue=spark_pow(10.0, -cast_double('LOG10P')),
        effect=cast_double('BETA'),
        standardError=cast_double('SE'),
        studySampleSize=to_int(pl.col('N')),
        studyCases=pl.lit(None, dtype=pl.Int32),
        studyCasesWithQualifyingVariants=pl.lit(None, dtype=pl.Int32),
        statisticalMethod=pl.concat_str('TEST', 'Mask', separator='_', ignore_nulls=True),
        statisticalMethodOverview=_method_overview(_lookup(pl.col('TEST'), GNH_TEST_DESC), '.'),
    )


def _meta(st13: pl.DataFrame) -> pl.DataFrame:
    return st13.select(
        **_study((None, None), 'dominant', 'Genes & Health + UK Biobank'),
        pValue=spark_pow(10.0, -cast_double('LOG10P')),
        effect=cast_double('BETA'),
        standardError=cast_double('SE'),
        studySampleSize=to_int(pl.col('N UKB')) + to_int(pl.col('N G&H')),
        studyCases=pl.lit(None, dtype=pl.Int32),
        studyCasesWithQualifyingVariants=pl.lit(None, dtype=pl.Int32),
        statisticalMethod=pl.concat_str(pl.lit('META.'), 'Mask'),
        statisticalMethodOverview=_method_overview(
            pl.lit('Burden'), ' meta-analysed between Genes & Health and UK Biobank.'
        ),
    )


def _recessive(st15: pl.DataFrame) -> pl.DataFrame:
    return st15.filter(
        (pl.col('Significant REC P-value') == 'Significant') & (pl.col('Variant consequence') == 'pLoF_pDM')
    ).select(
        **_study(_SOUTH_ASIAN, 'recessive', PROJECT_ID),
        pValue=cast_double('Recessive p-value'),
        effect=cast_double('Recessive log(OR)'),
        standardError=pl.lit(None, dtype=pl.Float64),
        studySampleSize=to_int(pl.col('Number for ExWAS')),
        studyCases=to_int(pl.col('Number of Cases')),
        studyCasesWithQualifyingVariants=to_int(pl.col('Biallelic Carriers')),
        statisticalMethod=pl.lit('REC.pLoF_pDM'),
        statisticalMethodOverview=pl.lit(
            'Recessive burden test carried out with biallelic pLoF and deleterious missense genotypes.'
        ),
    )


def parse_genes_and_health(path: str) -> pl.LazyFrame:
    """Genes & Health gene-based tests: additive (ST9), meta-analysed with UK Biobank (ST13), recessive (ST15).

    The study's EFO id is both `diseaseFromSourceId` and `curatedDiseaseFromSourceMappedId`. A p-value
    of 0.0 takes the smallest one above it over the three sheets. Quantitative traits report a beta,
    binary ones an odds ratio, `exp` of the log odds ratio.

    Args:
        path: the supplementary tables workbook.

    Returns:
        The evidence in `GENE_BURDEN_PARSED`.
    """
    sheets = {name: read_spreadsheet(path, name, infer_types=False) for name in ('ST9', 'ST13', 'ST15')}
    associations = pl.concat([_additive(sheets['ST9']), _meta(sheets['ST13']), _recessive(sheets['ST15'])])
    associations = associations.with_columns(effect=pl.when(~pl.col('effect').is_nan()).then(pl.col('effect')))
    associations = substitute_zero_pvalues(associations, 'pValue', label=PROJECT_ID)

    pvalue = pl.col('pValue')
    effect = pl.col('effect')
    lower = effect - pl.col('standardError')
    upper = effect + pl.col('standardError')
    quantitative = pl.col('isQuantitative')
    return (
        associations
        .select(
            datasourceId=pl.lit(DATASOURCE_ID),
            datatypeId=pl.lit(DATATYPE_ID),
            projectId=pl.lit(PROJECT_ID),
            literature=pl.lit(['41896352']),
            targetFromSourceId=pl.col('targetFromSourceId'),
            diseaseFromSource=pl.col('diseaseFromSource'),
            diseaseFromSourceId=pl.col('diseaseFromSourceId'),
            curatedDiseaseFromSourceMappedId=pl.col('curatedDiseaseFromSourceMappedId'),
            resourceScore=pvalue,
            pValueExponent=pvalue_exponent(pvalue),
            pValueMantissa=pvalue_mantissa(pvalue),
            beta=pl.when(quantitative).then(effect),
            betaConfidenceIntervalLower=pl.when(quantitative).then(lower),
            betaConfidenceIntervalUpper=pl.when(quantitative).then(upper),
            oddsRatio=pl.when(~quantitative).then(spark_exp(effect)),
            oddsRatioConfidenceIntervalLower=pl.when(~quantitative).then(spark_exp(lower)),
            oddsRatioConfidenceIntervalUpper=pl.when(~quantitative).then(spark_exp(upper)),
            ancestry=pl.col('ancestry'),
            ancestryId=pl.col('ancestryId'),
            allelicRequirements=pl.concat_list('allelicRequirements'),
            cohortId=pl.col('cohortId'),
            studySampleSize=pl.col('studySampleSize'),
            studyCases=pl.col('studyCases'),
            studyCasesWithQualifyingVariants=pl.col('studyCasesWithQualifyingVariants'),
            statisticalMethod=pl.col('statisticalMethod'),
            statisticalMethodOverview=pl.col('statisticalMethodOverview'),
        )
        .pipe(to_parsed)
        .pipe(spark_distinct)
        .lazy()
    )
