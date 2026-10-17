"""BRaVa consortium gene burden evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.gene_burden.common import DATASOURCE_ID, DATATYPE_ID, to_parsed
from pts.transformers.utils.dataset import read_spreadsheet
from pts.transformers.utils.spark_casts import cast_double, to_int
from pts.transformers.utils.spark_frames import spark_distinct
from pts.transformers.utils.spark_math import pvalue_exponent, pvalue_mantissa, substitute_zero_pvalues

PROJECT_ID = 'BRaVa Consortium'

MASK_ABBREVIATIONS = {
    'pLoF': 'pLoF',
    'damaging_missense_or_protein_altering': 'DM/PA',
    'pLoF;damaging_missense_or_protein_altering': 'pLoF;DM/PA',
}
MASK_DESCRIPTIONS = {
    'pLoF': 'pLoF variants',
    'damaging_missense_or_protein_altering': 'damaging missense or protein altering variants',
    'pLoF;damaging_missense_or_protein_altering': 'pLoF or damaging missense/protein-altering variants',
}
MAF_DESCRIPTIONS = {
    '0.001': 'with a MAF smaller than 0.1%',
    '0.0001': 'with a MAF smaller than 0.01%',
}
#: 'ALL', a pan-ancestry meta-analysis, has no label.
ANCESTRY_LABELS = {
    'EUR': 'European',
    'AFR': 'African',
    'EAS': 'East Asian',
    'AMR': 'Admixed American',
    'SAS': 'Central and South Asian',
    'MID': 'Middle Eastern',
    'non_EUR': 'non-European',
}
#: 'SAS' would need two ids, so it has none.
ANCESTRY_IDS = {
    'EUR': 'HANCESTRO_0005',
    'AFR': 'HANCESTRO_0010',
    'EAS': 'HANCESTRO_0009',
    'AMR': 'HANCESTRO_0014',
    'MID': 'HANCESTRO_0852',
}

_CAUCHY_PVALUE = 'Min P-value of significant Cauchy associations'
_KEYS = ['Phenotype ID', 'Ancestry Group']


def _lookup(expr: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    return expr.replace_strict(mapping, default=None, return_dtype=pl.String)


def _spark_sum(column: str) -> pl.Expr:
    """Spark's `sum` cast to int: null when every value is null."""
    return pl.when(pl.col(column).count() > 0).then(pl.col(column).cast(pl.Int64).sum()).cast(pl.Int32)


def _totals(contributions: pl.DataFrame) -> pl.DataFrame:
    """Per-biobank contributions summed by phenotype within each ancestry, 'ALL' and 'non_EUR'."""
    groups = pl.concat([
        contributions.with_columns(pl.col('Ancestry').alias('Ancestry Group')),
        contributions.with_columns(pl.lit('ALL').alias('Ancestry Group')),
        contributions.filter(pl.col('Ancestry') != 'EUR').with_columns(pl.lit('non_EUR').alias('Ancestry Group')),
    ])
    return groups.group_by(_KEYS, maintain_order=True).agg(
        studySampleSize=_spark_sum('sampleSize'), studyCases=_spark_sum('cases')
    )


def _sample_size_lookup(s4: pl.DataFrame, s5: pl.DataFrame) -> pl.DataFrame:
    """(Phenotype ID, Ancestry Group) to sample size and cases, from the binary (S4) and quantitative (S5) tables.

    Rows repeated verbatim are counted once. A phenotype is either binary or quantitative, so the
    key is unique.
    """
    binary = spark_distinct(s4).select(
        'Phenotype ID',
        'Ancestry',
        sampleSize=to_int(pl.col('N cases')) + to_int(pl.col('N controls')),
        cases=to_int(pl.col('N cases')),
    )
    quantitative = spark_distinct(s5).select(
        'Phenotype ID', 'Ancestry', sampleSize=to_int(pl.col('N')), cases=pl.lit(None, dtype=pl.Int32)
    )
    return pl.concat([_totals(binary), _totals(quantitative)])


def _granular(sheet: pl.DataFrame) -> pl.DataFrame:
    """Per mask, MAF, test and ancestry results (S14 quantitative, S15 binary)."""
    sheet = sheet.with_columns(cast_double('Pvalue'), cast_double('BETA Burden'), cast_double('SE Burden'))
    sheet = sheet.with_columns(pl.when(~pl.col('BETA Burden').is_nan()).then(pl.col('BETA Burden')).name.keep())
    sheet = substitute_zero_pvalues(sheet, 'Pvalue', label='BRaVa')
    group = pl.col('meta analyzed')
    beta = pl.col('BETA Burden')
    return sheet.select(
        'Gene ID',
        'Description',
        'Phenotype ID',
        group.alias('Ancestry Group'),
        resourceScore=pl.col('Pvalue'),
        beta=beta,
        betaConfidenceIntervalLower=beta - pl.col('SE Burden'),
        betaConfidenceIntervalUpper=beta + pl.col('SE Burden'),
        statisticalMethod=pl.concat_str(
            'class', _lookup(pl.col('Mask'), MASK_ABBREVIATIONS), 'max MAF', group, separator='.', ignore_nulls=True
        ),
        statisticalMethodOverview=pl.concat_str(
            'class',
            pl.lit(' test carried out with '),
            _lookup(pl.col('Mask'), MASK_DESCRIPTIONS),
            pl.lit(' '),
            _lookup(pl.col('max MAF'), MAF_DESCRIPTIONS),
            pl.lit(', meta-analyzed across BRaVa biobanks.'),
        ),
    )


def _cauchy(sheet: pl.DataFrame) -> pl.DataFrame:
    """Cauchy combinations across masks, MAFs and tests (S12 quantitative, S13 binary)."""
    sheet = sheet.with_columns(cast_double(_CAUCHY_PVALUE))
    sheet = substitute_zero_pvalues(sheet, _CAUCHY_PVALUE, label='BRaVa Cauchy')
    group = pl.col('Cauchy combination with minimum P-value')
    return sheet.select(
        'Gene ID',
        'Description',
        'Phenotype ID',
        group.alias('Ancestry Group'),
        resourceScore=pl.col(_CAUCHY_PVALUE),
        statisticalMethod=pl.concat_str(pl.lit('Cauchy.'), group),
        statisticalMethodOverview=pl.lit(
            'Cauchy combination test aggregating Burden and SKAT/SKAT-O results across variant masks and MAF cutoffs.'
        ),
    )


def parse_brava(path: str) -> pl.LazyFrame:
    """BRaVa gene burden evidence (PMID 42238450): granular results (S14, S15) and Cauchy combinations (S12, S13).

    A p-value of 0.0 takes the smallest one above it in its own sheet. Sample sizes and cases come
    from the per-biobank tables (S4 binary, S5 quantitative), summed for the evidence's ancestry
    group; a group with no known count has none.

    Args:
        path: the supplementary tables workbook.

    Returns:
        The evidence in `GENE_BURDEN_PARSED`.
    """
    sheets = {
        name: read_spreadsheet(path, f'Table {name}', infer_types=False)
        for name in ('S4', 'S5', 'S12', 'S13', 'S14', 'S15')
    }
    lookup = _sample_size_lookup(sheets['S4'], sheets['S5']).drop_nulls(_KEYS)
    associations = pl.concat(
        [_granular(sheets['S14']), _granular(sheets['S15']), _cauchy(sheets['S12']), _cauchy(sheets['S13'])],
        how='diagonal',
    )
    score = pl.col('resourceScore')
    group = pl.col('Ancestry Group')
    return (
        associations
        .select(
            _KEYS,
            datasourceId=pl.lit(DATASOURCE_ID),
            datatypeId=pl.lit(DATATYPE_ID),
            projectId=pl.lit(PROJECT_ID),
            cohortId=pl.lit(PROJECT_ID),
            literature=pl.lit(['42238450']),
            targetFromSourceId=pl.col('Gene ID'),
            diseaseFromSource=pl.col('Description'),
            ancestry=_lookup(group, ANCESTRY_LABELS),
            ancestryId=_lookup(group, ANCESTRY_IDS),
            resourceScore=score,
            pValueExponent=pvalue_exponent(score),
            pValueMantissa=pvalue_mantissa(score),
            beta=pl.col('beta'),
            betaConfidenceIntervalLower=pl.col('betaConfidenceIntervalLower'),
            betaConfidenceIntervalUpper=pl.col('betaConfidenceIntervalUpper'),
            statisticalMethod=pl.col('statisticalMethod'),
            statisticalMethodOverview=pl.col('statisticalMethodOverview'),
        )
        .join(lookup, on=_KEYS, how='left', validate='m:1', maintain_order='left')
        .pipe(to_parsed)
        .pipe(spark_distinct)
        .lazy()
    )
