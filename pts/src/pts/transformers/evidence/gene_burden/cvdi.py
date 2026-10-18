"""Broad CVDI Human Disease Portal gene burden evidence."""

from __future__ import annotations

import polars as pl
from loguru import logger

from pts.transformers.evidence.gene_burden.common import DATASOURCE_ID, DATATYPE_ID, to_parsed
from pts.transformers.utils.dataset import read_spreadsheet
from pts.transformers.utils.spark_casts import cast_double
from pts.transformers.utils.spark_frames import spark_distinct
from pts.transformers.utils.spark_math import pvalue_exponent, pvalue_mantissa, spark_double_to_int
from pts.transformers.utils.spark_strings import regexp_extract

PROJECT_ID = 'CVDI Human Disease Portal'

METHOD_DESCRIPTIONS = {
    'LOF + missense0.8 (MAF<0.1%)': (
        'Mixed-effects test carried out with LOF and predicted-deleterious missense variants '
        '(missense score > 0.8) with a MAF smaller than 0.1%.'
    ),
    'LOF + missense0.5 (MAF<0.001%)': (
        'Mixed-effects test carried out with LOF and predicted-deleterious missense variants '
        '(missense score > 0.5) with a MAF smaller than 0.001%.'
    ),
    'LOF (MAF<0.1%)': 'Mixed-effects test carried out with LOF variants with a MAF smaller than 0.1%.',
    'LOF + missense0.5 (MAF<0.1%)': (
        'Mixed-effects test carried out with LOF and predicted-deleterious missense variants '
        '(missense score > 0.5) with a MAF smaller than 0.1%.'
    ),
    'Cauchy': 'Combined test after combining mask-specific using Cauchy distribution.',
}
URL_PREFIX = (
    'https://hugeamp.org:8000/research.html?ancestry=mixed&cohort=UKB_450k_AoU_250k_MGB_53k_META_overlapcorrected'
    '&file=600Traits.csv&gene='
)
URL_SUFFIX = '&pageid=600_traits_app'

_INDEX = ['phenotype', 'Gene ID Ensembl', 'Gene']
_FIELDS = ['cMAC', 'OR [95%CI]', 'Meta P-value', 'Cauchy P-value']
_ANCESTRY = 'ALL ancestry'
_OTHER_COHORTS = ['AoU', 'UKB', 'META (no correction)', 'MGB']
_NUMBER = r'([0-9]+(?:\.[0-9]+)?)'
_ODDS_RATIO = rf'^{_NUMBER} \[{_NUMBER}; {_NUMBER}\]$'
_ODDS_COLUMNS = ['oddsRatio', 'oddsRatioConfidenceIntervalLower', 'oddsRatioConfidenceIntervalUpper']
_OLD_ODDS_PATTERNS = [r'([0-9]+\.[0-9]+)', r'\[([0-9]+\.[0-9]+)', r'; ([0-9]+\.[0-9]+)\]']

#: Each sheet column's header names, top level first; column `i` is `column_{i + 1}`.
Header = list[tuple[str, ...]]


def _multi_header(rows: list[tuple[str | None, ...]]) -> Header:
    """`pandas.read_excel`'s multi-row header: a blank name takes the one on its left, within the same name above."""
    width = len(rows[0])
    open_below = [True] * width
    levels = []
    for row in rows:
        names = ['' if name is None else name for name in row]
        last = names[0]
        for i in range(1, width):
            if not open_below[i]:
                last = names[i]
            if names[i]:
                open_below[i] = False
                last = names[i]
            else:
                names[i] = last
        levels.append(names)
    return list(zip(*levels, strict=True))


def _read(path: str, sheet: str, header_rows: int) -> tuple[Header, pl.DataFrame]:
    """A sheet's multi-row header and its rows, under a title row and over a footnote row."""
    raw = read_spreadsheet(path, sheet, infer_types=False, has_header=False)
    header = _multi_header(raw.slice(1, header_rows).rows())
    return header, raw.slice(1 + header_rows, max(raw.height - 2 - header_rows, 0))


def _only(columns: list[int], name: str) -> pl.Expr:
    if len(columns) != 1:
        msg = f'expected one CVDI column {name!r}, found {len(columns)}'
        raise ValueError(msg)
    return pl.col(f'column_{columns[0] + 1}').alias(name)


def _associations(path: str) -> pl.DataFrame:
    """ST6's pan-ancestry results: a row per association and method, the methods in sorted order."""
    header, rows = _read(path, 'ST6', 3)
    index = [_only([i for i, names in enumerate(header) if names[0] == name], name) for name in _INDEX]

    def fields(method: str) -> list[pl.Expr]:
        group = (_ANCESTRY, method)
        return [pl.col(f'column_{i + 1}').alias(names[2]) for i, names in enumerate(header) if names[:2] == group]

    methods = sorted({names[1] for names in header if names[0] == _ANCESTRY})
    associations = pl.concat(
        [rows.select(*index, *fields(method), method_name=pl.lit(method)) for method in methods], how='diagonal'
    )
    missing = [pl.lit(None, dtype=pl.String).alias(field) for field in _FIELDS if field not in associations.columns]
    return associations.with_columns(missing)


def _cutoffs(path: str) -> pl.DataFrame:
    """ST3's FDR 1% p-value cutoff of each mask; the overlap-corrected meta-analysis is the cohort left."""
    header, rows = _read(path, 'ST3', 2)
    missing = set(_OTHER_COHORTS) - {names[0] for names in header}
    if missing:
        msg = f'CVDI ST3 has no {sorted(missing)} cutoffs'
        raise ValueError(msg)
    kept = [(i, names[1]) for i, names in enumerate(header) if names[0] not in _OTHER_COHORTS]

    def column(name: str) -> pl.Expr:
        return _only([i for i, kept_name in kept if kept_name == name], name)

    return (
        rows
        .select(column('Significance cutoff').forward_fill(), column('Mask'), column('P cutoff'))
        .filter(pl.col('Significance cutoff') == 'FDR1%')
        .drop('Significance cutoff')
    )


def _with_odds_ratios(significant: pl.DataFrame) -> pl.DataFrame:
    """The odds ratio and its confidence interval, from `OR [95%CI]`'s `OR [lower; upper]`; else null.

    The rows where the old regexes, which take the first decimal and miss integral values, read
    them differently are counted.
    """
    odds = pl.col('OR [95%CI]')
    parsed = significant.with_columns(
        odds.str.extract(_ODDS_RATIO, group).cast(pl.Float64).alias(column)
        for group, column in enumerate(_ODDS_COLUMNS, start=1)
    )
    old = parsed.select(
        regexp_extract(odds, pattern, 1).cast(pl.Float64, strict=False).ne_missing(pl.col(column)).sum().alias(column)
        for pattern, column in zip(_OLD_ODDS_PATTERNS, _ODDS_COLUMNS, strict=True)
    )
    ratios, lower, upper = old.row(0)
    logger.info(
        f'{ratios} CVDI odds ratios and {lower + upper} confidence interval bounds ({lower} lower, {upper} upper) '
        'differ from the old regexes'
    )
    return parsed


def parse_cvdi(path: str) -> pl.LazyFrame:
    """CVDI gene burden evidence (PMID 39210047): the significant pan-ancestry results of ST6.

    A mask's result is significant at or below its FDR 1% p-value cutoff in ST3; the Cauchy
    combination is scored by its own p-value, every mask by the meta-analysis one.

    Args:
        path: the supplementary tables workbook.

    Returns:
        The evidence in `GENE_BURDEN_PARSED`.
    """
    gene_id = pl.col('Gene ID Ensembl')
    significant = (
        _associations(path)
        .join(_cutoffs(path), left_on='method_name', right_on='Mask', how='inner', maintain_order='left')
        .filter((gene_id != 'Gene ID Ensembl') & gene_id.is_not_null())
        .with_columns(cast_double(c) for c in ('cMAC', 'Meta P-value', 'Cauchy P-value', 'P cutoff'))
        .with_columns(
            resourceScore=pl
            .when(pl.col('method_name') == 'Cauchy')
            .then(pl.col('Cauchy P-value'))
            .otherwise(pl.col('Meta P-value'))
        )
        .filter(pl.col('resourceScore') <= pl.col('P cutoff'))
    )
    significant = _with_odds_ratios(significant)
    score = pl.col('resourceScore')
    return (
        significant
        .select(
            *_ODDS_COLUMNS,
            datasourceId=pl.lit(DATASOURCE_ID),
            datatypeId=pl.lit(DATATYPE_ID),
            projectId=pl.lit(PROJECT_ID),
            cohortId=pl.lit('UK Biobank 450k/All of Us/MGB'),
            diseaseFromSource=pl.col('phenotype').str.replace_all('_', ' ', literal=True),
            targetFromSourceId=gene_id,
            studySampleSize=pl.lit(748879),
            resourceScore=score,
            studyCasesWithQualifyingVariants=spark_double_to_int(pl.col('cMAC')),
            statisticalMethod=pl.col('method_name'),
            statisticalMethodOverview=pl.col('method_name').replace(METHOD_DESCRIPTIONS),
            urls=pl.concat_list(
                pl.struct(
                    url=pl.concat_str(pl.lit(URL_PREFIX), pl.col('Gene'), pl.lit(URL_SUFFIX)),
                    niceName=pl.lit('Broad CVDI Human Disease Portal'),
                )
            ),
            literature=pl.lit(['39210047']),
            pValueExponent=pvalue_exponent(score),
            pValueMantissa=pvalue_mantissa(score),
        )
        .pipe(to_parsed)
        .pipe(spark_distinct)
        .lazy()
    )
