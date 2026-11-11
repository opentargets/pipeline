"""BipEx (Bipolar Exome) gene burden evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.gene_burden.exome_browser import (
    TESTS,
    Study,
    most_significant,
    read_gene_results,
    to_evidence,
)

#: The flag BipEx sets on its 5% FDR genes.
SIGNIFICANT_FLAG = 'fdr_five_percent_significant'

STUDY = Study(
    project_id='BipEx',
    cohort_id='BipEx2',
    disease='bipolar disorder',
    disease_id='MONDO_0004985',
    browser='https://bipex.broadinstitute.org',
    literature=['35410376'],
)


def _test(test: str) -> list[pl.Expr]:
    return [
        pl.col(f'{test}_p_value'),
        pl.col(f'{test}_odds_ratio'),
        pl.col(f'{test}_odds_ratio_95_ci_lower_bound'),
        pl.col(f'{test}_odds_ratio_95_ci_upper_bound'),
        pl.col(f'{test}_case_carrier'),
    ]


def parse_bipex(path: str, release: str) -> pl.LazyFrame:
    """BipEx evidence: the genes flagged 5% FDR significant, each with its more significant test, PTV or PTV+missense.

    Args:
        path: the browser's `gene_results.tsv.bgz`.
        release: the download's date, as `releaseVersion`.

    Returns:
        The evidence in `GENE_BURDEN_PARSED`.
    """
    results = read_gene_results(path)
    flagged = results.filter(pl.col('flags').fill_null('').str.split(',').list.contains(SIGNIFICANT_FLAG))
    return to_evidence(most_significant(flagged, {test: _test(test) for test in TESTS}), STUDY, release)
