"""SCHEMA2 (Schizophrenia Exome Sequencing Meta-Analysis) gene burden evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.gene_burden.exome_browser import (
    TESTS,
    Study,
    most_significant,
    read_gene_results,
    to_evidence,
)

#: The 5% FDR cutoff the SCHEMA2 preprint reports on the case-control p-value.
P_VALUE_CUTOFF = 6.84e-5

STUDY = Study(
    project_id='SCHEMA',
    cohort_id='SCHEMA2',
    disease='schizophrenia',
    disease_id='MONDO_0005090',
    browser='https://schema.broadinstitute.org',
)

_INTERVAL = r'^(\S+) - (\S+)$'


def _test(test: str) -> list[pl.Expr]:
    interval = pl.col(f'{test}_odds_ratio_95_ci').str.extract_groups(_INTERVAL)
    return [
        pl.col(f'{test}_p_value'),
        pl.col(f'{test}_odds_ratio'),
        interval.struct.field('1'),
        interval.struct.field('2'),
        pl.col(f'{test}_case_carrier'),
    ]


def parse_schema(path: str, release: str) -> pl.LazyFrame:
    """SCHEMA2 evidence: the genes whose more significant test, PTV or PTV+missense, has p < 6.84e-5.

    The file's own case-control p-value is that minimum rounded to three figures; the unrounded
    p-value, and the odds ratio of the same test, are reported.

    Args:
        path: the browser's `gene_results.tsv.bgz`.
        release: the download's date, as `releaseVersion`.

    Returns:
        The evidence in `GENE_BURDEN_PARSED`.
    """
    results = read_gene_results(path)
    best = most_significant(results, {test: _test(test) for test in TESTS})
    return to_evidence(best.filter(pl.col('pValue') < P_VALUE_CUTOFF), STUDY, release)
