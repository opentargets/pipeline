"""Tests for the BipEx gene burden parser."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.gene_burden import GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.bipex import parse_bipex
from tests.evidence.gene_burden.fixtures import bipex_row, write_bipex


def _parse(tmp: Path, *rows: dict[str, str], release: str = '2026-08-07') -> pl.DataFrame:
    return parse_bipex(write_bipex(tmp, list(rows)), release).collect()


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, bipex_row()).schema == GENE_BURDEN_PARSED


def test_columns(tmp_path: Path) -> None:
    row = _parse(tmp_path, bipex_row()).row(0, named=True)
    assert row == {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'allelicRequirements': None,
        'targetFromSourceId': 'ENSG1',
        'diseaseFromSource': 'bipolar disorder',
        'pValueMantissa': 2.333,
        'pValueExponent': -11,
        'beta': None,
        'betaConfidenceIntervalLower': None,
        'betaConfidenceIntervalUpper': None,
        'oddsRatio': 7.5,
        'oddsRatioConfidenceIntervalLower': 3.25,
        'oddsRatioConfidenceIntervalUpper': 17.5,
        'resourceScore': 2.3328e-11,
        'ancestry': None,
        'ancestryId': None,
        'literature': None,
        'projectId': 'BipEx',
        'cohortId': 'BipEx2',
        'releaseVersion': '2026-08-07',
        'studySampleSize': 64435 + 168101,
        'studyCases': 64435,
        'studyCasesWithQualifyingVariants': 30,
        'statisticalMethod': 'ptv',
        'statisticalMethodOverview': 'Case-control burden test carried out with rare protein-truncating variants.',
        'urls': [{'url': 'https://bipex.broadinstitute.org/gene/ENSG1', 'niceName': 'BipEx browser'}],
        'sex': None,
        'diseaseFromSourceId': None,
        'curatedDiseaseFromSourceMappedId': 'MONDO_0004985',
    }


def test_reports_the_more_significant_test(tmp_path: Path) -> None:
    row = _parse(tmp_path, bipex_row(ptv_p_value='2.1427e-04', ptv_mis_p_value='1.3405e-11')).row(0, named=True)
    assert (
        row['statisticalMethod'],
        row['resourceScore'],
        row['oddsRatio'],
        row['oddsRatioConfidenceIntervalLower'],
        row['oddsRatioConfidenceIntervalUpper'],
        row['studyCasesWithQualifyingVariants'],
    ) == ('ptv_mis', 1.3405e-11, 2.5, 1.5, 4.0, 60)


def test_a_tie_reports_the_ptv_test(tmp_path: Path) -> None:
    row = _parse(tmp_path, bipex_row(ptv_mis_p_value='2.3328e-11')).row(0, named=True)
    assert row['statisticalMethod'] == 'ptv'


@pytest.mark.parametrize(
    ('flags', 'kept'),
    [
        ('bonferonni_significant,fdr_five_percent_significant', True),
        ('fdr_five_percent_significant', True),
        ('', False),
        ('bonferonni_significant', False),
    ],
)
def test_keeps_the_genes_flagged_five_percent_fdr(tmp_path: Path, flags: str, kept: bool) -> None:
    assert _parse(tmp_path, bipex_row(flags=flags)).height == int(kept)


def test_a_flagged_gene_without_a_ptv_test_is_dropped(tmp_path: Path) -> None:
    assert _parse(tmp_path, bipex_row(ptv_p_value='NA', ptv_mis_p_value='NA')).height == 0


def test_only_the_meta_analysis_is_read(tmp_path: Path) -> None:
    rows = _parse(tmp_path, bipex_row(), bipex_row(gene_id='ENSG2', group='EUR'))
    assert rows['targetFromSourceId'].to_list() == ['ENSG1']


def test_infinite_odds_ratio(tmp_path: Path) -> None:
    row = _parse(
        tmp_path,
        bipex_row(
            ptv_odds_ratio='Infinity', ptv_odds_ratio_95_ci_lower_bound='NaN', ptv_odds_ratio_95_ci_upper_bound='NaN'
        ),
    ).row(0, named=True)
    assert row['oddsRatio'] == math.inf
    assert math.isnan(row['oddsRatioConfidenceIntervalLower'])


def test_no_significant_rows_give_an_empty_frame(tmp_path: Path) -> None:
    rows = _parse(tmp_path, bipex_row(flags=''))
    assert rows.height == 0
    assert rows.schema == GENE_BURDEN_PARSED
