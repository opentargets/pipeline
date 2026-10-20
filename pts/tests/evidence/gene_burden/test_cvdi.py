"""Tests for the CVDI gene burden parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.schemas.gene_burden import GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.cvdi import parse_cvdi
from tests.evidence.gene_burden.fixtures import CVDI_CUTOFFS, CVDI_MASKS, cvdi_row, write_cvdi
from tests.evidence.gene_burden.logs import logged
from tests.evidence.gene_burden.spark_rows import expand, multiset

LOF = 'LOF (MAF<0.1%)'
LOF_M08 = 'LOF + missense0.8 (MAF<0.1%)'
M05 = 'missense0.5 (MAF<0.001%)'
M02 = 'missense0.2 (MAF<0.001%)'
#: ST3 names the lowercase ST6 masks capitalised.
CUTOFFS_WITH_M02 = [*CVDI_CUTOFFS[:4], (None, 'Missense0.2 (MAF<0.001%)', 3e-8), *CVDI_CUTOFFS[4:]]
NICE_NAME = 'Broad CVDI Human Disease Portal'
LOF_OVERVIEW = 'Mixed-effects test carried out with LOF variants with a MAF smaller than 0.1%.'
URL = (
    'https://hugeamp.org:8000/research.html?ancestry=mixed&cohort=UKB_450k_AoU_250k_MGB_53k_META_overlapcorrected'
    '&file=600Traits.csv&gene={}&pageid=600_traits_app'
)

#: Ruling 6: an integral odds ratio or bound, by gene.
INTEGRAL_ODDS = {'ENSG10': '2 [1.77; 2.25]', 'ENSG11': '74.78 [58.86; 95]', 'ENSG12': '2.49 [2; 3.11]'}
ODDS_COLUMNS = ['oddsRatio', 'oddsRatioConfidenceIntervalLower', 'oddsRatioConfidenceIntervalUpper']


def _lof(cells: dict[Any, Any] | None = None) -> dict[Any, Any]:
    """A row significant for the LOF mask alone, with `cells` set."""
    row = {(LOF, 'cMAC'): 12, (LOF, 'OR [95%CI]'): '2.5 [1.5; 3.5]', (LOF, 'Meta P-value'): 2.5e-9}
    return cvdi_row() | row | (cells or {})


def _parse(tmp: Path, *rows: dict[Any, Any], **kwargs: Any) -> pl.DataFrame:
    return parse_cvdi(write_cvdi(tmp, rows, **kwargs)).collect()


def _by_method(rows: pl.DataFrame) -> dict[str, dict[str, Any]]:
    return {r['statisticalMethod']: r for r in rows.iter_rows(named=True)}


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, _lof()).schema == GENE_BURDEN_PARSED


def test_mask_columns(tmp_path: Path) -> None:
    assert _parse(tmp_path, _lof()).rows(named=True) == [
        {
            'datasourceId': 'gene_burden',
            'datatypeId': 'genetic_association',
            'allelicRequirements': None,
            'targetFromSourceId': 'ENSG1',
            'diseaseFromSource': 'Type 2 diabetes',
            'pValueMantissa': 2.5,
            'pValueExponent': -9,
            'beta': None,
            'betaConfidenceIntervalLower': None,
            'betaConfidenceIntervalUpper': None,
            'oddsRatio': 2.5,
            'oddsRatioConfidenceIntervalLower': 1.5,
            'oddsRatioConfidenceIntervalUpper': 3.5,
            'resourceScore': 2.5e-9,
            'ancestry': None,
            'ancestryId': None,
            'literature': ['39210047'],
            'projectId': 'CVDI Human Disease Portal',
            'cohortId': 'UK Biobank 450k/All of Us/MGB',
            'releaseVersion': None,
            'studySampleSize': 748879,
            'studyCases': None,
            'studyCasesWithQualifyingVariants': 12,
            'statisticalMethod': LOF,
            'statisticalMethodOverview': LOF_OVERVIEW,
            'urls': [{'url': URL.format('G1'), 'niceName': NICE_NAME}],
            'sex': None,
            'diseaseFromSourceId': None,
            'curatedDiseaseFromSourceMappedId': None,
        }
    ]


def test_cauchy_has_no_odds_ratio(tmp_path: Path) -> None:
    rows = _by_method(_parse(tmp_path, _lof({('Cauchy', 'Cauchy P-value'): 4e-12})))
    assert set(rows) == {LOF, 'Cauchy'}
    cauchy = rows['Cauchy']
    assert cauchy['resourceScore'] == 4e-12
    assert (cauchy['oddsRatio'], cauchy['oddsRatioConfidenceIntervalLower']) == (None, None)
    assert cauchy['oddsRatioConfidenceIntervalUpper'] is None
    assert cauchy['studyCasesWithQualifyingVariants'] is None
    assert cauchy['statisticalMethodOverview'] == (
        'Combined test after combining mask-specific using Cauchy distribution.'
    )


def test_significance_uses_the_fdr1_cutoff_of_the_mask(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _lof({'Gene ID Ensembl': 'AT_CUTOFF', (LOF, 'Meta P-value'): 1e-6}),
        _lof({'Gene ID Ensembl': 'FDR5', (LOF, 'Meta P-value'): 1e-5}),
        _lof({'Gene ID Ensembl': 'M08', (LOF, 'Meta P-value'): None, (LOF_M08, 'Meta P-value'): 3e-7}),
        _lof({'Gene ID Ensembl': 'CAUCHY', (LOF, 'Meta P-value'): None, ('Cauchy', 'Cauchy P-value'): 2e-7}),
    )
    assert rows.select('targetFromSourceId', 'statisticalMethod').rows() == [
        ('CAUCHY', 'Cauchy'),
        ('AT_CUTOFF', LOF),
    ]


def test_cauchy_score_is_the_cauchy_pvalue(tmp_path: Path) -> None:
    rows = _parse(tmp_path, _lof({(LOF, 'Meta P-value'): None, ('Cauchy', 'Cauchy P-value'): 0.0}))
    assert rows.select('statisticalMethod', 'resourceScore', 'pValueMantissa', 'pValueExponent').rows() == [
        ('Cauchy', 0.0, None, None)
    ]


def test_other_ancestries_are_ignored(tmp_path: Path) -> None:
    rows = _parse(tmp_path, _lof({(LOF, 'Meta P-value'): 0.5}))
    assert rows.height == 0


def test_text_in_numeric_cells_is_null(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _lof(),
        _lof({
            'Gene ID Ensembl': 'ENSG2',
            (LOF, 'cMAC'): '-',
            (LOF, 'OR [95%CI]'): '-',
            (LOF, 'Meta P-value'): '-',
            ('Cauchy', 'Cauchy P-value'): 1e-12,
        }),
    )
    assert rows.select('targetFromSourceId', 'statisticalMethod', 'resourceScore').rows() == [
        ('ENSG2', 'Cauchy', 1e-12),
        ('ENSG1', LOF, 2.5e-9),
    ]


def test_numeric_cells_cast_as_spark(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _lof({(LOF, 'cMAC'): 12.0, (LOF, 'Meta P-value'): '1e-9'}),
        _lof({'Gene ID Ensembl': 'ENSG2', (LOF, 'cMAC'): 3e9}),
        _lof({'Gene ID Ensembl': 'ENSG3', (LOF, 'cMAC'): '7.9'}),
    )
    assert rows.select('targetFromSourceId', 'studyCasesWithQualifyingVariants', 'resourceScore').rows() == [
        ('ENSG1', 12, 1e-9),
        ('ENSG2', 2147483647, 2.5e-9),
        ('ENSG3', 7, 2.5e-9),
    ]


def test_boolean_cells_cast_as_the_old_typed_read(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _lof({(LOF, 'cMAC'): True}),
        _lof({'Gene ID Ensembl': 'ENSG2', (LOF, 'Meta P-value'): False}),
    )
    assert rows.select('targetFromSourceId', 'studyCasesWithQualifyingVariants', 'resourceScore').rows() == [
        ('ENSG1', 1, 2.5e-9),
        ('ENSG2', 12, 0.0),
    ]


def test_boolean_cutoffs_cast_as_the_old_typed_read(tmp_path: Path) -> None:
    cutoffs = [(label, mask, True if mask == LOF else cutoff) for label, mask, cutoff in CVDI_CUTOFFS]
    rows = _parse(tmp_path, _lof({(LOF, 'Meta P-value'): 0.5}), cutoffs=cutoffs)
    assert rows.select('statisticalMethod', 'resourceScore').rows() == [(LOF, 0.5)]


def test_a_missing_field_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Cauchy P-value'):
        _parse(tmp_path, _lof(), cauchy=False)


def test_header_repeats_and_rows_without_a_gene_id_are_dropped(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _lof({'phenotype': 'phenotype', 'Gene ID Ensembl': 'Gene ID Ensembl', 'Gene': 'Gene'}),
        _lof({'Gene ID Ensembl': None}),
        _lof(),
    )
    assert rows['targetFromSourceId'].to_list() == ['ENSG1']


def test_the_footer_row_is_skipped(tmp_path: Path) -> None:
    footer = ['Asthma', 'ENSG9', 'G9', 1, '2.5 [1.5; 3.5]', 1e-20]
    rows = _parse(tmp_path, _lof(), footer=footer)
    assert rows['targetFromSourceId'].to_list() == ['ENSG1']


def test_cutoff_labels_fill_down(tmp_path: Path) -> None:
    # without the fill, only the first mask of the FDR1% block would have a cutoff
    rows = _parse(tmp_path, _lof({(LOF, 'Meta P-value'): None, (LOF_M08, 'Meta P-value'): 1e-7}))
    assert rows['statisticalMethod'].to_list() == [LOF_M08]


def test_strings(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _lof({'phenotype': 'Cardiac_pacemaker/device_in_situ'}),
        _lof({'Gene ID Ensembl': 'ENSG2', 'phenotype': None, 'Gene': None}),
    )
    assert rows.select('diseaseFromSource', 'urls').rows() == [
        ('Cardiac pacemaker/device in situ', [{'url': URL.format('G1'), 'niceName': NICE_NAME}]),
        (None, [{'url': None, 'niceName': NICE_NAME}]),
    ]


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        pytest.param('2 [1.77; 2.25]', (2.0, 1.77, 2.25), id='integral_odds_ratio'),
        pytest.param('74.78 [58.86; 95]', (74.78, 58.86, 95.0), id='integral_upper'),
        pytest.param('2.49 [2; 3.11]', (2.49, 2.0, 3.11), id='integral_lower'),
        pytest.param('13.84 [7.67; 25.1]', (13.84, 7.67, 25.1), id='decimals'),
        pytest.param('NA [NA; NA]', (None, None, None), id='na'),
        pytest.param(None, (None, None, None), id='null'),
        pytest.param('2.5 [1.5;3.5]', (None, None, None), id='no_space'),
        pytest.param('2.5 [1.5; 3.5] ', (None, None, None), id='trailing_space'),
    ],
)
def test_odds_ratio_and_ci(tmp_path: Path, text: str | None, expected: tuple[float | None, ...]) -> None:
    rows = _parse(tmp_path, _lof({(LOF, 'OR [95%CI]'): text}))
    assert rows.select(ODDS_COLUMNS).row(0) == expected


def test_odds_ratios_unlike_the_old_regexes_counted(tmp_path: Path) -> None:
    rows = [_lof({'Gene ID Ensembl': gene, (LOF, 'OR [95%CI]'): text}) for gene, text in INTEGRAL_ODDS.items()]
    rows.append(_lof({'Gene ID Ensembl': 'ENSG13', (LOF, 'OR [95%CI]'): '3 [2; 4]'}))
    rows.append(_lof({'Gene ID Ensembl': 'ENSG14', (LOF, 'OR [95%CI]'): '3 [2; 4]', (LOF, 'Meta P-value'): 0.5}))
    # repeated rows are counted once, as the old step dropped duplicates before the cutoff
    _, log = logged(lambda: _parse(tmp_path, _lof(), *rows, rows[0], rows[3]))
    assert '2 CVDI odds ratios and 4 confidence interval bounds (2 lower, 2 upper) differ from the old regexes' in log


def test_methods_in_sorted_order(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _lof({(LOF_M08, 'Meta P-value'): 1e-8, ('Cauchy', 'Cauchy P-value'): 1e-9}),
        _lof({'Gene ID Ensembl': 'ENSG2', ('Cauchy', 'Cauchy P-value'): 1e-9}),
    )
    assert rows.select('statisticalMethod', 'targetFromSourceId').rows() == [
        ('Cauchy', 'ENSG1'),
        ('Cauchy', 'ENSG2'),
        (LOF, 'ENSG1'),
        (LOF, 'ENSG2'),
        (LOF_M08, 'ENSG1'),
    ]


def test_masks_matched_case_insensitively(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _lof({(LOF, 'Meta P-value'): None, (M05, 'Meta P-value'): 5e-8, (M02, 'Meta P-value'): 1e-8}),
        _lof({'Gene ID Ensembl': 'ENSG2', (LOF, 'Meta P-value'): None, (M05, 'Meta P-value'): 6e-8}),
        masks=[*CVDI_MASKS, M02],
        cutoffs=CUTOFFS_WITH_M02,
    )
    overview = (
        'Mixed-effects test carried out with predicted-deleterious missense variants (missense score > {}) '
        'with a MAF smaller than 0.001%.'
    )
    assert rows.select('targetFromSourceId', 'statisticalMethod', 'statisticalMethodOverview').rows() == [
        ('ENSG1', M02, overview.format('0.2')),
        ('ENSG1', M05, overview.format('0.5')),
    ]


def test_masks_matched_only_case_insensitively_counted(tmp_path: Path) -> None:
    _, log = logged(
        lambda: _parse(
            tmp_path,
            _lof({(M05, 'Meta P-value'): 5e-8, (M02, 'Meta P-value'): 1e-8}),
            _lof({'Gene ID Ensembl': 'ENSG2', (M05, 'Meta P-value'): 1e-9}),
            _lof({'Gene ID Ensembl': 'ENSG3', (M05, 'Meta P-value'): 1e-7}),
            _lof({'Gene ID Ensembl': 'ENSG2', (M05, 'Meta P-value'): 1e-9}),
            masks=[*CVDI_MASKS, M02],
            cutoffs=CUTOFFS_WITH_M02,
        )
    )
    assert (
        '3 significant CVDI rows from masks matched to a cutoff only case-insensitively, by mask: '
        'missense0.2 (MAF<0.001%): 1, missense0.5 (MAF<0.001%): 2'
    ) in log


def test_no_case_insensitive_match_counted(tmp_path: Path) -> None:
    _, log = logged(lambda: _parse(tmp_path, _lof()))
    assert '0 significant CVDI rows from masks matched to a cutoff only case-insensitively\n' in log


def test_a_mask_without_a_cutoff_is_dropped(tmp_path: Path) -> None:
    cutoffs = [c for c in CVDI_CUTOFFS if c[1] != LOF_M08]
    rows = _parse(tmp_path, _lof({(LOF_M08, 'Meta P-value'): 1e-9}), cutoffs=cutoffs)
    assert rows['statisticalMethod'].to_list() == [LOF]


def test_duplicates_are_dropped(tmp_path: Path) -> None:
    assert _parse(tmp_path, _lof(), _lof(), _lof({(LOF, 'cMAC'): 12.0})).height == 1


def test_a_sheet_without_rows_gives_an_empty_frame(tmp_path: Path) -> None:
    rows = _parse(tmp_path)
    assert rows.height == 0
    assert rows.schema == GENE_BURDEN_PARSED


PARITY_ROWS = [
    _lof(),
    _lof(),
    _lof({(LOF, 'cMAC'): 12.0, (LOF_M08, 'Meta P-value'): 1e-8, (LOF_M08, 'OR [95%CI]'): '13.84 [7.67; 25.1]'}),
    _lof({'Gene ID Ensembl': 'ENSG2', (LOF, 'OR [95%CI]'): 'NA [NA; NA]', (LOF, 'Meta P-value'): 1e-6}),
    _lof({'Gene ID Ensembl': 'ENSG3', (LOF, 'Meta P-value'): 1e-5, ('Cauchy', 'Cauchy P-value'): 0.0}),
    _lof({
        'Gene ID Ensembl': 'ENSG4',
        (LOF, 'cMAC'): '-',
        (LOF, 'OR [95%CI]'): '-',
        (LOF, 'Meta P-value'): '-',
        ('Cauchy', 'Cauchy P-value'): 1e-12,
    }),
    _lof({'phenotype': 'phenotype', 'Gene ID Ensembl': 'Gene ID Ensembl', 'Gene': 'Gene'}),
    _lof({'Gene ID Ensembl': None}),
    _lof({'Gene ID Ensembl': 'ENSG5', 'Gene': None, 'phenotype': None, (LOF, 'cMAC'): 3e9}),
    _lof({'Gene ID Ensembl': 'ENSG6', (LOF, 'Meta P-value'): '1e-9', (LOF, 'cMAC'): '7.9'}),
    _lof({'Gene ID Ensembl': 'ENSG7', 'phenotype': 'Cardiac_pacemaker/device_in_situ', (LOF, 'Meta P-value'): 5e-320}),
    _lof({'Gene ID Ensembl': 'ENSG8', (LOF, 'cMAC'): 'na', (LOF_M08, 'Meta P-value'): 'NA'}),
    _lof({'Gene ID Ensembl': 'ENSG15', (LOF, 'cMAC'): True, (LOF_M08, 'Meta P-value'): False}),
]


def test_matches_spark_baseline(tmp_path: Path) -> None:
    footer = ['Asthma', 'ENSG9', 'G9', 1, '2.5 [1.5; 3.5]', 1e-20]
    integral = [_lof({'Gene ID Ensembl': g, (LOF, 'OR [95%CI]'): text}) for g, text in INTEGRAL_ODDS.items()]
    lowercase = [
        _lof({'Gene ID Ensembl': 'ENSG20', (LOF, 'Meta P-value'): None, (M05, 'Meta P-value'): 1e-9}),
        _lof({'Gene ID Ensembl': 'ENSG21', (M02, 'Meta P-value'): 1e-9, (M02, 'OR [95%CI]'): '3 [2; 4.5]'}),
    ]
    path = write_cvdi(
        tmp_path,
        [*PARITY_ROWS, *integral, *lowercase],
        CUTOFFS_WITH_M02,
        masks=[*CVDI_MASKS, M02],
        footer=footer,
    )
    everything = parse_cvdi(path).collect()
    added = {M05, M02}
    new = everything.filter(~pl.col('statisticalMethod').is_in(added))

    # expected rows captured from the removed pyspark implementation, ruling 6 applied
    link = [{'niceName': NICE_NAME, 'url': URL.format('G1')}]
    no_link = [{'niceName': NICE_NAME, 'url': None}]
    cauchy_overview = 'Combined test after combining mask-specific using Cauchy distribution.'
    lof_m08_overview = (
        'Mixed-effects test carried out with LOF and predicted-deleterious missense variants (missense score > 0.8) '
        'with a MAF smaller than 0.1%.'
    )
    constant = {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'projectId': 'CVDI Human Disease Portal',
        'cohortId': 'UK Biobank 450k/All of Us/MGB',
        'studySampleSize': 748879,
        'literature': ['39210047'],
    }
    varying = [
        'diseaseFromSource',
        'targetFromSourceId',
        'resourceScore',
        'studyCasesWithQualifyingVariants',
        *ODDS_COLUMNS,
        'statisticalMethod',
        'statisticalMethodOverview',
        'urls',
        'pValueExponent',
        'pValueMantissa',
    ]
    rows = [
        ('Type 2 diabetes', 'ENSG1', 2.5e-09, 12, 2.5, 1.5, 3.5, LOF, LOF_OVERVIEW, link, -9, 2.5),
        ('Type 2 diabetes', 'ENSG2', 1e-06, 12, None, None, None, LOF, LOF_OVERVIEW, link, -7, 10.0),
        (None, 'ENSG5', 2.5e-09, 2147483647, 2.5, 1.5, 3.5, LOF, LOF_OVERVIEW, no_link, -9, 2.5),
        ('Type 2 diabetes', 'ENSG6', 1e-09, 7, 2.5, 1.5, 3.5, LOF, LOF_OVERVIEW, link, -10, 10.0),
        ('Cardiac pacemaker/device in situ', 'ENSG7', 5e-320, 12, 2.5, 1.5, 3.5, LOF, LOF_OVERVIEW, link, -320, 5.0),
        ('Type 2 diabetes', 'ENSG8', 2.5e-09, None, 2.5, 1.5, 3.5, LOF, LOF_OVERVIEW, link, -9, 2.5),
        ('Type 2 diabetes', 'ENSG15', 2.5e-09, 1, 2.5, 1.5, 3.5, LOF, LOF_OVERVIEW, link, -9, 2.5),
        # ruling 6: the old regexes gave (1.77, 1.77, 2.25)
        ('Type 2 diabetes', 'ENSG10', 2.5e-09, 12, 2.0, 1.77, 2.25, LOF, LOF_OVERVIEW, link, -9, 2.5),
        # ruling 6: the old regexes gave (74.78, 58.86, None)
        ('Type 2 diabetes', 'ENSG11', 2.5e-09, 12, 74.78, 58.86, 95.0, LOF, LOF_OVERVIEW, link, -9, 2.5),
        # ruling 6: the old regexes gave (2.49, None, 3.11)
        ('Type 2 diabetes', 'ENSG12', 2.5e-09, 12, 2.49, 2.0, 3.11, LOF, LOF_OVERVIEW, link, -9, 2.5),
        ('Type 2 diabetes', 'ENSG21', 2.5e-09, 12, 2.5, 1.5, 3.5, LOF, LOF_OVERVIEW, link, -9, 2.5),
        ('Type 2 diabetes', 'ENSG3', 0.0, None, None, None, None, 'Cauchy', cauchy_overview, link, None, None),
        ('Type 2 diabetes', 'ENSG4', 1e-12, None, None, None, None, 'Cauchy', cauchy_overview, link, -13, 10.0),
        ('Type 2 diabetes', 'ENSG1', 1e-08, None, 13.84, 7.67, 25.1, LOF_M08, lof_m08_overview, link, -9, 10.0),
        ('Type 2 diabetes', 'ENSG15', 0.0, None, None, None, None, LOF_M08, lof_m08_overview, link, None, None),
    ]
    old_rows = expand(constant, varying, rows)
    columns = list(old_rows[0])

    absent = sorted(set(new.columns) - set(columns))
    assert absent == sorted([
        'allelicRequirements',
        'ancestry',
        'ancestryId',
        'beta',
        'betaConfidenceIntervalLower',
        'betaConfidenceIntervalUpper',
        'curatedDiseaseFromSourceMappedId',
        'diseaseFromSourceId',
        'releaseVersion',
        'sex',
        'studyCases',
    ])
    assert new.select(absent).null_count().row(0) == (new.height,) * len(absent)
    assert multiset(new.rows(named=True), columns) == multiset(old_rows, columns)
    # ruling 8: masks the old step dropped
    extra = everything.filter(pl.col('statisticalMethod').is_in(added))
    assert extra.select('targetFromSourceId', 'statisticalMethod', *ODDS_COLUMNS).rows() == [
        ('ENSG21', M02, 3.0, 2.0, 4.5),
        ('ENSG20', M05, None, None, None),
    ]
