"""Tests for study deconvolution."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from polars.testing import assert_frame_equal

from pts.transformers.study import QualityFlag
from pts.transformers.study_deconvolution import _deconvolute_studies, study_deconvolution
from pts.transformers.utils.dataset import scan_dataset
from tests.study_fixtures import studies_frame


def test_deconvolute_studies_merges_flags_sumstats_and_granular_disease():
    """Duplicate sources propagate QC and select the most granular disease assignment."""
    studies = studies_frame([
        {
            'studyId': 's',
            'studyType': 'gwas',
            'hasSumstats': False,
            'qualityControls': [QualityFlag.NO_OT_CURATION, QualityFlag.SUMSTATS_NOT_AVAILABLE, 'top'],
            'traitFromSourceMappedIds': ['a', 'b'],
            'traitFromSource': 'broad',
            'analysisFlags': ['one'],
        },
        {
            'studyId': 's',
            'studyType': 'gwas',
            'hasSumstats': True,
            'qualityControls': [QualityFlag.NO_OT_CURATION, 'summary'],
            'traitFromSourceMappedIds': ['a'],
            'traitFromSource': 'specific',
            'analysisFlags': ['two'],
            'sumstatQCValues': [{'QCCheckName': 'test', 'QCCheckValue': 1.0}],
        },
    ])
    result = _deconvolute_studies(studies)
    assert isinstance(result, pl.LazyFrame)
    row = result.collect().row(0, named=True)
    assert set(row['qualityControls']) == {QualityFlag.NO_OT_CURATION, 'top', 'summary'}
    assert row['hasSumstats'] is True
    assert row['traitFromSourceMappedIds'] == ['a']
    assert row['traitFromSource'] == 'specific'
    assert sorted(row['analysisFlags']) == ['one', 'two']
    assert row['sumstatQCValues'] == [{'QCCheckName': 'test', 'QCCheckValue': 1.0}]


@pytest.mark.parametrize(
    ('sumstats', 'expected'), [([None, None], None), ([False, None], False), ([True, False], True)]
)
def test_deconvolute_studies_propagates_tristate_sumstats(sumstats, expected):
    """Summary statistics availability prefers true, then false, then null."""
    result = _deconvolute_studies(
        studies_frame([{'studyId': 's', 'studyType': 'gwas', 'hasSumstats': value} for value in sumstats])
    ).collect()
    assert result.height == 1
    assert result['hasSumstats'].to_list() == [expected]
    assert result['qualityControls'].to_list() == [[]]
    assert result['analysisFlags'].to_list() == [[]]


def test_deconvolute_studies_without_sumstats_removes_curation_flag():
    """Top-hit studies retain unavailable-sumstats flags but drop absent curation."""
    result = _deconvolute_studies(
        studies_frame([
            {
                'studyId': 's',
                'studyType': 'gwas',
                'hasSumstats': False,
                'qualityControls': [QualityFlag.NO_OT_CURATION, QualityFlag.SUMSTATS_NOT_AVAILABLE],
            }
        ])
    ).collect()
    assert result['qualityControls'].to_list() == [[QualityFlag.SUMSTATS_NOT_AVAILABLE]]


def test_deconvolute_studies_combines_ambiguous_types_and_keeps_distinct_studies():
    """Different study types remain visible to validation and separate IDs survive."""
    result = _deconvolute_studies(
        studies_frame([
            {'studyId': 's', 'studyType': 'gwas'},
            {'studyId': 's', 'studyType': 'eqtl'},
            {'studyId': 'other', 'studyType': 'gwas'},
        ])
    ).collect()
    assert result.height == 2
    assert set(result.filter(pl.col('studyId') == 's')['studyType'][0].split(',')) == {'gwas', 'eqtl'}


@pytest.mark.parametrize('unmapped', [None, []], ids=['null', 'empty'])
def test_deconvolute_studies_prefers_records_with_mapped_traits(unmapped):
    """A record without mapped traits must not win granularity over a populated one."""
    result = _deconvolute_studies(
        studies_frame([
            {'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': ['a', 'b'], 'traitFromSource': 'broad'},
            {'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': ['a'], 'traitFromSource': 'specific'},
            {'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': unmapped, 'traitFromSource': 'unmapped'},
        ])
    ).collect()
    assert result['traitFromSourceMappedIds'].to_list() == [['a']]
    assert result['traitFromSource'].to_list() == ['specific']


def test_deconvolute_studies_without_any_mapped_traits_keeps_every_candidate():
    """With no mapped traits anywhere, every record stays a candidate, as before."""
    result = _deconvolute_studies(
        studies_frame([
            {'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': None, 'traitFromSource': 'first'},
            {'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': None, 'traitFromSource': 'last'},
        ])
    ).collect()
    assert result['traitFromSourceMappedIds'].to_list() == [None]
    assert result['traitFromSource'].to_list() == ['last']


@pytest.mark.parametrize('seed', [1, 7, 42])
def test_deconvolution_is_independent_of_row_order(seed):
    """Explicit source ranks and canonical content break otherwise conflicting ties."""
    rows = (
        studies_frame([
            {
                'studyId': 's',
                'projectId': 'z',
                'traitFromSource': 'z',
                'traitFromSourceMappedIds': ['b'],
                'studyType': 'gwas',
                'qualityControls': ['z', 'a'],
                'analysisFlags': ['z', 'a', 'a'],
            },
            {
                'studyId': 's',
                'projectId': 'a',
                'traitFromSource': 'a',
                'traitFromSourceMappedIds': ['a'],
                'studyType': 'no_licence',
                'qualityControls': ['a', 'z'],
                'analysisFlags': ['a'],
            },
            {
                'studyId': 's',
                'projectId': 'preferred',
                'traitFromSource': 'last source',
                'traitFromSourceMappedIds': ['c'],
                'studyType': 'gwas',
            },
        ])
        .with_columns(pl.Series('_source_rank', [0, 0, 1]))
        .collect()
    )
    expected = _deconvolute_studies(rows.lazy()).collect()
    actual = _deconvolute_studies(rows.sample(fraction=1, shuffle=True, seed=seed).lazy()).collect()
    assert_frame_equal(actual, expected)
    row = actual.row(0, named=True)
    assert row['projectId'] == 'z'
    assert row['traitFromSource'] == 'last source'
    assert row['traitFromSourceMappedIds'] == ['c']
    assert row['studyType'] == 'gwas,no_licence'
    assert row['qualityControls'] == ['a', 'z']
    assert row['analysisFlags'] == ['a', 'a', 'a', 'z']
    assert '_source_rank' not in actual.columns


@pytest.mark.parametrize('order', [['first', 'second'], ['second', 'first']])
def test_study_deconvolution_prefers_the_first_listed_source(tmp_path, order):
    """Records of one study from several sources merge into one, taking conflicting fields from the first source."""
    paths = {}
    for project in ['first', 'second']:
        path = tmp_path / f'{project}.parquet'
        studies_frame([{'studyId': 's', 'studyType': 'gwas', 'projectId': project}]).collect().write_parquet(path)
        paths[project] = str(path)
    output_path = tmp_path / 'output'

    study_deconvolution(
        source={'study': [paths[project] for project in order]},
        destination=str(output_path),
        settings={},
        config=Mock(spec=Config),
    )

    result = scan_dataset(str(output_path)).collect()
    assert result['studyId'].to_list() == ['s']
    assert result['projectId'].to_list() == [order[0]]
    assert '_source_rank' not in result.columns
