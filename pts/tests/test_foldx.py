"""FoldX transformation semantics and real dataset input/output."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from pts.schemas.variant_annotation import amino_acid_variant_schema
from pts.transformers.foldx import (
    FoldxSettings,
    aggregate_foldx_predictions,
    filter_foldx_predictions,
    foldx,
    normalise_foldx_score,
)


def test_conflicting_confidence_excludes_entire_protein_before_threshold_filtering():
    """A filtered-out conflicting residue still invalidates other sites in its protein."""
    frame = pl.DataFrame({
        'protein_acc': ['P1', 'P1', 'P1', 'P2', 'P2'],
        'position': [10, 10, 20, 10, 10],
        'wild_type': ['A', 'A', 'A', 'A', 'A'],
        'mutated_type': ['V', 'V', 'V', 'V', 'L'],
        'plddt': [0.9, 0.2, 0.9, 0.9, 0.9],
        'foldx_ddg': [1.0, 1.0, 1.0, 1.0, 1.0],
    }).lazy()
    result = frame.pipe(filter_foldx_predictions, 0.7).pipe(aggregate_foldx_predictions).collect()
    assert result['uniprotAccession'].to_list() == ['P2', 'P2']
    assert sorted(result['aminoAcidChange'].to_list()) == ['A10L', 'A10V']


def test_null_confidence_is_not_a_distinct_conflicting_value():
    """Spark collect_set ignores nulls when deciding whether a protein is ambiguous."""
    frame = pl.DataFrame({
        'protein_acc': ['P1', 'P1', 'P1'],
        'position': [10, 10, 10],
        'wild_type': ['A', 'A', 'A'],
        'mutated_type': ['V', 'V', 'V'],
        'plddt': [None, 0.9, 0.9],
        'foldx_ddg': [1.0, 1.0, 1.0],
    }).lazy()
    result = frame.pipe(filter_foldx_predictions, 0.7).pipe(aggregate_foldx_predictions).collect()
    assert result.height == 1
    assert len(result['variantEffect'][0]) == 1


@pytest.mark.parametrize(
    ('confidence', 'threshold', 'expected_rows'),
    [
        pytest.param(0.7, 0.7, 0, id='at-threshold'),
        pytest.param(0.70001, 0.7, 1, id='above-threshold'),
        pytest.param(0.69, 0.7, 0, id='below-threshold'),
        pytest.param(None, 0.7, 0, id='null-confidence'),
        pytest.param(0.6, 0.5, 1, id='custom-threshold'),
        pytest.param(0.6, 0.6, 0, id='at-custom-threshold'),
    ],
)
def test_confidence_threshold_is_strict(confidence, threshold, expected_rows):
    frame = pl.DataFrame(
        {
            'protein_acc': ['P1'],
            'position': [10],
            'wild_type': ['A'],
            'mutated_type': ['V'],
            'plddt': [confidence],
            'foldx_ddg': [1.0],
        },
        schema_overrides={'plddt': pl.Float64},
    ).lazy()
    result = frame.pipe(filter_foldx_predictions, threshold).pipe(aggregate_foldx_predictions).collect()
    assert result.height == expected_rows
    assert result.schema == amino_acid_variant_schema


@pytest.mark.parametrize(
    ('score', 'expected'),
    [(-3, 1), (-2, 1), (-1, 0.5), (0, 0), (1, 0.5), (2, 1), (3, 1), (None, None), (float('nan'), 1)],
)
def test_score_normalisation(score, expected):
    result = pl.DataFrame({'score': [score]}, schema={'score': pl.Float32}).select(
        normalise_foldx_score(pl.col('score')).alias('normalised')
    )
    assert result['normalised'][0] == expected
    assert result.schema['normalised'] == pl.Float64


def test_f_precision_dont_affect_duplicate():
    frame = pl.DataFrame({
        'protein_acc': ['P1', 'P1'],
        'position': [10, 10],
        'wild_type': ['A', 'A'],
        'mutated_type': ['V', 'V'],
        'plddt': [0.9, 0.9],
        'foldx_ddg': [0.1, 0.1 + 1e-10],
    }).lazy()
    result = frame.pipe(filter_foldx_predictions, 0.7).pipe(aggregate_foldx_predictions).collect()
    assert result.height == 1
    effects = result['variantEffect'][0].to_list()
    assert len(effects) == 1
    assert effects[0]['score'] == pytest.approx(0.1)


def test_null_score():
    frame = pl.DataFrame({
        'protein_acc': ['P1', 'P1'],
        'position': [10, 10],
        'wild_type': ['A', 'A'],
        'mutated_type': ['V', 'V'],
        'plddt': [0.9, 0.9],
        'foldx_ddg': [0.1, None],
    }).lazy()
    result = frame.pipe(filter_foldx_predictions, 0.7).pipe(aggregate_foldx_predictions).collect()
    assert result.height == 1
    effects = result['variantEffect'][0].to_list()
    assert len(effects) == 2
    assert effects[0]['score'] is effects[0]['normalisedScore'] is None
    assert effects[1]['score'] == pytest.approx(0.1)


@pytest.mark.parametrize(
    'scores',
    [
        pytest.param([3.0, 1.0, -1.0, 3.0], id='duplicates-apart'),
        pytest.param([-1.0, 3.0, 3.0, 1.0], id='duplicates-together'),
        pytest.param([1.0, 3.0, -1.0, 3.0], id='mixed-order'),
    ],
)
def test_effect_lists_are_deterministic_and_duplicate_scores_collapse(scores):
    frame = pl.DataFrame({
        'protein_acc': ['P1', 'P1', 'P1', 'P1'],
        'position': [10, 10, 10, 10],
        'wild_type': ['A', 'A', 'A', 'A'],
        'mutated_type': ['V', 'V', 'V', 'V'],
        'plddt': [0.9, 0.9, 0.9, 0.9],
        'foldx_ddg': scores,
    }).lazy()
    actual = frame.pipe(filter_foldx_predictions, 0.7).pipe(aggregate_foldx_predictions).collect()
    expected = pl.DataFrame(
        {
            'uniprotAccession': ['P1'],
            'aminoAcidChange': ['A10V'],
            'variantEffect': [
                [
                    {
                        'method': 'FoldX',
                        'assessment': None,
                        'score': -1.0,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 0.5,
                    },
                    {
                        'method': 'FoldX',
                        'assessment': None,
                        'score': 1.0,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 0.5,
                    },
                    {
                        'method': 'FoldX',
                        'assessment': None,
                        'score': 3.0,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 1.0,
                    },
                ]
            ],
        },
        schema=amino_acid_variant_schema,
    )
    assert_frame_equal(expected, actual)


def test_null_accessions_do_not_match_anti_join_and_null_substitutions_survive():
    """Null join keys do not match, and concat propagates null source components."""
    frame = pl.DataFrame({
        'protein_acc': [None, None, 'P2'],
        'position': [10, 10, 10],
        'wild_type': ['A', 'A', 'A'],
        'mutated_type': ['V', 'V', None],
        'plddt': [0.8, 0.9, 0.9],
        'foldx_ddg': [1.0, 1.0, 1.0],
    }).lazy()
    result = frame.pipe(filter_foldx_predictions, 0.7).pipe(aggregate_foldx_predictions).collect()
    assert result.height == 2
    assert result.filter(pl.col('uniprotAccession').is_null()).height == 1
    assert result.filter(pl.col('uniprotAccession') == 'P2')['aminoAcidChange'][0] is None


def test_real_step_writes_annotation_dataset(tmp_path):
    source = tmp_path / 'raw.parquet'
    pl.DataFrame({
        'protein_acc': ['P1', 'P1', 'low'],
        'position': [10, 10, 10],
        'wild_type': ['A', 'A', 'A'],
        'mutated_type': ['V', 'V', 'V'],
        'plddt': [0.9, 0.9, 0.1],
        'foldx_ddg': [1.0, 1.0, 1.0],
    }).write_parquet(source)
    destination = tmp_path / 'annotations'
    foldx(str(source), str(destination), {'plddt_threshold': 0.7}, Mock(spec=Config))
    result = pl.read_parquet(destination / '*.parquet')
    assert result.schema == amino_acid_variant_schema
    assert result['uniprotAccession'].to_list() == ['P1']
    assert result['aminoAcidChange'].to_list() == ['A10V']
    assert result['variantEffect'][0][0]['normalisedScore'] == 0.5


@pytest.mark.parametrize(
    'settings', [{'unknown': 1}, {'plddt_threshold': float('nan')}, {'plddt_threshold': float('inf')}]
)
def test_settings_reject_unknown_options_and_nonfinite_thresholds(settings):
    with pytest.raises(ValidationError):
        FoldxSettings.model_validate(settings)


def test_raw_export_string_types_preserve_ambiguity_and_numeric_filtering():
    """Raw Parquet stores numbers as strings; ambiguity still compares source strings."""
    frame = pl.DataFrame({
        'protein_acc': ['P1', 'P1', 'P2', 'P3'],
        'position': ['1', '1', '002', '3'],
        'wild_type': ['A'] * 4,
        'mutated_type': ['V'] * 4,
        'plddt': ['0.8', '0.80', '0.9', 'invalid'],
        'foldx_ddg': ['1', '1', '1.25', '1'],
    }).lazy()
    result = frame.pipe(filter_foldx_predictions, 0.7).pipe(aggregate_foldx_predictions).collect()
    assert result['uniprotAccession'].to_list() == ['P2']
    assert result['aminoAcidChange'].to_list() == ['A002V']
    assert result['variantEffect'][0][0]['normalisedScore'] == 0.625
