"""FoldX transformation semantics and real dataset input/output."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from pts.schemas.amino_acid_variants import amino_acid_variants_schema
from pts.transformers.foldx import (
    FoldxSettings,
    aggregate_foldx_predictions,
    filter_foldx_predictions,
    foldx,
    normalise_foldx_score,
)


@pytest.fixture
def doctest_namespace():
    """These Polars tests do not need the shared Spark doctest session."""
    return {}


def predictions(rows):
    """Build raw predictions with explicit source types and useful defaults."""
    defaults = {
        'protein_acc': 'P1',
        'position': 10,
        'wild_type': 'A',
        'mutated_type': 'V',
        'plddt': 0.9,
        'foldx_ddg': 1.0,
    }
    return pl.DataFrame(
        [defaults | row for row in rows],
        schema={
            'protein_acc': pl.String,
            'position': pl.Int64,
            'wild_type': pl.String,
            'mutated_type': pl.String,
            'plddt': pl.Float64,
            'foldx_ddg': pl.Float64,
        },
    ).lazy()


def transform(frame, threshold=0.7):
    return frame.pipe(filter_foldx_predictions, threshold).pipe(aggregate_foldx_predictions)


def test_ambiguity_uses_low_confidence_rows_and_excludes_whole_protein():
    """A filtered-out conflicting residue still invalidates other sites in its protein."""
    frame = predictions([
        {'plddt': 0.9},
        {'plddt': 0.2},
        {'position': 20},
        {'protein_acc': 'P2'},
        {'protein_acc': 'P2', 'mutated_type': 'L'},
    ])
    result = transform(frame).collect()
    assert result['uniprotAccession'].to_list() == ['P2', 'P2']
    assert sorted(result['aminoAcidChange'].to_list()) == ['A10L', 'A10V']


def test_null_confidence_is_not_a_distinct_conflicting_value():
    """Spark collect_set ignores nulls when deciding whether a protein is ambiguous."""
    result = transform(predictions([{'plddt': None}, {'plddt': 0.9}, {'plddt': 0.9}])).collect()
    assert result.height == 1
    assert len(result['variantEffect'][0]) == 1


@pytest.mark.parametrize(('confidence', 'kept'), [(0.7, False), (0.70001, True), (0.69, False), (None, False)])
def test_confidence_threshold_is_strict(confidence, kept):
    assert transform(predictions([{'plddt': confidence}])).collect().height == int(kept)


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


def test_float32_cast_precedes_deduplication_and_missing_scores_are_retained():
    frame = predictions([{'foldx_ddg': 0.1}, {'foldx_ddg': 0.1 + 1e-10}, {'foldx_ddg': None}])
    result = transform(frame).collect()
    assert result.schema == amino_acid_variants_schema
    effects = result['variantEffect'][0].to_list()
    assert len(effects) == 2
    assert all(effect['method'] == 'FoldX' for effect in effects)
    assert all(effect['assessment'] is effect['assessmentFlag'] is effect['targetId'] is None for effect in effects)
    assert effects[0]['score'] is effects[0]['normalisedScore'] is None
    assert effects[1]['normalisedScore'] == effects[1]['score'] / 2


@pytest.mark.parametrize('seed', [1, 42])
def test_effect_lists_are_deterministic_and_duplicate_scores_collapse(seed):
    frame = predictions([{'foldx_ddg': value} for value in [3, 1, -1, 3]])
    expected = transform(frame).collect()
    actual = transform(frame.collect().sample(fraction=1, shuffle=True, seed=seed).lazy()).collect()
    assert_frame_equal(expected, actual)
    assert [e['score'] for e in actual['variantEffect'][0].to_list()] == [-1, 1, 3]


def test_empty_output_preserves_schema():
    result = transform(predictions([{'plddt': 0.1}])).collect()
    assert result.is_empty()
    assert result.schema == amino_acid_variants_schema


def test_null_accessions_do_not_match_anti_join_and_null_substitutions_survive():
    """Null join keys do not match, and concat propagates null source components."""
    result = transform(
        predictions([
            {'protein_acc': None, 'plddt': 0.8},
            {'protein_acc': None, 'plddt': 0.9},
            {'protein_acc': 'P2', 'mutated_type': None},
        ])
    ).collect()
    assert result.height == 2
    assert result.filter(pl.col('uniprotAccession').is_null()).height == 1
    assert result.filter(pl.col('uniprotAccession') == 'P2')['aminoAcidChange'][0] is None


def test_real_step_writes_annotation_dataset(tmp_path):
    source = tmp_path / 'raw.parquet'
    predictions([{}, {}, {'protein_acc': 'low', 'plddt': 0.1}]).collect().write_parquet(source)
    destination = tmp_path / 'annotations'
    foldx(str(source), str(destination), {'plddt_threshold': 0.7}, Mock(spec=Config))
    result = pl.read_parquet(destination / '*.parquet')
    assert result.schema == amino_acid_variants_schema
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
    result = transform(frame).collect()
    assert result['uniprotAccession'].to_list() == ['P2']
    assert result['aminoAcidChange'].to_list() == ['A002V']
    assert result['variantEffect'][0][0]['normalisedScore'] == 0.625
