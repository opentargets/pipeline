"""Tests of lazy variant annotation transformations."""

import hashlib

import polars as pl
import pytest

from pts.schemas.variant import variant_schema
from pts.transformers.utils.variant import hash_variant_index_ids
from pts.transformers.variant_utils.annotations import (
    add_annotation,
    annotate_with_amino_acid_consequences,
    normalize_variant_effects,
)


@pytest.fixture
def doctest_namespace():
    """Avoid constructing a Spark session for Polars tests."""
    return {}


def effect(method='VEP', score=1.0, assessment=None):
    """Build one predictor assessment."""
    return {
        'method': method,
        'assessment': assessment,
        'score': score,
        'assessmentFlag': None,
        'targetId': None,
        'normalisedScore': None,
    }


@pytest.mark.parametrize(
    ('method', 'score', 'assessment', 'expected'),
    [
        ('LOFTEE', None, 'HC', 1.0),
        ('LOFTEE', None, 'LC', 0.85),
        ('LOFTEE', 1.0, 'other', None),
        ('SIFT', 0.0, 'deleterious', 1.0),
        ('SIFT', 0.05, 'deleterious', 0.5),
        ('SIFT', 0.0, 'deleterious_low_confidence', 0.5),
        ('SIFT', 1.0, 'tolerated_low_confidence', -0.5),
        ('SIFT', 1.0, 'tolerated', -1.0),
        ('SIFT', 1.0, 'deleterious', None),
        ('PolyPhen', 0.0, 'benign', -1.0),
        ('PolyPhen', 0.446, 'benign', -0.25),
        ('PolyPhen', 0.908, 'probably_damaging', 0.25),
        ('PolyPhen', 1.0, 'unknown', None),
        ('AlphaMissense', 0.0, None, -1.0),
        ('AlphaMissense', 0.06, None, -0.25),
        ('AlphaMissense', 0.77, None, 0.25),
        ('AlphaMissense', 1.0, None, 1.0),
        ('CADD', 0.0, None, -1.0),
        ('CADD', 10.0, None, 0.0),
        ('CADD', 20.0, None, 0.5),
        ('CADD', 30.0, None, 0.75),
        ('CADD', 81.0, None, 1.0),
        ('CADD', 132.0, None, 1.25),
        ('Pangolin', -0.14, None, 0.5),
        ('Pangolin', -1.0, None, 1.0),
        ('LossOfFunctionCuration', None, 'lof', 1.0),
        ('LossOfFunctionCuration', None, 'likely_lof', 0.5),
        ('LossOfFunctionCuration', None, 'uncertain', 0.0),
        ('LossOfFunctionCuration', None, 'likely_not_lof', -0.5),
        ('LossOfFunctionCuration', None, 'not_lof', -1.0),
        ('SpliceAI', 0.2, None, 0.2),
        ('VEP', 0.4, None, 0.4),
        ('GERP', -5.0, None, 0.0),
        ('GERP', -3.0, None, 0.0),
        ('GERP', 0.0, None, 0.25),
        ('GERP', 2.0, None, 0.75),
        ('GERP', 6.0, None, 1.0),
        ('GERP', 8.0, None, 1.0),
        ('FoldX', -2.0, None, 1.0),
        ('FoldX', 1.0, None, 0.5),
        ('unknown', 1.0, None, None),
        ('CADD', None, None, None),
    ],
)
def test_normalization(method, score, assessment, expected):
    frame = pl.DataFrame(
        {'variantEffect': [[effect(method, score, assessment)]]},
        schema={'variantEffect': variant_schema['variantEffect']},
    ).lazy()
    result = normalize_variant_effects(frame).collect()['variantEffect'][0][0]
    if expected is None:
        assert result['normalisedScore'] is None
    else:
        assert result['normalisedScore'] == pytest.approx(expected, abs=1e-6)
    assert result['method'] == method
    assert result['assessment'] == assessment


def test_null_and_empty_predictors():
    frame = pl.DataFrame({'variantEffect': [None, []]}, schema={'variantEffect': variant_schema['variantEffect']})
    assert normalize_variant_effects(frame.lazy()).collect()['variantEffect'].to_list() == [None, []]


def test_identifier_hashing():
    frame = pl.DataFrame({
        'variantId': ['short', 'abcdefghij', 'missing', None],
        'chromosome': ['1', '2', None, None],
        'position': [1, 2, 3, None],
    })
    result = frame.select(
        hash_variant_index_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), 10).alias('variantId')
    )['variantId'].to_list()
    assert result == [
        'short',
        'OTVAR_2_2_' + hashlib.md5(b'abcdefghij').hexdigest(),
        'OTVAR_' + hashlib.md5(b'missing').hexdigest(),
        None,
    ]


def test_annotation_merge():
    left = pl.DataFrame(
        {
            'variantId': ['a', 'b', None],
            'rsIds': [['rs1', 'rs1'], None, []],
            'variantDescription': ['old', None, None],
            'chromosome': ['1', None, None],
        },
        schema={
            'variantId': pl.String,
            'rsIds': pl.List(pl.String),
            'variantDescription': pl.String,
            'chromosome': pl.String,
        },
    )
    right = pl.DataFrame({
        'variantId': ['a', 'a', 'b', None],
        'rsIds': [['rs2', 'rs1'], [], ['rs3'], ['no']],
        'variantDescription': ['new', None, None, 'no'],
        'chromosome': ['2', '2', '3', '4'],
    })
    result = add_annotation(left.lazy(), right.lazy()).collect()
    assert result.height == 4
    assert result['rsIds'].to_list() == [['rs1', 'rs2'], ['rs1'], ['rs3'], []]
    assert result['chromosome'].to_list() == ['1', '1', '3', None]
    assert result['variantDescription'].to_list() == ['old new', 'old', '', '']


def test_struct_annotation_union_order():
    dtype = variant_schema['dbXrefs']
    left = pl.DataFrame(
        {'variantId': ['a'], 'dbXrefs': [[{'id': 'a', 'source': 'x'}, {'id': 'c', 'source': 'x'}]]},
        schema={'variantId': pl.String, 'dbXrefs': dtype},
    )
    right = pl.DataFrame(
        {'variantId': ['a'], 'dbXrefs': [[{'id': 'b', 'source': 'x'}, {'id': 'c', 'source': 'x'}]]},
        schema={'variantId': pl.String, 'dbXrefs': dtype},
    )
    assert [item['id'] for item in add_annotation(left.lazy(), right.lazy()).collect()['dbXrefs'][0]] == ['c', 'a', 'b']


def test_amino_acid_annotation():
    transcript_type = variant_schema['transcriptConsequences']
    assert isinstance(transcript_type, pl.List) and isinstance(transcript_type.inner, pl.Struct)
    fields = transcript_type.inner.fields

    def transcript(change, accessions):
        return {
            **dict.fromkeys([field.name for field in fields]),
            'aminoAcidChange': change,
            'uniprotAccessions': accessions,
        }

    frame = pl.DataFrame(
        {
            'variantId': ['a', 'b'],
            'variantEffect': [[effect()], []],
            'transcriptConsequences': [[transcript(None, ['ignored']), transcript('A/T', ['P1', 'P2'])], []],
        },
        schema={
            'variantId': pl.String,
            'variantEffect': variant_schema['variantEffect'],
            'transcriptConsequences': variant_schema['transcriptConsequences'],
        },
    )
    annotation = pl.DataFrame(
        {
            'aminoAcidChange': ['A/T', 'A/T'],
            'uniprotAccession': ['P1', 'P2'],
            'variantEffect': [[effect('FoldX')], [effect('FoldX'), effect('AlphaMissense')]],
        },
        schema={
            'aminoAcidChange': pl.String,
            'uniprotAccession': pl.String,
            'variantEffect': variant_schema['variantEffect'],
        },
    )
    result = annotate_with_amino_acid_consequences(frame.lazy(), annotation.lazy()).collect().sort('variantId')
    assert result['variantEffect'].list.len().to_list() == [3, 0]
    assert [item['method'] for item in result['variantEffect'][0]] == ['VEP', 'FoldX', 'AlphaMissense']

    # Predictions for substitutions no variant carries, or with null keys, change nothing.
    unrelated = pl.DataFrame(
        {
            'aminoAcidChange': ['C/G', 'A/T', None],
            'uniprotAccession': ['P1', 'P9', 'P1'],
            'variantEffect': [[effect('FoldX')]] * 3,
        },
        schema=annotation.schema,
    )
    padded = annotate_with_amino_acid_consequences(frame.lazy(), pl.concat([annotation, unrelated]).lazy())
    assert padded.collect().sort('variantId').equals(result)


def test_null_array_and_null_key_annotations():
    """Null join keys never match and missing arrays remain null until initialization."""
    frame = pl.DataFrame(
        {'variantId': ['a', None], 'rsIds': [None, None]},
        schema={'variantId': pl.String, 'rsIds': pl.List(pl.String)},
    )
    annotation = pl.DataFrame(
        {'variantId': ['a', None], 'rsIds': [None, ['rs1']]},
        schema={'variantId': pl.String, 'rsIds': pl.List(pl.String)},
    )
    assert add_annotation(frame.lazy(), annotation.lazy()).collect()['rsIds'].to_list() == [None, None]


def test_hash_uses_character_length():
    """Identifier thresholds count characters, not UTF-8 bytes."""
    frame = pl.DataFrame({'variantId': ['éé'], 'chromosome': ['X'], 'position': [4]})
    assert (
        frame.select(
            hash_variant_index_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), 3).alias('variantId')
        )['variantId'][0]
        == 'éé'
    )
    assert frame.select(
        hash_variant_index_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), 2).alias('variantId')
    )['variantId'][0] == ('OTVAR_X_4_' + hashlib.md5('éé'.encode()).hexdigest())
