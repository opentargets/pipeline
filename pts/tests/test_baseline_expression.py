"""Tests for the baseline expression transformer."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pts.transformers.baseline_expression import (
    BaselineExpressionFlags,
    build,
    normalise_label,
    parent_by_biosample,
    parent_by_label,
    prettify_label,
)

PROTEIN = pl.List(pl.Struct({'id': pl.String, 'source': pl.String}))


# Outputs of the spark job's `pretty` on labels from the curated parent files.
@pytest.mark.parametrize(
    ('label', 'expected'),
    [
        ('AdiposeVisceralOmentum', 'adipose visceral omentum'),
        ('BrainFrontalCortexBA9', 'brain frontal cortex BA9'),
        ('BRAINANTERIORCINGULATECORTEXBA24', 'BRAINANTERIORCINGULATECORTEXBA24'),
        ('ArteryTibial', 'artery tibial'),
        ('B cell, naive', 'B cell naive'),
        ('CD4-positive_T cell', 'CD4 positive T cell'),
        ('lung2x', 'lung 2 x'),
        ('Brain  Cortex', 'brain cortex'),
        ('HTMLParser', 'HTML parser'),
        ('', ''),
    ],
)
def test_prettify_label(label: str, expected: str) -> None:
    assert pl.select(prettify_label(pl.lit(label))).item() == expected


def test_normalise_label() -> None:
    assert pl.select(normalise_label(pl.lit('Gall_Bladder (2)'))).item() == 'gallbladder2'


def test_parent_by_biosample_prefers_the_most_specific_candidate() -> None:
    biosample = pl.DataFrame({
        'biosampleId': ['ROOT', 'ORGAN', 'PART', 'OTHER'],
        'ancestors': [[], ['ROOT'], ['ORGAN', 'ROOT'], None],
    })
    parents = parent_by_biosample(biosample, ['ROOT', 'ORGAN', 'MISSING']).sort('biosampleId')
    expected = pl.DataFrame({'biosampleId': ['ORGAN', 'PART', 'ROOT'], 'parentId': ['ORGAN', 'ORGAN', 'ROOT']})
    assert_frame_equal(parents, expected)


def test_parent_by_label_takes_smallest_parent_and_leftmost_spelling() -> None:
    parents = pl.DataFrame({
        'BiosampleId': ['UBERON_2', 'UBERON_1'],
        'Label': ['Gall bladder', 'Liver'],
        'GTEx': ['Gallbladder', 'Liver; Gall_bladder'],
    })
    labels = parent_by_label(parents).sort('_norm')
    expected = pl.DataFrame({
        '_norm': ['gallbladder', 'liver'],
        '_labelParentId': ['UBERON_1', 'UBERON_1'],
        '_pretty': ['gall bladder', 'liver'],
    })
    assert_frame_equal(labels, expected)


def _target() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            'id': ['ENSG1', 'ENSG2', 'ENSG3'],
            'proteinIds': [
                [{'id': 'P1', 'source': 'uniprot'}],
                [{'id': 'P2', 'source': 'uniprot'}],
                [{'id': 'P2', 'source': 'uniprot'}],
            ],
            'approvedSymbol': ['GENE1', None, 'GENE3'],
        },
        schema={'id': pl.String, 'proteinIds': PROTEIN, 'approvedSymbol': pl.String},
    )


def _biosample() -> pl.DataFrame:
    return pl.DataFrame({
        'biosampleId': ['UBERON_LIVER', 'UBERON_LOBE', 'UBERON_OTHER', 'CL_T'],
        'ancestors': [[], ['UBERON_LIVER'], [], []],
    })


def test_build_resolves_and_flags() -> None:
    expression = pl.LazyFrame({
        'targetId': ['GENE1', 'ENSG9', None, 'ENSG1'],
        'tissueBiosampleId': ['UBERON_LOBE', 'UBERON_LOBE', 'UBERON_OTHER', None],
        'celltypeBiosampleId': [None, None, None, 'CL_NOPE'],
        'datasourceId': ['GTEx', 'GTEx', 'PRIDE', 'DICE'],
        'tissueBiosampleFromSource': ['Liver lobe', 'Liver lobe', 'LIVER', None],
        'celltypeBiosampleFromSource': [None, None, None, 'T cell'],
        'targetFromSourceId': [None, None, 'P2', None],
        'median': [1.0, 2.0, 3.0, 4.0],
    })
    tissue = pl.DataFrame({'BiosampleId': ['UBERON_LIVER'], 'Label': ['Liver']})
    celltype = pl.DataFrame({'BiosampleId': ['CL_T'], 'Label': ['T-cell']})

    result = build(expression, _target(), _biosample(), tissue, celltype).collect().sort('median', 'targetId')

    assert result.get_column('targetId').to_list() == ['ENSG1', 'ENSG9', 'ENSG2', 'ENSG3', 'ENSG1']
    assert result.get_column('tissueBiosampleParentId').to_list() == [
        'UBERON_LIVER',
        'UBERON_LIVER',
        'UBERON_LIVER',
        'UBERON_LIVER',
        None,
    ]
    assert result.get_column('tissueBiosampleFromSource').to_list() == [
        'Liver lobe',
        'Liver lobe',
        'liver',
        'liver',
        None,
    ]
    assert result.get_column('celltypeBiosampleParentId').to_list() == [None, None, None, None, 'CL_T']
    assert result.get_column('celltypeBiosampleFromSource').to_list() == [None, None, None, None, 'T cell']
    assert result.get_column('qualityControls').to_list() == [
        [],
        [BaselineExpressionFlags.INVALID_TARGET.value],
        [],
        [],
        [BaselineExpressionFlags.INVALID_BIOSAMPLE.value],
    ]
    assert result.columns[:3] == ['celltypeBiosampleId', 'tissueBiosampleId', 'targetId']
    assert result.columns[-1] == 'qualityControls'
