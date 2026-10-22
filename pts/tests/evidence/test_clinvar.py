"""Tests for ClinVar and ClinVar somatic evidence."""

from __future__ import annotations

import polars as pl
import pytest

from pts.transformers.evidence.clinvar import DIRECTION_ON_TARGET, DIRECTION_ON_TRAIT, SCORE


def test_score_sums_significance_and_confidence() -> None:
    df = pl.DataFrame({
        'clinicalSignificances': [['pathogenic'], [None], ['unknown_value'], None, ['likely pathogenic', 'pathogenic']],
        'confidence': [
            'practice guideline',
            'criteria provided, single submitter',
            'unknown_conf',
            None,
            'criteria provided, single submitter',
        ],
    })
    # the last row has two matching, non-equal-scoring elements: the best one counts, they do not add up
    assert df.select(SCORE)['clinicalSignificances'].to_list() == pytest.approx([1.0, 0.02, 0.0, 0.0, 0.92])


def test_direction_on_trait_null_when_both_pathogenic_and_protective() -> None:
    df = pl.DataFrame({
        'clinicalSignificances': [['pathogenic'], ['protective'], ['pathogenic', 'protective'], [None], ['Pathogenic']]
    })
    # the capitalised last row pins the case-insensitive match
    assert df.select(DIRECTION_ON_TRAIT.alias('d'))['d'].to_list() == ['risk', 'protect', None, None, 'risk']


def test_direction_on_target_is_lof_for_loss_of_function_consequences() -> None:
    df = pl.DataFrame({'variantFunctionalConsequenceId': ['SO_0001589', 'SO_0001893', 'SO_9999999', None]})
    assert df.select(DIRECTION_ON_TARGET.alias('d'))['d'].to_list() == ['LoF', 'LoF', None, None]














def test_score_sum_is_exact_to_two_decimals() -> None:
    df = pl.DataFrame({
        'clinicalSignificances': [['pathogenic'], ['likely pathogenic']],
        'confidence': ['criteria provided, multiple submitters, no conflicts'] * 2,
    })
    # spark adds decimal literals, so 0.9 + 0.05 is 0.95 and not 0.9500000000000001
    assert df.select(SCORE)['clinicalSignificances'].to_list() == [0.95, 0.75]
