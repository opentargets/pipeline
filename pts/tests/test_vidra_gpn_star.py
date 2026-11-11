"""Tests of the VIDRA GPN-Star transform and its ClinVar calibration."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pts.pyspark.vidra_utils.gpn_star import (
    auroc,
    benign_score,
    calibrate,
    clinvar_missense_labels,
    fit_logistic,
    variant_scores,
)

SINGLE = 'criteria provided, single submitter'
MULTIPLE = 'criteria provided, multiple submitters, no conflicts'


def _effect(method: str, score: float | None, target: str | None = None) -> dict[str, object]:
    return {'method': method, 'score': score, 'targetId': target}


def test_benign_score_is_one_minus_sigmoid_and_keeps_nulls() -> None:
    frame = pl.DataFrame({'llr': [0.0, -10.0, 10.0, None]}).select(benign_score(pl.col('llr'), 0.0, 1.0).alias('s'))
    scores = frame['s'].to_list()
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(1.0, abs=1e-4)
    assert scores[2] == pytest.approx(0.0, abs=1e-4)
    assert scores[3] is None


def test_benign_score_decreasing_in_llr_when_b_negative() -> None:
    # The fitted b is negative (lower LLR = more constrained = more pathogenic), so the benign
    # score must rise with the LLR.
    frame = pl.DataFrame({'llr': [-8.0, -4.0, 0.0, 4.0]}).select(benign_score(pl.col('llr'), -2.0, -0.5).alias('s'))
    assert frame['s'].to_list() == sorted(frame['s'].to_list())


def test_clinvar_missense_labels_review_filters_and_conflicts() -> None:
    eva = pl.LazyFrame({
        'variantId': ['p', 'b', 'both', 'both', 'vus', 'nocrit', 'conflict', 'lof', 'multi', None],
        'targetId': ['g'] * 10,
        'variantFunctionalConsequenceId': ['SO_0001583'] * 7 + ['SO_0001587', 'SO_0001583', 'SO_0001583'],
        'clinicalSignificances': [
            ['pathogenic'],
            ['likely benign'],
            ['pathogenic'],
            ['benign'],
            ['uncertain significance'],
            ['pathogenic'],
            ['pathogenic'],
            ['pathogenic'],
            ['likely pathogenic'],
            ['pathogenic'],
        ],
        'confidence': [
            SINGLE,
            SINGLE,
            SINGLE,
            MULTIPLE,
            SINGLE,
            'no assertion criteria provided',
            'criteria provided, conflicting classifications',
            SINGLE,
            'reviewed by expert panel',
            SINGLE,
        ],
    })
    labels = clinvar_missense_labels(eva).sort('variantId').collect()
    assert labels.to_dict(as_series=False) == {
        'variantId': ['b', 'multi', 'p'],
        'targetId': ['g'] * 3,
        'label': [0, 1, 1],
    }

    # 'both' loses its single-submitter pathogenic row and keeps the multiple-submitter benign one
    strict = clinvar_missense_labels(eva, min_review='multiple').sort('variantId').collect()
    assert strict.to_dict(as_series=False) == {'variantId': ['both', 'multi'], 'targetId': ['g'] * 2, 'label': [0, 1]}

    with pytest.raises(ValueError, match='min_review'):
        clinvar_missense_labels(eva, min_review='any')


def test_variant_scores_picks_gpn_star_and_gene_matched_alphamissense() -> None:
    index = pl.LazyFrame({
        'variantId': ['v1', 'v2', 'v3', 'unrelated'],
        'variantEffect': [
            [_effect('GPN-Star', -3.0), _effect('AlphaMissense', 0.9, 'g'), _effect('AlphaMissense', 0.2, 'h')],
            [_effect('VEP', 0.5)],
            None,
            [_effect('GPN-Star', 1.0)],
        ],
    })
    wanted = pl.LazyFrame({'variantId': ['v1', 'v2', 'v3'], 'targetId': ['g', 'g', 'g']})
    scored = variant_scores(index, wanted).sort('variantId').collect()
    assert scored['llr'].to_list() == [-3.0, None, None]
    assert scored['am_pathogenicity'].to_list() == [0.9, None, None]


def test_fit_logistic_recovers_parameters_and_balanced_shifts_intercept() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(-3.0, 3.0, size=30000)
    true_a, true_b = -1.5, -0.8
    y = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-(true_a + true_b * x)))).astype(int)
    a, b = fit_logistic(x, y)
    assert a == pytest.approx(true_a, abs=0.1)
    assert b == pytest.approx(true_b, abs=0.05)
    a_balanced, b_balanced = fit_logistic(x, y, balanced=True)
    assert b_balanced == pytest.approx(true_b, abs=0.1)
    # balancing moves the intercept towards 50:50 prevalence: down for this pathogenic-rich sample
    assert y.mean() > 0.5
    assert a_balanced < a

    with pytest.raises(ValueError, match='both classes'):
        fit_logistic(x, np.zeros_like(y))


def test_auroc_handles_ties_and_perfect_separation() -> None:
    assert auroc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 0, 1, 1])) == 0.5


def test_calibrate_end_to_end_on_synthetic_release() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    llr = rng.normal(-4.0, 3.0, size=n)
    label = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-2.0 - 0.9 * llr)))).astype(int)
    ids = [f'1_{i}_A_G' for i in range(n)]
    eva = pl.LazyFrame({
        'variantId': [*ids, '1_999999_A_G'],
        'targetId': ['g'] * (n + 1),
        'variantFunctionalConsequenceId': ['SO_0001583'] * (n + 1),
        'clinicalSignificances': [['pathogenic'] if v else ['benign'] for v in label] + [['pathogenic']],
        'confidence': [SINGLE] * (n + 1),
    })
    effects = [[_effect('GPN-Star', float(v)), _effect('AlphaMissense', float(1 / (1 + np.exp(v))), 'g')] for v in llr]
    variant = pl.LazyFrame({'variantId': [*ids, '1_999999_A_G'], 'variantEffect': [*effects, [_effect('VEP', 0.1)]]})

    result = calibrate(variant, eva)
    assert result.n_pathogenic + result.n_benign == n
    assert result.n_without_llr == 1
    assert result.b < 0
    assert result.auroc_gpn_star > 0.85
    assert result.n_with_alphamissense == n
    assert result.auroc_alphamissense == result.auroc_gpn_star_on_alphamissense_subset  # same monotone score
    quantiles = result.benign_score_quantiles
    assert quantiles['gpn_star_pathogenic'][2] < quantiles['gpn_star_benign'][2]
    assert set(result.to_dict()) >= {'a', 'b', 'auroc_gpn_star', 'benign_score_quantiles'}
