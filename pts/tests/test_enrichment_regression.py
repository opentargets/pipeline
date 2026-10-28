"""Tests for the enrichment regression, checked against fitting each model with statsmodels."""

import numpy as np
import pytest
import statsmodels.api as sm
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests

from pts.transformers.enrichment import regression

N_GENES = 300
N_UNITS = 150


@pytest.fixture
def data() -> dict[str, np.ndarray]:
    """Outcomes that are mostly zero with a few large values, like L2G, and skewed predictors."""
    rng = np.random.default_rng(7)
    w = rng.normal(size=(N_GENES, 2))
    x = rng.gamma(0.5, size=(N_GENES, 4))
    y = np.where(rng.random((N_GENES, N_UNITS)) < 0.1, rng.random((N_GENES, N_UNITS)), 0.0)
    y[:, 0] += 0.3 * x[:, 1]
    labels = rng.integers(0, 40, size=N_GENES)
    return {'c': np.column_stack([np.ones(N_GENES), w]), 'x': x, 'y': y, 'labels': labels}


def _full_fit(data: dict[str, np.ndarray], unit: int, biosample: int, **fit_kwargs):
    design = np.column_stack([data['x'][:, biosample], data['c']])
    return sm.OLS(data['y'][:, unit], design).fit(**fit_kwargs)


def _fit(data: dict[str, np.ndarray], standard_errors: str, clusters=None) -> regression.Fit:
    return regression.Design(data['c'], data['x'], standard_errors, clusters).fit(data['y'])


def test_slopes_match_the_full_model_with_covariates(data) -> None:
    beta = _fit(data, 'gene').beta
    for unit, biosample in [(0, 1), (3, 0), (N_UNITS - 1, 3)]:
        assert beta[unit, biosample] == pytest.approx(_full_fit(data, unit, biosample).params[0])


def test_classical_se_matches_the_full_model(data) -> None:
    se = _fit(data, 'classical').standard_error
    for unit, biosample in [(0, 1), (5, 2)]:
        assert se[unit, biosample] == pytest.approx(_full_fit(data, unit, biosample).bse[0])


def test_per_gene_sandwich_matches_statsmodels_hc0(data) -> None:
    se = _fit(data, 'gene').standard_error
    for unit, biosample in [(0, 1), (7, 0), (N_UNITS - 1, 3)]:
        expected = _full_fit(data, unit, biosample, cov_type='HC0').bse[0]
        assert se[unit, biosample] == pytest.approx(expected)


def test_region_sandwich_matches_statsmodels_cluster_without_small_sample_correction(data) -> None:
    # more units than one internal batch, so batch boundaries are covered
    assert N_UNITS > regression._CLUSTER_BATCH
    se = _fit(data, 'region', regression.Clusters.from_labels(data['labels'])).standard_error
    for unit, biosample in [(0, 1), (regression._CLUSTER_BATCH, 2), (N_UNITS - 1, 3)]:
        expected = _full_fit(
            data, unit, biosample, cov_type='cluster', cov_kwds={'groups': data['labels'], 'use_correction': False}
        ).bse[0]
        assert se[unit, biosample] == pytest.approx(expected)


def test_region_sandwich_with_one_gene_per_cluster_is_hc0(data) -> None:
    per_gene = regression.Clusters.from_labels(np.arange(N_GENES))
    np.testing.assert_allclose(
        _fit(data, 'region', per_gene).standard_error, _fit(data, 'gene').standard_error, rtol=1e-10
    )


def test_region_standard_errors_need_clusters(data) -> None:
    with pytest.raises(ValueError, match='clusters'):
        regression.Design(data['c'], data['x'], 'region')


def test_without_covariates_t_ranks_biosamples_like_correlation(data) -> None:
    fit = regression.Design(np.ones((N_GENES, 1)), data['x'], 'classical').fit(data['y'][:, [0]])
    t = (fit.beta / fit.standard_error)[0]
    r = np.array([np.corrcoef(data['y'][:, 0], data['x'][:, k])[0, 1] for k in range(data['x'].shape[1])])
    assert spearmanr(t, r).statistic == pytest.approx(1.0)
    assert np.argsort(t).tolist() == np.argsort(r).tolist()


@pytest.mark.parametrize('standard_errors', ['region', 'gene', 'classical'])
def test_a_predictor_the_covariates_explain_gives_nan(data, standard_errors) -> None:
    x = np.column_stack([data['x'][:, 0], 2 * data['c'][:, 1] - data['c'][:, 2]])
    clusters = regression.Clusters.from_labels(data['labels'])
    fit = regression.Design(data['c'], x, standard_errors, clusters).fit(data['y'])
    assert np.isnan(fit.beta[:, 1]).all()
    assert np.isnan(fit.standard_error[:, 1]).all()
    assert np.isfinite(fit.beta[:, 0]).all()
    assert np.isfinite(fit.standard_error[:, 0]).all()


def test_clusters_sum_rows_within_each_label() -> None:
    labels = np.array(['b', 'a', 'b', 'c', 'a'])
    values = np.arange(10, dtype=float).reshape(5, 2)
    clusters = regression.Clusters.from_labels(labels)
    assert clusters.count == 3
    np.testing.assert_array_equal(clusters.sum(values), [[2 + 8, 3 + 9], [0 + 4, 1 + 5], [6, 7]])


def test_region_labels_use_fixed_windows() -> None:
    labels = regression.region_labels(np.array(['1', '1', '1', 'X']), np.array([999_999, 1_000_000, 0, 5]), 1_000_000)
    assert labels.tolist() == ['1:0', '1:1', '1:0', 'X:0']


def test_upper_tail_p_matches_the_normal_survival_function() -> None:
    z = np.array([[-2.0, 0.0, 1.5], [9.0, np.nan, 10.0]])
    p = regression.upper_tail_p(z)
    np.testing.assert_allclose(p[np.isfinite(z)], norm.sf(z[np.isfinite(z)]), rtol=1e-12)
    assert np.isnan(p[1, 1])
    assert p[1, 2] > 0  # 1 - cdf(10) would round to exactly 0


def test_multiple_testing_matches_statsmodels_per_row_ignoring_nan() -> None:
    rng = np.random.default_rng(1)
    p = rng.random((6, 9)) ** 3
    p[2, [0, 4]] = np.nan
    q = regression.benjamini_hochberg(p)
    bonf = regression.bonferroni(p)
    for row in range(p.shape[0]):
        finite = np.isfinite(p[row])
        np.testing.assert_allclose(q[row, finite], multipletests(p[row, finite], method='fdr_bh')[1])
        np.testing.assert_allclose(bonf[row, finite], multipletests(p[row, finite], method='bonferroni')[1])
        assert np.isnan(q[row, ~finite]).all()
        assert np.isnan(bonf[row, ~finite]).all()
