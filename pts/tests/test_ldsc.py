"""Tests for the LD score regression port.

The port was checked against github.com/bulik/ldsc to 1e-8 on synthetic data (see the branch's PR);
these tests pin its pieces and that it recovers what it is given.
"""

import math

import numpy as np
import pytest

from pts.transformers.genetic_correlation.ldsc import jackknife, regression
from pts.transformers.genetic_correlation.ldsc.rg import genetic_correlation

N_SNPS = 6000
M = 1.1e6
H2 = (0.25, 0.4)
RHO = 0.3


@pytest.fixture
def traits() -> dict[str, np.ndarray | float]:
    """Two traits simulated under the LDSC model, with different sample sizes."""
    rng = np.random.default_rng(11)
    ld = rng.gamma(4.0, 25.0, size=N_SNPS)
    n1, n2 = 50_000.0, 120_000.0
    g1 = rng.normal(size=N_SNPS) * np.sqrt(H2[0] * ld / M)
    g2 = RHO * math.sqrt(H2[1] / H2[0]) * g1 + rng.normal(size=N_SNPS) * np.sqrt((1 - RHO**2) * H2[1] * ld / M)
    return {
        'z1': math.sqrt(n1) * g1 + rng.normal(size=N_SNPS),
        'z2': math.sqrt(n2) * g2 + rng.normal(size=N_SNPS),
        'n1': np.full(N_SNPS, n1),
        'n2': np.full(N_SNPS, n2),
        'ld': ld,
        'w_ld': ld,
        'm': M,
    }


def test_block_jackknife_matches_leaving_each_block_out() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 2))
    y = (x @ np.array([[1.5], [-0.5]])) + rng.normal(size=(300, 1))
    boundaries = jackknife.separators(300, 10)
    fit = jackknife.least_squares(x, y, boundaries)
    assert fit.est == pytest.approx(np.linalg.lstsq(x, y, rcond=None)[0].reshape((1, 2)))
    for i in range(10):
        keep = np.ones(300, dtype=bool)
        keep[boundaries[i] : boundaries[i + 1]] = False
        refit = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0].reshape(2)
        assert fit.delete_values[i] == pytest.approx(refit)
    assert fit.jackknife_se.shape == (1, 2)
    assert np.all(fit.jackknife_se > 0)


def test_ratio_jackknife_uses_ratio_pseudovalues() -> None:
    numerator = np.array([2.0, 4.0, 6.0])
    denominator = np.array([1.0, 2.0, 2.0])
    est, se = jackknife.ratio(2.5, numerator, denominator)
    values = 3 * 2.5 - 2 * numerator / denominator
    assert est == pytest.approx(values.mean())
    assert se == pytest.approx(math.sqrt(values.var(ddof=1) / 3))
    assert all(math.isnan(v) for v in jackknife.ratio(1.0, numerator, np.array([1.0, 0.0, 1.0])))


def test_unmask_separators_maps_back_to_full_rows() -> None:
    mask = np.array([True, False, True, True, False, True, True])
    masked = jackknife.separators(5, 2)  # [0, 2, 5] over the five kept rows
    assert jackknife.unmask_separators(masked, mask).tolist() == [0, 3, 7]


def test_heritability_recovers_the_simulated_value(traits: dict) -> None:
    chisq = np.square(traits['z1'])
    fit = regression.heritability(chisq, traits['ld'], traits['w_ld'], traits['n1'], M, n_blocks=100, twostep=30.0)
    assert abs(fit.total - H2[0]) < 4 * fit.total_se
    assert abs(fit.intercept - 1.0) < 4 * fit.intercept_se
    assert fit.lambda_gc > 1
    assert fit.mean_chisq == pytest.approx(chisq.mean())
    assert fit.total_delete_values.shape == (100,)


def test_heritability_with_a_fixed_intercept(traits: dict) -> None:
    fit = regression.heritability(np.square(traits['z1']), traits['ld'], traits['w_ld'], traits['n1'], M, intercept=1.0)
    assert fit.intercept == 1.0
    assert math.isnan(fit.intercept_se)
    assert abs(fit.total - H2[0]) < 4 * fit.total_se


def test_genetic_correlation_recovers_the_simulated_value(traits: dict) -> None:
    result = genetic_correlation(**traits, n_blocks=100)
    assert abs(result.rg - RHO) < 4 * result.rg_se
    assert abs(result.intercept) < 4 * result.intercept_se
    assert abs(result.h2_1 - H2[0]) < 4 * result.h2_1_se
    assert abs(result.h2_2 - H2[1]) < 4 * result.h2_2_se
    assert result.genetic_covariance == pytest.approx(result.rg * math.sqrt(result.h2_1 * result.h2_2))
    assert result.n_snps == N_SNPS


def test_genetic_correlation_without_the_two_step_estimator(traits: dict) -> None:
    result = genetic_correlation(**traits, n_blocks=100, twostep=None)
    assert abs(result.rg - RHO) < 4 * result.rg_se


def test_rg_is_undefined_when_a_heritability_is_not_positive(traits: dict) -> None:
    # chi-square falling with LD score: a negative slope, so a negative heritability
    rng = np.random.default_rng(5)
    ld = traits['ld']
    z1 = rng.normal(size=N_SNPS) * np.sqrt(2 - ld / ld.max())
    result = genetic_correlation(z1, traits['z2'], traits['n1'], traits['n2'], ld, ld, M, n_blocks=100)
    assert result.h2_1 < 0
    assert math.isnan(result.rg)
    assert math.isnan(result.rg_se)
    assert not math.isnan(result.genetic_covariance)


def test_genetic_correlation_rejects_bad_inputs(traits: dict) -> None:
    with pytest.raises(ValueError, match='one length'):
        genetic_correlation(
            traits['z1'][:-1], traits['z2'], traits['n1'], traits['n2'], traits['ld'], traits['w_ld'], M
        )
    with pytest.raises(ValueError, match='jackknife blocks'):
        genetic_correlation(*(traits[k][:50] for k in ('z1', 'z2', 'n1', 'n2', 'ld', 'w_ld')), M, n_blocks=100)
