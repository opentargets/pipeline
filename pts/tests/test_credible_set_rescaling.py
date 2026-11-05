"""Annotation: the rescaled effect size of every validated variant, from its p-value, frequency and study design."""

import math
from typing import Any

import polars as pl
import pytest

from pts.schemas.credible_set import credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetVariantFlag
from pts.transformers.credible_set.rescaling import rescaled_effect_size
from pts.transformers.study import QualityFlag as StudyQualityFlag

MEASUREMENT = StudyQualityFlag.MEASUREMENT_STUDY_DESIGN.value
CASE_CONTROL = StudyQualityFlag.CASE_CONTROL_STUDY_DESIGN.value
RESCALED = {CredibleSetVariantFlag.RESCALED_FROM_CONDITIONAL, CredibleSetVariantFlag.RESCALED_FROM_MARGINAL}

ROW = {
    'pValueMantissa': 1.0,
    'pValueExponent': -10,
    'effectAlleleFrequency': 0.2,
    'reportedBeta': 0.2,
    'conditionalBeta': None,
}
"""A variant with p = 1e-10, AF = 0.2 and a positive marginal effect."""


def _variants(*rows: dict[str, Any]) -> pl.LazyFrame:
    """Variants of credible set `cs`, each `ROW` updated with the given values, with an earlier flag."""
    defaults = {'credibleSetId': 'cs', 'qualityControls': [CredibleSetVariantFlag.EAF_FROM_SOURCE.value]}
    return pl.LazyFrame(
        [
            {
                name: {**defaults, 'variantId': f'1_{i + 1}_A_T', **ROW, **row}.get(name)
                for name in credible_set_variant_schema
            }
            for i, row in enumerate(rows)
        ],
        schema=credible_set_variant_schema,
    )


SETS = pl.LazyFrame({'id': ['cs'], 'studyId': ['s']})


def _studies(
    design: str | None = MEASUREMENT, n_samples: int | None = 100_000, n_cases: int | None = None
) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            'studyId': ['s'],
            'nSamples': [n_samples],
            'nCases': [n_cases],
            'qualityControls': [[design] if design else []],
        },
        schema={'studyId': pl.String, 'nSamples': pl.Int32, 'nCases': pl.Int32, 'qualityControls': pl.List(pl.String)},
    )


def _rescale(variants: pl.LazyFrame, studies: pl.LazyFrame | None = None) -> list[dict[str, Any]]:
    result = rescaled_effect_size(variants, SETS, studies if studies is not None else _studies()).collect()
    assert result.schema == credible_set_variant_schema
    assert all(CredibleSetVariantFlag.EAF_FROM_SOURCE in qc for qc in result['qualityControls'])
    return [
        {**row, 'flags': set(row['qualityControls']) & RESCALED}
        for row in result.select(
            'rescaledAbsBeta', 'rescaledStandardError', 'rescaledDirection', 'qualityControls'
        ).to_dicts()
    ]


def _not_rescaled(row: dict[str, Any]) -> bool:
    values = (row['rescaledAbsBeta'], row['rescaledStandardError'], row['rescaledDirection'])
    return values == (None, None, None) and row['flags'] == set()


def test_a_quantitative_trait_is_rescaled_to_its_standard_deviation() -> None:
    [row] = _rescale(_variants({}))
    assert row['rescaledStandardError'] == pytest.approx(0.00559, abs=5e-6)
    assert row['rescaledAbsBeta'] == pytest.approx(0.03615, abs=5e-6)
    assert row['rescaledAbsBeta'] / row['rescaledStandardError'] == pytest.approx(6.467, abs=5e-4)
    assert row['rescaledDirection'] == 1
    assert row['flags'] == {CredibleSetVariantFlag.RESCALED_FROM_MARGINAL}


def test_a_binary_trait_is_rescaled_to_the_log_odds_with_its_prevalence() -> None:
    [row] = _rescale(_variants({}), _studies(CASE_CONTROL, n_cases=20_000))
    assert row['rescaledStandardError'] == pytest.approx(0.013975, abs=5e-6)
    assert row['rescaledAbsBeta'] == pytest.approx(0.09038, abs=5e-5)


def test_the_direction_prefers_the_conditional_effect() -> None:
    [row] = _rescale(_variants({'conditionalBeta': -0.1, 'reportedBeta': 0.2}))
    assert row['rescaledDirection'] == -1
    assert row['flags'] == {CredibleSetVariantFlag.RESCALED_FROM_CONDITIONAL}


def test_the_direction_falls_back_to_the_marginal_effect() -> None:
    [row] = _rescale(_variants({'conditionalBeta': None, 'reportedBeta': -0.2}))
    assert row['rescaledDirection'] == -1
    assert row['flags'] == {CredibleSetVariantFlag.RESCALED_FROM_MARGINAL}


@pytest.mark.parametrize(
    'values',
    [
        {'effectAlleleFrequency': 0.0},
        {'effectAlleleFrequency': 1.0},
        {'effectAlleleFrequency': None},
        {'pValueMantissa': None},
        {'pValueExponent': None},
        {'reportedBeta': None, 'conditionalBeta': None},
    ],
    ids=['AF 0', 'AF 1', 'no AF', 'no mantissa', 'no exponent', 'no beta'],
)
def test_a_variant_without_the_statistics_is_not_rescaled(values: dict[str, Any]) -> None:
    [row] = _rescale(_variants(values))
    assert _not_rescaled(row)


@pytest.mark.parametrize(
    'studies',
    [
        _studies(StudyQualityFlag.ONE_ONLY_CASE_OR_CONTROL.value),
        _studies(StudyQualityFlag.CASE_CONTROL_SUM_NEQ_SAMPLE_SIZE.value),
        _studies(None),
        _studies(n_samples=None),
        _studies(CASE_CONTROL, n_cases=None),
        _studies(CASE_CONTROL, n_cases=0),
        _studies(CASE_CONTROL, n_cases=100_000),
        _studies(n_samples=0),
    ],
    ids=[
        'only cases or controls',
        'case-control sum mismatch',
        'no design',
        'no sample size',
        'no cases',
        'prevalence 0',
        'prevalence 1',
        'sample size 0',
    ],
)
def test_a_study_without_a_supported_design_gives_no_rescaled_value(studies: pl.LazyFrame) -> None:
    [row] = _rescale(_variants({}), studies)
    assert _not_rescaled(row)


def test_a_variant_of_a_set_without_a_study_is_not_rescaled() -> None:
    other = pl.LazyFrame({
        'studyId': ['other'],
        'nSamples': [100_000],
        'nCases': [None],
        'qualityControls': [[MEASUREMENT]],
    })
    [row] = _rescale(_variants({}), other.cast({'nSamples': pl.Int32, 'nCases': pl.Int32}))
    assert _not_rescaled(row)


def test_an_extreme_p_value_gives_a_finite_rescaled_effect() -> None:
    [row] = _rescale(_variants({'pValueExponent': -1000}))
    assert math.isfinite(row['rescaledAbsBeta'])
    assert row['rescaledAbsBeta'] > 0


def test_a_source_rescaled_value_is_replaced_and_rows_keep_their_order() -> None:
    variants = _variants(
        {'rescaledAbsBeta': 9.0, 'rescaledStandardError': 9.0, 'rescaledDirection': 1, 'effectAlleleFrequency': None},
        {},
        {'pValueExponent': -20},
    )
    rows = _rescale(variants)
    assert _not_rescaled(rows[0])
    assert rows[1]['rescaledAbsBeta'] < rows[2]['rescaledAbsBeta']


@pytest.mark.parametrize('non_finite', [math.nan, math.inf, -math.inf], ids=['nan', 'inf', '-inf'])
def test_a_non_finite_beta_is_missing(non_finite: float) -> None:
    [row] = _rescale(_variants({'reportedBeta': non_finite, 'conditionalBeta': None}))
    assert _not_rescaled(row)


@pytest.mark.parametrize('non_finite', [math.nan, math.inf], ids=['nan', 'inf'])
def test_a_non_finite_conditional_beta_falls_back_to_the_marginal_one(non_finite: float) -> None:
    [row] = _rescale(_variants({'conditionalBeta': non_finite, 'reportedBeta': -0.2}))
    assert row['rescaledDirection'] == -1
    assert row['flags'] == {CredibleSetVariantFlag.RESCALED_FROM_MARGINAL}
