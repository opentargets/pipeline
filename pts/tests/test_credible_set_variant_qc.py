"""Variant QC: each check of a credible set variant, run through the stage on the worked examples of the PRD."""

import math

import polars as pl
import pytest

from pts.schemas.credible_set import credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetVariantFlag

CS_ID = '0' * 32
EXCEEDS = CredibleSetVariantFlag.EXCEEDS_COVERAGE_95
ABNORMAL = CredibleSetVariantFlag.ABNORMAL_PIP


def _variants(pips: list[float], credible_set_id: str = CS_ID, names: list[str] | None = None) -> pl.LazyFrame:
    names = names or [f'1_{100 + index}_A_T' for index in range(len(pips))]
    rows = [
        {'credibleSetId': credible_set_id, 'variantId': name, 'posteriorProbability': pip, 'qualityControls': []}
        for name, pip in zip(names, pips, strict=True)
    ]
    return pl.LazyFrame([{name: row.get(name) for name in credible_set_variant_schema} for row in rows]).cast(
        credible_set_variant_schema
    )


def _with(frame: pl.LazyFrame, flag: str) -> list[float]:
    """The PIPs of the rows carrying `flag`, rounded for comparison."""
    flagged = frame.filter(pl.col('qualityControls').list.contains(flag)).collect()
    return [round(pip, 2) for pip in flagged['posteriorProbability']]


def test_the_cut_keeps_the_variant_that_reaches_coverage(qc_variants) -> None:
    result = qc_variants(_variants([0.60, 0.30, 0.09, 0.01]))
    assert _with(result, EXCEEDS) == [0.01]
    assert _with(result, ABNORMAL) == []


def test_a_variant_after_coverage_is_reached_exceeds_it(qc_variants) -> None:
    result = qc_variants(_variants([0.97, 0.02]))
    assert _with(result, EXCEEDS) == [0.02]
    assert _with(result, ABNORMAL) == []


def test_a_set_below_coverage_is_abnormal_on_every_row(qc_variants) -> None:
    result = qc_variants(_variants([0.50, 0.30]))
    assert _with(result, EXCEEDS) == []
    assert _with(result, ABNORMAL) == [0.50, 0.30]


def test_a_kept_sum_above_one_is_abnormal(qc_variants) -> None:
    result = qc_variants(_variants([0.60, 0.60]))
    assert _with(result, ABNORMAL) == [0.60, 0.60]


@pytest.mark.parametrize('bad_pip', [math.nan, 0.0, -0.1, 1.1])
def test_an_invalid_pip_makes_every_row_of_its_set_abnormal(qc_variants, bad_pip: float) -> None:
    other = _variants([0.96], credible_set_id='1' * 32)
    result = qc_variants(pl.concat([_variants([0.70, bad_pip, 0.30]), other]))
    flagged = result.filter(pl.col('qualityControls').list.contains(ABNORMAL)).collect()
    assert flagged['credibleSetId'].to_list() == [CS_ID] * 3


@pytest.mark.parametrize('order', [['1_1_A_T', '1_2_A_T', '1_3_A_T'], ['1_3_A_T', '1_2_A_T', '1_1_A_T']])
def test_equal_pips_at_the_cut_are_ordered_by_variant_id(qc_variants, order: list[str]) -> None:
    pips = {'1_1_A_T': 0.90, '1_2_A_T': 0.06, '1_3_A_T': 0.06}
    result = qc_variants(_variants([pips[name] for name in order], names=order))
    exceeding = result.filter(pl.col('qualityControls').list.contains(EXCEEDS)).collect()
    assert exceeding['variantId'].to_list() == ['1_3_A_T']


def test_the_cut_compares_the_stored_pips_without_tolerance(qc_variants) -> None:
    """0.90 + 0.05 in single precision is 0.94999998, below the coverage, so nothing is cut.

    The sources store their 95% sets; a tolerance on the cut would remove members they kept.
    """
    result = qc_variants(_variants([0.90, 0.05, 0.05]))
    assert _with(result, EXCEEDS) == []


@pytest.mark.parametrize(
    ('credible_set_id', 'invalid'),
    [(CS_ID, False), ('0123456789abcdef' * 2, False), ('0123456789ABCDEF' * 2, True), ('0' * 31, True), ('', True)],
)
def test_invalid_cs_id(qc_variants, credible_set_id: str, invalid: bool) -> None:
    result = qc_variants(_variants([1.0], credible_set_id=credible_set_id)).collect()
    assert result['qualityControls'].list.contains(CredibleSetVariantFlag.INVALID_CS_ID).item() is invalid


def test_flagging_keeps_rows_columns_and_existing_flags_and_is_idempotent(qc_variants) -> None:
    frame = _variants([0.60, 0.30, 0.09, 0.01]).with_columns(
        pl.lit([CredibleSetVariantFlag.NOT_IN_LOCUS.value]).alias('qualityControls')
    )
    once = qc_variants(frame).collect()
    twice = qc_variants(once.lazy()).collect()

    assert once.height == 4
    assert once.drop('qualityControls').equals(frame.collect().drop('qualityControls'))
    assert once['qualityControls'].list.contains(CredibleSetVariantFlag.NOT_IN_LOCUS).all()
    assert twice.equals(once)


def _row(**values: float | int | None) -> pl.LazyFrame:
    """One variant row with the given statistics; the other columns are null."""
    row = {'credibleSetId': CS_ID, 'variantId': '1_100_A_T', 'posteriorProbability': 1.0, 'qualityControls': []}
    return pl.LazyFrame([{name: {**row, **values}.get(name) for name in credible_set_variant_schema}]).cast(
        credible_set_variant_schema
    )


def _carries(frame: pl.LazyFrame, flag: str) -> bool:
    return frame.select(pl.col('qualityControls').list.contains(flag)).collect().item()


@pytest.mark.parametrize(
    ('reported', 'conditional', 'mismatch'),
    [
        (0.2, -0.1, True),
        (-0.2, 0.1, True),
        (0.2, 0.1, False),
        (0.2, None, False),
        (None, -0.1, False),
        (0.0, -0.1, False),
    ],
)
def test_effect_direction_mismatch(
    qc_variants, reported: float | None, conditional: float | None, mismatch: bool
) -> None:
    frame = qc_variants(_row(reportedBeta=reported, conditionalBeta=conditional))
    assert _carries(frame, CredibleSetVariantFlag.EFFECT_DIRECTION_MISMATCH) is mismatch


@pytest.mark.parametrize(
    ('beta', 'standard_error', 'incomplete'),
    [(0.2, None, True), (None, 0.01, True), (0.2, 0.01, False), (None, None, False)],
)
def test_incomplete_reported_and_conditional_stats(
    qc_variants, beta: float | None, standard_error: float | None, incomplete: bool
) -> None:
    reported = qc_variants(_row(reportedBeta=beta, reportedStandardError=standard_error))
    conditional = qc_variants(_row(conditionalBeta=beta, conditionalStandardError=standard_error))
    assert _carries(reported, CredibleSetVariantFlag.INCOMPLETE_REPORTED_STATS) is incomplete
    assert _carries(reported, CredibleSetVariantFlag.INCOMPLETE_CONDITIONAL_STATS) is False
    assert _carries(conditional, CredibleSetVariantFlag.INCOMPLETE_CONDITIONAL_STATS) is incomplete
    assert _carries(conditional, CredibleSetVariantFlag.INCOMPLETE_REPORTED_STATS) is False


@pytest.mark.parametrize(
    ('mantissa', 'exponent', 'missing'),
    [(None, -10, True), (5.0, None, True), (None, None, True), (5.0, -10, False)],
)
def test_missing_pvalue(qc_variants, mantissa: float | None, exponent: int | None, missing: bool) -> None:
    frame = qc_variants(_row(pValueMantissa=mantissa, pValueExponent=exponent))
    assert _carries(frame, CredibleSetVariantFlag.MISSING_PVALUE) is missing


def test_variant_qc_adds_the_metadata_flags(qc_variants) -> None:
    frame = qc_variants(_row(reportedBeta=0.2, conditionalBeta=-0.1)).collect()
    assert set(frame['qualityControls'].item()) == {
        CredibleSetVariantFlag.EFFECT_DIRECTION_MISMATCH,
        CredibleSetVariantFlag.INCOMPLETE_REPORTED_STATS,
        CredibleSetVariantFlag.INCOMPLETE_CONDITIONAL_STATS,
        CredibleSetVariantFlag.MISSING_PVALUE,
    }


DUPLICATED = CredibleSetVariantFlag.DUPLICATED_VARIANT


def test_every_copy_of_a_variant_repeated_in_its_set_is_duplicated(qc_variants) -> None:
    variants = _variants([0.5, 0.4, 0.1], names=['1_100_A_T', '1_100_A_T', '1_200_A_T'])
    assert _with(qc_variants(variants), DUPLICATED) == [0.5, 0.4]


def test_the_same_variant_in_two_sets_is_not_duplicated(qc_variants) -> None:
    variants = pl.concat([_variants([1.0], 'a' * 32, ['1_100_A_T']), _variants([1.0], 'b' * 32, ['1_100_A_T'])])
    assert _with(qc_variants(variants), DUPLICATED) == []


def test_variant_qc_flags_duplicated_variants(qc_variants) -> None:
    variants = _variants([0.5, 0.5], names=['1_100_A_T', '1_100_A_T'])
    assert _with(qc_variants(variants), DUPLICATED) == [0.5, 0.5]
