"""Tests for GPN-Star cut-offs, normalisation and assessment."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from pts.transformers.variant.gpn_star import GpnStarCutoffs, assess_gpn_star, normalise_gpn_star

CUTOFFS = GpnStarCutoffs(top_10_percent=-2.0, top_1_percent=-4.0, top_0_1_percent=-6.0, minimum=-10.0)


@pytest.mark.parametrize(
    ('llr', 'expected'),
    [
        (0.0, 0.0),
        (-2.0, 0.0),
        (-3.0, 0.25),
        (-4.0, 0.5),
        (-5.0, 0.625),
        (-6.0, 0.75),
        (-8.0, 0.875),
        (-10.0, 1.0),
        (-12.0, 1.0),
        (None, None),
    ],
)
def test_normalise_gpn_star(llr: float | None, expected: float | None) -> None:
    frame = pl.DataFrame({'llr': [llr]}, schema={'llr': pl.Float32})
    result = frame.select(normalise_gpn_star(pl.col('llr'), CUTOFFS).alias('score'))
    assert result.schema['score'] == pl.Float64
    value = result['score'][0]
    assert value == pytest.approx(expected) if expected is not None else value is None


@pytest.mark.parametrize(
    ('llr', 'expected'),
    [
        (-1.0, 'not constrained'),
        (-2.0, 'not constrained'),
        (-2.5, 'constrained (top 10%)'),
        (-4.0, 'constrained (top 10%)'),
        (-4.5, 'constrained (top 1%)'),
        (-6.0, 'constrained (top 1%)'),
        (-6.5, 'constrained (top 0.1%)'),
        (None, None),
    ],
)
def test_assess_gpn_star(llr: float | None, expected: str | None) -> None:
    frame = pl.DataFrame({'llr': [llr]}, schema={'llr': pl.Float32})
    result = frame.select(assess_gpn_star(pl.col('llr'), CUTOFFS).alias('assessment'))
    assert result['assessment'][0] == expected


def test_cutoffs_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        GpnStarCutoffs(top_10_percent=-4, top_1_percent=-2, top_0_1_percent=-6, minimum=-10)


def test_cutoffs_minimum_must_be_below_top_0_1_percent() -> None:
    with pytest.raises(ValidationError):
        GpnStarCutoffs(top_10_percent=-2, top_1_percent=-4, top_0_1_percent=-6, minimum=-6)
