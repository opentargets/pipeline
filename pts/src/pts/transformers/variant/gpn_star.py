"""GPN-Star variant effect scores: cut-offs, normalisation and assessment."""

from __future__ import annotations

from typing import Self

import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator

from pts.transformers.utils.variant import min_max_scaler

METHOD = 'GPN-Star'


class GpnStarCutoffs(BaseModel):
    """Calibrated log-likelihood-ratio cut-offs; the more negative the score, the more constrained."""

    model_config = ConfigDict(extra='forbid')

    top_10_percent: float
    """Score at the edge of the 10% most constrained variants."""
    top_1_percent: float
    """Score at the edge of the 1% most constrained variants."""
    top_0_1_percent: float
    """Score at the edge of the 0.1% most constrained variants."""
    minimum: float
    """Score at and below which the normalised score saturates at 1."""

    @model_validator(mode='after')
    def _check_order(self) -> Self:
        if not self.minimum < self.top_0_1_percent < self.top_1_percent < self.top_10_percent:
            raise ValueError('cut-offs must satisfy minimum < top_0_1_percent < top_1_percent < top_10_percent')
        return self


def normalise_gpn_star(llr: pl.Expr, cutoffs: GpnStarCutoffs) -> pl.Expr:
    """Map the calibrated score onto [0, 1], higher meaning more constrained."""
    value = llr.cast(pl.Float64)
    return (
        pl
        .when(value >= cutoffs.top_10_percent)
        .then(0.0)
        .when(value >= cutoffs.top_1_percent)
        .then(min_max_scaler(value, cutoffs.top_10_percent, cutoffs.top_1_percent, 0.0, 0.5))
        .when(value >= cutoffs.top_0_1_percent)
        .then(min_max_scaler(value, cutoffs.top_1_percent, cutoffs.top_0_1_percent, 0.5, 0.75))
        .when(value.is_not_null())
        .then(min_max_scaler(value, cutoffs.top_0_1_percent, cutoffs.minimum, 0.75, 1.0).clip(upper_bound=1.0))
    )


def assess_gpn_star(llr: pl.Expr, cutoffs: GpnStarCutoffs) -> pl.Expr:
    """Label the score by the constraint bin it falls into."""
    return (
        pl
        .when(llr >= cutoffs.top_10_percent)
        .then(pl.lit('not constrained'))
        .when(llr >= cutoffs.top_1_percent)
        .then(pl.lit('constrained (top 10%)'))
        .when(llr >= cutoffs.top_0_1_percent)
        .then(pl.lit('constrained (top 1%)'))
        .when(llr.is_not_null())
        .then(pl.lit('constrained (top 0.1%)'))
    )
