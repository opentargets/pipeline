"""Tests for the association harmonic sum."""

from __future__ import annotations

from functools import reduce
from math import isclose

import polars as pl

from pts.transformers.association.harmonic import MAX_HARMONIC_SUM, harmonic_sum


def _hs(scores: list[float]) -> float:
    df = pl.DataFrame({'s': [scores]}, schema={'s': pl.List(pl.Float64)})
    return df.select(harmonic_sum(pl.col('s')))['s'][0]


def test_harmonic_sum_matches_spark_doctest() -> None:
    assert isclose(_hs([0.1, 0.5, 0.1, 0.1, 0.9, 0.85]), 1.1810833333333335, abs_tol=1e-12)


def test_harmonic_sum_spark_evidence_case() -> None:
    assert isclose(_hs([0.5, 0.5, 0.2]), 0.6472222225533591, abs_tol=1e-9)


def test_harmonic_sum_caps_at_50() -> None:
    assert isclose(_hs([1.0] * 60), sum(1 / i**2 for i in range(1, 51)), abs_tol=1e-12)


def test_harmonic_sum_empty_is_zero() -> None:
    assert _hs([]) == 0.0


def test_max_harmonic_sum_value() -> None:
    assert reduce(lambda acc, iv: acc + iv[1] / iv[0] ** 2, enumerate([1] * 1000, start=1), 0.0) == MAX_HARMONIC_SUM
