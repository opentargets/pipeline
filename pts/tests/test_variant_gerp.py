"""Tests for the GERP numeric maximum over a bigWig."""

from __future__ import annotations

from typing import Any

import polars as pl
import pyBigWig  # ty: ignore[unresolved-import] -- compiled module without stubs
import pytest

from pts.transformers.variant.gerp import KentBigWig, gerp_max

SCORES = {
    # one-based positions of the bases that carry a score
    11: -0.466,
    12: -1.75,
    13: -1.87,
    21: 7.0e-05,
    22: 1.97,
    23: -2.0,
    31: 0.5,
    32: 1.5,
}
CONTIG_LENGTH = 100


@pytest.fixture(scope='module')
def bigwig(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A bigWig of contig 22, one single-base interval per scored position."""
    path = str(tmp_path_factory.mktemp('gerp') / 'gerp.bw')
    handle = pyBigWig.open(path, 'w')
    handle.addHeader([('22', CONTIG_LENGTH)])
    positions = sorted(SCORES)
    handle.addEntries(
        ['22'] * len(positions),
        [position - 1 for position in positions],
        ends=list(positions),
        values=[SCORES[position] for position in positions],
    )
    handle.close()
    return path


def spans(*pairs: tuple[int, int]) -> pl.DataFrame:
    """Spans in VEP's coordinates: an insertion ends before it starts."""
    return pl.DataFrame(pairs, schema={'vepStart': pl.Int64, 'vepEnd': pl.Int64}, orient='row')


def values(bigwig: str, *pairs: tuple[int, int], chromosome: str = '22') -> list[Any]:
    result = gerp_max(bigwig, chromosome, spans(*pairs))
    assert result.dtype == pl.Float64
    return [None if value is None else pytest.approx(value, abs=1e-6) for value in result]


def test_numeric_max_not_string_sort(bigwig: str):
    # VEP sorts the values as strings, so -1.87 beats -0.466; the numeric maximum is -0.466
    assert values(bigwig, (11, 13)) == [pytest.approx(-0.466, abs=1e-6)]


def test_exponent_values(bigwig: str):
    # VEP's string sort puts 7e-05 above 1.97
    assert values(bigwig, (21, 22)) == [pytest.approx(1.97, abs=1e-6)]


def test_single_base_and_row_order(bigwig: str):
    assert values(bigwig, (32, 32), (11, 11), (31, 31)) == [
        pytest.approx(1.5, abs=1e-6),
        pytest.approx(-0.466, abs=1e-6),
        pytest.approx(0.5, abs=1e-6),
    ]


def test_insertion_uses_flanking_bases(bigwig: str):
    # an insertion between 22 and 23 ends before it starts; the flanking bases score 1.97 and -2.0
    assert values(bigwig, (23, 22)) == [pytest.approx(1.97, abs=1e-6)]
    # between 12 and 13 the flanking scores are -1.75 and -1.87
    assert values(bigwig, (13, 12)) == [pytest.approx(-1.75, abs=1e-6)]


def test_uncovered_span_is_null(bigwig: str):
    assert values(bigwig, (50, 55), (1, 1)) == [None, None]


def test_span_beyond_the_contig_end(bigwig: str):
    assert values(bigwig, (99, 120), (101, 105)) == [None, None]


def test_missing_contig_is_null(bigwig: str):
    assert values(bigwig, (11, 13), chromosome='X') == [None]


def test_kent_summary_max_matches_exact_for_short_spans(bigwig: str):
    with KentBigWig(bigwig) as reader:
        assert reader.summary_max('22', 10, 13) == pytest.approx(-0.466, abs=1e-6)
        assert reader.summary_max('X', 10, 13) is None
        assert reader.summary_mean('22', 20, 22) == pytest.approx((7.0e-05 + 1.97) / 2, abs=1e-6)
