"""Tests for the GERP numeric maximum over a bigWig."""

from __future__ import annotations

from typing import Any

import numpy as np
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


def _gerp_max_per_span(bigwig: str, chromosome: str, frame: pl.DataFrame) -> pl.Series:
    """The previous implementation: one `intervals` read per distinct span."""
    found: dict[tuple[int, int], float | None] = {}
    keys = list(zip(frame['vepStart'], frame['vepEnd'], strict=True))
    with pyBigWig.open(bigwig) as handle:
        length = (handle.chroms() or {}).get(chromosome)
        if length is not None:
            for start, end in dict.fromkeys(keys):
                low, high = (start - 2, end + 1) if start - 1 == end else (start - 1, end)
                low, high = max(low, 0), min(high, length)
                intervals = handle.intervals(chromosome, low, high) if low < high else None
                found[start, end] = max(value for _, _, value in intervals) if intervals else None
    return pl.Series('gerp', [found.get(key) for key in keys], dtype=pl.Float64)


@pytest.mark.parametrize('seed', range(5))
def test_windowed_reads_equal_one_read_per_span(tmp_path, seed: int):
    """Random sparse scores and spans (insertions, repeats, overlaps, both contig ends), small windows."""
    rng = np.random.default_rng(seed)
    length = 5_000
    path = str(tmp_path / 'random.bw')
    handle = pyBigWig.open(path, 'w')
    handle.addHeader([('1', length), ('2', 10)])
    covered = np.sort(rng.choice(length, size=3_000, replace=False))
    handle.addEntries(
        ['1'] * covered.size,
        covered.tolist(),
        ends=(covered + 1).tolist(),
        values=np.round(rng.normal(0, 3, covered.size), 3).tolist(),
    )
    handle.close()
    starts = rng.integers(-2, length + 5, 2_000)
    lengths = rng.choice([0, 1, 1, 1, 2, 7, 40], starts.size)
    frame = pl.DataFrame({'vepStart': starts, 'vepEnd': starts + lengths - 1}).sample(fraction=1.0, seed=seed)
    expected = _gerp_max_per_span(path, '1', frame)
    for gap, longest in ((0, 1), (16, 64), (4096, 1 << 22)):
        assert gerp_max(path, '1', frame, gap=gap, longest=longest).equals(expected)
    assert gerp_max(path, 'missing', frame).null_count() == frame.height
