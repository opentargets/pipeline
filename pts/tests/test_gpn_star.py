"""Tests for GPN-Star cut-offs, normalisation and assessment."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from pts.schemas.variant_annotation import variant_effect_dtype
from pts.transformers.variant.gpn_star import (
    KEY,
    METHOD,
    GpnStarCutoffs,
    add_gpn_star,
    assess_gpn_star,
    cutoffs_from_histogram,
    gpn_star_files,
    llr_histogram,
    normalise_gpn_star,
    read_gpn_star,
)

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


def _write_scores(path: Path, rows: list[tuple[int, str, str, float]]) -> str:
    frame = pl.DataFrame(
        {
            'chrom': ['1'] * len(rows),
            'pos': [row[0] for row in rows],
            'ref': [row[1] for row in rows],
            'alt': [row[2] for row in rows],
            'llr_calibrated': [row[3] for row in rows],
            'abs_llr_calibrated': [abs(row[3]) for row in rows],
        },
        schema={
            'chrom': pl.String,
            'pos': pl.Int64,
            'ref': pl.String,
            'alt': pl.String,
            'llr_calibrated': pl.Float32,
            'abs_llr_calibrated': pl.Float32,
        },
    )
    frame.write_parquet(path)
    return str(path)


def test_gpn_star_files_by_chromosome(tmp_path: Path) -> None:
    one = _write_scores(tmp_path / 'llr_chr1.parquet', [(1, 'A', 'C', -1.0)])
    x = _write_scores(tmp_path / 'llr_chrX.parquet', [(1, 'A', 'C', -1.0)])
    assert gpn_star_files(str(tmp_path)) == {'1': one, 'X': x}


def test_gpn_star_files_reject_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='llr_chr'):
        gpn_star_files(str(tmp_path))


def test_cutoffs_from_histogram() -> None:
    values = [float(v) for v in range(-10000, 0)]
    histogram = pl.DataFrame(
        {'llr': values, 'count': [1] * len(values)}, schema={'llr': pl.Float64, 'count': pl.UInt64}
    )
    cutoffs = cutoffs_from_histogram(histogram)
    assert cutoffs.top_0_1_percent == -9991.0
    assert cutoffs.top_1_percent == -9901.0
    assert cutoffs.top_10_percent == -9001.0
    assert cutoffs.minimum == -10000.0


def test_llr_histogram_sums_files_and_rounds(tmp_path: Path) -> None:
    first = _write_scores(tmp_path / 'a.parquet', [(1, 'A', 'C', -1.0004), (2, 'A', 'C', -1.0001)])
    second = _write_scores(tmp_path / 'b.parquet', [(1, 'A', 'C', -1.0002), (2, 'A', 'C', -2.0)])
    histogram = llr_histogram([first, second])
    assert histogram.schema == pl.Schema({'llr': pl.Float64, 'count': pl.UInt64})
    assert histogram.to_dict(as_series=False) == {'llr': [-2.0, -1.0], 'count': [1, 3]}


def test_llr_histogram_ignores_unscored_rows(tmp_path: Path) -> None:
    file = tmp_path / 'llr_chr1.parquet'
    pl.DataFrame({'llr_calibrated': [-1.0, None, float('nan')]}, schema={'llr_calibrated': pl.Float32}).write_parquet(
        file
    )
    assert llr_histogram([str(file)]).to_dict(as_series=False) == {'llr': [-1.0], 'count': [1]}


def _three_alts(position: int) -> list[tuple[int, str, str, float]]:
    return [(position, 'A', alt, -float(position) / 100) for alt in 'CGT']


def test_read_gpn_star_keeps_requested_positions(tmp_path: Path) -> None:
    file = _write_scores(tmp_path / 'llr_chr1.parquet', _three_alts(100) + _three_alts(200))
    result = read_gpn_star(file, pl.Series([100], dtype=pl.Int32), CUTOFFS)
    assert result.height == 3
    assert result['position'].to_list() == [100] * 3
    assert result.schema['position'] == pl.Int32
    assert result.schema['gpnStarEffect'] == variant_effect_dtype.inner
    effects = result['gpnStarEffect']
    assert effects.struct.field('method').to_list() == [METHOD] * 3
    assert effects.struct.field('score').dtype == pl.Float32


def test_read_gpn_star_rejects_duplicates(tmp_path: Path) -> None:
    file = _write_scores(tmp_path / 'llr_chr1.parquet', [(100, 'A', 'C', -1.0), (100, 'A', 'C', -2.0)])
    with pytest.raises(ValueError, match='duplicate'):
        read_gpn_star(file, pl.Series([100], dtype=pl.Int32), CUTOFFS)


def _effect(method: str) -> dict[str, object]:
    return {
        'method': method,
        'assessment': None,
        'score': 0.5,
        'assessmentFlag': None,
        'targetId': None,
        'normalisedScore': 0.5,
    }


def _index(rows: list[tuple[int, str, str, list[dict[str, object]] | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'position': [row[0] for row in rows],
            'referenceAllele': [row[1] for row in rows],
            'alternateAllele': [row[2] for row in rows],
            'variantEffect': [row[3] for row in rows],
        },
        schema={
            'position': pl.Int32,
            'referenceAllele': pl.String,
            'alternateAllele': pl.String,
            'variantEffect': variant_effect_dtype,
        },
    )


def _scores(rows: list[tuple[int, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'position': [row[0] for row in rows],
            'referenceAllele': [row[1] for row in rows],
            'alternateAllele': [row[2] for row in rows],
            'gpnStarEffect': [_effect(METHOD) for _ in rows],
        },
        schema={
            'position': pl.Int32,
            'referenceAllele': pl.String,
            'alternateAllele': pl.String,
            'gpnStarEffect': variant_effect_dtype.inner,
        },
    )


def test_add_gpn_star_matches_allele_not_just_position() -> None:
    frame = _index([(100, 'A', 'G', []), (100, 'A', 'C', [])])
    result = add_gpn_star(frame, _scores([(100, 'A', 'C'), (100, 'A', 'T')]))
    assert [len(effects) for effects in result['variantEffect'].to_list()] == [0, 1]


def test_add_gpn_star_skips_indels_and_keeps_rows() -> None:
    old = [_effect('VEP')]
    frame = _index([(100, 'A', 'C', old), (100, 'A', 'AT', old), (300, 'G', 'T', old)])
    result = add_gpn_star(frame, _scores([(100, 'A', 'C')]))
    assert result.columns == frame.columns
    assert result.height == 3
    assert result['alternateAllele'].to_list() == ['C', 'AT', 'T']
    effects = result['variantEffect'].to_list()
    assert effects[1] == old
    assert effects[2] == old
    assert [effect['method'] for effect in effects[0]] == ['VEP', METHOD]


def test_add_gpn_star_on_multichunk_frame() -> None:
    first = _index([(100, 'A', 'C', [_effect('VEP')])])
    second = _index([(200, 'G', 'T', None), (300, 'G', 'T', [])])
    scores = _scores([(100, 'A', 'C'), (200, 'G', 'T')])
    chunked = pl.concat([first, second], rechunk=False)
    assert chunked.n_chunks() > 1
    result = add_gpn_star(chunked, scores)
    assert result.equals(add_gpn_star(pl.concat([first, second]), scores))
    assert [len(effects or []) for effects in result['variantEffect'].to_list()] == [2, 1, 0]


@pytest.mark.skipif(pl.thread_pool_size() < 2, reason='a single-threaded join yields one chunk')
def test_add_gpn_star_on_a_batch_mixing_matched_and_unmatched_rows(tmp_path: Path) -> None:
    positions = range(1, 5001)
    file = _write_scores(tmp_path / 'llr_chr1.parquet', [(position, 'A', 'C', -1.0) for position in positions])
    old = [_effect('VEP')]
    frame = _index([
        (position, 'A', 'C' if position % 2 else 'G', [None, [], old][position % 3]) for position in positions
    ])
    scores = read_gpn_star(file, frame['position'].unique(), CUTOFFS)
    assert frame.join(scores, on=KEY, how='left', maintain_order='left')['gpnStarEffect'].n_chunks() > 1
    result = add_gpn_star(frame, scores)
    assert result.select(KEY).equals(frame.select(KEY))
    for position, before, after in zip(
        positions, frame['variantEffect'].to_list(), result['variantEffect'].to_list(), strict=True
    ):
        if position % 2:
            assert [effect['method'] for effect in after] == [effect['method'] for effect in before or []] + [METHOD]
        else:
            assert after == before


def test_read_gpn_star_drops_unscored_rows(tmp_path: Path) -> None:
    file = tmp_path / 'llr_chr1.parquet'
    pl.DataFrame(
        {
            'pos': [100, 100, 100],
            'ref': ['A'] * 3,
            'alt': ['C', 'G', 'T'],
            'llr_calibrated': [-1.0, None, float('nan')],
        },
        schema={'pos': pl.Int64, 'ref': pl.String, 'alt': pl.String, 'llr_calibrated': pl.Float32},
    ).write_parquet(file)
    result = read_gpn_star(str(file), pl.Series([100], dtype=pl.Int32), CUTOFFS)
    assert result['alternateAllele'].to_list() == ['C']


def test_assess_and_normalise_agree_at_a_float32_cutoff() -> None:
    cutoffs = GpnStarCutoffs(top_10_percent=-0.1, top_1_percent=-3.3, top_0_1_percent=-6.3, minimum=-13.7)
    frame = pl.DataFrame({'llr': [-0.1, -3.3, -6.3]}, schema={'llr': pl.Float32})
    result = frame.select(
        assess_gpn_star(pl.col('llr'), cutoffs).alias('assessment'),
        normalise_gpn_star(pl.col('llr'), cutoffs).alias('score'),
    )
    for assessment, score in result.iter_rows():
        assert (assessment == 'not constrained') == (score == 0.0)
