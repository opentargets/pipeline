"""Tests for the polars equivalents of spark's string functions."""

import math
import random
import struct

import polars as pl

from pts.transformers.utils.spark_strings import java_double_to_string, regexp_extract, spark_trim, spark_upper

DOUBLES = [
    0.0,
    -0.0,
    1.0,
    12.5,
    0.001,
    0.00099,
    9999999.0,
    1e7,
    1e-7,
    1.6688e-308,
    5.498e-320,
    1.7976931348623157e308,
    math.nan,
    math.inf,
    -math.inf,
    0.1 + 0.2,
    12.508500099182129,
    1e-5,
    -1.5e-10,
    -123456789.123,
    1e15,
    1e16,
    5.5410480786836218e17,
    2.3780253101263498e18,
    1e22,
]


def _spark_cast(spark, values: list[float]) -> list[str | None]:
    rows = spark.createDataFrame([(v,) for v in values], 'v DOUBLE').selectExpr('CAST(v AS STRING) AS s').collect()
    return [r['s'] for r in rows]


def _java(values: list[float | None]) -> list[str | None]:
    df = pl.DataFrame({'v': values}, schema={'v': pl.Float64})
    return df.select(java_double_to_string(pl.col('v')).alias('v'))['v'].to_list()


def test_spark_trim_strips_spaces_only() -> None:
    out = pl.select(spark_trim(pl.lit(pl.Series(['  a b  ', '\ta\t', '\xa0a\xa0', None])))).to_series()
    assert out.to_list() == ['a b', '\ta\t', '\xa0a\xa0', None]


def test_regexp_extract_is_empty_on_no_match_and_null_on_null() -> None:
    value = pl.lit(pl.Series(['PMID:123', 'none', None]))
    assert pl.select(regexp_extract(value, r'PMID:([0-9]+)', 1)).to_series().to_list() == ['123', '', None]


def test_java_double_to_string_matches_spark(spark) -> None:
    assert _java(DOUBLES) == _spark_cast(spark, DOUBLES)


def _from_bits(bits: int) -> float:
    return struct.unpack('<d', struct.pack('<Q', bits))[0]


def _random_doubles() -> list[float]:
    """Seeded doubles below 2**63 in magnitude: random bits, subnormals, integers, decimals, the notation edges."""
    rng = random.Random(20261018)
    values: list[float] = []
    while len(values) < 2000:
        value = _from_bits(rng.getrandbits(64))
        if math.isfinite(value) and abs(value) < 2.0**63:
            values.append(value)
    values += [_from_bits(rng.getrandbits(52) | rng.getrandbits(1) << 63) for _ in range(500)]
    values += [float(rng.randint(-(2**62), 2**62) >> rng.randint(0, 61)) for _ in range(1000)]
    values += [float(f'{rng.randint(1, 99999)}e{rng.randint(-12, 12)}') for _ in range(500)]
    for edge in (1e-3, -1e-3, 1e7, -1e7):
        values += [edge * (1 + rng.uniform(-1e-6, 1e-6)) for _ in range(250)]
        below, above = edge, edge
        for _ in range(20):
            values.append(below)
            below = math.nextafter(below, 0.0)
            above = math.nextafter(above, math.copysign(math.inf, edge))
            values.append(above)
    return values


def test_java_double_to_string_matches_spark_on_random_doubles(spark) -> None:
    values = _random_doubles()
    pairs = zip(values, _java(values), _spark_cast(spark, values), strict=True)
    mismatches = [(v, j, s) for v, j, s in pairs if j != s]
    assert mismatches == []


def _signed(values: list[float]) -> list[float]:
    return values + [-v for v in values]


def _mismatches(spark, values: list[float]) -> list[tuple[float, str | None, str | None]]:
    pairs = zip(values, _java(values), _spark_cast(spark, values), strict=True)
    return [(v, j, s) for v, j, s in pairs if j != s]


def test_java_double_to_string_matches_spark_on_every_power_of_two(spark) -> None:
    powers = [math.ldexp(1.0, e) for e in range(-1074, 1024)]
    assert _java([2.0**-24, 2.0**-1074]) == ['5.9604644775390625E-8', '4.9E-324']
    assert _mismatches(spark, _signed(powers)) == []


def test_java_double_to_string_matches_spark_next_to_every_power_of_two(spark) -> None:
    powers = [math.ldexp(1.0, e) for e in range(-1074, 1024)]
    neighbours = [math.nextafter(p, 0.0) for p in powers] + [math.nextafter(p, math.inf) for p in powers]
    assert _mismatches(spark, _signed([v for v in neighbours if v not in (0.0, math.inf)])) == []


def test_java_double_to_string_matches_spark_on_the_smallest_subnormals(spark) -> None:
    assert _mismatches(spark, _signed([math.ldexp(float(n), -1074) for n in range(1, 4096)])) == []


def test_java_double_to_string_is_shortest_where_java_17_is_not(spark) -> None:
    # above 2**63 only: a long overflow in java's digit loop, and a candidate exactly half a gap away
    values = [2.4255049468349816e25, 1.50073856281376e19]
    assert _spark_cast(spark, values) == ['2.4255049468349815E25', '1.5007385628137599E19']
    assert _java(values) == ['2.4255049468349816E25', '1.50073856281376E19']


def test_java_double_to_string_keeps_null() -> None:
    assert _java([None, 1e-5]) == [None, '1.0E-5']


def test_java_double_to_string_broadcasts_a_scalar() -> None:
    assert pl.select(java_double_to_string(pl.lit(1.5e-9))).to_series().to_list() == ['1.5E-9']
    assert pl.select(java_double_to_string(pl.lit(2.5))).to_series().to_list() == ['2.5']


def test_spark_upper_matches_spark(spark) -> None:
    labels = ['Straße', 'ﬁx', 'İı', 'ΟΔΟΣ ς', 'a\ua7d3b', '\ua7cf\ua7d5', '\U00016ebb\U00016ed3', '', None]
    labels += [chr(c) for c in range(0x80, 0x20000) if not 0xD800 <= c <= 0xDFFF]
    rows = spark.createDataFrame([(v,) for v in labels], 'v string').selectExpr('upper(v) AS u').collect()
    out = pl.select(spark_upper(pl.lit(pl.Series(labels, dtype=pl.String)))).to_series().to_list()
    assert out == [r['u'] for r in rows]
