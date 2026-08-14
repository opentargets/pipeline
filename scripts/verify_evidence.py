#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.41.2"]
# ///
# ruff: noqa: T201
r"""Compare a polars `evidence_postprocess` output against a pyspark baseline and/or a published release.

Deliberately takes PATHS, not a datasource name: it never imports `pts.pyspark`, so it keeps
working after that module is deleted, and pointing it at any two releases compares them.

`--baseline` (a fresh pyspark run on the same staged inputs) is the PRIMARY comparison: the
staged inputs are not guaranteed to be the exact snapshot that produced `--published`, so a
diff against published alone is not evidence of a regression -- see reactome in
`.superpowers/sdd/plan/task-12-requirements.md`. `--published` is a secondary sanity check.
Both are optional so the tool still works with whichever is available; at least one is required.

`id` is excluded from every value-level check. Measured: the documented id formula does not
reproduce the pyspark implementation's own stored ids (0 of 4,223 rows on intogen, in both a
fresh pyspark run and the published output), while the two id SETS are identical -- so the
polars port computing the formula correctly is *expected* to diverge from both baselines there.

Rows are NOT compared by joining on the natural key: measured on intogen, the key is not
unique (2,590 keys for 4,223 rows), so a key join is many-to-many and produces a cartesian
artifact that silently drops real mismatches. The natural key is used only for a set-overlap
count. Row values are compared as a sorted multiset (a per-row hash, order-independent, exact
match count) over every shared, same-dtype column except `id` and `score`. `score` is excluded
from that hash and compared separately at a relative tolerance (`--tolerance`, default 1e-12):
`log10` differs by up to 2 ULP between spark's JVM and polars' Rust, so bit-exactness is not a
realistic bar -- see the requirements doc for the measurements behind both choices.

usage:
    verify_evidence.py --new work/output/evidence_intogen/evidence_intogen.parquet \
        --baseline work/spike/baseline/intogen/evidence \
        --published work/spike/published/evidence_intogen \
        --key targetFromSourceId,diseaseFromSourceMappedId
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

DEFAULT_SCORE_TOLERANCE = 1e-12


def _parquet_source(path: str) -> str | list[str]:
    """The parquet part(s) at `path`, spark's `_`-prefix metadata skip applied.

    `path` is either a single parquet file (the polars port's `sink_parquet` output) or a
    directory of parts (a pyspark `.write.parquet()` output, or a published release's GCS-style
    export) -- both shapes are used across `--new`/`--baseline`/`--published` in practice, so
    this accepts either rather than assuming one. Spark's writer/reader treats a leading `_` as
    metadata (`_SUCCESS`), not data; matched here so a stray metadata-shaped file can't silently
    join the comparison, same convention as `pts.transformers.evidence_postprocess._parquet_parts`.

    Args:
        path: location of one side's evidence output.

    Returns:
        `path` unchanged when it is a file, otherwise its sorted non-`_`-prefixed parts.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if `path` is a directory with no matching part.
    """
    p = Path(path)
    if p.is_file():
        return str(p)
    if not p.is_dir():
        msg = f'{path} does not exist'
        raise FileNotFoundError(msg)
    parts = sorted(str(f) for f in p.glob('*.parquet') if not f.name.startswith('_'))
    if not parts:
        msg = f'no parquet files found in {path}'
        raise ValueError(msg)
    return parts


def _columns(schema: pl.Schema, other: pl.Schema) -> tuple[list[str], list[str], list[str]]:
    """Split `schema` against `other` into (matching-dtype, dtype-mismatched, only-in-schema).

    Args:
        schema: the schema to categorise.
        other: the schema to compare it against.

    Returns:
        `(shared_matching, shared_mismatched, only_in_schema)` column name lists, `schema`'s
        own order preserved within each.
    """
    shared_matching, shared_mismatched, only = [], [], []
    for name in schema.names():
        if name not in other:
            only.append(name)
        elif schema[name] == other[name]:
            shared_matching.append(name)
        else:
            shared_mismatched.append(name)
    return shared_matching, shared_mismatched, only


def _print_row_counts(new: pl.LazyFrame, other: pl.LazyFrame, other_label: str) -> None:
    n_new = new.select(pl.len()).collect().item()
    n_other = other.select(pl.len()).collect().item()
    print(f'rows: new={n_new:,} {other_label}={n_other:,} delta={n_new - n_other:+,}')


def _print_columns(new_schema: pl.Schema, other_schema: pl.Schema, other_label: str) -> tuple[list[str], bool]:
    """Print the column-set/dtype report and return the comparable columns (matching dtype, shared).

    Returns:
        `(comparable, clean)` -- `comparable` excludes `id`; `clean` is False if any column set
        or dtype differs, `id` dtype mismatches included (still worth surfacing even though `id`
        is excluded from every value-level check below).
    """
    shared_matching, shared_mismatched, only_new = _columns(new_schema, other_schema)
    _, _, only_other = _columns(other_schema, new_schema)
    print(f'columns: new={len(new_schema)} {other_label}={len(other_schema)}')
    print(f'  only new       : {only_new}')
    print(f'  only {other_label:10}: {only_other}')
    for name in shared_mismatched:
        print(f'  dtype differs: {name} new={new_schema[name]} {other_label}={other_schema[name]}')
    clean = not (only_new or only_other or shared_mismatched)
    comparable = [c for c in shared_matching if c != 'id']
    return comparable, clean


def _print_null_distinct(new: pl.LazyFrame, other: pl.LazyFrame, other_label: str, scalar_columns: list[str]) -> bool:
    """Print per-column null/distinct counts for the scalar (non-nested) comparable columns.

    Returns:
        True if every column's null and distinct counts match on both sides.
    """
    if not scalar_columns:
        print('\n(no shared scalar columns to check null/distinct counts on)')
        return True

    def agg(lf: pl.LazyFrame) -> dict[str, int]:
        exprs = [pl.col(c).null_count().alias(f'{c}__n') for c in scalar_columns]
        exprs += [pl.col(c).n_unique().alias(f'{c}__u') for c in scalar_columns]
        return lf.select(exprs).collect().row(0, named=True)

    a, b = agg(new), agg(other)
    print(
        f'\n{"column":30} {"nulls new":>14} {f"nulls {other_label}":>14} '
        f'{"uniq new":>12} {f"uniq {other_label}":>12}  ok'
    )
    clean = True
    for c in scalar_columns:
        ok = a[f'{c}__n'] == b[f'{c}__n'] and a[f'{c}__u'] == b[f'{c}__u']
        clean &= ok
        print(
            f'{c:30} {a[f"{c}__n"]:>14,} {b[f"{c}__n"]:>14,} {a[f"{c}__u"]:>12,} {b[f"{c}__u"]:>12,}  '
            f'{"YES" if ok else "NO"}'
        )
    return clean


def _row_multiset(lf: pl.LazyFrame, columns: list[str]) -> pl.DataFrame:
    """A per-row hash of `columns`, collapsed to (hash, count) -- the multiset, not the set.

    `DataFrame.hash_rows` hashes every column together, nested list/struct columns included, so
    this needs no per-type serialisation logic. Collision risk is the usual 64-bit-hash one --
    negligible at these row counts, and no worse than any other hash-based dedup in this codebase.
    """
    return lf.select(columns).collect().hash_rows().value_counts().rename({'': 'hash'})


def _print_multiset(new: pl.LazyFrame, other: pl.LazyFrame, other_label: str, columns: list[str]) -> bool:
    """Print the sorted-multiset row comparison over `columns` (already excludes id and score).

    Returns:
        True if the two multisets are identical (same hashes, same counts each).
    """
    if not columns:
        print('\n(no shared columns left to compare row-for-row after excluding id/score)')
        return True
    new_counts = _row_multiset(new, columns)
    other_counts = _row_multiset(other, columns)
    joined = new_counts.join(other_counts, on='hash', how='full', suffix='_other')
    both = joined.filter(pl.col('count').eq(pl.col('count_other'))).select(pl.col('count').sum()).item() or 0
    only_new = (
        joined.filter(pl.col('count_other').is_null() | (pl.col('count') > pl.col('count_other')))
        .select((pl.col('count').fill_null(0) - pl.col('count_other').fill_null(0)).clip(lower_bound=0).sum())
        .item()
        or 0
    )
    only_other = (
        joined.filter(pl.col('count').is_null() | (pl.col('count_other') > pl.col('count')))
        .select((pl.col('count_other').fill_null(0) - pl.col('count').fill_null(0)).clip(lower_bound=0).sum())
        .item()
        or 0
    )
    print(f'\nrow multiset (excludes id, score; {len(columns)} columns): matching={both:,}')
    print(f'  only new={only_new:,}   only {other_label}={only_other:,}')
    return only_new == 0 and only_other == 0


def _print_score(
    new: pl.LazyFrame, other: pl.LazyFrame, other_label: str, join_columns: list[str], tolerance: float
) -> bool:
    """Pair rows on `join_columns` (the same columns the multiset used) and diff `score`.

    Returns:
        True if `score` is missing from either side (nothing to check) or every paired delta is
        within `tolerance` relative to the `other` side's value.
    """
    print(f'\nscore: relative tolerance={tolerance:g}')
    if 'score' not in new.collect_schema() or 'score' not in other.collect_schema():
        print('  score column missing on one side, skipped')
        return True
    joined = (
        new.select([*join_columns, 'score'])
        # nulls_equal=True: several join columns are nullable (e.g. publicationDate, 938 of
        # intogen's 4,223 rows), and SQL-style null-excludes-null semantics silently dropped a
        # quarter of the pairs here on a real run -- measured 3,228 of 4,223 paired without it.
        # A null in both sides is the same missing value, not two unknowns, for this pairing.
        .join(other.select([*join_columns, 'score']), on=join_columns, how='inner', suffix='_other', nulls_equal=True)
        .with_columns(delta=(pl.col('score') - pl.col('score_other')).abs())
        .with_columns(
            relative=pl.when(pl.col('score_other') == 0)
            .then(pl.col('delta'))
            .otherwise(pl.col('delta') / pl.col('score_other').abs())
        )
        .collect()
    )
    if joined.is_empty():
        print('  no rows paired on the shared non-score columns, skipped')
        return True
    max_delta = joined['delta'].max()
    max_relative = joined['relative'].max()
    mismatched = joined.filter(pl.col('relative') > tolerance).height
    print(f'  paired rows={joined.height:,}  max absolute delta={max_delta:.3e}  max relative delta={max_relative:.3e}')
    print(f'  rows exceeding tolerance={mismatched:,}')
    return mismatched == 0


def _print_key_overlap(new: pl.LazyFrame, other: pl.LazyFrame, other_label: str, key: list[str]) -> None:
    """Print natural-key set overlap. Not used for value agreement -- the key is not unique."""
    kn = new.select(key).unique().collect()
    ko = other.select(key).unique().collect()
    both = kn.join(ko, on=key, how='inner', nulls_equal=True).height
    print(f'\nnatural key {key}: new={kn.height:,} {other_label}={ko.height:,} both={both:,}')
    print(f'  only new={kn.height - both:,}   only {other_label}={ko.height - both:,}')


def compare(new: pl.LazyFrame, other: pl.LazyFrame, other_label: str, key: list[str], tolerance: float) -> bool:
    """Run every check for one `--new` vs `--baseline`/`--published` side, printing as it goes.

    Args:
        new: the polars port's output.
        other: the side to compare it against.
        other_label: `'baseline'` or `'published'`, used in every printed line.
        key: natural-key columns, for the set-overlap count only.
        tolerance: relative tolerance for the `score` column.

    Returns:
        True if this side is a clean match -- columns, null/distinct counts, the row multiset,
        `score` within tolerance, and total natural-key overlap.
    """
    print(f'\n{"=" * 20} new vs {other_label} {"=" * 20}')
    _print_row_counts(new, other, other_label)
    comparable, columns_clean = _print_columns(new.collect_schema(), other.collect_schema(), other_label)
    scalar_columns = [c for c in comparable if not isinstance(new.collect_schema()[c], (pl.List, pl.Struct))]
    nulls_clean = _print_null_distinct(new, other, other_label, scalar_columns)
    multiset_columns = [c for c in comparable if c != 'score']
    multiset_clean = _print_multiset(new, other, other_label, multiset_columns)
    score_clean = _print_score(new, other, other_label, multiset_columns, tolerance)
    _print_key_overlap(new, other, other_label, key)
    clean = columns_clean and nulls_clean and multiset_clean and score_clean
    print(f'\n{other_label}: {"MATCH" if clean else "MISMATCH"}')
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--new', required=True, help='polars evidence_postprocess output')
    parser.add_argument('--baseline', help='pyspark evidence_postprocess output on the same inputs (primary)')
    parser.add_argument('--published', help='released evidence output (secondary sanity check)')
    parser.add_argument('--key', required=True, help='comma-separated natural key columns, for overlap counts only')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_SCORE_TOLERANCE, help='relative tolerance for score')
    args = parser.parse_args()

    if not args.baseline and not args.published:
        parser.error('at least one of --baseline or --published is required')

    key = args.key.split(',')
    new = pl.scan_parquet(_parquet_source(args.new))

    checked, results = [], []
    for label, path in (('baseline', args.baseline), ('published', args.published)):
        if path is None:
            continue
        other = pl.scan_parquet(_parquet_source(path))
        results.append(compare(new, other, label, key, args.tolerance))
        checked.append(label)

    print(f'\n{"=" * 20} coverage {"=" * 20}')
    print(f'checked against: {checked}')
    if 'baseline' not in checked:
        print('no --baseline given: this run only checked against published, which is NOT proof the')
        print('port matches pyspark -- the staged inputs are not guaranteed to be the snapshot that')
        print('produced --published (see task-12-requirements.md).')

    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
