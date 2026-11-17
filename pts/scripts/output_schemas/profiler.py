# ruff: noqa: T201
"""Profile parquet datasets at every depth: dtype, nulls, NaN, empty lists, min/max, low-cardinality values, ID shapes.

Usage: python profiler.py <output_dir> <profiles_dir> [dataset ...]
Writes one <dataset>.json per dataset; skips datasets already profiled.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

#: Candidate value shapes, most specific first; a string value gets the first one every value fully matches.
PATTERNS = {
    'ensembl_gene': r'^ENSG\d{11}$',
    'ensembl_transcript': r'^ENST\d{11}$',
    'ensembl_protein': r'^ENSP\d{11}$',
    'sha1': r'^[0-9a-f]{40}$',
    'md5': r'^[0-9a-f]{32}$',
    'chembl': r'^CHEMBL\d+$',
    'rsid': r'^rs\d+$',
    'variant': r'^(\d+|X|Y|MT)_\d+_[ACGTN]+_[ACGTN]+$',
    'variant_or_hash': r'^((\d+|X|Y|MT)_\d+_[ACGTN]+_[ACGTN]+|OTVAR_[0-9a-f]{32})$',
    'iso_date': r'^\d{4}-\d{2}-\d{2}$',
    'digits': r'^\d+$',
    'ontology': r'^[A-Za-z]+_[A-Za-z0-9]+$',
    'curie': r'^[A-Za-z][A-Za-z0-9.]*:[A-Za-z0-9_.-]+$',
}

#: At most this many distinct values makes a string column an enum candidate.
MAX_ENUM = 20


def stats(dtype: pl.DataType, e: pl.Expr) -> dict[str, pl.Expr]:
    out = {'n': e.len(), 'nulls': e.null_count()}
    if dtype.is_float():
        clean = e.fill_nan(None)
        out |= {'nan': e.is_nan().sum(), 'min': clean.min(), 'max': clean.max(), 'inf': e.is_infinite().sum()}
    elif dtype.is_integer() or dtype == pl.Date:
        out |= {'min': e.min(), 'max': e.max()}
    elif isinstance(dtype, pl.List):
        out |= {'empty': (e.list.len() == 0).sum(), 'maxlen': e.list.len().max()}
    elif dtype == pl.String:
        # capped: more than MAX_ENUM distinct values is all that matters
        out |= {
            'values': e.drop_nulls().unique().head(MAX_ENUM + 1).implode(),
            'emptystr': (e.str.len_chars() == 0).sum(),
        }
        out |= {f'pat:{name}': e.str.contains(rx).sum() for name, rx in PATTERNS.items()}
    return out


def merge(into: dict, name: str, value: Any) -> None:
    if name == 'min':
        into[name] = value if into.get(name) is None else (into[name] if value is None else min(into[name], value))
    elif name in {'max', 'maxlen'}:
        into[name] = value if into.get(name) is None else (into[name] if value is None else max(into[name], value))
    elif name == 'values':
        if into.get('values', ()) is None:
            return
        merged = set(into.get('values', ())) | set(value or ())
        into['values'] = merged if len(merged) <= MAX_ENUM else None
    else:
        into[name] = into.get(name, 0) + (value or 0)


def dtype_json(dtype: Any) -> object:
    if isinstance(dtype, pl.List):
        return {'list': dtype_json(dtype.inner)}
    if isinstance(dtype, pl.Struct):
        return {'struct': [[f.name, dtype_json(f.dtype)] for f in dtype.fields]}
    return str(dtype)


def profile(location: Path) -> dict:
    parts = sorted(location.rglob('*.parquet'))
    schema = pl.scan_parquet(parts, hive_partitioning=False).collect_schema()
    values: dict[str, dict] = {}
    rows = 0
    for part in parts:
        frame = pl.read_parquet(part, hive_partitioning=False)
        rows += frame.height
        for name in schema:
            _profile_values(frame.get_column(name), name, values)
    for v in values.values():
        if 'values' in v:
            v['values'] = sorted(v['values']) if v['values'] is not None else None
            if v['values'] is None:
                del v['values']
    return {'rows': rows, 'schema': {n: dtype_json(d) for n, d in schema.items()}, 'paths': values}


def _profile_values(series: pl.Series, path: str, values: dict[str, dict]) -> None:
    """Profile one value and, recursively, the values inside it.

    Each nesting level is unpacked once and profiled as a column of its own.
    """
    stat_exprs = stats(series.dtype, pl.col(path))
    row = series.to_frame(path).select(expr.alias(stat) for stat, expr in stat_exprs.items()).row(0, named=True)
    into = values.setdefault(path, {'dtype': str(series.dtype)})
    for stat, value in row.items():
        merge(into, stat, value.to_list() if isinstance(value, pl.Series) else value)
    if isinstance(series.dtype, pl.List):
        _profile_values(series.explode(keep_nulls=False, empty_as_null=False), f'{path}[]', values)
    elif isinstance(series.dtype, pl.Struct):
        # values of a field only where the struct itself is present
        present = series.drop_nulls()
        for field in series.dtype.fields:
            _profile_values(present.struct.field(field.name), f'{path}/{field.name}', values)


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[3:] or sorted(p.name for p in root.iterdir() if p.is_dir())
    for name in names:
        target = out / f'{name}.json'
        if target.exists():
            continue
        t = time.time()
        try:
            result = profile(root / name)
        except Exception as e:
            print(f'FAILED {name}: {type(e).__name__}: {str(e)[:300]}', flush=True)
            continue
        target.write_text(json.dumps(result, indent=1, default=str))
        print(f'{name}: {result["rows"]:,} rows, {len(result["paths"])} paths, {time.time() - t:.1f}s', flush=True)


if __name__ == '__main__':
    main()
