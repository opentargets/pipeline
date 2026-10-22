"""Validate every discovered model against its 26.09 output, one parquet part at a time.

Usage: python validate_outputs.py <output_dir> [dataset ...]
Row rules hold per part; primary-key uniqueness is checked on the whole dataset separately (generate.py).
"""

import sys
import time
from pathlib import Path

import pandera.errors as pe
import polars as pl

from pts.schemas.dataset import to_pandera
from pts.schemas.dataset.discovery import discover_datasets

root = Path(sys.argv[1])
datasets = discover_datasets()
names = sys.argv[2:] or sorted(datasets)
for name in names:
    schema = to_pandera(datasets[name])
    # uniqueness is a whole-dataset property; drop it for per-part checks
    schema = schema.update_columns({c: {'unique': False} for c in schema.columns})
    schema.unique = None
    parts = sorted((root / name).rglob('*.parquet'))
    t = time.time()
    failures = []
    rows = 0
    for part in parts:
        df = pl.read_parquet(part)
        rows += df.height
        try:
            schema.validate(df, lazy=True)
        except pe.SchemaErrors as e:
            failures.append((part.name, str(e)[:1500]))
            break
    status = 'OK' if not failures else 'FAILED'
    print(f'{status} {name}: {rows:,} rows in {len(parts)} parts, {time.time() - t:.1f}s', flush=True)
    for part, message in failures:
        print(f'   {part}: {message}', flush=True)
