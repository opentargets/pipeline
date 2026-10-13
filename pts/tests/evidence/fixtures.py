"""Small disease, target and literature inputs for the evidence step tests."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

TARGET_SCHEMA = {
    'id': pl.String,
    'biotype': pl.String,
    'approvedSymbol': pl.String,
    'proteinIds': pl.List(pl.Struct({'id': pl.String, 'source': pl.String})),
    'hallmarks': pl.Struct({'attributes': pl.List(pl.Struct({'description': pl.String}))}),
}


def write_disease(path: Path, ids: list[str], obsolete: dict[str, list[str]] | None = None) -> str:
    obsolete = obsolete or {}
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {'id': ids, 'obsoleteTerms': [obsolete.get(i, []) for i in ids]},
        schema={'id': pl.String, 'obsoleteTerms': pl.List(pl.String)},
    ).write_parquet(path / 'disease.parquet')
    return str(path)


def write_target(
    path: Path,
    ids: list[str],
    symbols: dict[str, str] | None = None,
    hallmarks: dict[str, list[str]] | None = None,
) -> str:
    symbols = symbols or {}
    hallmarks = hallmarks or {}
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            'id': ids,
            'biotype': ['protein_coding'] * len(ids),
            'approvedSymbol': [symbols.get(i) for i in ids],
            'proteinIds': [[] for _ in ids],
            'hallmarks': [
                {'attributes': [{'description': d} for d in hallmarks[i]]} if i in hallmarks else None for i in ids
            ],
        },
        schema=TARGET_SCHEMA,
    ).write_parquet(path / 'target.parquet')
    return str(path)


def write_literature(path: Path, dates: dict[str, str]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    rows = [{'source': 'MED', 'pmid': pmid, 'id': pmid, 'firstPublicationDate': date} for pmid, date in dates.items()]
    (path / 'part-00000.json').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return str(path)
