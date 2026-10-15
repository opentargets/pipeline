"""Tests for literature_cooccurrence step."""

from types import SimpleNamespace

import polars as pl
from pyspark.sql import Row

from pts.pyspark import literature_cooccurrence as step


def test_writes_only_the_cooccurrences(spark, tmp_path, monkeypatch):
    spark.createDataFrame([Row(pmid='1', label='BRCA1')]).write.parquet(str(tmp_path / 'match'))
    cooccurrence = spark.createDataFrame([
        Row(pmid='1', mappedId1='ENSG001', mappedId2='EFO_0000311', evidenceScore=0.8, section='abstract'),
    ])
    matches = []

    class _MatchMapped:
        def __init__(self, match):
            matches.append(match.collect())

        def generate_target_disease_cooccurrences(self):
            return SimpleNamespace(df=cooccurrence)

    monkeypatch.setattr(step, 'Session', lambda **_: SimpleNamespace(load_data=lambda path: spark.read.parquet(path)))
    monkeypatch.setattr(step, 'MatchMapped', _MatchMapped)

    step.literature_cooccurrence(
        {'match': str(tmp_path / 'match')},
        {'cooccurrence': str(tmp_path / 'cooccurrence')},
        {'cooccurrence_coalesce': 1},
        {},
    )

    assert matches == [[Row(pmid='1', label='BRCA1')]]
    written = pl.read_parquet(tmp_path / 'cooccurrence' / '*.parquet')
    assert written.to_dicts() == [
        {'pmid': '1', 'mappedId1': 'ENSG001', 'mappedId2': 'EFO_0000311', 'evidenceScore': 0.8, 'section': 'abstract'}
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ['cooccurrence', 'match']
