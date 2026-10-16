"""The gnomAD constraint builder."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from pts.pyspark.target import _build_genetic_constraints
from pts.target.constraint import COLUMNS, RANK, build_constraint, ntile, read_constraint
from tests.target.parity import rows

METRICS = [c for c in COLUMNS if c not in ('gene_id', 'canonical', 'transcript_type', RANK)]


def _row(gene_id: str, rank: str, canonical: str = 'true', transcript_type: str = 'protein_coding') -> dict:
    return {
        'gene_id': gene_id,
        'canonical': canonical,
        'transcript_type': transcript_type,
        RANK: rank,
        **dict.fromkeys(METRICS, '1'),
    }


PARITY_RECORDS = [_row(f'ENSG{i:011d}', str(i)) for i in range(1, 8)] + [
    _row('ENSGNA', 'NA'),
    _row('ENSGX', '9') | {'syn.z_score': ' 2.5 ', 'mis.obs': '3.0', 'lof.exp': '1e2', 'lof.oe': 'x'},
    _row('ENSGF', '10', canonical='false'),
]


def _frame(records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(records, schema=dict.fromkeys(COLUMNS, pl.String))


def _lof(raw: pl.DataFrame) -> dict:
    return {
        r['id']: next(c for c in r['constraint'] if c['constraintType'] == 'lof')
        for r in build_constraint(raw).to_dicts()
    }


def _tsv(lines: list[list[str]]) -> bytes:
    return ''.join('\t'.join(line) + '\n' for line in lines).encode()


def test_read_keeps_the_used_columns_as_text(tmp_path: Path) -> None:
    path = tmp_path / 'gnomad_constraint_metrics.tsv.bgz'
    header = ['extra', *COLUMNS]
    lines = [['x', *[_row(f'ENSG{i}', str(i))[c] for c in COLUMNS]] for i in (1, 2)]
    # bgzip is a concatenation of gzip members
    path.write_bytes(gzip.compress(_tsv([header, lines[0]])) + gzip.compress(_tsv([lines[1]])))
    raw = read_constraint(str(path))
    assert raw.columns == COLUMNS
    assert set(raw.schema.values()) == {pl.String}
    assert raw['gene_id'].to_list() == ['ENSG1', 'ENSG2']


def test_syn_mis_lof_structure() -> None:
    record = _row('ENSG00000141510', '1000') | {
        'syn.z_score': '1.23',
        'syn.obs': '95',
        'mis.oe': '0.9',
        'lof.pLI': '0.99',
        'lof.oe_ci.upper': 'NA',
    }
    (gene,) = build_constraint(_frame([record])).to_dicts()
    syn, mis, lof = gene['constraint']
    assert [syn['constraintType'], mis['constraintType'], lof['constraintType']] == ['syn', 'mis', 'lof']
    assert syn['score'] == pl.Series([1.23], dtype=pl.Float32)[0]
    assert syn['obs'] == 95
    assert syn['upperRank'] is None
    assert lof['upperRank'] == 1000
    assert lof['oeUpper'] is None


def test_only_canonical_transcripts_with_a_type() -> None:
    raw = _frame([_row('A', '1'), _row('B', '2', canonical='false'), _row('C', '3', transcript_type='NA')])
    assert build_constraint(raw)['id'].to_list() == ['A']


def test_sextiles_ignore_unranked_genes() -> None:
    ranked = [_row(f'ENSG{i:011d}', str(i)) for i in range(1, 13)]
    unranked = [_row(f'ENSGNA{i:09d}', 'NA') for i in range(6)]
    lof = _lof(_frame(ranked + unranked))
    assert sorted(lof[r['gene_id']]['upperBin6'] for r in ranked) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert all(lof[r['gene_id']]['upperBin6'] is None for r in unranked)


def test_deciles_are_computed_locally() -> None:
    ranked = [_row(f'ENSG{i:011d}', str(i)) for i in range(1, 21)]
    lof = _lof(_frame([*ranked, _row('ENSGNA', 'NA')]))
    assert sorted(lof[r['gene_id']]['upperBin'] for r in ranked) == [i // 2 for i in range(20)]
    assert lof['ENSGNA']['upperBin'] is None


def test_ntile_spreads_the_remainder_over_the_first_buckets() -> None:
    result = pl.DataFrame({'p': list(range(7))}).select(ntile(3, pl.len().cast(pl.Int64), pl.col('p')))
    assert result.to_series().to_list() == [0, 0, 0, 1, 1, 2, 2]
    fewer = pl.DataFrame({'p': list(range(2))}).select(ntile(6, pl.len().cast(pl.Int64), pl.col('p')))
    assert fewer.to_series().to_list() == [0, 1]


def test_matches_spark(spark) -> None:
    sdf = spark.createDataFrame(PARITY_RECORDS, ', '.join(f'`{c}` string' for c in COLUMNS))
    assert rows(build_constraint(_frame(PARITY_RECORDS)), 'id') == rows(_build_genetic_constraints(sdf), 'id')
