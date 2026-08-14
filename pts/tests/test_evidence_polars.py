"""Tests for the polars evidence post-processing."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.utils.validation_lut import (
    build_disease_lut,
    build_publication_lut,
    build_target_lut,
)

TARGET_SCHEMA = {
    'id': pl.String,
    'biotype': pl.String,
    'approvedSymbol': pl.String,
    'proteinIds': pl.List(pl.Struct({'id': pl.String, 'source': pl.String})),
    'hallmarks': pl.Struct({'attributes': pl.List(pl.Struct({'description': pl.String}))}),
}


def _write_parquet(directory: Path, frame: pl.DataFrame) -> str:
    """Write a frame as a single part file, the way the upstream steps lay a dataset out."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / '_SUCCESS').touch()
    frame.write_parquet(directory / 'part-00000.parquet')
    return str(directory)


def _write_ndjson(directory: Path, *parts: list[dict]) -> str:
    """Write one gzipped newline delimited json file per part."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, rows in enumerate(parts):
        payload = '\n'.join(json.dumps(row) for row in rows).encode()
        (directory / f'part-{index:05d}.json.gz').write_bytes(gzip.compress(payload))
    return str(directory)


def _disease(directory: Path, rows: dict) -> str:
    return _write_parquet(
        directory, pl.DataFrame(rows, schema={'id': pl.String, 'obsoleteTerms': pl.List(pl.String)})
    )


def _target(directory: Path, rows: dict) -> str:
    return _write_parquet(directory, pl.DataFrame(rows, schema=TARGET_SCHEMA))


def _hallmarks(*descriptions: str | None) -> dict:
    return {'attributes': [{'description': d} for d in descriptions]}


# --------------------------------------------------------------------------- disease


def test_disease_lut_includes_obsolete_terms(tmp_path: Path) -> None:
    lut = build_disease_lut(_disease(tmp_path / 'disease', {'id': ['EFO_1'], 'obsoleteTerms': [['OLD_1', 'OLD_2']]}))

    assert set(lut['diseaseFromSourceMappedId']) == {'EFO_1', 'OLD_1', 'OLD_2'}
    assert set(lut['diseaseId']) == {'EFO_1'}


def test_disease_lut_handles_null_obsolete_terms(tmp_path: Path) -> None:
    lut = build_disease_lut(_disease(tmp_path / 'disease', {'id': ['EFO_1'], 'obsoleteTerms': [None]}))

    assert lut.to_dicts() == [{'diseaseId': 'EFO_1', 'diseaseFromSourceMappedId': 'EFO_1'}]


def test_disease_lut_keeps_duplicate_rows(tmp_path: Path) -> None:
    """Spark's `_prepare_disease_lut` has no `distinct()`, and 10 real 26.06 diseases hit this.

    Deduplicating would drop the second copy of the evidence the duplicate fans out,
    and that copy is what `validate_uniqueness` later flags as DUPLICATED. Dropping it
    here moves rows out of `failed_evidence`, so the duplicates have to survive.
    """
    path = _disease(tmp_path / 'disease', {'id': ['EFO_1'], 'obsoleteTerms': [['EFO_1', 'OLD_1']]})

    assert sorted(build_disease_lut(path)['diseaseFromSourceMappedId']) == ['EFO_1', 'EFO_1', 'OLD_1']


# --------------------------------------------------------------------------- target


def test_target_lut_keys_on_every_identifier(tmp_path: Path) -> None:
    path = _target(
        tmp_path / 'target',
        {
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'approvedSymbol': ['ABC1'],
            'proteinIds': [[{'id': 'P1', 'source': 'uniprot'}, {'id': 'P2', 'source': 'uniprot'}]],
            'hallmarks': [_hallmarks('oncogene role')],
        },
    )
    lut = build_target_lut(path)

    assert set(lut['targetFromSourceId']) == {'ENSG1', 'P1', 'P2', 'ABC1'}
    assert set(lut['TSorOncogene']) == {'oncogene'}
    assert set(lut['biotype']) == {'protein_coding'}


def test_target_lut_deduplicates_repeated_identifiers(tmp_path: Path) -> None:
    """Spark's `_prepare_target_lut` does `array_distinct` then `distinct`; both are needed."""
    path = _target(
        tmp_path / 'target',
        {
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'approvedSymbol': ['ENSG1'],
            'proteinIds': [[{'id': 'ENSG1', 'source': 'uniprot'}]],
            'hallmarks': [None],
        },
    )

    assert build_target_lut(path)['targetFromSourceId'].to_list() == ['ENSG1']


def test_target_lut_handles_missing_symbol_and_proteins(tmp_path: Path) -> None:
    """A null `proteinIds` nulls the whole `concat_list`, so it has to be filled first."""
    path = _target(
        tmp_path / 'target',
        {
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'approvedSymbol': [None],
            'proteinIds': [None],
            'hallmarks': [None],
        },
    )
    lut = build_target_lut(path)

    assert sorted(lut['targetFromSourceId'], key=lambda v: (v is None, v)) == ['ENSG1', None]
    assert set(lut['TSorOncogene']) == {None}


@pytest.mark.parametrize(
    ('hallmarks', 'expected'),
    [
        (_hallmarks('oncogene role'), 'oncogene'),
        (_hallmarks('tsg role'), 'tsg'),
        (_hallmarks('acts as oncogene and tsg'), 'bivalent'),
        (_hallmarks('oncogene role', 'tsg role'), 'bivalent'),
        (_hallmarks('ONCOGENE in caps'), 'oncogene'),
        (_hallmarks('unrelated annotation'), None),
        (_hallmarks(None), None),
        (_hallmarks(), None),
        (None, None),
    ],
)
def test_target_lut_cancer_gene_assessment(tmp_path: Path, hallmarks: dict | None, expected: str | None) -> None:
    path = _target(
        tmp_path / 'target',
        {
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'approvedSymbol': ['ABC1'],
            'proteinIds': [[]],
            'hallmarks': [hallmarks],
        },
    )

    assert build_target_lut(path)['TSorOncogene'].to_list() == [expected, expected]


# --------------------------------------------------------------------------- publication


def _publication(pmid: str | None, identifier: str, pmcid: str | None, date: str, source: str = 'MED') -> dict:
    return {'pmid': pmid, 'id': identifier, 'pmcid': pmcid, 'firstPublicationDate': date, 'source': source}


def test_publication_lut_keys_on_every_identifier(tmp_path: Path) -> None:
    path = _write_ndjson(tmp_path / 'literature', [_publication('123', 'MED_1', 'PMC1', '2020-01-01')])

    assert build_publication_lut(path).sort('publicationId').to_dicts() == [
        {'publicationDate': '2020-01-01', 'publicationId': '123'},
        {'publicationDate': '2020-01-01', 'publicationId': 'MED_1'},
        {'publicationDate': '2020-01-01', 'publicationId': 'PMC1'},
    ]


def test_publication_lut_drops_null_identifiers(tmp_path: Path) -> None:
    path = _write_ndjson(tmp_path / 'literature', [_publication(None, 'MED_1', None, '2020-01-01')])

    assert build_publication_lut(path)['publicationId'].to_list() == ['MED_1']


def test_publication_lut_filters_on_source(tmp_path: Path) -> None:
    path = _write_ndjson(
        tmp_path / 'literature',
        [
            _publication(None, 'MED_1', None, '2020-01-01', source='MED'),
            _publication(None, 'PPR_1', None, '2020-01-01', source='PPR'),
            _publication(None, 'AGR_1', None, '2020-01-01', source='AGR'),
            _publication(None, 'PAT_1', None, '2020-01-01', source='PAT'),
        ],
    )

    assert sorted(build_publication_lut(path)['publicationId']) == ['AGR_1', 'MED_1', 'PPR_1']


def test_publication_lut_deduplicates(tmp_path: Path) -> None:
    """`pmid` and `id` carry the same value for most MED records — 44% of the real rows."""
    path = _write_ndjson(tmp_path / 'literature', [_publication('123', '123', None, '2020-01-01')])

    assert build_publication_lut(path)['publicationId'].to_list() == ['123']


def test_publication_lut_reads_a_column_absent_from_the_leading_rows(tmp_path: Path) -> None:
    """Polars infers ndjson dtypes from a sample, and the real export has files whose first rows have no pmid.

    Inference then types the column Null, which fails the read outright, or, under
    `ignore_errors`, discards every value in it — 386,627 pmids in one file of the
    26.06 export. The reader pins the schema so the answer does not depend on where
    the nulls happen to fall.
    """
    rows = [_publication(None, f'MED_{i}', None, '2020-01-01') for i in range(200)]
    rows.append(_publication('999', 'MED_999', None, '2020-01-01'))
    path = _write_ndjson(tmp_path / 'literature', rows)

    assert '999' in set(build_publication_lut(path)['publicationId'])


def test_publication_lut_reads_every_part(tmp_path: Path) -> None:
    """Each part infers its own schema, so parts have to be read under one pinned schema."""
    path = _write_ndjson(
        tmp_path / 'literature',
        [_publication(None, 'MED_1', None, '2020-01-01')],
        [_publication('222', 'MED_2', None, '2021-01-01')],
    )

    assert sorted(build_publication_lut(path)['publicationId']) == ['222', 'MED_1', 'MED_2']


# --------------------------------------------------------------------------- reading


@pytest.mark.parametrize(
    ('builder', 'pattern'),
    [(build_disease_lut, '*.parquet'), (build_target_lut, '*.parquet'), (build_publication_lut, '*.json*')],
)
def test_builders_fail_loudly_on_an_empty_dataset(tmp_path: Path, builder, pattern: str) -> None:
    empty = tmp_path / 'empty'
    empty.mkdir()
    (empty / '_SUCCESS').touch()

    with pytest.raises(ValueError, match=f'no {pattern.replace("*", ".")} '):
        builder(str(empty))
