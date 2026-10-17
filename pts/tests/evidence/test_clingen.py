"""Tests for ClinGen evidence."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.clingen import CLINGEN, COLUMNS, clingen, parse_clingen
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol, normalise_term
from tests.evidence.fixtures import write_disease, write_target


def _csv_row(values: list[str]) -> str:
    return ','.join(f'"{v}"' for v in values) + '\n'


#: The real file's preamble: three metadata rows, then the header between two separator rows.
SEPARATOR = _csv_row(['+' * len(c) for c in COLUMNS])
HEADER = (
    _csv_row(['CLINGEN GENE DISEASE VALIDITY CURATIONS'] + [''] * 9)
    + _csv_row(['FILE CREATED: 2026-09-24'] + [''] * 9)
    + _csv_row(['WEBPAGE: https://search.clinicalgenome.org/kb/gene-validity'] + [''] * 9)
    + SEPARATOR
    + _csv_row(COLUMNS)
    + SEPARATOR
)

ROWS = [
    # mapped by label
    (
        'CFTR ',
        'HGNC:1884',
        'cystic fibrosis',
        'MONDO:9999999',
        'AR',
        'SOP10',
        'Definitive',
        'https://r/1',
        '2024-03-14T16:00:00.000Z',
        'Panel A',
    ),
    # mapped by MONDO id only
    (
        'BRCA1',
        'HGNC:1100',
        'unknown label',
        'MONDO:0007254',
        'AD',
        'SOP10',
        'Moderate',
        'https://r/2',
        '2018-06-07T00:00:00',
        'Panel B',
    ),
    # mapped by neither
    (
        'BRCA1',
        'HGNC:1100',
        'nothing at all',
        'MONDO:0000000',
        'AD',
        'SOP10',
        'Refuted',
        'https://r/3',
        'not a date',
        'Panel B',
    ),
]

LUT_SCHEMA = {
    'entityKind': pl.String,
    'entityType': pl.String,
    'entityLabelNormalised': pl.String,
    'entityIds': pl.List(pl.Struct({'entityId': pl.String, 'entitySource': pl.String})),
}


def _write_csv(path: Path) -> str:
    body = ''.join(_csv_row(list(row)) for row in ROWS)
    path.write_text(HEADER + body)
    return str(path)


def _write_lut(path: Path, kind: str, label: str, entity_id: str) -> str:
    path.mkdir(parents=True)
    pl.DataFrame(
        [(kind, 'DS', label, [{'entityId': entity_id, 'entitySource': 'name'}])], schema=LUT_SCHEMA, orient='row'
    ).write_parquet(path / 'part-0.parquet')
    return str(path)


def test_parse_skips_the_preamble_and_shapes_raw_evidence(tmp_path: Path) -> None:
    raw = parse_clingen(_write_csv(tmp_path / 'clingen.csv')).collect()
    assert raw.height == len(ROWS)
    first = raw.row(0, named=True)
    assert first == {
        'datasourceId': 'clingen',
        'datatypeId': 'genetic_literature',
        'targetFromSourceId': 'CFTR',
        'diseaseFromSource': 'cystic fibrosis',
        'diseaseFromSourceId': 'MONDO:9999999',
        'allelicRequirements': ['AR'],
        'urls': [{'url': 'https://r/1'}],
        'confidence': 'Definitive',
        'releaseDate': '2024-03-14',
        'studyId': 'Panel A',
    }


def test_release_date_takes_either_timestamp_form_and_nulls_garbage(tmp_path: Path) -> None:
    raw = parse_clingen(_write_csv(tmp_path / 'clingen.csv')).collect()
    assert raw['releaseDate'].to_list() == ['2024-03-14', '2018-06-07', None]


@pytest.mark.parametrize(
    ('confidence', 'score'),
    [('Definitive', 1.0), ('Strong', 1.0), ('Moderate', 0.5), ('Limited', 0.01), ('Refuted', 0.01), ('Other', None)],
)
def test_score_follows_the_classification(confidence: str, score: float | None) -> None:
    df = pl.DataFrame({'confidence': [confidence]})
    assert df.select(CLINGEN.score).item() == score


def test_maps_scores_and_splits_valid_from_failed(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    dst = tmp_path / 'out'
    clingen(
        {
            'evidence': _write_csv(src / 'clingen.csv'),
            'ontoma_disease_label_lut': _write_lut(
                src / 'label_lut', 'label', normalise_term('cystic fibrosis'), 'MONDO_0009061'
            ),
            'ontoma_disease_id_lut': _write_lut(
                src / 'id_lut', 'id', normalise_symbol('MONDO:0007254'), 'MONDO_0007254'
            ),
            'disease': write_disease(src / 'disease', ['MONDO_0009061', 'MONDO_0007254']),
            'target': write_target(src / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'CFTR', 'ENSG2': 'BRCA1'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid').sort('targetId')
    assert valid.select('targetId', 'diseaseId', 'score', 'evidenceDate').rows() == [
        ('ENSG1', 'MONDO_0009061', 1.0, '2024-03-14'),
        ('ENSG2', 'MONDO_0007254', 0.5, '2018-06-07'),
    ]
    failed = pl.read_parquet(dst / 'failed')
    assert failed['diseaseFromSource'].to_list() == ['nothing at all']
    assert failed['diseaseFromSourceMappedId'].to_list() == [None]
