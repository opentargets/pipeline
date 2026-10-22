"""Tests for ClinGen evidence."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.clingen import CLINGEN, COLUMNS, parse_clingen


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

def _write_csv(path: Path) -> str:
    body = ''.join(_csv_row(list(row)) for row in ROWS)
    path.write_text(HEADER + body)
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


