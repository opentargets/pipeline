from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pts.transformers.clinical import map_entities
from pts.transformers.clinical import mapping as mapping_module
from tests.clinical.mapping_fixtures import (
    CHEMBL_CURATION,
    DISEASE,
    DRUG,
    NER_CACHE,
    REPORT_SCHEMA,
    REPORTS,
    ner_stub,
)
from tests.clinical.mapping_ontoma_rows import MAPPED, MAPPED_COLUMNS, NER_CACHE_WRITTEN, NER_EXTRACTED_LABELS


@pytest.fixture
def ner_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The labels each drug NER call was given; NER is the fixtures' stub."""
    calls: list[list[str]] = []

    def extract(labels: pl.Series, batch_size: int) -> pl.Series:
        calls.append(labels.to_list())
        return ner_stub(labels, batch_size)

    monkeypatch.setattr(mapping_module, 'extract_drug_entities', extract)
    return calls


def _rows(df: pl.DataFrame) -> list[tuple]:
    return df.sort(df.columns, nulls_last=True).rows()


def _cache(path: Path, frame: pl.DataFrame = NER_CACHE) -> str:
    frame.write_parquet(path)
    return str(path)


def _cache_rows(path: str) -> list[tuple]:
    return _rows(pl.read_parquet(path).with_columns(pl.col('extracted_drugs').list.sort()))


def _reports(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=REPORT_SCHEMA, orient='row')


def _map(reports: pl.DataFrame, cache: str | None, curation: pl.DataFrame | None = None) -> pl.DataFrame:
    return map_entities(reports, DISEASE, DRUG, curation, cache, 8)


def _without_preset_fan_out(rows: list[tuple]) -> list[tuple]:
    """`rows` with each repeat of a row whose report had preset ids dropped."""
    preset = set(REPORTS.filter(pl.col('drugId').is_not_null() | pl.col('diseaseId').is_not_null())['id'])
    kept = [row for row in rows if row[0] not in preset]
    return sorted([*kept, *{row for row in rows if row[0] in preset}], key=lambda row: [(v is None, v) for v in row])


def test_map_entities_matches_ontoma(tmp_path: Path, ner_calls: list[list[str]]) -> None:
    """Mira's rows, except that preset-id rows appear once, unlike mira."""
    cache = _cache(tmp_path / 'ner.parquet')
    mapped = map_entities(REPORTS, DISEASE, DRUG, CHEMBL_CURATION, cache, 8)
    assert mapped.columns == MAPPED_COLUMNS
    expected = _without_preset_fan_out(MAPPED)
    assert len(expected) == len(MAPPED) - 3
    assert _rows(mapped) == expected
    assert sorted(label for call in ner_calls for label in call) == NER_EXTRACTED_LABELS
    assert _cache_rows(cache) == NER_CACHE_WRITTEN


def test_preset_id_row_not_duplicated(ner_calls: list[list[str]]) -> None:
    reports = _reports(
        ('NCT05', 'Shared Brand', 'shared syndrome', None, None),
        ('CHEMBL_IND1', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', None),
    )
    mapped = _map(reports, None)
    assert _rows(mapped.select('id', 'drugId', 'diseaseId')) == [
        ('CHEMBL_IND1', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'CHEMBL_PRESET', 'EFO_0000001'),
        ('CHEMBL_IND2', 'CHEMBL_PRESET', 'EFO_0000002'),
        ('NCT05', 'CHEMBL2', 'EFO_0000001'),
        ('NCT05', 'CHEMBL2', 'EFO_0000002'),
        ('NCT05', 'CHEMBL3', 'EFO_0000001'),
        ('NCT05', 'CHEMBL3', 'EFO_0000002'),
    ]


def test_curation_and_preset_ids_win(tmp_path: Path, ner_calls: list[list[str]]) -> None:
    reports = _reports(
        ('NCT01', 'Aspirin', 'Cystic fibrosis', None, None),
        ('NCT02', 'Ivacaftor 150 mg tablet', 'CF', None, None),
        ('NCT12', 'Aspirin', 'Cystic fibrosis', None, None),
        ('NCT14', 'Ivacaftor 150 mg tablet', 'CF', None, None),
        ('CHEMBL_IND1', 'Aspirin', 'Cystic fibrosis', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'Ivacaftor 150 mg tablet', 'CF', 'CHEMBL_PRESET', 'EFO_PRESET'),
    )
    curation = pl.concat([
        CHEMBL_CURATION,
        pl.DataFrame(
            [('NCT14', 'Ivacaftor 150 mg tablet', 'CHEMBL_CURATED_NER', 'CF', 'EFO_CURATED_NER')],
            schema=CHEMBL_CURATION.schema,
            orient='row',
        ),
    ])
    mapped = _map(reports, _cache(tmp_path / 'ner.parquet'), curation)
    assert _rows(mapped.select('id', 'drugId', 'diseaseId')) == [
        ('CHEMBL_IND1', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('NCT01', 'CHEMBL25', 'MONDO_0009061'),
        ('NCT02', 'CHEMBL2010601', 'MONDO_0009061'),
        ('NCT12', 'CHEMBL_CURATED', 'EFO_CURATED_3'),
        ('NCT14', 'CHEMBL_CURATED_NER', 'EFO_CURATED_NER'),
    ]


def test_ner_cache_hit_not_recomputed(tmp_path: Path, ner_calls: list[list[str]]) -> None:
    cache = _cache(tmp_path / 'ner.parquet')
    before = Path(cache).read_bytes()
    mapped = _map(_reports(('NCT09', 'aspirin tabs', 'Cystic fibrosis', None, None)), cache)
    assert mapped.select('drugId', 'diseaseId').rows() == [('CHEMBL25', 'MONDO_0009061')]
    assert ner_calls == []
    assert Path(cache).read_bytes() == before


def test_ner_cache_written_by_pandas_is_read(tmp_path: Path, ner_calls: list[list[str]]) -> None:
    path = tmp_path / 'ner.parquet'
    table = pa.table(
        {
            'query_label': pa.array(['aspirin tabs'], type=pa.large_string()),
            'extracted_drugs': pa.array([['acetylsalicylic acid']], type=pa.list_(pa.large_string())),
        },
        metadata={b'pandas': b'{"index_columns": [], "columns": []}'},
    )
    pq.write_table(table, path)
    mapped = _map(_reports(('NCT09', 'aspirin tabs', 'Cystic fibrosis', None, None)), str(path))
    assert mapped.select('drugId').item() == 'CHEMBL25'
    assert ner_calls == []


def test_ner_cache_miss_computed_and_written(tmp_path: Path, ner_calls: list[list[str]]) -> None:
    cache = _cache(tmp_path / 'ner.parquet')
    reports = _reports(
        ('NCT07', 'Ivacaftor 150 mg tablet', 'CF', None, None),
        ('NCT09', 'aspirin tabs', 'Cystic fibrosis', None, None),
    )
    mapped = _map(reports, cache)
    assert _rows(mapped.select('id', 'drugId')) == [('NCT07', 'CHEMBL2010601'), ('NCT09', 'CHEMBL25')]
    assert ner_calls == [['Ivacaftor 150 mg tablet']]
    assert _cache_rows(cache) == sorted([*NER_CACHE.rows(), ('Ivacaftor 150 mg tablet', ['ivacaftor'])])


@pytest.mark.parametrize('content', [None, b'not parquet', b''], ids=['missing', 'unreadable', 'empty file'])
def test_missing_or_unreadable_cache_is_empty(
    tmp_path: Path, ner_calls: list[list[str]], content: bytes | None
) -> None:
    path = tmp_path / 'ner' / 'cache.parquet'
    if content is not None:
        path.parent.mkdir()
        path.write_bytes(content)
    mapped = _map(_reports(('NCT09', 'aspirin tabs', 'Cystic fibrosis', None, None)), str(path))
    assert mapped.select('drugId').item() is None
    assert ner_calls == [['aspirin tabs']]
    assert _cache_rows(str(path)) == [('aspirin tabs', [])]


def test_no_cache_path_computes_and_writes_nothing(tmp_path: Path, ner_calls: list[list[str]]) -> None:
    mapped = _map(_reports(('NCT07', 'Ivacaftor 150 mg tablet', 'CF', None, None)), None)
    assert mapped.select('drugId').item() == 'CHEMBL2010601'
    assert ner_calls == [['Ivacaftor 150 mg tablet']]
    assert list(tmp_path.iterdir()) == []


def test_ner_ids_fill_drug_labels_only(tmp_path: Path, ner_calls: list[list[str]]) -> None:
    reports = _reports(('NCT01', 'aspirin tabs', 'aspirin tabs', None, None))
    mapped = _map(reports, _cache(tmp_path / 'ner.parquet'))
    assert mapped.select('drugId', 'diseaseId').rows() == [('CHEMBL25', None)]


def test_all_mapped_skips_lookup(ner_calls: list[list[str]]) -> None:
    reports = _reports(('CHEMBL_IND1', 'Aspirin', None, 'CHEMBL_PRESET', None))
    mapped = _map(reports.drop('diseaseId'), None)
    assert mapped.rows() == [('CHEMBL_IND1', 'Aspirin', None, 'CHEMBL_PRESET', None)]
    assert mapped.columns == MAPPED_COLUMNS
    assert ner_calls == []
