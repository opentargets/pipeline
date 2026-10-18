"""Tests for literature label mapping with the polars disease mapping, checked against OnToma's output."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from pts.pyspark.literature_utils.dataset.match import Match
from pts.pyspark.literature_utils.dataset.match_mapped import MatchMapped
from pts.pyspark.literature_utils.label_mapping import lookup_table, map_distinct_labels
from pts.schemas.ontoma_lut import RAW_ENTITY_LUT
from pts.transformers.ontoma_lut.build import (
    build_lut,
    curation_label_lut,
    disease_id_lut,
    disease_label_lut,
    drug_label_lut,
    target_label_lut,
)
from pts.transformers.utils.dataset import write_dataset
from tests.literature_label_mapping_spark_rows import CONSTANTS, EXTRA_LUT, MAPPED, match_rows
from tests.ontoma_lut_fixtures import CURATION, DISEASE, DRUG, TARGET

KEY = ['pmid', 'text', 'label', 'type', 'startInSentence']


@pytest.fixture(scope='module')
def lut() -> pl.DataFrame:
    return build_lut([
        disease_label_lut(DISEASE),
        disease_id_lut(DISEASE),
        curation_label_lut(CURATION, 'ot_curation', DISEASE),
        target_label_lut(TARGET),
        drug_label_lut(DRUG),
        pl.DataFrame(EXTRA_LUT, schema=RAW_ENTITY_LUT, orient='row'),
    ])


@pytest.fixture(scope='module')
def lut_path(lut: pl.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp('lut') / 'lut')
    write_dataset(lut, path)
    return path


def _sorted(rows: list[tuple]) -> list[tuple]:
    return sorted(rows, key=lambda row: [(value is not None, repr(value)) for value in row])


def _map_labels(spark, rows: list[dict], lut_path: str) -> MatchMapped:
    df = spark.createDataFrame(rows, Match.get_schema())
    return Match(_df=df, _schema=Match.get_schema()).map_labels(
        session=SimpleNamespace(spark=spark), label_lut_path=lut_path, label_col_name='label', type_col_name='type'
    )


def _mapped_rows(mapped: MatchMapped) -> list[tuple]:
    rows = []
    for row in mapped.df.collect():
        ids = None if row.entityIds is None else sorted((i.entityId, i.entitySource) for i in row.entityIds)
        rows.append((*(row[c] for c in KEY), row.entityLabelNormalised, ids, row.mappedId, row.isMapped))
    return rows


def _labels(*pairs: tuple[str | None, str | None]) -> pl.DataFrame:
    return pl.DataFrame(list(pairs), schema={'label': pl.String, 'type': pl.String}, orient='row')


def _distinct_rows(mapped: pl.DataFrame) -> list[tuple]:
    return _sorted([
        (
            label,
            entity_type,
            normalised,
            None if ids is None else tuple((i['entityId'], i['entitySource']) for i in ids),
        )
        for label, entity_type, normalised, ids in mapped.rows()
    ])


def _expected_distinct(*pairs: tuple[str | None, str | None]) -> list[tuple]:
    return _sorted(list({(r[2], r[3], r[5], None if r[6] is None else tuple(r[6])) for r in MAPPED if r[2:4] in pairs}))


def test_map_labels_matches_ontoma(spark, lut_path: str) -> None:
    mapped = _map_labels(spark, match_rows(), lut_path)
    assert mapped.df.columns == [
        *Match.get_schema().fieldNames(),
        'entityLabelNormalised',
        'entityIds',
        'mappedId',
        'isMapped',
    ]
    assert _sorted(_mapped_rows(mapped)) == _sorted(MAPPED)
    ends = {(r.label, r.startInSentence, r.endInSentence) for r in mapped.df.collect()}
    assert ends == {(label, start, start + len(label or '')) for _, _, label, _, start, *_ in MAPPED}
    constants = mapped.df.select(*CONSTANTS).distinct().collect()
    assert [row.asDict() for row in constants] == [CONSTANTS]


def test_map_labels_schema(spark, lut_path: str) -> None:
    mapped = _map_labels(spark, match_rows()[:3], lut_path)
    expected = MatchMapped.get_schema()
    assert [(f.name, f.dataType) for f in mapped.df.schema.fields] == [
        (f.name, f.dataType) for f in expected.fields if f.name not in {'validReasons', 'isValid'}
    ]


def test_map_distinct_labels_one_row_per_spelling(lut: pl.DataFrame) -> None:
    pairs = (('Gaucher-Lefèvre disease', 'DS'), ('Ñandú kinase', 'GP'), ('Cystic fibrosis', 'DS'))
    mapped = map_distinct_labels(_labels(*pairs), lut)
    assert mapped.columns == ['label', 'type', 'entityLabelNormalised', 'entityIds']
    assert mapped.schema['entityIds'] == pl.List(pl.Struct({'entityId': pl.String, 'entitySource': pl.String}))
    rows = _distinct_rows(mapped)
    assert rows == _expected_distinct(*pairs)
    assert [(n, ids is None) for label, _, n, ids in rows if label == 'Gaucher-Lefèvre disease'] == [
        ('diseasgaucherlefevr', False),
        ('diseasgaucherlefvr', True),
    ]


def test_spelling_rows_split_mapped_and_unmapped(spark, lut_path: str) -> None:
    rows = [r for r in match_rows() if r['label'] == 'Gaucher-Lefèvre disease']
    mapped = _sorted(_mapped_rows(_map_labels(spark, rows, lut_path)))
    assert [(r[5], r[7], r[8]) for r in mapped] == [
        ('diseasgaucherlefevr', 'MONDO_0000020', True),
        ('diseasgaucherlefvr', None, False),
    ]


def test_identical_rows_collapse(spark, lut_path: str) -> None:
    row = match_rows()[0]
    mapped = _mapped_rows(_map_labels(spark, [row, row, row, {**row, 'startInSentence': 5}], lut_path))
    assert sorted(r[4] for r in mapped) == [0, 5]


def test_null_and_unknown_types(lut: pl.DataFrame) -> None:
    pairs = (
        (None, 'DS'),
        (None, None),
        ('', None),
        ('Cystic fibrosis', None),
        ('Cystic fibrosis', 'XX'),
        ('Cystic fibrosis', 'GP'),
        ('TNF', ''),
    )
    rows = _distinct_rows(map_distinct_labels(_labels(*pairs), lut))
    assert rows == _expected_distinct(*pairs)
    assert {ids for *_, ids in rows} == {None}
    assert [(label, normalised) for label, _, normalised, _ in rows if label is None] == [(None, None), (None, None)]


def test_null_label_rows_kept(spark, lut_path: str) -> None:
    rows = [r for r in match_rows() if r['label'] is None]
    assert _sorted(_mapped_rows(_map_labels(spark, rows, lut_path))) == _sorted([r for r in MAPPED if r[2] is None])


def test_empty_normalised_label_unmapped(spark, lut: pl.DataFrame, lut_path: str) -> None:
    labels = ('', ' ', '#', '--', 'of the', 'ΑΣ', 'Cystic fibrosis #')
    mapped = map_distinct_labels(_labels(*((label, 'DS') for label in labels)), lut)
    assert mapped.height == len(labels)
    assert set(mapped['entityLabelNormalised']) == {''}
    assert mapped['entityIds'].is_null().all()
    rows = [r for r in match_rows() if r['type'] == 'DS' and r['label'] in labels]
    out = _mapped_rows(_map_labels(spark, rows, lut_path))
    assert len(out) == len(labels)
    assert {(r[5], r[6], r[7], r[8]) for r in out} == {('', None, None, False)}


def test_null_entity_id_is_unmapped(spark, lut_path: str) -> None:
    rows = [r for r in match_rows() if r['label'] == 'NOID']
    assert _mapped_rows(_map_labels(spark, rows, lut_path)) == [
        ('1', 's2', 'NOID', 'GP', 9, 'noid', [(None, 'approved_symbol')], None, False)
    ]


@pytest.mark.parametrize(('kind', 'error'), [('missing', 'not found'), ('empty', 'no \\*.parquet files found')])
def test_unreadable_lut_fails(spark, tmp_path: Path, kind: str, error: str) -> None:
    path = tmp_path / 'lut'
    if kind == 'empty':
        path.mkdir()
    mapped = _map_labels(spark, match_rows()[:3], str(path))
    with pytest.raises(Exception, match=error):
        mapped.df.collect()


def test_lookup_table_read_once(lut: pl.DataFrame, tmp_path: Path) -> None:
    path = str(tmp_path / 'lut')
    write_dataset(lut, path)
    first = lookup_table(path)
    shutil.rmtree(path)
    assert lookup_table(path) is first
    assert first.equals(lut.filter(pl.col('entityKind') == 'label').select(first.columns))
