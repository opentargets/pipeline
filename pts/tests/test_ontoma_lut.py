"""Tests for the polars port of OnToma's lookup-table builders, checked against OnToma's output on the same fixtures."""

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from ontoma import spark_nlp_coordinate

from pts.schemas.ontoma_lut import ENTITY_LUT, RAW_ENTITY_LUT
from pts.transformers.ontoma_lut.build import (
    build_lut,
    curation_label_lut,
    disease_id_lut,
    disease_label_lut,
    drug_label_lut,
    target_label_lut,
)
from pts.transformers.utils.dataset import write_dataset
from pts.transformers.utils.disease_mapping import normalise as normalise_module
from tests.ontoma_lut_fixtures import CURATION, DISEASE, DRUG, TARGET
from tests.ontoma_lut_spark_rows import (
    CURATION_LABEL,
    DISEASE_ID,
    DISEASE_LABEL,
    DRUG_LABEL,
    LUT,
    TARGET_LABEL,
)


def test_disease_label_lut_matches_ontoma() -> None:
    assert disease_label_lut(DISEASE).rows() == DISEASE_LABEL


def test_disease_id_lut_matches_ontoma() -> None:
    assert disease_id_lut(DISEASE).rows() == DISEASE_ID


def test_curation_label_lut_matches_ontoma() -> None:
    out = curation_label_lut(CURATION, 'ot_curation', DISEASE)
    assert out.rows() == CURATION_LABEL
    ids = {(label, entity_id) for entity_id, label, *_ in CURATION_LABEL}
    assert ('remapped by rank', 'MONDO_0000003') in ids
    assert ('obsolete label', 'MONDO_0000001') in ids
    assert ('label one', 'MONDO_0000010') in ids
    assert not {label for _, label in ids} & {'two-id label', 'no match', 'unicode word', 'carriage return', '1)'}


def test_target_label_lut_matches_ontoma() -> None:
    assert target_label_lut(TARGET).rows() == TARGET_LABEL
    assert not {row for row in TARGET_LABEL if row[0] in {'ENSG02', 'ENSG03'}}


def test_drug_label_lut_matches_ontoma() -> None:
    assert drug_label_lut(DRUG).rows() == DRUG_LABEL
    labels = {(entity_id, label) for entity_id, label, *_ in DRUG_LABEL}
    assert {('CHEMBL1', 'symkevi'), ('CHEMBL1', 'Trikafta'), ('CHEMBL4', 'ChEMBL4'), ('CHEMBL6', 'CHEMBL6')} <= labels
    assert not {entity_id for entity_id, _ in labels} & {'CHEMBL2', 'CHEMBL3', 'CHEMBL8'}


@pytest.mark.parametrize(
    ('build', 'frame'),
    [(disease_label_lut, DISEASE), (disease_id_lut, DISEASE), (target_label_lut, TARGET), (drug_label_lut, DRUG)],
)
def test_schema_is_raw_entity_lut(build, frame) -> None:
    assert build(frame).schema == RAW_ENTITY_LUT
    assert build(frame.clear()).schema == RAW_ENTITY_LUT


def test_curation_schema_is_raw_entity_lut() -> None:
    assert curation_label_lut(CURATION, 'eva_clinvar', DISEASE).schema == RAW_ENTITY_LUT
    assert curation_label_lut(CURATION.clear(), 'eva_clinvar', DISEASE).schema == RAW_ENTITY_LUT


def _raw(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=RAW_ENTITY_LUT, orient='row')


def _ids(lut: pl.DataFrame) -> dict[tuple, list[tuple]]:
    return {
        (kind, entity_type, label): [(i['entityId'], i['entitySource']) for i in ids]
        for kind, entity_type, label, ids in lut.rows()
    }


def test_empty_normalised_label_dropped() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_1', 'the', 1.0, 'term', 'name', 'DS', 'label'),
            ('MONDO_2', '--', 1.0, 'symbol', 'name', 'DS', 'label'),
            ('MONDO_3', 'Cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
        )
    ])
    assert _ids(lut) == {('label', 'DS', 'cysticfibrosi'): [('MONDO_3', 'name')]}


def test_ties_kept_and_lower_scores_dropped() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_2', 'Cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
            ('MONDO_3', 'cystic-fibrosis', 0.999, 'term', 'exact_synonym', 'DS', 'label'),
        ),
        _raw(('MONDO_1', 'fibrosis, cystic', 1.0, 'term', 'ot_curation', 'DS', 'label')),
    ])
    assert _ids(lut) == {('label', 'DS', 'cysticfibrosi'): [('MONDO_1', 'ot_curation'), ('MONDO_2', 'name')]}


def test_same_id_two_sources_two_structs() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_1', 'Cystic fibrosis', 1.0, 'term', 'ot_curation', 'DS', 'label'),
            ('MONDO_1', 'Cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
            ('MONDO_1', 'cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
        )
    ])
    assert _ids(lut) == {('label', 'DS', 'cysticfibrosi'): [('MONDO_1', 'name'), ('MONDO_1', 'ot_curation')]}


def test_types_ranked_separately() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_1', 'Foo', 1.0, 'term', 'name', 'DS', 'label'),
            ('ENSG01', 'FOO', 0.999, 'symbol', 'symbol_synonym', 'GP', 'label'),
            ('MONDO_2', 'FOO', 0.998, 'symbol', 'crossref', 'DS', 'id'),
        )
    ])
    assert _ids(lut) == {
        ('id', 'DS', 'foo'): [('MONDO_2', 'crossref')],
        ('label', 'DS', 'foo'): [('MONDO_1', 'name')],
        ('label', 'GP', 'foo'): [('ENSG01', 'symbol_synonym')],
    }


def test_null_scores_rank_last() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_1', 'Foo', 0.5, 'term', 'name', 'DS', 'label'),
            ('MONDO_2', 'foo', None, 'term', 'name', 'DS', 'label'),
            ('ENSG01', 'Foo', None, 'term', 'name', 'GP', 'label'),
            ('ENSG02', 'foo', None, 'term', 'name', 'GP', 'label'),
        )
    ])
    assert _ids(lut) == {
        ('label', 'DS', 'foo'): [('MONDO_1', 'name')],
        ('label', 'GP', 'foo'): [('ENSG01', 'name'), ('ENSG02', 'name')],
    }


def test_each_label_normalised_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def normalise(label: str, track: str) -> str:
        calls.append((label, track))
        return label.lower()

    monkeypatch.setattr(normalise_module, 'normalise', normalise)
    build_lut([
        _raw(
            ('MONDO_1', 'Foo', 1.0, 'term', 'name', 'DS', 'label'), ('MONDO_2', 'Foo', 1.0, 'term', 'name', 'DS', 'id')
        ),
        _raw(
            ('ENSG01', 'Foo', 1.0, 'term', 'name', 'GP', 'label'),
            ('ENSG01', 'Foo', 1.0, 'symbol', 'name', 'GP', 'label'),
        ),
    ])
    assert sorted(calls) == [('Foo', 'symbol'), ('Foo', 'term')]


def test_lut_schema_and_order() -> None:
    raw = [disease_label_lut(DISEASE), disease_id_lut(DISEASE), target_label_lut(TARGET), drug_label_lut(DRUG)]
    lut = build_lut(raw)
    assert lut.schema == ENTITY_LUT
    assert lut.equals(lut.sort(ENTITY_LUT.names()[:3]))
    assert lut.select(ENTITY_LUT.names()[:3]).is_unique().all()
    assert lut['entityIds'].equals(lut['entityIds'].list.sort())
    assert build_lut([frame.clear() for frame in raw]).schema == ENTITY_LUT


_SPARK_NLP_JAR = (
    Path.home() / '.ivy2.5.2' / 'jars' / (spark_nlp_coordinate().replace(':', '_', 1).replace(':', '-') + '.jar')
)

# OnToma's constructor needs spark-nlp in the driver JVM, which the shared test session has not loaded.
_ONTOMA = """
import sys
import traceback
from pathlib import Path

import pyspark.sql.functions as f
from ontoma import OnToma, spark_nlp_coordinate
from pyspark.sql import SparkSession

root = Path(sys.argv[1])
spark = (
    SparkSession.builder.master('local[1]')
    .config('spark.jars.packages', spark_nlp_coordinate())
    .config('spark.sql.shuffle.partitions', '1')
    .getOrCreate()
)
try:
    loaded = OnToma(spark, cache_dir=str(root / 'polars_lut'))
    query = spark.createDataFrame([('Cystic fibrosis',), (' Mucoviscidosis ',)], 'label string')
    loaded.map_entities(query, 'mapped', 'label', 'label', type_col=f.lit('DS')).write.parquet(str(root / 'mapped'))
except Exception:
    (root / 'load_error.txt').write_text(traceback.format_exc())
"""


@pytest.fixture(scope='module')
def ontoma(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('ontoma')
    write_dataset(build_lut(_raw_luts()), str(root / 'polars_lut'))
    env = {**os.environ, 'PYSPARK_PYTHON': sys.executable}
    subprocess.run([sys.executable, '-c', _ONTOMA, str(root)], check=True, env=env)
    return root


def _raw_luts() -> list[pl.DataFrame]:
    return [
        disease_label_lut(DISEASE),
        disease_id_lut(DISEASE),
        curation_label_lut(CURATION, 'ot_curation', DISEASE),
        target_label_lut(TARGET),
        drug_label_lut(DRUG),
    ]


needs_spark_nlp = pytest.mark.skipif(
    not _SPARK_NLP_JAR.exists(), reason=f'spark-nlp is not in the ivy cache: {_SPARK_NLP_JAR}'
)


def test_build_lut_matches_ontoma() -> None:
    assert len(LUT) > 50
    assert build_lut(_raw_luts()).rows() == LUT


@pytest.mark.slow
@needs_spark_nlp
def test_ontoma_loads_polars_lut(ontoma: Path) -> None:
    assert not (ontoma / 'load_error.txt').exists(), (ontoma / 'load_error.txt').read_text()
    mapped = pl.read_parquet(ontoma / 'mapped' / '*.parquet').sort('label')
    assert mapped.rows() == [(' Mucoviscidosis ', ['MONDO_0000001']), ('Cystic fibrosis', ['MONDO_0000001'])]
