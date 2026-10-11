"""Tests for the pharmacogenetics module's curated phenotype lookup."""

import json
from unittest.mock import MagicMock

import pytest
from loguru import logger
from pyspark.sql import DataFrame, SparkSession

from pts.pyspark import pharmacogenetics as pgx
from pts.pyspark.pharmacogenetics import annotate_phenotype, find_unparsed_texts

PGX_SCHEMA = 'genotypeAnnotationText string, phenotypeText string, phenotypeFromSourceId string, studyId string'
LUT_SCHEMA = 'genotypeAnnotationText string, phenotypeText array<string>'


@pytest.fixture
def pgx_df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            ('parsed', 'verbose', None, 's1'),
            ('parsed empty', 'verbose', None, 's2'),
            ('unparsed b', 'verbose', None, 's3'),
            ('unparsed a', 'verbose', None, 's4'),
            (None, 'verbose', None, 's5'),
        ],
        PGX_SCHEMA,
    )


@pytest.fixture
def lut_df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [('parsed', ['increased response']), ('parsed empty', [])],
        LUT_SCHEMA,
    )


class TestFindUnparsedTexts:
    def test_texts_missing_from_the_lookup_are_returned_sorted(self, pgx_df, lut_df) -> None:
        assert find_unparsed_texts(pgx_df, lut_df) == ['unparsed a', 'unparsed b']

    def test_empty_when_everything_is_parsed(self, spark, lut_df) -> None:
        df = spark.createDataFrame([('parsed', None, None, 's1'), ('parsed empty', None, None, 's2')], PGX_SCHEMA)
        assert find_unparsed_texts(df, lut_df) == []


class TestAnnotatePhenotype:
    def test_unparsed_rows_are_kept_with_a_null_phenotype(self, pgx_df, lut_df) -> None:
        rows = {r.studyId: r.phenotypeText for r in annotate_phenotype(pgx_df, lut_df).collect()}
        assert rows == {
            's1': 'increased response',
            's2': None,
            's3': None,
            's4': None,
            's5': None,
        }


class TestStep:
    """The step writes the unparsed texts and warns, without calling any external API."""

    def _run(self, spark, monkeypatch, tmp_path, pgx_df, lut_df) -> tuple[list[str], list[str]]:
        frames = {'clinpgx': pgx_df, 'phenotypes': lut_df}
        session = MagicMock()
        session.spark = spark
        session.load_data.side_effect = lambda path, **_: frames.get(path, MagicMock())
        monkeypatch.setattr(pgx, 'Session', lambda **_: session)
        monkeypatch.setattr(pgx, 'add_variantid_column', lambda df: df)
        monkeypatch.setattr(pgx, 'add_efo_mapping', lambda **kw: kw['evidence_df'])
        monkeypatch.setattr(pgx, 'enrich_with_drug_and_target_info', lambda df, *_: df)

        unparsed_path = tmp_path / 'unparsed_texts.json'
        messages: list[str] = []
        sink_id = logger.add(messages.append, level='WARNING')
        try:
            pgx.pharmacogenetics(
                source={
                    'clinpgx': 'clinpgx',
                    'phenotypes': 'phenotypes',
                    'chembl_molecule': 'chembl_molecule',
                    'drug_mechanism_of_action': 'moa',
                    'ontoma_disease_label_lut': 'lut',
                },
                destination={'associations': str(tmp_path / 'out'), 'unparsed_texts': str(unparsed_path)},
                settings={'partition_count': 1},
                properties={},
            )
        finally:
            logger.remove(sink_id)
        return json.loads(unparsed_path.read_text()), messages

    def test_unparsed_texts_are_written_and_warned(self, spark, monkeypatch, tmp_path, pgx_df, lut_df) -> None:
        unparsed, messages = self._run(spark, monkeypatch, tmp_path, pgx_df, lut_df)
        assert unparsed == ['unparsed a', 'unparsed b']
        assert any('2 genotypeAnnotationText' in m for m in messages)
        out = spark.read.parquet(str(tmp_path / 'out'))
        assert out.count() == 5
        assert out.where('phenotypeText is null').count() == 4

    def test_an_empty_list_is_written_when_everything_is_parsed(self, spark, monkeypatch, tmp_path, lut_df) -> None:
        df = spark.createDataFrame([('parsed', None, None, 's1')], PGX_SCHEMA)
        unparsed, messages = self._run(spark, monkeypatch, tmp_path, df, lut_df)
        assert unparsed == []
        assert messages == []
