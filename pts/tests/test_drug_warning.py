"""Tests for drug_warning, which now joins raw ChEMBL tables."""

import pytest
from pyspark.sql import SparkSession

from pts.pyspark.drug_warning import process_drug_warnings


@pytest.fixture
def tables(spark: SparkSession) -> dict:
    warnings = spark.createDataFrame(
        [
            (10, 100, 1, 'Withdrawn', 'Cardiotoxicity', 'France', 'bad things', 2009, 'term', 'EFO_1', 'EFO_2'),
            (11, 101, 3, 'Warning', None, 'US', None, None, None, None, None),
        ],
        'warning_id int, record_id int, molregno int, warning_type string, warning_class string, '
        'warning_country string, warning_description string, warning_year int, efo_term string, '
        'efo_id string, efo_id_for_warning_class string',
    )
    refs = spark.createDataFrame(
        [(1, 10, 'ISBN', 'ref-a', 'http://a'), (2, 10, 'DOI', 'ref-b', 'http://b')],
        'warnref_id int, warning_id int, ref_type string, ref_id string, ref_url string',
    )
    molecules = spark.createDataFrame(
        [(1, 'CHEMBL1'), (2, 'CHEMBL2'), (3, 'CHEMBL3')], 'molregno int, chembl_id string'
    )
    hierarchy = spark.createDataFrame([(1, 2), (2, 2), (3, 3)], 'molregno int, parent_molregno int')
    return {'warnings': warnings, 'refs': refs, 'molecules': molecules, 'hierarchy': hierarchy}


def rows_by_id(df) -> dict:
    return {r['id']: r.asDict(recursive=True) for r in df.collect()}


class TestProcessDrugWarnings:
    def test_one_row_per_warning(self, tables: dict) -> None:
        result = process_drug_warnings(**tables)
        # assert on the raw count BEFORE keying, so a fan-out cannot hide
        assert result.count() == 2
        assert sorted(rows_by_id(result)) == [10, 11]

    def test_scalar_fields(self, tables: dict) -> None:
        w = rows_by_id(process_drug_warnings(**tables))[10]
        assert w['warningType'] == 'Withdrawn'
        assert w['toxicityClass'] == 'Cardiotoxicity'
        assert w['country'] == 'France'
        assert w['description'] == 'bad things'
        assert w['year'] == 2009
        assert w['efoTerm'] == 'term'
        assert w['efoId'] == 'EFO_1'
        assert w['efoIdForWarningClass'] == 'EFO_2'

    def test_chembl_ids_has_molecule_and_parent(self, tables: dict) -> None:
        # Order is [molecule, parent] by construction; assert it directly rather
        # than sorting, so swapping the f.array(...) arguments would fail here.
        assert rows_by_id(process_drug_warnings(**tables))[10]['chemblIds'] == ['CHEMBL1', 'CHEMBL2']

    def test_chembl_ids_deduplicates_a_self_parent(self, tables: dict) -> None:
        assert rows_by_id(process_drug_warnings(**tables))[11]['chemblIds'] == ['CHEMBL3']

    def test_references(self, tables: dict) -> None:
        refs = rows_by_id(process_drug_warnings(**tables))[10]['references']
        assert {r['source'] for r in refs} == {'ISBN', 'DOI'}
        assert {r['id'] for r in refs} == {'ref-a', 'ref-b'}
        assert {r['url'] for r in refs} == {'http://a', 'http://b'}

    def test_no_references_is_an_empty_array_not_null(self, tables: dict) -> None:
        assert rows_by_id(process_drug_warnings(**tables))[11]['references'] == []
