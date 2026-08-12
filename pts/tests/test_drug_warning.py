"""Tests for drug_warning, which now joins raw ChEMBL tables."""

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import (
    ArrayType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.drug_warning import _deduplicate_warnings, process_drug_warnings

REFERENCE_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('source', StringType()),
    StructField('url', StringType()),
])

WARNING_SCHEMA = StructType([
    StructField('chemblIds', ArrayType(StringType())),
    StructField('toxicityClass', StringType()),
    StructField('country', StringType()),
    StructField('description', StringType()),
    StructField('id', LongType()),
    StructField('references', ArrayType(REFERENCE_SCHEMA)),
    StructField('warningType', StringType()),
    StructField('year', LongType()),
    StructField('efoTerm', StringType()),
    StructField('efoId', StringType()),
    StructField('efoIdForWarningClass', StringType()),
])


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


class TestDeduplicateWarnings:
    @pytest.mark.slow
    def test_parent_and_child_warning_are_merged(self, spark):
        """A parent's native warning and its child's rolled-up copy collapse into one row.

        Reproduces the CHEMBL479 case: warning 3961 belongs to CHEMBL479 alone,
        warning 4517 belongs to CHEMBL1200916 but ChEMBL's own
        `_metadata.all_molecule_chembl_ids` rolls it up to include CHEMBL479 too.
        """
        ref = [Row(id='22941581', source='PubMed', url='http://europepmc.org/abstract/MED/22941581')]
        data = [
            Row(
                chemblIds=['CHEMBL479'], toxicityClass='cardiotoxicity', country='Worldwide',
                description='Cardiac arrythmias', id=3961, references=ref, warningType='Withdrawn',
                year=2005, efoTerm='cardiac arrhythmia', efoId='EFO:0004269',
                efoIdForWarningClass='EFO:1001482',
            ),
            Row(
                chemblIds=['CHEMBL1200916', 'CHEMBL479'], toxicityClass='cardiotoxicity', country='Worldwide',
                description='Cardiac arrythmias', id=4517, references=ref, warningType='Withdrawn',
                year=2005, efoTerm='cardiac arrhythmia', efoId='EFO:0004269',
                efoIdForWarningClass='EFO:1001482',
            ),
        ]
        df = spark.createDataFrame(data, schema=WARNING_SCHEMA)

        result = _deduplicate_warnings(df)
        rows = result.collect()

        assert len(rows) == 1
        assert rows[0]['id'] == 3961
        assert sorted(rows[0]['chemblIds']) == ['CHEMBL1200916', 'CHEMBL479']

    @pytest.mark.slow
    def test_distinct_warnings_are_not_merged(self, spark):
        """Two genuinely different warnings on the same drug must both survive."""
        data = [
            Row(
                chemblIds=['CHEMBL479'], toxicityClass='cardiotoxicity', country='Worldwide',
                description='Cardiac arrythmias', id=3961, references=[], warningType='Withdrawn',
                year=2005, efoTerm='cardiac arrhythmia', efoId='EFO:0004269',
                efoIdForWarningClass='EFO:1001482',
            ),
            Row(
                chemblIds=['CHEMBL479'], toxicityClass=None, country='United States',
                description=None, id=100, references=[], warningType='Black Box Warning',
                year=None, efoTerm=None, efoId=None, efoIdForWarningClass=None,
            ),
        ]
        df = spark.createDataFrame(data, schema=WARNING_SCHEMA)

        result = _deduplicate_warnings(df)

        assert result.count() == 2
        assert {r['warningType'] for r in result.collect()} == {'Withdrawn', 'Black Box Warning'}

    @pytest.mark.slow
    def test_child_drug_still_sees_the_warning(self, spark):
        """The child molecule's own page must still resolve the merged warning."""
        ref = [Row(id='22941581', source='PubMed', url='http://europepmc.org/abstract/MED/22941581')]
        data = [
            Row(
                chemblIds=['CHEMBL479'], toxicityClass='cardiotoxicity', country='Worldwide',
                description='Cardiac arrythmias', id=3961, references=ref, warningType='Withdrawn',
                year=2005, efoTerm='cardiac arrhythmia', efoId='EFO:0004269',
                efoIdForWarningClass='EFO:1001482',
            ),
            Row(
                chemblIds=['CHEMBL1200916', 'CHEMBL479'], toxicityClass='cardiotoxicity', country='Worldwide',
                description='Cardiac arrythmias', id=4517, references=ref, warningType='Withdrawn',
                year=2005, efoTerm='cardiac arrhythmia', efoId='EFO:0004269',
                efoIdForWarningClass='EFO:1001482',
            ),
        ]
        df = spark.createDataFrame(data, schema=WARNING_SCHEMA)

        result = _deduplicate_warnings(df)
        exploded = result.withColumn('drugId', f.explode('chemblIds'))

        assert exploded.filter(f.col('drugId') == 'CHEMBL1200916').count() == 1
