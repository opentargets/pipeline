"""Tests for the drug_warning module."""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as f
from pyspark.sql.types import (
    ArrayType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.drug_warning import _deduplicate_warnings

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
