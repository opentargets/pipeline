"""Tests for the drug_mechanism_of_action module."""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as f
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.drug_mechanism_of_action import _consolidate_duplicate_references

REFERENCE_SCHEMA = StructType([
    StructField('source', StringType()),
    StructField('ids', ArrayType(StringType())),
    StructField('urls', ArrayType(StringType())),
])

MECHANISM_SCHEMA = StructType([
    StructField('mechanismOfAction', StringType()),
    StructField('actionType', StringType()),
    StructField('chemblIds', ArrayType(StringType())),
    StructField('references', ArrayType(REFERENCE_SCHEMA)),
    StructField('targetName', StringType()),
    StructField('targetType', StringType()),
    StructField('targets', ArrayType(StringType())),
])


class TestConsolidateDuplicateReferences:
    @pytest.mark.slow
    def test_parent_and_child_mechanism_are_merged(self, spark):
        """A parent's native mechanism and its child's rolled-up copy collapse into one row.

        Reproduces the CHEMBL479 case: the mechanism is recorded once for
        CHEMBL479 alone, and once for CHEMBL1200916, whose ChEMBL
        `_metadata.all_molecule_chembl_ids` rolls it up to include CHEMBL479 too.
        """
        refs = [Row(source='PubMed', ids=['111'], urls=['u1'])]
        data = [
            Row(
                mechanismOfAction='Serotonin 2a (5-HT2a) receptor antagonist', actionType='ANTAGONIST',
                chemblIds=['CHEMBL479'], references=refs, targetName='5-HT2a',
                targetType='single protein', targets=['ENSG1'],
            ),
            Row(
                mechanismOfAction='Serotonin 2a (5-HT2a) receptor antagonist', actionType='ANTAGONIST',
                chemblIds=['CHEMBL1200916', 'CHEMBL479'], references=refs, targetName='5-HT2a',
                targetType='single protein', targets=['ENSG1'],
            ),
        ]
        df = spark.createDataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)
        rows = result.collect()

        assert len(rows) == 1
        assert sorted(rows[0]['chemblIds']) == ['CHEMBL1200916', 'CHEMBL479']

    @pytest.mark.slow
    def test_distinct_mechanisms_are_not_merged(self, spark):
        """Two genuinely different mechanisms on the same drug must both survive."""
        refs = [Row(source='PubMed', ids=['111'], urls=['u1'])]
        data = [
            Row(
                mechanismOfAction='Serotonin 2a (5-HT2a) receptor antagonist', actionType='ANTAGONIST',
                chemblIds=['CHEMBL479'], references=refs, targetName='5-HT2a',
                targetType='single protein', targets=['ENSG1'],
            ),
            Row(
                mechanismOfAction='Dopamine D2 receptor antagonist', actionType='ANTAGONIST',
                chemblIds=['CHEMBL479'], references=refs, targetName='D2',
                targetType='single protein', targets=['ENSG2'],
            ),
        ]
        df = spark.createDataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)

        assert result.count() == 2
        assert {r['mechanismOfAction'] for r in result.collect()} == {
            'Serotonin 2a (5-HT2a) receptor antagonist',
            'Dopamine D2 receptor antagonist',
        }

    @pytest.mark.slow
    def test_child_drug_still_sees_the_mechanism(self, spark):
        """The child molecule's own page must still resolve the merged mechanism."""
        refs = [Row(source='PubMed', ids=['111'], urls=['u1'])]
        data = [
            Row(
                mechanismOfAction='Serotonin 2a (5-HT2a) receptor antagonist', actionType='ANTAGONIST',
                chemblIds=['CHEMBL479'], references=refs, targetName='5-HT2a',
                targetType='single protein', targets=['ENSG1'],
            ),
            Row(
                mechanismOfAction='Serotonin 2a (5-HT2a) receptor antagonist', actionType='ANTAGONIST',
                chemblIds=['CHEMBL1200916', 'CHEMBL479'], references=refs, targetName='5-HT2a',
                targetType='single protein', targets=['ENSG1'],
            ),
        ]
        df = spark.createDataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)
        exploded = result.withColumn('drugId', f.explode('chemblIds'))

        assert exploded.filter(f.col('drugId') == 'CHEMBL1200916').count() == 1
