"""Tests for drug_mechanism_of_action, which now joins raw ChEMBL tables."""

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.drug_mechanism_of_action import (
    _chembl_target,
    _consolidate_duplicate_references,
    _with_target_chembl_id,
    process_mechanism_of_action,
)

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


@pytest.fixture
def tables(spark: SparkSession) -> dict:
    drug_mechanism = spark.createDataFrame(
        [
            (100, 1000, 1, 'Inhibits enzyme X', 20, 'INHIBITOR'),
            (101, 1001, 3, 'Blocks receptor', 21, 'ANTAGONIST'),
        ],
        'mec_id int, record_id int, molregno int, mechanism_of_action string, tid int, action_type string',
    )
    mechanism_refs = spark.createDataFrame(
        [(1, 100, 'PMID', '12345', 'http://pmid/12345'), (2, 100, 'DOI', '10.1/xyz', 'http://doi/xyz')],
        'mecref_id int, mec_id int, ref_type string, ref_id string, ref_url string',
    )
    molecule_dictionary = spark.createDataFrame(
        [
            (1, 'CHEMBL1', 'MolA', 'Small molecule'),
            (2, 'CHEMBL2', 'ParentA', 'Small molecule'),
            (3, 'CHEMBL3', 'MolB', 'Small molecule'),
        ],
        'molregno int, chembl_id string, pref_name string, molecule_type string',
    )
    molecule_hierarchy = spark.createDataFrame([(1, 2), (2, 2), (3, 3)], 'molregno int, parent_molregno int')
    target_dictionary = spark.createDataFrame(
        [
            (20, 'CHEMBL_T20', 'Target Twenty', 'SINGLE PROTEIN'),
            (21, 'CHEMBL_T21', 'Target TwentyOne', 'SINGLE PROTEIN'),
        ],
        'tid int, chembl_id string, pref_name string, target_type string',
    )
    target_components = spark.createDataFrame(
        [(2001, 20, 300), (2002, 20, 301), (2003, 21, 302)],
        'targcomp_id int, tid int, component_id int',
    )
    component_sequences = spark.createDataFrame(
        [(300, 'P100'), (301, 'P200'), (302, 'P300')],
        'component_id int, accession string',
    )
    gene_df = spark.createDataFrame(
        [('ENSG1', ['P100'], None), ('ENSG2', None, ['P200']), ('ENSG3', None, ['P300'])],
        'id string, uniprot_trembl array<string>, uniprot_swissprot array<string>',
    )
    return {
        'drug_mechanism': drug_mechanism,
        'mechanism_refs': mechanism_refs,
        'molecule_dictionary': molecule_dictionary,
        'molecule_hierarchy': molecule_hierarchy,
        'target_dictionary': target_dictionary,
        'target_components': target_components,
        'component_sequences': component_sequences,
        'gene_df': gene_df,
    }


def rows_by_target(df) -> dict:
    return {r['targetName']: r.asDict(recursive=True) for r in df.collect()}


class TestProcessMechanismOfAction:
    def test_one_row_per_mec_id(self, tables: dict) -> None:
        result = process_mechanism_of_action(**tables)
        # assert on the raw count BEFORE keying, so a fan-out through target_components cannot hide
        assert result.count() == 2

    def test_scalar_fields(self, tables: dict) -> None:
        rows = rows_by_target(process_mechanism_of_action(**tables))
        assert rows['Target Twenty']['mechanismOfAction'] == 'Inhibits enzyme X'
        assert rows['Target Twenty']['actionType'] == 'INHIBITOR'
        assert rows['Target TwentyOne']['mechanismOfAction'] == 'Blocks receptor'
        assert rows['Target TwentyOne']['actionType'] == 'ANTAGONIST'

    def test_chembl_ids_has_molecule_and_parent(self, tables: dict) -> None:
        # Order is [molecule, parent] by construction; assert it directly rather
        # than sorting, so swapping the f.array(...) arguments would fail here.
        rows = rows_by_target(process_mechanism_of_action(**tables))
        assert rows['Target Twenty']['chemblIds'] == ['CHEMBL1', 'CHEMBL2']

    def test_chembl_ids_deduplicates_a_self_parent(self, tables: dict) -> None:
        rows = rows_by_target(process_mechanism_of_action(**tables))
        assert rows['Target TwentyOne']['chemblIds'] == ['CHEMBL3']

    def test_references_grouped_by_ref_type(self, tables: dict) -> None:
        refs = rows_by_target(process_mechanism_of_action(**tables))['Target Twenty']['references']
        by_source = {r['source']: r for r in refs}
        assert by_source['PMID']['ids'] == ['12345']
        assert by_source['PMID']['urls'] == ['http://pmid/12345']
        assert by_source['DOI']['ids'] == ['10.1/xyz']
        assert by_source['DOI']['urls'] == ['http://doi/xyz']

    def test_no_references_is_an_empty_array_not_null(self, tables: dict) -> None:
        rows = rows_by_target(process_mechanism_of_action(**tables))
        assert rows['Target TwentyOne']['references'] == []

    def test_target_components_join_produces_all_component_accessions(self, tables: dict) -> None:
        rows = rows_by_target(process_mechanism_of_action(**tables))
        # tid=20 has two components (accessions P100 and P200); both must resolve to genes
        # without multiplying the mechanism row.
        assert sorted(rows['Target Twenty']['targets']) == ['ENSG1', 'ENSG2']
        assert rows['Target Twenty']['targetType'] == 'single protein'


class TestWithTargetChemblId:
    def test_null_tid_yields_null_target_chembl_id_without_dropping_the_row(self, spark: SparkSession) -> None:
        mechanism = spark.createDataFrame([(100, 20), (101, None)], 'mec_id int, tid int')
        target_dictionary = spark.createDataFrame(
            [(20, 'CHEMBL_T20', 'Target Twenty', 'SINGLE PROTEIN')],
            'tid int, chembl_id string, pref_name string, target_type string',
        )
        result = _with_target_chembl_id(mechanism, target_dictionary)
        rows = {r['mec_id']: r['target_chembl_id'] for r in result.collect()}
        assert rows == {100: 'CHEMBL_T20', 101: None}


class TestChemblTarget:
    def test_two_components_yield_one_row_with_both_genes(self, tables: dict) -> None:
        result = _chembl_target(
            tables['target_dictionary'],
            tables['target_components'],
            tables['component_sequences'],
            tables['gene_df'],
        )
        rows = rows_by_target(result)
        assert len(rows) == 2
        assert rows['Target Twenty']['target_chembl_id'] == 'CHEMBL_T20'
        assert sorted(rows['Target Twenty']['targets']) == ['ENSG1', 'ENSG2']
        assert rows['Target TwentyOne']['targets'] == ['ENSG3']


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
