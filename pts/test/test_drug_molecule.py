"""Tests for the drug_molecule module."""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as f
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.drug_molecule import (
    _compute_max_phase_per_drug,
    _generate_description,
    _join_semantic,
    _process_clinical_report_indications,
    process_drug_index,
)

# --- Schemas used to build test DataFrames ---

CLINICAL_REPORT_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('clinicalStage', StringType()),
    StructField(
        'drugs',
        ArrayType(
            StructType([
                StructField('drugFromSource', StringType()),
                StructField('drugId', StringType()),
            ])
        ),
    ),
    StructField(
        'diseases',
        ArrayType(
            StructType([
                StructField('diseaseFromSource', StringType()),
                StructField('diseaseId', StringType()),
            ])
        ),
    ),
    StructField('qualityControls', ArrayType(StringType())),
])

MOLECULE_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('name', StringType()),
    StructField('drugType', StringType()),
    StructField('canonicalSmiles', StringType()),
    StructField('inchiKey', StringType()),
    StructField('parentId', StringType()),
    StructField('tradeNames', ArrayType(StringType())),
    StructField('synonyms', ArrayType(StringType())),
    StructField(
        'crossReferences',
        ArrayType(
            StructType([
                StructField('source', StringType()),
                StructField('ids', ArrayType(StringType())),
            ])
        ),
    ),
    StructField('childChemblIds', ArrayType(StringType())),
    StructField('description', StringType()),
])

DISEASE_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('name', StringType()),
])

CHEMICAL_PROBES_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('drugFromSourceId', StringType()),
    StructField('drugId', StringType()),
])

MECHANISM_SCHEMA = StructType([
    StructField('chemblIds', ArrayType(StringType())),
    StructField('actionType', StringType()),
])


# --- Fixtures ---


@pytest.fixture
def clinical_report_df(spark):
    """A clinical report with multiple drugs, diseases, and stages."""
    data = [
        Row(
            id='report1',
            clinicalStage='APPROVAL',
            drugs=[
                Row(drugFromSource='Drug A', drugId='CHEMBL1'),
            ],
            diseases=[
                Row(diseaseFromSource='Disease X', diseaseId='EFO_0001'),
            ],
            qualityControls=[],
        ),
        Row(
            id='report2',
            clinicalStage='PHASE_3',
            drugs=[
                Row(drugFromSource='Drug A', drugId='CHEMBL1'),
                Row(drugFromSource='Drug B', drugId='CHEMBL2'),
            ],
            diseases=[
                Row(diseaseFromSource='Disease Y', diseaseId='EFO_0002'),
            ],
            qualityControls=[],
        ),
        Row(
            id='report3',
            clinicalStage='PHASE_1',
            drugs=[
                Row(drugFromSource='Drug C', drugId='CHEMBL3'),
            ],
            diseases=[
                Row(diseaseFromSource='Disease X', diseaseId='EFO_0001'),
            ],
            qualityControls=[],
        ),
        # Report with null drugId should be filtered out
        Row(
            id='report4',
            clinicalStage='PHASE_2',
            drugs=[
                Row(drugFromSource='Unknown Drug', drugId=None),
            ],
            diseases=[
                Row(diseaseFromSource='Disease Z', diseaseId='EFO_0003'),
            ],
            qualityControls=[],
        ),
    ]
    return spark.createDataFrame(data, schema=CLINICAL_REPORT_SCHEMA)


@pytest.fixture
def disease_df(spark):
    """Disease reference data."""
    data = [
        Row(id='EFO_0001', name='Disease X'),
        Row(id='EFO_0002', name='Disease Y'),
        Row(id='EFO_0003', name='Disease Z'),
    ]
    return spark.createDataFrame(data, schema=DISEASE_SCHEMA)


@pytest.fixture
def molecule_df(spark):
    """Molecule data with various cross-references."""
    data = [
        Row(
            id='CHEMBL1',
            name='Drug A',
            drugType='Small molecule',
            canonicalSmiles='C',
            inchiKey='INCHI1',
            parentId='CHEMBL1',
            tradeNames=['TradeA'],
            synonyms=['SynA'],
            crossReferences=[Row(source='drugbank', ids=['DB001'])],
            childChemblIds=[],
            description=None,
        ),
        Row(
            id='CHEMBL2',
            name='Drug B',
            drugType='Antibody',
            canonicalSmiles=None,
            inchiKey=None,
            parentId='CHEMBL2',
            tradeNames=None,
            synonyms=None,
            crossReferences=[],
            childChemblIds=[],
            description=None,
        ),
        Row(
            id='CHEMBL3',
            name='Drug C',
            drugType='Small molecule',
            canonicalSmiles='CC',
            inchiKey='INCHI3',
            parentId='CHEMBL3',
            tradeNames=None,
            synonyms=None,
            crossReferences=[],
            childChemblIds=[],
            description=None,
        ),
        # A molecule with drugbank xref but no clinical reports (should get UNKNOWN phase)
        Row(
            id='CHEMBL888',
            name='Drug D',
            drugType='Small molecule',
            canonicalSmiles='CCCC',
            inchiKey='INCHI888',
            parentId='CHEMBL888',
            tradeNames=None,
            synonyms=None,
            crossReferences=[Row(source='drugbank', ids=['DB888'])],
            childChemblIds=[],
            description=None,
        ),
        # A molecule that is NOT a drug (no drugbank, no clinical reports, no mechanism, no probe)
        Row(
            id='CHEMBL999',
            name='Not A Drug',
            drugType='Small molecule',
            canonicalSmiles='CCC',
            inchiKey='INCHI999',
            parentId='CHEMBL999',
            tradeNames=None,
            synonyms=None,
            crossReferences=[],
            childChemblIds=[],
            description=None,
        ),
    ]
    return spark.createDataFrame(data, schema=MOLECULE_SCHEMA)


@pytest.fixture
def chemical_probes_df(spark):
    """Chemical probes data."""
    data = [
        Row(id='A-1155463', drugFromSourceId='PD001', drugId='CHEMBL3'),
        Row(id='Some Compound', drugFromSourceId='PD002', drugId=None),  # null drugId
    ]
    return spark.createDataFrame(data, schema=CHEMICAL_PROBES_SCHEMA)


@pytest.fixture
def mechanism_df(spark):
    """Mechanism of action data."""
    data = [
        Row(chemblIds=['CHEMBL1', 'CHEMBL2'], actionType='INHIBITOR'),
    ]
    return spark.createDataFrame(data, schema=MECHANISM_SCHEMA)


# --- Tests for _compute_max_phase_per_drug ---


class TestComputeMaxPhasePerDrug:
    @pytest.mark.slow
    def test_basic_max_phase(self, spark, clinical_report_df):
        """CHEMBL1 has APPROVAL and PHASE_3 -> max should be 'APPROVAL'."""
        result = _compute_max_phase_per_drug(clinical_report_df)
        rows = {r['id']: r['maximumClinicalStage'] for r in result.collect()}

        assert rows['CHEMBL1'] == 'APPROVAL'
        assert rows['CHEMBL2'] == 'PHASE_3'
        assert rows['CHEMBL3'] == 'PHASE_1'

    @pytest.mark.slow
    def test_null_drug_ids_are_excluded(self, spark, clinical_report_df):
        """Drugs with null drugId should not appear in results."""
        result = _compute_max_phase_per_drug(clinical_report_df)
        ids = [r['id'] for r in result.collect()]
        assert all(drug_id is not None for drug_id in ids)

    @pytest.mark.slow
    def test_withdrawal_maps_to_approval(self, spark):
        """WITHDRAWAL stage should be treated as APPROVAL for max computation."""
        data = [
            Row(
                id='report_w',
                clinicalStage='WITHDRAWAL',
                drugs=[Row(drugFromSource='Drug W', drugId='CHEMBL_W')],
                diseases=[Row(diseaseFromSource='Disease', diseaseId='EFO_0001')],
                qualityControls=[],
            ),
        ]
        cr = spark.createDataFrame(data, schema=CLINICAL_REPORT_SCHEMA)
        result = _compute_max_phase_per_drug(cr)
        rows = {r['id']: r['maximumClinicalStage'] for r in result.collect()}
        assert rows['CHEMBL_W'] == 'APPROVAL'

    def test_phase_4_maps_to_approval(self, spark):
        """PHASE_4 stage should be treated as APPROVAL for max computation."""
        data = [
            Row(
                id='report_p4',
                clinicalStage='PHASE_4',
                drugs=[Row(drugFromSource='Drug P4', drugId='CHEMBL_P4')],
                diseases=[Row(diseaseFromSource='Disease', diseaseId='EFO_0001')],
                qualityControls=[],
            ),
        ]
        cr = spark.createDataFrame(data, schema=CLINICAL_REPORT_SCHEMA)
        result = _compute_max_phase_per_drug(cr)
        rows = {r['id']: r['maximumClinicalStage'] for r in result.collect()}
        assert rows['CHEMBL_P4'] == 'APPROVAL'


# --- Tests for _process_clinical_report_indications ---


class TestProcessClinicalReportIndications:
    @pytest.mark.slow
    def test_basic_indications(self, spark, clinical_report_df, disease_df):
        """Check correct indications are generated per drug."""
        result = _process_clinical_report_indications(clinical_report_df, disease_df)
        rows = {r['id']: r['indications'] for r in result.collect()}

        # CHEMBL1 should have indications for EFO_0001 (approved) and EFO_0002 (phase III)
        chembl1_indications = {(i['disease'], i['maxClinicalStage']) for i in rows['CHEMBL1']}
        assert ('EFO_0001', 'APPROVAL') in chembl1_indications
        assert ('EFO_0002', 'PHASE_3') in chembl1_indications

        # CHEMBL3 should have one indication for EFO_0001 (phase I)
        chembl3_indications = {(i['disease'], i['maxClinicalStage']) for i in rows['CHEMBL3']}
        assert ('EFO_0001', 'PHASE_1') in chembl3_indications

    def test_null_drug_or_disease_excluded(self, spark):
        """Rows where drugId or diseaseId is null should be excluded."""
        data = [
            Row(
                id='report_null',
                clinicalStage='PHASE_2',
                drugs=[Row(drugFromSource='Drug', drugId=None)],
                diseases=[Row(diseaseFromSource='Disease', diseaseId='EFO_0001')],
                qualityControls=[],
            ),
            Row(
                id='report_null2',
                clinicalStage='PHASE_2',
                drugs=[Row(drugFromSource='Drug', drugId='CHEMBL_X')],
                diseases=[Row(diseaseFromSource='Disease', diseaseId=None)],
                qualityControls=[],
            ),
        ]
        cr = spark.createDataFrame(data, schema=CLINICAL_REPORT_SCHEMA)
        disease = spark.createDataFrame(
            [Row(id='EFO_0001', name='Disease')],
            schema=DISEASE_SCHEMA,
        )
        result = _process_clinical_report_indications(cr, disease)
        assert result.count() == 0

    @pytest.mark.slow
    def test_efo_name_is_lowercase_trimmed(self, spark, clinical_report_df, disease_df):
        """EfoName should be lowercase and trimmed."""
        result = _process_clinical_report_indications(clinical_report_df, disease_df)
        rows = {r['id']: r['indications'] for r in result.collect()}
        for indications in rows.values():
            for ind in indications:
                if ind['efoName'] is not None:
                    assert ind['efoName'] == ind['efoName'].strip().lower()


# --- Tests for _generate_description ---


class TestGenerateDescription:
    def test_approved_drug_single_indication(self):
        """Drug with approved stage and one approved indication."""
        result = _generate_description(
            'Small molecule',
            'APPROVAL',
            ['APPROVAL'],
            ['rheumatoid arthritis'],
        )
        assert 'Small molecule drug' in result
        assert 'Approval' in result
        assert 'approval for rheumatoid arthritis' in result

    def test_phase_3_drug(self):
        """Drug in phase III with one investigational indication."""
        result = _generate_description(
            'Antibody',
            'PHASE_3',
            ['PHASE_3'],
            ['breast cancer'],
        )
        assert 'Antibody drug' in result
        assert 'Phase 3' in result
        assert '1 investigational indication' in result

    def test_multiple_approved_indications(self):
        """Drug with many approved indications shows count."""
        result = _generate_description(
            'Small molecule',
            'APPROVAL',
            ['APPROVAL', 'APPROVAL', 'APPROVAL'],
            ['disease a', 'disease b', 'disease c'],
        )
        assert 'approval for 3 indications' in result

    def test_two_approved_indications_listed(self):
        """Drug with exactly two approved indications lists them."""
        result = _generate_description(
            'Small molecule',
            'APPROVAL',
            ['APPROVAL', 'APPROVAL'],
            ['disease a', 'disease b'],
        )
        assert 'disease a' in result
        assert 'disease b' in result

    def test_mixed_approved_and_investigational(self):
        """Drug with both approved and investigational indications."""
        result = _generate_description(
            'Small molecule',
            'APPROVAL',
            ['APPROVAL', 'PHASE_2'],
            ['disease a', 'disease b'],
        )
        assert 'approval for disease a' in result
        assert '1 investigational indication' in result

    def test_none_drug_type(self):
        """None drug type defaults to 'Unknown'."""
        result = _generate_description(None, 'PHASE_1', [], [])
        assert result.startswith('Unknown drug')

    def test_no_phase_no_indications(self):
        """Drug with no clinical data."""
        result = _generate_description('Small molecule', None, [], [])
        assert result == 'Small molecule drug.'

    def test_multi_indication_phrase(self):
        """Drug with multiple indications includes 'across all indications'."""
        result = _generate_description(
            'Small molecule',
            'APPROVAL',
            ['APPROVAL', 'PHASE_3'],
            ['disease a', 'disease b'],
        )
        assert 'across all indications' in result

    def test_no_withdrawal_or_blackbox_in_description(self):
        """Description should not contain withdrawal or black box references."""
        result = _generate_description(
            'Small molecule',
            'APPROVAL',
            ['APPROVAL'],
            ['some disease'],
        )
        assert 'withdrawal' not in result.lower()
        assert 'black box' not in result.lower()


# --- Tests for _join_semantic ---


class TestJoinSemantic:
    def test_empty_list(self):
        assert not _join_semantic([])

    def test_single_item(self):
        assert _join_semantic(['alpha']) == 'alpha'

    def test_two_items(self):
        assert _join_semantic(['alpha', 'beta']) == 'alpha and beta'

    def test_three_items(self):
        assert _join_semantic(['a', 'b', 'c']) == 'a, b and c'


# --- Tests for process_drug_index ---


class TestProcessDrugIndex:
    @pytest.mark.slow
    def test_non_drug_molecules_excluded(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """CHEMBL999 has no drugbank ref, no clinical reports, no mechanism, no probe -> excluded."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        ids = [r['id'] for r in result.collect()]
        assert 'CHEMBL999' not in ids

    @pytest.mark.slow
    def test_drug_with_drugbank_included(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """CHEMBL1 has a drugbank cross-reference -> included."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        ids = [r['id'] for r in result.collect()]
        assert 'CHEMBL1' in ids

    @pytest.mark.slow
    def test_drug_in_clinical_reports_included(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """CHEMBL2 appears in clinical reports -> included."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        ids = [r['id'] for r in result.collect()]
        assert 'CHEMBL2' in ids

    @pytest.mark.slow
    def test_chemical_probe_included(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """CHEMBL3 is a chemical probe -> included."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        ids = [r['id'] for r in result.collect()]
        assert 'CHEMBL3' in ids

    @pytest.mark.slow
    def test_chemical_probe_gets_probes_drugs_xref(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """CHEMBL3 is a chemical probe -> should have probes&drugs cross-reference with probe ID."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        chembl3 = result.filter(f.col('id') == 'CHEMBL3').collect()[0]
        xrefs = {xref['source']: xref['ids'] for xref in chembl3['crossReferences']}
        assert 'Probes&Drugs' in xrefs

    @pytest.mark.slow
    def test_non_probe_has_no_probes_drugs_xref(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """CHEMBL1 is not a chemical probe -> should not have probes&drugs cross-reference."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        chembl1 = result.filter(f.col('id') == 'CHEMBL1').collect()[0]
        xref_sources = [xref['source'] for xref in chembl1['crossReferences']]
        assert 'probes&drugs' not in xref_sources

    def test_max_phase_is_string(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """MaximumClinicalTrialPhase should be a string, not a double."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        phase_field = result.schema['maximumClinicalStage']
        assert phase_field.dataType == StringType()

    @pytest.mark.slow
    def test_drugs_without_clinical_reports_get_unknown_phase(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """Drugs not in clinical reports should have maximumClinicalStage='UNKNOWN'."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        null_phases = result.filter(f.col('maximumClinicalStage').isNull()).count()
        assert null_phases == 0
        # CHEMBL888 has drugbank xref but no clinical reports -> unknown
        chembl888 = result.filter(f.col('id') == 'CHEMBL888').collect()[0]
        assert chembl888['maximumClinicalStage'] == 'UNKNOWN'

    def test_no_blackbox_or_withdrawal_columns(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """Output should not contain blackBoxWarning or hasBeenWithdrawn columns."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        assert 'blackBoxWarning' not in result.columns
        assert 'hasBeenWithdrawn' not in result.columns

    def test_no_intermediate_columns(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """Intermediate columns should be dropped from final output."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        assert 'chemicalProbeDrugId' not in result.columns
        assert 'hasMechanismOfAction' not in result.columns
        assert 'indications' not in result.columns

    @pytest.mark.slow
    def test_description_is_populated(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """All drugs in the output should have a non-null description."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        null_descriptions = result.filter(f.col('description').isNull()).count()
        assert null_descriptions == 0

    @pytest.mark.slow
    def test_no_duplicate_ids(
        self,
        spark,
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    ):
        """Output should have no duplicate drug IDs."""
        result = process_drug_index(
            molecule_df,
            chemical_probes_df,
            mechanism_df,
            clinical_report_df,
            disease_df,
        )
        total = result.count()
        distinct = result.select('id').distinct().count()
        assert total == distinct
