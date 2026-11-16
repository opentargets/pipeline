"""Tests for the drug_molecule transformer."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config

from pts.schemas.disease import schema as disease_index_schema
from pts.schemas.drug_molecule import drug_molecule_schema, molecule_schema
from pts.transformers.drug_molecule import (
    _generate_description,
    _join_semantic,
    _process_clinical_report_indications,
    compute_max_phase_per_drug,
    drug_molecule,
    process_drug_index,
    roll_up_to_parents,
)

# --- Schemas used to build test DataFrames ---

# the two datasets that declare a schema are taken from it, so a column renamed upstream
# breaks these tests rather than leaving them asserting against a stale shape
MOLECULE_SCHEMA = molecule_schema

DISEASE_SCHEMA = {name: disease_index_schema[name] for name in ('id', 'name')}

# clinical_report, chemical_probes and drug_mechanism_of_action declare no schema, so
# these are hand written and hold only the columns this step reads
DRUG_LIST = pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String}))
DISEASE_LIST = pl.List(pl.Struct({'diseaseFromSource': pl.String, 'diseaseId': pl.String}))

CLINICAL_REPORT_SCHEMA = {
    'id': pl.String,
    'clinicalStage': pl.String,
    'drugs': DRUG_LIST,
    'diseases': DISEASE_LIST,
    'qualityControls': pl.List(pl.String),
}

CHEMICAL_PROBES_SCHEMA = {'id': pl.String, 'drugFromSourceId': pl.String, 'drugId': pl.String}

MECHANISM_SCHEMA = {'chemblIds': pl.List(pl.String), 'actionType': pl.String}


def _report(report_id, stage, drugs, diseases, quality_controls=None):
    return {
        'id': report_id,
        'clinicalStage': stage,
        'drugs': [{'drugFromSource': label, 'drugId': drug_id} for label, drug_id in drugs],
        'diseases': (
            None
            if diseases is None
            else [{'diseaseFromSource': label, 'diseaseId': disease_id} for label, disease_id in diseases]
        ),
        'qualityControls': quality_controls or [],
    }


def _molecule(molecule_id, **overrides):
    row = {
        'id': molecule_id,
        'canonicalSmiles': None,
        'inchiKey': None,
        'molblock': None,
        'drugType': 'Small molecule',
        'name': molecule_id,
        'parentId': molecule_id,
        'synonyms': None,
        'tradeNames': None,
        'crossReferences': None,
        'childChemblIds': [],
    }
    row.update(overrides)
    return row


# --- Fixtures ---


@pytest.fixture
def clinical_report_df():
    """A clinical report with multiple drugs, diseases, and stages."""
    return pl.DataFrame(
        [
            _report('report1', 'APPROVAL', [('Drug A', 'CHEMBL1')], [('Disease X', 'EFO_0001')]),
            _report(
                'report2',
                'PHASE_3',
                [('Drug A', 'CHEMBL1'), ('Drug B', 'CHEMBL2')],
                [('Disease Y', 'EFO_0002')],
            ),
            _report('report3', 'PHASE_1', [('Drug C', 'CHEMBL3')], [('Disease X', 'EFO_0001')]),
            # a null drugId should be filtered out
            _report('report4', 'PHASE_2', [('Unknown Drug', None)], [('Disease Z', 'EFO_0003')]),
        ],
        schema=CLINICAL_REPORT_SCHEMA,
        orient='row',
    )


@pytest.fixture
def disease_df():
    """Disease reference data."""
    return pl.DataFrame(
        [
            {'id': 'EFO_0001', 'name': 'Disease X'},
            {'id': 'EFO_0002', 'name': 'Disease Y'},
            {'id': 'EFO_0003', 'name': 'Disease Z'},
        ],
        schema=DISEASE_SCHEMA,
        orient='row',
    )


@pytest.fixture
def molecule_df():
    """Molecule data with various cross-references."""
    return pl.DataFrame(
        [
            _molecule(
                'CHEMBL1',
                canonicalSmiles='C',
                inchiKey='INCHI1',
                molblock='MOLBLOCK_CHEMBL1',
                tradeNames=[{'label': 'TradeA', 'source': 'ChEMBL'}],
                synonyms=[{'label': 'SynA', 'source': 'ChEMBL'}],
                crossReferences=[{'source': 'drugbank', 'ids': ['DB001']}],
            ),
            _molecule('CHEMBL2', drugType='Antibody', crossReferences=[]),
            _molecule('CHEMBL3', canonicalSmiles='CC', inchiKey='INCHI3', molblock='MOLBLOCK_CHEMBL3'),
            # drugbank xref but no clinical report, so it should get the UNKNOWN stage
            _molecule('CHEMBL888', crossReferences=[{'source': 'drugbank', 'ids': ['DB888']}]),
            # no drugbank, no clinical report, no mechanism, no probe: not a drug
            _molecule('CHEMBL999', name='Not A Drug'),
        ],
        schema=MOLECULE_SCHEMA,
        orient='row',
    )


@pytest.fixture
def chemical_probes_df():
    """Chemical probes data."""
    return pl.DataFrame(
        [
            {'id': 'A-1155463', 'drugFromSourceId': 'PD001', 'drugId': 'CHEMBL3'},
            {'id': 'Some Compound', 'drugFromSourceId': 'PD002', 'drugId': None},
        ],
        schema=CHEMICAL_PROBES_SCHEMA,
        orient='row',
    )


@pytest.fixture
def mechanism_df():
    """Mechanism of action data."""
    return pl.DataFrame(
        [{'chemblIds': ['CHEMBL1', 'CHEMBL2'], 'actionType': 'INHIBITOR'}],
        schema=MECHANISM_SCHEMA,
        orient='row',
    )


@pytest.fixture
def drug_index_result(molecule_df, chemical_probes_df, mechanism_df, clinical_report_df, disease_df):
    """The drug index shared across the TestProcessDrugIndex cases."""
    return process_drug_index(
        molecule_df.lazy(), chemical_probes_df, mechanism_df, clinical_report_df, disease_df
    )


def _by_id(frame: pl.DataFrame, column: str) -> dict:
    return dict(zip(frame['id'].to_list(), frame[column].to_list(), strict=True))


# --- Tests for compute_max_phase_per_drug ---


class TestComputeMaxPhasePerDrug:
    def test_basic_max_phase(self, clinical_report_df):
        """CHEMBL1 has APPROVAL and PHASE_3, so the max is APPROVAL."""
        stages = _by_id(compute_max_phase_per_drug(clinical_report_df), 'maximumClinicalStage')
        assert stages['CHEMBL1'] == 'APPROVAL'
        assert stages['CHEMBL2'] == 'PHASE_3'
        assert stages['CHEMBL3'] == 'PHASE_1'

    def test_null_drug_ids_are_excluded(self, clinical_report_df):
        """Drugs with a null drugId should not appear in the results."""
        result = compute_max_phase_per_drug(clinical_report_df)
        assert result['id'].null_count() == 0
        assert 'CHEMBL_MISSING' not in result['id'].to_list()

    @pytest.mark.parametrize('stage', ['WITHDRAWAL', 'PHASE_4'])
    def test_stage_folds_into_approval(self, stage):
        """WITHDRAWAL and PHASE_4 are both treated as APPROVAL for the max computation."""
        report = pl.DataFrame(
            [_report('r', stage, [('Drug W', 'CHEMBL_W')], [('Disease', 'EFO_0001')])],
            schema=CLINICAL_REPORT_SCHEMA,
            orient='row',
        )
        stages = _by_id(compute_max_phase_per_drug(report), 'maximumClinicalStage')
        assert stages['CHEMBL_W'] == 'APPROVAL'

    def test_unrecognised_stage_falls_back_to_unknown(self):
        """A stage that is not in the rank table ranks as UNKNOWN rather than failing."""
        report = pl.DataFrame(
            [_report('r', 'NOT_A_STAGE', [('Drug', 'CHEMBL_U')], [('Disease', 'EFO_0001')])],
            schema=CLINICAL_REPORT_SCHEMA,
            orient='row',
        )
        stages = _by_id(compute_max_phase_per_drug(report), 'maximumClinicalStage')
        assert stages['CHEMBL_U'] == 'UNKNOWN'


# --- Tests for roll_up_to_parents ---


def _stages(rows):
    return pl.DataFrame(rows, schema={'id': pl.String, 'maximumClinicalStage': pl.String}, orient='row')


def _family(rows):
    return pl.DataFrame(rows, schema={'id': pl.String, 'parentId': pl.String}, orient='row')


class TestRollUpToParents:
    def test_parent_takes_the_best_stage_of_its_family(self):
        stages = _stages([('P', 'UNKNOWN'), ('SALT', 'APPROVAL'), ('HYDRATE', 'PHASE_3')])
        family = _family([('P', None), ('SALT', 'P'), ('HYDRATE', 'P')])
        assert _by_id(roll_up_to_parents(stages, family), 'maximumClinicalStage')['P'] == 'APPROVAL'

    def test_children_keep_their_own_stage(self):
        stages = _stages([('P', 'APPROVAL'), ('SALT', 'PHASE_3')])
        family = _family([('P', None), ('SALT', 'P')])
        rolled = _by_id(roll_up_to_parents(stages, family), 'maximumClinicalStage')
        assert rolled == {'P': 'APPROVAL', 'SALT': 'PHASE_3'}

    def test_parent_without_reports_gets_its_childs_stage(self):
        stages = _stages([('SALT', 'PHASE_2')])
        family = _family([('P', None), ('SALT', 'P')])
        rolled = _by_id(roll_up_to_parents(stages, family), 'maximumClinicalStage')
        assert rolled == {'P': 'PHASE_2', 'SALT': 'PHASE_2'}

    def test_parent_better_than_its_children_keeps_its_stage(self):
        stages = _stages([('P', 'APPROVAL'), ('SALT', 'PHASE_1')])
        family = _family([('P', 'P'), ('SALT', 'P')])
        assert _by_id(roll_up_to_parents(stages, family), 'maximumClinicalStage')['P'] == 'APPROVAL'

    def test_ids_outside_the_molecules_keep_their_stage(self):
        stages = _stages([('ORPHAN', 'PHASE_1')])
        rolled = _by_id(roll_up_to_parents(stages, _family([('P', None)])), 'maximumClinicalStage')
        assert rolled == {'ORPHAN': 'PHASE_1'}

    def test_one_row_per_id(self):
        stages = _stages([('P', 'PHASE_1'), ('A', 'APPROVAL'), ('B', 'PHASE_2')])
        family = _family([('P', None), ('A', 'P'), ('B', 'P')])
        assert roll_up_to_parents(stages, family)['id'].sort().to_list() == ['A', 'B', 'P']


# --- Tests for _process_clinical_report_indications ---


class TestProcessClinicalReportIndications:
    def test_basic_indications(self, clinical_report_df, disease_df):
        """Each drug gets one indication struct per disease, at its best stage."""
        rows = _by_id(_process_clinical_report_indications(clinical_report_df, disease_df), 'indications')

        chembl1 = {(i['disease'], i['maxClinicalStage']) for i in rows['CHEMBL1']}
        assert chembl1 == {('EFO_0001', 'APPROVAL'), ('EFO_0002', 'PHASE_3')}

        chembl3 = {(i['disease'], i['maxClinicalStage']) for i in rows['CHEMBL3']}
        assert chembl3 == {('EFO_0001', 'PHASE_1')}

    def test_null_drug_or_disease_excluded(self):
        """Rows where drugId or diseaseId is null should be excluded."""
        reports = pl.DataFrame(
            [
                _report('r1', 'PHASE_2', [('Drug', None)], [('Disease', 'EFO_0001')]),
                _report('r2', 'PHASE_2', [('Drug', 'CHEMBL_X')], [('Disease', None)]),
            ],
            schema=CLINICAL_REPORT_SCHEMA,
            orient='row',
        )
        disease = pl.DataFrame([{'id': 'EFO_0001', 'name': 'Disease'}], schema=DISEASE_SCHEMA, orient='row')
        assert _process_clinical_report_indications(reports, disease).height == 0

    def test_null_diseases_array_is_dropped(self):
        """A report with a null diseases array contributes no indication.

        Exploding a null array yields a null row, which the diseaseId filter removes.
        """
        reports = pl.DataFrame(
            [_report('r', 'PHASE_2', [('Drug', 'CHEMBL_X')], None)],
            schema=CLINICAL_REPORT_SCHEMA,
            orient='row',
        )
        disease = pl.DataFrame([{'id': 'EFO_0001', 'name': 'Disease'}], schema=DISEASE_SCHEMA, orient='row')
        assert _process_clinical_report_indications(reports, disease).height == 0

    def test_efo_name_is_lowercased_and_space_trimmed(self):
        """Names are lowercased and stripped of leading and trailing spaces."""
        reports = pl.DataFrame(
            [_report('r', 'PHASE_2', [('Drug', 'CHEMBL_X')], [('Disease', 'EFO_0001')])],
            schema=CLINICAL_REPORT_SCHEMA,
            orient='row',
        )
        disease = pl.DataFrame([{'id': 'EFO_0001', 'name': '  Disease X  '}], schema=DISEASE_SCHEMA, orient='row')
        rows = _by_id(_process_clinical_report_indications(reports, disease), 'indications')
        assert rows['CHEMBL_X'][0]['efoName'] == 'disease x'

    def test_non_space_whitespace_is_kept(self):
        """Only the space character is stripped, so other whitespace survives the trim.

        A bare `strip_chars()` would take these too, changing the name rather than
        trimming it.
        """
        reports = pl.DataFrame(
            [_report('r', 'PHASE_2', [('Drug', 'CHEMBL_X')], [('Disease', 'EFO_0001')])],
            schema=CLINICAL_REPORT_SCHEMA,
            orient='row',
        )
        disease = pl.DataFrame(
            [{'id': 'EFO_0001', 'name': '\xa0Disease X\xa0'}], schema=DISEASE_SCHEMA, orient='row'
        )
        rows = _by_id(_process_clinical_report_indications(reports, disease), 'indications')
        assert rows['CHEMBL_X'][0]['efoName'] == '\xa0disease x\xa0'

    def test_indications_are_sorted_by_disease_id(self, disease_df):
        """The indication list is ordered by disease id regardless of the input order."""
        reports = pl.DataFrame(
            [
                _report('r1', 'PHASE_2', [('Drug', 'CHEMBL_X')], [('Disease Z', 'EFO_0003')]),
                _report('r2', 'PHASE_2', [('Drug', 'CHEMBL_X')], [('Disease X', 'EFO_0001')]),
                _report('r3', 'PHASE_2', [('Drug', 'CHEMBL_X')], [('Disease Y', 'EFO_0002')]),
            ],
            schema=CLINICAL_REPORT_SCHEMA,
            orient='row',
        )
        rows = _by_id(_process_clinical_report_indications(reports, disease_df), 'indications')
        assert [i['disease'] for i in rows['CHEMBL_X']] == ['EFO_0001', 'EFO_0002', 'EFO_0003']


# --- Tests for _generate_description ---


class TestGenerateDescription:
    def test_approved_drug_single_indication(self):
        result = _generate_description('Small molecule', 'APPROVAL', ['APPROVAL'], ['rheumatoid arthritis'])
        assert result == (
            'Small molecule drug with a maximum clinical stage of Approval, '
            'with an approval for rheumatoid arthritis.'
        )

    def test_phase_3_drug(self):
        result = _generate_description('Antibody', 'PHASE_3', ['PHASE_3'], ['breast cancer'])
        assert result == (
            'Antibody drug with a maximum clinical stage of Phase 3, with 1 investigational indication.'
        )

    def test_multiple_approved_indications(self):
        result = _generate_description(
            'Small molecule',
            'APPROVAL',
            ['APPROVAL', 'APPROVAL', 'APPROVAL'],
            ['disease a', 'disease b', 'disease c'],
        )
        assert 'approval for 3 indications' in result

    def test_two_approved_indications_are_listed_alphabetically(self):
        """Named labels are sorted, so the sentence does not depend on the input order."""
        result = _generate_description(
            'Small molecule', 'APPROVAL', ['APPROVAL', 'APPROVAL'], ['zebra disease', 'aardvark disease']
        )
        assert 'approval for aardvark disease and zebra disease' in result

    def test_mixed_approved_and_investigational(self):
        result = _generate_description(
            'Small molecule', 'APPROVAL', ['APPROVAL', 'PHASE_2'], ['disease a', 'disease b']
        )
        assert 'approval for disease a' in result
        assert '1 investigational indication' in result

    def test_duplicate_indications_are_counted_once(self):
        """Repeated (stage, label) pairs count once."""
        result = _generate_description(
            'Small molecule', 'PHASE_2', ['PHASE_2', 'PHASE_2'], ['disease a', 'disease a']
        )
        assert 'with 1 investigational indication.' in result

    def test_none_drug_type(self):
        assert _generate_description(None, 'PHASE_1', [], []).startswith('Unknown drug')

    def test_no_phase_no_indications(self):
        assert _generate_description('Small molecule', None, [], []) == 'Small molecule drug.'

    def test_multi_indication_phrase(self):
        result = _generate_description(
            'Small molecule', 'APPROVAL', ['APPROVAL', 'PHASE_3'], ['disease a', 'disease b']
        )
        assert 'across all indications' in result

    def test_single_indication_has_no_multi_phrase(self):
        result = _generate_description('Small molecule', 'APPROVAL', ['APPROVAL'], ['disease a'])
        assert 'across all indications' not in result

    def test_no_withdrawal_or_blackbox_in_description(self):
        result = _generate_description('Small molecule', 'APPROVAL', ['APPROVAL'], ['some disease'])
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
    def test_only_drugs_are_kept(self, drug_index_result):
        """Drugbank xref, clinical report, mechanism or probe qualifies; nothing else does."""
        assert sorted(drug_index_result['id'].to_list()) == ['CHEMBL1', 'CHEMBL2', 'CHEMBL3', 'CHEMBL888']

    def test_chemical_probe_gets_probes_drugs_xref(self, drug_index_result):
        """CHEMBL3 is a probe with no existing cross-references, so one is created."""
        xrefs = _by_id(drug_index_result, 'crossReferences')['CHEMBL3']
        assert xrefs == [{'source': 'Probes&Drugs', 'ids': ['PD001']}]

    def test_probes_drugs_xref_is_appended_to_existing(self, molecule_df, mechanism_df, clinical_report_df, disease_df):
        """A probe that already has cross-references keeps them and gains Probes&Drugs."""
        probes = pl.DataFrame(
            [{'id': 'p', 'drugFromSourceId': 'PD009', 'drugId': 'CHEMBL1'}],
            schema=CHEMICAL_PROBES_SCHEMA,
            orient='row',
        )
        result = process_drug_index(molecule_df.lazy(), probes, mechanism_df, clinical_report_df, disease_df)
        assert _by_id(result, 'crossReferences')['CHEMBL1'] == [
            {'source': 'drugbank', 'ids': ['DB001']},
            {'source': 'Probes&Drugs', 'ids': ['PD009']},
        ]

    def test_non_probe_keeps_its_cross_references(self, drug_index_result):
        xrefs = _by_id(drug_index_result, 'crossReferences')
        assert xrefs['CHEMBL1'] == [{'source': 'drugbank', 'ids': ['DB001']}]
        assert xrefs['CHEMBL2'] == []

    def test_drugs_without_clinical_reports_get_unknown_stage(self, drug_index_result):
        stages = _by_id(drug_index_result, 'maximumClinicalStage')
        assert stages['CHEMBL888'] == 'UNKNOWN'
        assert drug_index_result['maximumClinicalStage'].null_count() == 0

    def test_drug_with_only_an_electronic_mechanism_qualifies(
        self, molecule_df, chemical_probes_df, clinical_report_df, disease_df
    ):
        """CHEMBL999 has no drugbank xref, report or probe; an electronic mechanism makes it a drug."""
        mechanisms = pl.DataFrame(
            [
                {'chemblIds': ['CHEMBL1'], 'actionType': 'INHIBITOR', 'curation': 'manual'},
                {'chemblIds': ['CHEMBL999'], 'actionType': 'INHIBITOR', 'curation': 'electronic'},
            ],
            schema={**MECHANISM_SCHEMA, 'curation': pl.String},
            orient='row',
        )
        result = process_drug_index(molecule_df.lazy(), chemical_probes_df, mechanisms, clinical_report_df, disease_df)
        assert 'CHEMBL999' in result['id'].to_list()
        assert _by_id(result, 'maximumClinicalStage')['CHEMBL999'] == 'UNKNOWN'
        assert dict(result.schema) == drug_molecule_schema

    def test_parent_stage_is_rolled_up_from_its_salt(self, chemical_probes_df, mechanism_df, disease_df):
        drugbank = [{'source': 'drugbank', 'ids': ['DB1']}]
        molecules = pl.DataFrame(
            [
                _molecule('CHEMBL_P', parentId=None, childChemblIds=['CHEMBL_S'], crossReferences=drugbank),
                _molecule('CHEMBL_S', parentId='CHEMBL_P'),
            ],
            schema=MOLECULE_SCHEMA,
            orient='row',
        )
        reports = pl.DataFrame(
            [
                _report('r1', 'APPROVAL', [('Salt', 'CHEMBL_S')], [('Disease X', 'EFO_0001')]),
                _report('r2', 'PHASE_2', [('Parent', 'CHEMBL_P')], [('Disease Y', 'EFO_0002')]),
            ],
            schema=CLINICAL_REPORT_SCHEMA,
            orient='row',
        )
        result = process_drug_index(molecules.lazy(), chemical_probes_df, mechanism_df, reports, disease_df)
        assert _by_id(result, 'maximumClinicalStage') == {'CHEMBL_P': 'APPROVAL', 'CHEMBL_S': 'APPROVAL'}

    def test_output_matches_the_declared_schema(self, drug_index_result):
        assert dict(drug_index_result.schema) == drug_molecule_schema

    def test_intermediate_columns_are_dropped(self, drug_index_result):
        for column in ('_isChemicalProbe', '_hasMechanismOfAction', '_probeIds', 'indications'):
            assert column not in drug_index_result.columns

    def test_description_is_populated(self, drug_index_result):
        assert drug_index_result['description'].null_count() == 0

    def test_description_uses_the_undefaulted_stage(self, drug_index_result):
        """CHEMBL888 has no clinical report, so it gets no phase clause despite the UNKNOWN default."""
        assert _by_id(drug_index_result, 'description')['CHEMBL888'] == 'Small molecule drug.'

    def test_no_duplicate_ids(self, drug_index_result):
        assert drug_index_result['id'].n_unique() == drug_index_result.height

    def test_rows_are_sorted_by_id(self, drug_index_result):
        assert drug_index_result['id'].to_list() == sorted(drug_index_result['id'].to_list())

    def test_molblock_passed_through(self, drug_index_result):
        molblocks = _by_id(drug_index_result, 'molblock')
        assert molblocks['CHEMBL1'] == 'MOLBLOCK_CHEMBL1'
        assert molblocks['CHEMBL2'] is None


# --- Tests for the drug_molecule step ---


def test_step_excludes_reports_with_an_invalid_qc_flag(
    tmp_path, molecule_df, chemical_probes_df, mechanism_df, clinical_report_df, disease_df
):
    """A report with a listed flag goes to `excluded` and sets no stage; an unlisted flag keeps the report."""
    flagged = _report(
        'flagged', 'PHASE_4', [('Drug C', 'CHEMBL3')], [('Disease X', 'EFO_0001')], ['PHASE_IV_NOT_APPROVED']
    )
    unlisted = _report('unlisted', 'PHASE_4', [('Drug B', 'CHEMBL2')], [('Disease Y', 'EFO_0002')], ['NO_DISEASE'])
    reports = pl.concat([
        clinical_report_df,
        pl.DataFrame([flagged, unlisted], schema=CLINICAL_REPORT_SCHEMA, orient='row'),
    ])
    frames = {
        'molecule': molecule_df,
        'chemical_probes': chemical_probes_df,
        'mechanism_of_action': mechanism_df,
        'clinical_report': reports,
        'disease': disease_df,
    }
    source = {}
    for name, frame in frames.items():
        path = tmp_path / f'{name}.parquet'
        frame.write_parquet(path)
        source[name] = str(path)
    destination = {'output': str(tmp_path / 'output'), 'excluded': str(tmp_path / 'excluded')}

    drug_molecule(
        source=source,
        destination=destination,
        settings={'invalid_clinical_report_qc': ['PHASE_IV_NOT_APPROVED']},
        config=Mock(spec=Config),
    )

    output = pl.read_parquet(list((tmp_path / 'output').glob('*.parquet')))
    excluded = pl.read_parquet(list((tmp_path / 'excluded').glob('*.parquet')))
    assert excluded['id'].to_list() == ['flagged']
    stages = _by_id(output, 'maximumClinicalStage')
    assert stages['CHEMBL3'] == 'PHASE_1'
    assert stages['CHEMBL2'] == 'APPROVAL'
