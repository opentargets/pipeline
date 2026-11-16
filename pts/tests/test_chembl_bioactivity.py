"""Tests for chembl_bioactivity, which builds the bioactivity dataset from raw ChEMBL activities."""

import polars as pl
import pytest

from pts.postgres import _check_projection
from pts.schemas.chembl_bioactivity import chembl_bioactivity_schema
from pts.transformers.chembl_bioactivity import DEGRADATION_MODALITY, FLOAT_COLUMNS, TABLES, process_bioactivity

PROTEIN_ID = pl.Struct({'id': pl.String, 'source': pl.String})

ACTIVITY_SCHEMA = {
    'activity_id': pl.Int64,
    'assay_id': pl.Int64,
    'molregno': pl.Int64,
    'doc_id': pl.Int64,
    'standard_type': pl.String,
    'standard_value': pl.Float64,
    'standard_units': pl.String,
    'standard_text_value': pl.String,
    'pchembl_value': pl.Float64,
    'activity_comment': pl.String,
    'action_type': pl.String,
    'modality': pl.String,
    'potential_duplicate': pl.Int16,
}

TPD = DEGRADATION_MODALITY


def _activity(activity_id: int, assay_id: int, standard_type: str, **fields) -> dict:
    row = dict.fromkeys(ACTIVITY_SCHEMA)
    row.update(activity_id=activity_id, assay_id=assay_id, molregno=1, doc_id=1, standard_type=standard_type)
    row.update(fields)
    return row


ACTIVITIES = [
    # bioactivity
    _activity(
        101,
        1,
        'IC50',
        standard_value=41,
        standard_units='nM',
        pchembl_value=7.39,
        action_type='BINDING AGENT',
        activity_comment='Antagonist',
    ),
    _activity(
        102,
        1,
        'Ki',
        standard_value=10,
        standard_units='nM',
        pchembl_value=8,
        molregno=2,
        activity_comment=' Not Active',
    ),
    _activity(103, 2, 'IC50', standard_value=1000, standard_units='nM', pchembl_value=6),  # rat target
    _activity(104, 1, 'IC50', standard_value=100, pchembl_value=7, potential_duplicate=1),
    _activity(105, 3, 'IC50', standard_value=100, pchembl_value=7),  # cell-line target
    _activity(106, 4, 'IC50', standard_value=100, pchembl_value=7),  # ADMET assay
    _activity(107, 5, 'IC50', standard_value=100, standard_units='nM', pchembl_value=7),  # target with no gene
    _activity(108, 1, 'Inhibition', standard_value=50, standard_units='%'),  # no pChEMBL
    _activity(
        109,
        6,
        'EC50',
        standard_value=300,
        standard_units='nM',
        pchembl_value=6.5,
        molregno=3,
        activity_comment='Inconclusive',
    ),  # complex
    # passes every bioactivity prefilter, but is a degradation activity
    _activity(209, 1, 'IC50', standard_value=100, standard_units='nM', pchembl_value=7, modality=TPD),
]

ASSAYS = [
    # assay_id, chembl_id, assay_type, description, assay_organism, confidence_score, tid, variant_id
    (1, 'CHEMBL_A1', 'B', 'Assay 1', 'Homo sapiens', 9, 1, 1),
    (2, 'CHEMBL_A2', 'B', 'Assay 2', None, 9, 2, None),
    (3, 'CHEMBL_A3', 'F', 'Assay 3', None, 1, 3, None),
    (4, 'CHEMBL_A4', 'A', 'Assay 4', None, 9, 1, None),
    (5, 'CHEMBL_A5', 'B', 'Assay 5', None, 9, 5, None),
    (6, 'CHEMBL_A6', 'F', 'Assay 6', None, 7, 4, 2),
]

TARGETS = [
    # tid, chembl_id, pref_name, target_type, tax_id
    (1, 'CHEMBL_T1', 'Epidermal growth factor receptor', 'SINGLE PROTEIN', 9606),
    (2, 'CHEMBL_T2', 'Rat receptor', 'SINGLE PROTEIN', 10116),
    (3, 'CHEMBL_T3', 'HeLa', 'CELL-LINE', 9606),
    (4, 'CHEMBL_T4', 'Complex AB', 'PROTEIN COMPLEX', 9606),
    (5, 'CHEMBL_T5', 'Unmapped protein', 'SINGLE PROTEIN', 9606),
]

TARGET_COMPONENTS = [
    # tid, component_id
    (1, 1), (2, 2), (4, 3), (4, 1), (5, 5),
]  # fmt: skip

COMPONENT_SEQUENCES = [
    # component_id, accession
    (1, 'P00533'),
    (2, 'PRAT01'),
    (3, 'PSHARED'),
    (5, 'PNOGENE'),
]

GENES = [
    # id, proteinIds
    ('ENSG_EGFR', [('P00533', 'uniprot_swissprot')]),
    ('ENSG_A', [('PSHARED', 'uniprot_trembl')]),
    ('ENSG_B', [('PSHARED', 'uniprot_trembl')]),
]


def _frame(rows: list, schema: dict) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, orient='row')


@pytest.fixture
def tables() -> dict[str, pl.DataFrame]:
    return {
        'activities': pl.DataFrame(ACTIVITIES, schema=ACTIVITY_SCHEMA),
        'assays': _frame(
            ASSAYS,
            {
                'assay_id': pl.Int64,
                'chembl_id': pl.String,
                'assay_type': pl.String,
                'description': pl.String,
                'assay_organism': pl.String,
                'confidence_score': pl.Int16,
                'tid': pl.Int64,
                'variant_id': pl.Int64,
            },
        ),
        # activity 102 has no ligand_eff row
        'ligand_eff': _frame([(101, 0.45, 5.1)], {'activity_id': pl.Int64, 'le': pl.Float64, 'lle': pl.Float64}),
        'target_dictionary': _frame(
            TARGETS,
            {
                'tid': pl.Int64,
                'chembl_id': pl.String,
                'pref_name': pl.String,
                'target_type': pl.String,
                'tax_id': pl.Int64,
            },
        ),
        'target_components': _frame(TARGET_COMPONENTS, {'tid': pl.Int64, 'component_id': pl.Int64}),
        'component_sequences': _frame(COMPONENT_SEQUENCES, {'component_id': pl.Int64, 'accession': pl.String}),
        'binding_sites': _frame([(1, 1)], {'site_id': pl.Int64, 'tid': pl.Int64}),
        'site_components': _frame([(1, 1)], {'site_id': pl.Int64, 'component_id': pl.Int64}),
        'molecule_dictionary': _frame(
            [(1, 'CHEMBL_M1'), (2, 'CHEMBL_M2'), (3, 'CHEMBL_M3')], {'molregno': pl.Int64, 'chembl_id': pl.String}
        ),
        'docs': _frame(
            [(1, 'CHEMBL_D1', 12127526, 2002, 'PUBLICATION')],
            {
                'doc_id': pl.Int64,
                'chembl_id': pl.String,
                'pubmed_id': pl.Int64,
                'year': pl.Int32,
                'doc_type': pl.String,
            },
        ),
        'variant_sequences': _frame(
            [(1, 'T790M'), (2, 'UNDEFINED MUTATION')], {'variant_id': pl.Int64, 'mutation': pl.String}
        ),
    }


@pytest.fixture
def gene_df() -> pl.DataFrame:
    return pl.DataFrame(
        [(gene, [{'id': i, 'source': s} for i, s in proteins]) for gene, proteins in GENES],
        schema={'id': pl.String, 'proteinIds': pl.List(PROTEIN_ID)},
        orient='row',
    )


DRUG_SCHEMA = {'id': pl.String, 'parentId': pl.String, 'childChemblIds': pl.List(pl.String)}


@pytest.fixture
def drug_df() -> pl.DataFrame:
    # CHEMBL_M1 is a salt of CHEMBL_D1, which lists it as a child; CHEMBL_M3 is not a drug
    return _frame([('CHEMBL_D1', None, ['CHEMBL_M1']), ('CHEMBL_M2', None, None)], DRUG_SCHEMA)


@pytest.fixture
def moa_df() -> pl.DataFrame:
    return _frame(
        [
            ('INHIBITOR', ['CHEMBL_D1', 'CHEMBL_M1'], ['ENSG_EGFR']),
            ('INHIBITOR', ['CHEMBL_D1'], ['ENSG_EGFR', 'ENSG_A']),  # a second mechanism, same action type
            ('ANTAGONIST', ['CHEMBL_M3'], ['ENSG_B']),
            ('DEGRADER', ['CHEMBL_M3'], ['ENSG_B']),
            ('AGONIST', ['CHEMBL_M2'], ['ENSG_XIAP']),  # a gene the activity's target does not have
        ],
        {'actionType': pl.String, 'chemblIds': pl.List(pl.String), 'targets': pl.List(pl.String)},
    )


@pytest.fixture
def output(tables, gene_df, drug_df, moa_df) -> pl.DataFrame:
    return process_bioactivity(tables, gene_df, drug_df, moa_df)


def _by_id(frame: pl.DataFrame) -> dict[int, dict]:
    """The rows of single-target activities, by activity id."""
    single = frame.filter(pl.col('activityId').is_unique())
    return {row['activityId']: row for row in single.iter_rows(named=True)}


def _by_key(frame: pl.DataFrame) -> dict[tuple[int, str], dict]:
    return {(row['activityId'], row['targetId']): row for row in frame.iter_rows(named=True)}


def _credited(frame: pl.DataFrame, activity_id: int) -> list[tuple[str, bool]]:
    rows = frame.filter(pl.col('activityId') == activity_id)
    return [(row['targetId'], row['complexFlag']) for row in rows.iter_rows(named=True)]


def test_tables_read_every_column_the_step_uses(tables):
    assert {name: list(frame.columns) for name, frame in tables.items()} == TABLES


class TestBioactivity:
    def test_schema(self, output):
        assert output.schema == chembl_bioactivity_schema

    def test_prefilters_keep_only_human_mapped_protein_rows_with_pchembl(self, output):
        # 103 rat, 104 duplicate, 105 cell line, 106 ADMET, 107 unmapped, 108 no pChEMBL, 209 degradation
        assert output['activityId'].unique(maintain_order=True).to_list() == [101, 102, 109]

    def test_one_row_per_activity_and_target_sorted(self, output):
        keys = output.select('activityId', 'targetId')
        assert keys.is_unique().all()
        assert keys.rows() == sorted(keys.rows())

    def test_measurement_class(self, output):
        classes = dict(output.select('activityId', 'measurementClass').unique().iter_rows())
        assert classes == {101: 'inhibition', 102: 'affinity', 109: 'functional_potency'}

    def test_activity_outcome_from_depositor_comments_only(self, output):
        outcomes = dict(output.select('activityId', 'activityOutcome').unique().iter_rows())
        # free text from a publication gives no outcome; depositors' words are normalised
        assert outcomes == {101: None, 102: 'inactive', 109: 'inconclusive'}

    def test_unknown_pchembl_type_fails(self, tables, gene_df, drug_df, moa_df):
        tables['activities'] = pl.DataFrame(
            [*ACTIVITIES, _activity(110, 1, 'Kb', standard_value=1, pchembl_value=9)], schema=ACTIVITY_SCHEMA
        )
        with pytest.raises(ValueError, match="'Kb'"):
            process_bioactivity(tables, gene_df, drug_df, moa_df)

    def test_unit_other_than_nm_fails(self, tables, gene_df, drug_df, moa_df):
        tables['activities'] = pl.DataFrame(
            [*ACTIVITIES, _activity(110, 1, 'IC50', standard_value=1, standard_units='uM', pchembl_value=6)],
            schema=ACTIVITY_SCHEMA,
        )
        with pytest.raises(ValueError, match="'uM'"):
            process_bioactivity(tables, gene_df, drug_df, moa_df)

    def test_single_protein_credits_its_gene(self, output):
        assert _credited(output, 101) == [('ENSG_EGFR', False)]

    def test_complex_without_binding_site_credits_every_gene(self, output):
        # an accession shared by two genes credits both
        assert _credited(output, 109) == [('ENSG_A', True), ('ENSG_B', True), ('ENSG_EGFR', True)]

    def test_complex_with_binding_site_credits_the_site_genes(self, tables, gene_df, drug_df, moa_df):
        tables['binding_sites'] = _frame([(1, 1), (2, 4)], {'site_id': pl.Int64, 'tid': pl.Int64})
        tables['site_components'] = _frame([(1, 1), (2, 3)], {'site_id': pl.Int64, 'component_id': pl.Int64})
        output = process_bioactivity(tables, gene_df, drug_df, moa_df)
        # the site is on PSHARED, which maps to two genes: still more than one gene credited
        assert _credited(output, 109) == [('ENSG_A', True), ('ENSG_B', True)]

    def test_obsolete_labelled_accession_resolves(self, tables, gene_df, drug_df, moa_df):
        # target.proteinIds labels every accession of a UniProt entry uniprot_obsolete, current ones too
        gene_df = pl.concat([
            gene_df,
            pl.DataFrame(
                [('ENSG_OBS', [{'id': 'PNOGENE', 'source': 'uniprot_obsolete'}])],
                schema=gene_df.schema,
                orient='row',
            ),
        ])
        output = process_bioactivity(tables, gene_df, drug_df, moa_df)
        assert _credited(output, 107) == [('ENSG_OBS', False)]

    def test_variant_placeholder_is_null(self, output):
        assert _by_id(output)[101]['variantFromSource'] == 'T790M'
        assert output.filter(pl.col('activityId') == 109)['variantFromSource'].to_list() == [None] * 3

    def test_shared_annotations(self, output):
        row = _by_id(output)[101]
        assert (row['targetFromSourceId'], row['assayId']) == ('CHEMBL_T1', 'CHEMBL_A1')
        assert (row['documentId'], row['pubmedId'], row['year'], row['docType']) == (
            'CHEMBL_D1',
            12127526,
            2002,
            'PUBLICATION',
        )
        assert (row['standardMeasurementType'], row['standardMeasurementValue'], row['pchemblValue']) == (
            'IC50',
            41.0,
            7.39,
        )
        assert (row['assayDescription'], row['assayOrganism']) == ('Assay 1', 'Homo sapiens')

    def test_drug_ids_resolve_salts_to_the_parent(self, output):
        drugs = output.select('activityId', 'drugFromSourceId', 'drugId').unique().sort('activityId')
        assert drugs.rows() == [
            (101, 'CHEMBL_M1', 'CHEMBL_D1'),
            (102, 'CHEMBL_M2', 'CHEMBL_M2'),
            (109, 'CHEMBL_M3', None),
        ]

    def test_parent_missing_from_drug_molecule_falls_back_to_the_molecule(self, tables, gene_df, moa_df):
        # CHEMBL_M2's parent is not in the drug molecule dataset, so drugId must not point to it
        drugs = _frame([('CHEMBL_D1', None, ['CHEMBL_M1']), ('CHEMBL_M2', 'CHEMBL_GONE', None)], DRUG_SCHEMA)
        row = _by_id(process_bioactivity(tables, gene_df, drugs, moa_df))[102]
        assert (row['drugFromSourceId'], row['drugId']) == ('CHEMBL_M2', 'CHEMBL_M2')

    def test_molecule_resolving_to_two_drugs_fails(self, tables, gene_df, moa_df):
        drugs = _frame([('CHEMBL_D1', None, ['CHEMBL_M1']), ('CHEMBL_D2', None, ['CHEMBL_M1'])], DRUG_SCHEMA)
        with pytest.raises(ValueError, match='CHEMBL_M1'):
            process_bioactivity(tables, gene_df, drugs, moa_df)

    def test_action_types_from_mechanisms_on_the_same_drug_and_gene(self, output):
        rows = _by_key(output)
        # 101: two mechanisms of the parent on EGFR, one action type; the assay's own is kept apart
        row = rows[101, 'ENSG_EGFR']
        assert (row['assayActionType'], row['actionTypes']) == ('BINDING AGENT', ['INHIBITOR'])
        # 102: the drug's only mechanism is on a gene its target does not have
        assert (rows[102, 'ENSG_EGFR']['assayActionType'], rows[102, 'ENSG_EGFR']['actionTypes']) == (None, [])
        # 109: only the complex's gene with the mechanism carries its two action types
        assert [rows[109, g]['actionTypes'] for g in ('ENSG_A', 'ENSG_B', 'ENSG_EGFR')] == [
            [],
            ['ANTAGONIST', 'DEGRADER'],
            [],
        ]

    def test_unmapped_components_are_left_out(self, tables, gene_df, drug_df, moa_df):
        # an extra component on the EGFR target whose accession maps to no gene
        tables['target_components'] = pl.concat([
            tables['target_components'],
            _frame([(1, 13)], {'tid': pl.Int64, 'component_id': pl.Int64}),
        ])
        tables['component_sequences'] = pl.concat([
            tables['component_sequences'],
            _frame([(13, 'PNONE')], {'component_id': pl.Int64, 'accession': pl.String}),
        ])
        # the binding site names P00533, so only EGFR is credited and the row is not a complex
        assert _credited(process_bioactivity(tables, gene_df, drug_df, moa_df), 101) == [('ENSG_EGFR', False)]

    def test_ligand_efficiency_is_null_without_a_ligand_eff_row(self, output):
        rows = _by_id(output)
        assert (rows[101]['le'], rows[101]['lle']) == (0.45, 5.1)
        assert (rows[102]['le'], rows[102]['lle']) == (None, None)


def test_float_columns_are_read_by_the_step():
    _check_projection(TABLES, FLOAT_COLUMNS, 'float_columns', needs_columns=True)
