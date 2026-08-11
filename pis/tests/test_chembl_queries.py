"""Run the ChEMBL rebuild queries against a small fixture database."""

from contextlib import closing

import duckdb
import pytest
from pixeltable_pgserver.postgres_server import PostgresServer, get_server

from pis.tasks.postgres_export import _load_query

pytestmark = pytest.mark.pgserver

SCHEMA = """
CREATE TABLE molecule_dictionary (molregno int PRIMARY KEY, chembl_id text, pref_name text, molecule_type text);
CREATE TABLE molecule_hierarchy (molregno int PRIMARY KEY, parent_molregno int, active_molregno int);
CREATE TABLE drug_warning (
    warning_id int PRIMARY KEY, record_id int, molregno int, warning_type text, warning_class text,
    warning_country text, warning_description text, warning_year int,
    efo_term text, efo_id text, efo_id_for_warning_class text
);
CREATE TABLE warning_refs (warnref_id int PRIMARY KEY, warning_id int, ref_type text, ref_id text, ref_url text);
"""

DATA = """
INSERT INTO molecule_dictionary VALUES
    (1, 'CHEMBL1', 'child drug', 'Small molecule'),
    (2, 'CHEMBL2', 'parent drug', 'Small molecule'),
    -- ChEMBL 37 has 18 pref_names with a trailing space; chembl_molecule.sql trims
    (3, 'CHEMBL3', '  lone drug ', 'Small molecule');
INSERT INTO molecule_hierarchy VALUES (1, 2, 2), (2, 2, 2), (3, 3, 3);
INSERT INTO drug_warning VALUES
    (10, 100, 1, 'Withdrawn', 'Cardiotoxicity', 'France', 'bad things', 2009, 'term', 'EFO_1', 'EFO_2'),
    (11, 101, 3, 'Warning', NULL, 'US', NULL, NULL, NULL, NULL, NULL);
INSERT INTO warning_refs VALUES
    (1, 10, 'ISBN', 'ref-a', 'http://a'),
    (2, 10, 'DOI', 'ref-b', 'http://b');
"""

SCHEMA += """
CREATE TABLE target_dictionary (tid int PRIMARY KEY, target_type text, pref_name text, chembl_id text);
CREATE TABLE drug_mechanism (
    mec_id int PRIMARY KEY, record_id int, molregno int, mechanism_of_action text, tid int, action_type text
);
CREATE TABLE mechanism_refs (mecref_id int PRIMARY KEY, mec_id int, ref_type text, ref_id text, ref_url text);
"""

DATA += """
INSERT INTO target_dictionary VALUES (500, 'SINGLE PROTEIN', 'A target', 'CHEMBL_T1');
INSERT INTO drug_mechanism VALUES
    (20, 200, 1, 'Kinase inhibitor', 500, 'INHIBITOR'),
    (21, 201, 3, 'Receptor agonist', NULL, NULL);
INSERT INTO mechanism_refs VALUES (1, 20, 'PubMed', '12345', 'http://pm/12345');
"""

SCHEMA += """
CREATE TABLE compound_structures (
    molregno int PRIMARY KEY, molfile text, standard_inchi text, standard_inchi_key text, canonical_smiles text
);
CREATE TABLE molecule_synonyms (molsyn_id int PRIMARY KEY, molregno int, syn_type text, synonyms text);
CREATE TABLE source (src_id int PRIMARY KEY, src_short_name text, src_description text);
CREATE TABLE compound_records (
    record_id int PRIMARY KEY, molregno int, doc_id int, src_id int, src_compound_id text
);
"""

DATA += """
-- compound_structures is inconsistent about the molblock terminator: molregno 1
-- stops at `M  END`, molregno 2 already carries the trailing newline
INSERT INTO compound_structures VALUES
    (1, 'MOLBLOCK1' || chr(10) || 'M  END', 'InChI=1S/x', 'INCHIKEY1', 'CCO'),
    (2, 'MOLBLOCK2' || chr(10) || 'M  END' || chr(10), 'InChI=1S/y', 'INCHIKEY2', 'CCC');
INSERT INTO molecule_synonyms VALUES
    (1, 1, 'TRADE_NAME', 'Tradey'),
    (2, 1, 'INN', 'childium');
INSERT INTO source VALUES (1, 'LITERATURE', 'Scientific Literature'), (63, 'INN', 'INN');
INSERT INTO compound_records VALUES
    (100, 1, 900, 63, '24616'),
    (101, 1, 901, 1, 'IGNORED');
"""

SCHEMA += """
CREATE TABLE component_sequences (component_id int PRIMARY KEY, accession text, component_type text);
CREATE TABLE target_components (targcomp_id int PRIMARY KEY, tid int, component_id int, homologue int);
CREATE TABLE protein_classification (
    protein_class_id int PRIMARY KEY, parent_id int, pref_name text, short_name text, class_level int
);
CREATE TABLE component_class (comp_class_id int PRIMARY KEY, component_id int, protein_class_id int);
"""

DATA += """
INSERT INTO target_dictionary VALUES
    (501, 'SINGLE PROTEIN', 'Single target', 'CHEMBL_T2'),
    (502, 'PROTEIN COMPLEX', 'Complex target', 'CHEMBL_T3'),
    (503, 'CELL-LINE', 'No components', 'CHEMBL_T4'),
    (504, 'NUCLEIC-ACID', 'Unclassified target', 'CHEMBL_T5');
-- 27 of ChEMBL 37's components have no accession, and the RNA one here is also
-- the component with no component_class row
INSERT INTO component_sequences VALUES
    (1, 'P00001', 'PROTEIN'),
    (2, 'P00002', 'PROTEIN'),
    (3, NULL, 'RNA');
-- targcomp_id runs against component_id for target 502, so that a query ordering
-- by the wrong one comes out backwards
INSERT INTO target_components VALUES
    (1, 501, 1, 0),
    (2, 502, 2, 0),
    (3, 502, 1, 0),
    (4, 504, 3, 0);
-- protein_class_id 0 is the tree's root in ChEMBL 37 and sits at class_level 0,
-- below l1; it must never surface as a level
INSERT INTO protein_classification VALUES
    (0, NULL, 'Protein class', 'pc', 0),
    (10, 0, 'Enzyme', 'enz', 1),
    (11, 10, 'Kinase', 'kin', 2),
    (12, 11, 'Protein Kinase', 'pk', 3),
    (20, 0, 'Transporter', 'tra', 1);
INSERT INTO component_class VALUES
    (1, 1, 12),
    (2, 1, 20),
    (3, 2, 11);
"""


@pytest.fixture(scope='module')
def chembl(tmp_path_factory: pytest.TempPathFactory) -> PostgresServer:
    """A postgres server holding a miniature ChEMBL."""
    server = get_server(tmp_path_factory.mktemp('chembl') / 'pgdata', cleanup_mode='delete')
    server.psql(SCHEMA)
    server.psql(DATA)
    return server


def run_query(server: PostgresServer, name: str) -> list[dict]:
    """Run a shipped SQL file against the fixture database and return rows as dicts."""
    with closing(duckdb.connect()) as con:
        con.execute('LOAD postgres')
        con.execute(f"ATTACH '{server.get_uri(database='postgres')}' AS pg (TYPE postgres, READ_ONLY)")
        con.execute('USE pg."public"')
        result = con.execute(_load_query(name))
        columns = [d[0] for d in result.description]
        return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


class TestDrugWarning:
    @pytest.fixture(scope='class')
    def raw(self, chembl: PostgresServer) -> list[dict]:
        return run_query(chembl, 'chembl_drug_warning')

    @pytest.fixture(scope='class')
    def rows(self, raw: list[dict]) -> dict[int, dict]:
        return {r['warning_id']: r for r in raw}

    def test_one_row_per_warning(self, raw: list[dict], rows: dict[int, dict]) -> None:
        # assert on the list BEFORE the dict collapses duplicates
        assert len(raw) == 2
        assert sorted(rows) == [10, 11]

    def test_scalar_fields(self, rows: dict[int, dict]) -> None:
        w = rows[10]
        assert w['warning_type'] == 'Withdrawn'
        assert w['warning_class'] == 'Cardiotoxicity'
        assert w['warning_country'] == 'France'
        assert w['warning_description'] == 'bad things'
        assert w['warning_year'] == 2009
        assert w['efo_id'] == 'EFO_1'
        assert w['efo_term'] == 'term'
        assert w['efo_id_for_warning_class'] == 'EFO_2'

    def test_molecule_and_parent(self, rows: dict[int, dict]) -> None:
        assert rows[10]['molecule_chembl_id'] == 'CHEMBL1'
        assert rows[10]['parent_molecule_chembl_id'] == 'CHEMBL2'

    def test_all_molecule_chembl_ids_has_both(self, rows: dict[int, dict]) -> None:
        assert sorted(rows[10]['_metadata']['all_molecule_chembl_ids']) == ['CHEMBL1', 'CHEMBL2']

    def test_all_molecule_chembl_ids_deduplicates(self, rows: dict[int, dict]) -> None:
        # CHEMBL3 is its own parent
        assert rows[11]['_metadata']['all_molecule_chembl_ids'] == ['CHEMBL3']

    def test_refs(self, rows: dict[int, dict]) -> None:
        refs = rows[10]['warning_refs']
        assert len(refs) == 2
        assert {r['ref_type'] for r in refs} == {'ISBN', 'DOI'}
        assert {r['ref_id'] for r in refs} == {'ref-a', 'ref-b'}

    def test_no_refs_is_an_empty_list_not_null(self, rows: dict[int, dict]) -> None:
        assert rows[11]['warning_refs'] == []


class TestMechanism:
    @pytest.fixture(scope='class')
    def raw(self, chembl: PostgresServer) -> list[dict]:
        return run_query(chembl, 'chembl_mechanism')

    @pytest.fixture(scope='class')
    def rows(self, raw: list[dict]) -> dict[int, dict]:
        return {r['record_id']: r for r in raw}

    def test_one_row_per_mechanism(self, raw: list[dict], rows: dict[int, dict]) -> None:
        # assert on the list BEFORE the dict collapses duplicates
        assert len(raw) == 2
        assert sorted(rows) == [200, 201]

    def test_scalar_fields(self, rows: dict[int, dict]) -> None:
        m = rows[200]
        assert m['mechanism_of_action'] == 'Kinase inhibitor'
        assert m['action_type'] == 'INHIBITOR'
        assert m['molecule_chembl_id'] == 'CHEMBL1'
        assert m['parent_molecule_chembl_id'] == 'CHEMBL2'

    def test_target(self, rows: dict[int, dict]) -> None:
        assert rows[200]['target_chembl_id'] == 'CHEMBL_T1'

    def test_missing_target_is_null(self, rows: dict[int, dict]) -> None:
        assert rows[201]['target_chembl_id'] is None

    def test_all_molecule_chembl_ids(self, rows: dict[int, dict]) -> None:
        assert sorted(rows[200]['_metadata']['all_molecule_chembl_ids']) == ['CHEMBL1', 'CHEMBL2']

    def test_refs(self, rows: dict[int, dict]) -> None:
        refs = rows[200]['mechanism_refs']
        assert len(refs) == 1
        assert refs[0]['ref_type'] == 'PubMed'
        assert refs[0]['ref_id'] == '12345'

    def test_no_refs_is_an_empty_list_not_null(self, rows: dict[int, dict]) -> None:
        assert rows[201]['mechanism_refs'] == []


class TestMolecule:
    @pytest.fixture(scope='class')
    def raw(self, chembl: PostgresServer) -> list[dict]:
        return run_query(chembl, 'chembl_molecule')

    @pytest.fixture(scope='class')
    def rows(self, raw: list[dict]) -> dict[str, dict]:
        return {r['molecule_chembl_id']: r for r in raw}

    def test_one_row_per_molecule(self, raw: list[dict], rows: dict[str, dict]) -> None:
        # assert on the list BEFORE the dict collapses duplicates
        assert len(raw) == 3
        assert sorted(rows) == ['CHEMBL1', 'CHEMBL2', 'CHEMBL3']

    def test_scalar_fields(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL1']['pref_name'] == 'child drug'
        assert rows['CHEMBL1']['molecule_type'] == 'Small molecule'

    def test_pref_name_is_trimmed(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL3']['pref_name'] == 'lone drug'

    def test_structures(self, rows: dict[str, dict]) -> None:
        s = rows['CHEMBL1']['molecule_structures']
        assert s['canonical_smiles'] == 'CCO'
        assert s['standard_inchi_key'] == 'INCHIKEY1'
        assert s['molfile'] == 'MOLBLOCK1\nM  END\n'

    def test_molfile_terminator_is_normalised_to_one_newline(self, rows: dict[str, dict]) -> None:
        # pts truncates the molblock with `(?s)(\nM  END\n).*`, which only matches
        # when `M  END` is followed by a newline. Both source shapes must come out
        # with exactly one, so neither a missing nor a doubled newline reaches pts.
        assert rows['CHEMBL1']['molecule_structures']['molfile'] == 'MOLBLOCK1\nM  END\n'
        assert rows['CHEMBL2']['molecule_structures']['molfile'] == 'MOLBLOCK2\nM  END\n'

    def test_standard_inchi_is_pruned(self, rows: dict[str, dict]) -> None:
        assert 'standard_inchi' not in rows['CHEMBL1']['molecule_structures']

    def test_missing_structures_is_null(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL3']['molecule_structures'] is None

    def test_hierarchy_has_only_parent(self, rows: dict[str, dict]) -> None:
        h = rows['CHEMBL1']['molecule_hierarchy']
        assert h['parent_chembl_id'] == 'CHEMBL2'
        assert set(h) == {'parent_chembl_id'}

    def test_synonyms(self, rows: dict[str, dict]) -> None:
        syns = rows['CHEMBL1']['molecule_synonyms']
        assert {s['molecule_synonym'] for s in syns} == {'Tradey', 'childium'}
        assert {s['syn_type'] for s in syns} == {'TRADE_NAME', 'INN'}

    def test_no_synonyms_is_an_empty_list_not_null(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL3']['molecule_synonyms'] == []

    def test_cross_references_is_empty_for_every_molecule(self, raw: list[dict]) -> None:
        # the Elasticsearch document's cross_references cannot be rebuilt from the
        # relational schema, so the field is emitted empty. CHEMBL1 has a non
        # literature compound_records row precisely so that a future attempt to
        # populate it from compound_records fails here rather than in production.
        assert [r['cross_references'] for r in raw] == [[], [], []]

    def test_dead_fields_are_absent(self, rows: dict[str, dict]) -> None:
        for dead in ('first_approval', 'max_phase', 'withdrawn_flag', 'black_box_warning'):
            assert dead not in rows['CHEMBL1']


class TestTarget:
    @pytest.fixture(scope='class')
    def raw(self, chembl: PostgresServer) -> list[dict]:
        return run_query(chembl, 'chembl_target')

    @pytest.fixture(scope='class')
    def rows(self, raw: list[dict]) -> dict[str, dict]:
        return {r['target_chembl_id']: r for r in raw}

    def test_one_row_per_target(self, raw: list[dict], rows: dict[str, dict]) -> None:
        # assert on the list BEFORE the dict collapses duplicates
        assert len(raw) == 5
        assert sorted(rows) == ['CHEMBL_T1', 'CHEMBL_T2', 'CHEMBL_T3', 'CHEMBL_T4', 'CHEMBL_T5']

    def test_scalar_fields(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL_T2']['pref_name'] == 'Single target'
        assert rows['CHEMBL_T2']['target_type'] == 'SINGLE PROTEIN'

    def test_components_carry_only_accession(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL_T2']['target_components'] == [{'accession': 'P00001'}]

    def test_target_with_no_components_is_an_empty_list(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL_T4']['target_components'] == []
        assert rows['CHEMBL_T4']['_metadata']['protein_classification'] == []

    def test_components_are_ordered_by_component_id(self, rows: dict[str, dict]) -> None:
        # target 502 lists component 2 first by targcomp_id, but the Elasticsearch
        # document orders target_components by component_id in all 18552 targets
        assert [c['accession'] for c in rows['CHEMBL_T3']['target_components']] == ['P00001', 'P00002']

    def test_every_class_of_a_component_is_kept(self, rows: dict[str, dict]) -> None:
        # component 1 carries both Protein Kinase (12) and Transporter (20), so a
        # single-component target ends up with two classifications and one component
        single = rows['CHEMBL_T2']
        assert len(single['target_components']) == 1
        assert [c['protein_class_id'] for c in single['_metadata']['protein_classification']] == [12, 20]

    def test_classification_groups_by_component_then_class_id(self, rows: dict[str, dict]) -> None:
        # component 1 (classes 12 and 20) before component 2 (class 11)
        classes = rows['CHEMBL_T3']['_metadata']['protein_classification']
        assert [c['protein_class_id'] for c in classes] == [12, 20, 11]

    def test_ancestors_are_flattened_into_levels(self, rows: dict[str, dict]) -> None:
        pc = rows['CHEMBL_T2']['_metadata']['protein_classification'][0]
        assert pc['l1'] == 'Enzyme'
        assert pc['l2'] == 'Kinase'
        assert pc['l3'] == 'Protein Kinase'
        assert pc['l4'] is None
        assert pc['l5'] is None
        assert pc['l6'] is None

    def test_the_root_of_the_class_tree_is_not_a_level(self, rows: dict[str, dict]) -> None:
        # every class descends from protein_class_id 0, 'Protein class', at level 0
        for target in rows.values():
            for pc in target['_metadata']['protein_classification']:
                assert 'Protein class' not in pc.values()

    def test_component_without_a_class_contributes_no_entry(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL_T5']['target_components'] == [{'accession': None}]
        assert rows['CHEMBL_T5']['_metadata']['protein_classification'] == []

    def test_classification_keeps_only_what_pts_reads(self, rows: dict[str, dict]) -> None:
        pc = rows['CHEMBL_T2']['_metadata']['protein_classification'][0]
        assert set(pc) == {'protein_class_id', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6'}
