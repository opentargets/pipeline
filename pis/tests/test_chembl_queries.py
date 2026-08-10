"""Run the ChEMBL rebuild queries against a small fixture database."""

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
    (3, 'CHEMBL3', 'lone drug', 'Small molecule');
INSERT INTO molecule_hierarchy VALUES (1, 2, 2), (2, 2, 2), (3, 3, 3);
INSERT INTO drug_warning VALUES
    (10, 100, 1, 'Withdrawn', 'Cardiotoxicity', 'France', 'bad things', 2009, 'term', 'EFO_1', 'EFO_2'),
    (11, 101, 3, 'Warning', NULL, 'US', NULL, NULL, NULL, NULL, NULL);
INSERT INTO warning_refs VALUES
    (1, 10, 'ISBN', 'ref-a', 'http://a'),
    (2, 10, 'DOI', 'ref-b', 'http://b');
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
    con = duckdb.connect()
    con.execute('LOAD postgres')
    con.execute(f"ATTACH '{server.get_uri(database='postgres')}' AS pg (TYPE postgres, READ_ONLY)")
    con.execute('USE pg."public"')
    result = con.execute(_load_query(name))
    columns = [d[0] for d in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


class TestDrugWarning:
    @pytest.fixture(scope='class')
    def rows(self, chembl: PostgresServer) -> dict[int, dict]:
        return {r['warning_id']: r for r in run_query(chembl, 'chembl_drug_warning')}

    def test_one_row_per_warning(self, rows: dict[int, dict]) -> None:
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
