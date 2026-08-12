"""Tests for chembl_target, which stages the five ChEMBL tables `target` reads.

No joins, no flattening: this step restores five tables from the ChEMBL dump and
writes each straight to parquet, at the names and columns `target` already expects.
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from pixeltable_pgserver.postgres_server import get_server

from pts.transformers.chembl_target import TABLES, chembl_target

EXPECTED_TABLES = {
    'target_dictionary', 'target_components', 'component_sequences', 'component_class', 'protein_classification',
}


def test_tables_pins_the_five_names_and_columns() -> None:
    """The columns config.yaml declared for these tables before the export step was removed."""
    assert TABLES == {
        'target_dictionary': ['tid', 'chembl_id', 'pref_name', 'target_type'],
        'target_components': ['targcomp_id', 'tid', 'component_id'],
        'component_sequences': ['component_id', 'accession'],
        'component_class': ['comp_class_id', 'component_id', 'protein_class_id'],
        'protein_classification': ['protein_class_id', 'parent_id', 'pref_name', 'class_level'],
    }


@pytest.mark.pgserver
class TestChemblTarget:
    """Restore a real dump into a real server and read it, with nothing mocked."""

    @pytest.fixture(scope='class')
    def dump(self, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
        path = tmp_path_factory.mktemp('source')
        server = get_server(path / 'pgdata', cleanup_mode='delete')
        try:
            server.psql(
                'CREATE SCHEMA public;'
                'CREATE TABLE public.target_dictionary '
                '(tid int, chembl_id text, pref_name text, target_type text);'
                'CREATE TABLE public.target_components (targcomp_id int, tid int, component_id int);'
                'CREATE TABLE public.component_sequences (component_id int, accession text);'
                'CREATE TABLE public.component_class '
                '(comp_class_id int, component_id int, protein_class_id int);'
                'CREATE TABLE public.protein_classification '
                '(protein_class_id int, parent_id int, pref_name text, class_level int);'
                "INSERT INTO public.target_dictionary VALUES "
                "(20, 'CHEMBL_T20', 'Target Twenty', 'SINGLE PROTEIN');"
                'INSERT INTO public.target_components VALUES (2001, 20, 300);'
                "INSERT INTO public.component_sequences VALUES (300, 'P100');"
                'INSERT INTO public.component_class VALUES (5001, 300, 900);'
                "INSERT INTO public.protein_classification VALUES (900, NULL, 'Enzyme', 0);"
            )
            dump = path / 'chembl.dmp'
            subprocess.run([str(server.bin_path / 'pg_dump'), '-Fc', '-f', str(dump), server.get_uri()], check=True)
            yield dump
        finally:
            server.cleanup()

    def test_it_writes_the_five_tables(self, dump: Path, tmp_path: Path) -> None:
        """`target` reads exactly these five, by these names."""
        destination = {name: tmp_path / f'{name}.parquet' for name in TABLES}
        chembl_target(dump, destination, {}, None)

        written = {p.stem for p in tmp_path.glob('*.parquet')}
        assert written == EXPECTED_TABLES

    def test_each_file_has_only_the_declared_columns(self, dump: Path, tmp_path: Path) -> None:
        destination = {name: tmp_path / f'{name}.parquet' for name in TABLES}
        chembl_target(dump, destination, {}, None)

        for name, columns in TABLES.items():
            assert pl.read_parquet(destination[name]).columns == columns

    def test_no_join_no_flatten_the_rows_pass_through_untouched(self, dump: Path, tmp_path: Path) -> None:
        destination = {name: tmp_path / f'{name}.parquet' for name in TABLES}
        chembl_target(dump, destination, {}, None)

        target_dictionary = pl.read_parquet(destination['target_dictionary'])
        assert target_dictionary.to_dicts() == [
            {'tid': 20, 'chembl_id': 'CHEMBL_T20', 'pref_name': 'Target Twenty', 'target_type': 'SINGLE PROTEIN'}
        ]

        protein_classification = pl.read_parquet(destination['protein_classification'])
        assert protein_classification.to_dicts() == [
            {'protein_class_id': 900, 'parent_id': None, 'pref_name': 'Enzyme', 'class_level': 0}
        ]
