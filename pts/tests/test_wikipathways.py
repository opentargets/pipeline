"""Tests for the polars wikipathways transformer."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.schemas.reactome import reactome_schema
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.wikipathways import (
    _build_documents,
    _parse_pathways,
    _read_gmt,
    _validate,
    wikipathways,
)


def _gmt(tmp_path: Path, *lines: str) -> str:
    path = tmp_path / 'wikipathways.gmt'
    path.write_text(''.join(f'{line}\n' for line in lines))
    return str(path)


def _line(wp_id: str, name: str, species: str = 'Homo sapiens', release: str = 'WikiPathways_20260910') -> str:
    return f'{name}%{release}%{wp_id}%{species}\thttps://www.wikipathways.org/instance/{wp_id}\t1\t2'


class TestReadGmt:
    def test_lines_with_different_gene_counts_are_read(self, tmp_path: Path) -> None:
        path = _gmt(
            tmp_path,
            'A%WikiPathways_20260910%WP1%Homo sapiens\turl1\t1',
            'B%WikiPathways_20260910%WP2%Homo sapiens\turl2\t1\t2\t3\t4',
        )
        result = _read_gmt(path)
        assert result.columns == ['header', 'url']
        assert result['url'].to_list() == ['url1', 'url2']

    def test_double_quote_in_name_is_literal(self, tmp_path: Path) -> None:
        path = _gmt(tmp_path, _line('WP1', 'The "quoted" pathway'))
        pathways = _parse_pathways(_read_gmt(path))
        assert pathways['name'][0] == 'The "quoted" pathway'


class TestParsePathways:
    def test_splits_header(self) -> None:
        gmt = pl.DataFrame({'header': ['Apoptosis%WikiPathways_20260910%WP254%Homo sapiens'], 'url': ['u']})
        row = _parse_pathways(gmt).row(0, named=True)
        assert row == {
            'name': 'Apoptosis',
            'release': 'WikiPathways_20260910',
            'id': 'WP254',
            'species': 'Homo sapiens',
        }

    def test_percent_in_name_is_kept(self) -> None:
        gmt = pl.DataFrame({'header': ['50% pathway%WikiPathways_20260910%WP1%Homo sapiens'], 'url': ['u']})
        row = _parse_pathways(gmt).row(0, named=True)
        assert row['name'] == '50% pathway'
        assert row['id'] == 'WP1'

    def test_malformed_header_yields_null_id(self) -> None:
        gmt = pl.DataFrame({'header': ['not a gmt header'], 'url': ['u']})
        assert _parse_pathways(gmt)['id'][0] is None


def _pathways(*rows: tuple[str, str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=['name', 'release', 'id', 'species'], orient='row')


class TestValidate:
    def test_valid_input_passes(self) -> None:
        _validate(_pathways(('A', 'WikiPathways_1', 'WP1', 'Homo sapiens')), min_pathways=1)

    def test_malformed_header_fails(self) -> None:
        with pytest.raises(ValueError, match='header'):
            _validate(_pathways((None, None, None, None)), min_pathways=0)

    def test_mixed_releases_fail(self) -> None:
        pathways = _pathways(
            ('A', 'WikiPathways_1', 'WP1', 'Homo sapiens'),
            ('B', 'WikiPathways_2', 'WP2', 'Homo sapiens'),
        )
        with pytest.raises(ValueError, match='one WikiPathways release'):
            _validate(pathways, min_pathways=0)

    def test_duplicate_ids_fail(self) -> None:
        pathways = _pathways(
            ('A', 'WikiPathways_1', 'WP1', 'Homo sapiens'),
            ('B', 'WikiPathways_1', 'WP1', 'Homo sapiens'),
        )
        with pytest.raises(ValueError, match='duplicate'):
            _validate(pathways, min_pathways=0)

    def test_too_few_human_pathways_fail(self) -> None:
        pathways = _pathways(
            ('A', 'WikiPathways_1', 'WP1', 'Homo sapiens'),
            ('B', 'WikiPathways_1', 'WP2', 'Mus musculus'),
        )
        with pytest.raises(ValueError, match='expected at least 2'):
            _validate(pathways, min_pathways=2)


class TestBuildDocuments:
    def test_matches_reactome_schema(self) -> None:
        result = _build_documents(_pathways(('A', 'WikiPathways_1', 'WP1', 'Homo sapiens')))
        assert result.schema == pl.Schema(reactome_schema)

    def test_keeps_only_homo_sapiens(self) -> None:
        pathways = _pathways(
            ('A', 'WikiPathways_1', 'WP1', 'Homo sapiens'),
            ('B', 'WikiPathways_1', 'WP2', 'Mus musculus'),
        )
        assert _build_documents(pathways)['id'].to_list() == ['WP1']

    def test_every_pathway_is_its_own_root(self) -> None:
        row = _build_documents(_pathways(('A', 'WikiPathways_1', 'WP1', 'Homo sapiens'))).row(0, named=True)
        assert row == {
            'id': 'WP1',
            'label': 'A',
            'ancestors': [],
            'descendants': [],
            'children': [],
            'parents': [],
            'path': [['WP1']],
        }

    def test_rows_are_sorted_by_id(self) -> None:
        pathways = _pathways(
            ('B', 'WikiPathways_1', 'WP2', 'Homo sapiens'),
            ('A', 'WikiPathways_1', 'WP1', 'Homo sapiens'),
        )
        assert _build_documents(pathways)['id'].to_list() == ['WP1', 'WP2']


class TestWikipathways:
    def test_writes_dataset(self, tmp_path: Path) -> None:
        source = _gmt(tmp_path, _line('WP2', 'B'), _line('WP1', 'A'), _line('WP3', 'C', species='Mus musculus'))
        destination = str(tmp_path / 'output')
        wikipathways(source, destination, {'min_pathways': 2}, None)  # type: ignore[arg-type]
        result = scan_dataset(destination).collect()
        assert result['id'].to_list() == ['WP1', 'WP2']
        assert result['label'].to_list() == ['A', 'B']
