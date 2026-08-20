"""Tests for the polars reactome transformer."""

from __future__ import annotations

import polars as pl

from pts.transformers.reactome import _build_graph_documents, _clean_pathways


def _pathways(*rows: tuple[str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=['column_1', 'column_2', 'column_3'], orient='row')


def _relations(*rows: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=['src', 'dst'], orient='row')


class TestCleanPathways:
    def test_keeps_only_homo_sapiens(self) -> None:
        df = _pathways(
            ('R-HSA-1', 'Pathway A', 'Homo sapiens'),
            ('R-MMU-1', 'Pathway B', 'Mus musculus'),
        )
        result = _clean_pathways(df)
        assert result.height == 1
        assert result['id'][0] == 'R-HSA-1'

    def test_drops_species_column(self) -> None:
        df = _pathways(('R-HSA-1', 'Pathway A', 'Homo sapiens'))
        result = _clean_pathways(df)
        assert set(result.columns) == {'id', 'name'}


class TestBuildGraphDocuments:
    def test_root_node_has_no_ancestors(self) -> None:
        vertices = pl.DataFrame({'id': ['R-1'], 'name': ['Root']})
        result = _build_graph_documents(vertices, _relations())
        row = result.row(by_predicate=pl.col('id') == 'R-1', named=True)
        assert row['ancestors'] == []
        assert row['children'] == []

    def test_child_has_parent_as_ancestor(self) -> None:
        vertices = pl.DataFrame({'id': ['R-1', 'R-2'], 'name': ['Root', 'Child']})
        result = _build_graph_documents(vertices, _relations(('R-1', 'R-2')))
        child = result.row(by_predicate=pl.col('id') == 'R-2', named=True)
        root = result.row(by_predicate=pl.col('id') == 'R-1', named=True)
        assert 'R-1' in child['ancestors']
        assert 'R-1' in child['parents']
        assert 'R-2' in root['descendants']
        assert any('R-1' in p and 'R-2' in p for p in child['path'])

    def test_cycle_guard_removes_back_edge(self) -> None:
        vertices = pl.DataFrame({'id': ['R-1', 'R-2'], 'name': ['A', 'B']})
        # R-1 -> R-2 -> R-1 is a cycle
        result = _build_graph_documents(vertices, _relations(('R-1', 'R-2'), ('R-2', 'R-1')))
        assert result.height == 2

    def test_relation_only_nodes_are_not_emitted(self) -> None:
        """Nodes that appear only in the relations file are graph context, not output rows."""
        vertices = pl.DataFrame({'id': ['R-HSA-1'], 'name': ['Human']})
        result = _build_graph_documents(vertices, _relations(('R-MMU-1', 'R-MMU-2')))
        assert result['id'].to_list() == ['R-HSA-1']

    def test_ancestry_columns_are_sorted(self) -> None:
        vertices = pl.DataFrame({'id': ['R-1', 'R-2', 'R-3', 'R-4'], 'name': ['A', 'B', 'C', 'D']})
        edges = _relations(('R-1', 'R-4'), ('R-3', 'R-4'), ('R-2', 'R-4'))
        row = _build_graph_documents(vertices, edges).row(by_predicate=pl.col('id') == 'R-4', named=True)
        assert row['ancestors'] == ['R-1', 'R-2', 'R-3']
        assert row['parents'] == ['R-1', 'R-2', 'R-3']

    def test_output_is_stable_across_repeated_builds(self) -> None:
        """Set-backed traversals must not leak iteration order into the output."""
        vertices = pl.DataFrame({
            'id': [f'R-{i}' for i in range(30)],
            'name': [f'P{i}' for i in range(30)],
        })
        edges = _relations(*[(f'R-{i}', f'R-{i + 1}') for i in range(29)])
        first = _build_graph_documents(vertices, edges)
        second = _build_graph_documents(vertices, edges)
        assert first.equals(second)
