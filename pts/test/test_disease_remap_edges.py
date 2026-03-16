"""Tests for remap_edges."""

from __future__ import annotations

import polars as pl
import pytest

from pts.schemas.ontology import node
from pts.transformers.disease import _IAO_REPLACED_BY, annotate_name_duplicates, remap_edges

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    id_: str,
    lbl: str,
    *,
    deprecated: bool | None = None,
    bpv: list[dict] | None = None,
) -> dict:
    return {
        'id': id_,
        'lbl': lbl,
        'type': 'CLASS',
        'meta': {
            'basicPropertyValues': bpv or [],
            'comments': [],
            'definition': {'val': None, 'xrefs': []},
            'deprecated': deprecated,
            'subsets': [],
            'synonyms': [],
            'xrefs': [],
        },
    }


def _make_nodes(*nodes_: dict) -> pl.DataFrame:
    return pl.from_dicts(list(nodes_), schema=node)


def _make_edges(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    """Build an edge DataFrame from (sub, pred, obj) tuples."""
    if not rows:
        return pl.DataFrame(
            {'sub': [], 'pred': [], 'obj': []},
            schema={'sub': pl.String, 'pred': pl.String, 'obj': pl.String},
        )
    subs, preds, objs = zip(*rows, strict=True)
    return pl.DataFrame({'sub': list(subs), 'pred': list(preds), 'obj': list(objs)})


# Shared URL constants
EFO_001 = 'http://www.ebi.ac.uk/efo/EFO_001'
EFO_002 = 'http://www.ebi.ac.uk/efo/EFO_002'
HP_001 = 'http://purl.obolibrary.org/obo/HP_001'   # superseded by EFO_001
HP_002 = 'http://purl.obolibrary.org/obo/HP_002'   # superseded by EFO_002
HP_003 = 'http://purl.obolibrary.org/obo/HP_003'   # canonical (unique name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def annotated_nodes() -> pl.DataFrame:
    """Node table with two name-collision pairs already annotated."""
    return annotate_name_duplicates(_make_nodes(
        _make_node(EFO_001, 'acidosis'),
        _make_node(HP_001, 'Acidosis'),   # superseded by EFO_001
        _make_node(EFO_002, 'fever'),
        _make_node(HP_002, 'Fever'),       # superseded by EFO_002
        _make_node(HP_003, 'unique term'),  # no collision
    ))


@pytest.fixture
def edges() -> pl.DataFrame:
    return _make_edges([
        (HP_001, 'is_a', EFO_002),   # sub is superseded
        (EFO_001, 'is_a', HP_002),    # obj is superseded
        (HP_001, 'is_a', HP_002),    # both superseded → becomes EFO_001 -> EFO_002
        (HP_001, 'is_a', EFO_001),   # remaps to EFO_001 -> EFO_001 (self-loop → drop)
        (EFO_001, 'is_a', HP_003),    # no remapping needed
        (HP_003, 'is_a', EFO_002),   # no remapping needed
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemapEdges:
    """Tests for remap_edges."""

    @pytest.fixture(autouse=True)
    def _run(self, edges: pl.DataFrame, annotated_nodes: pl.DataFrame) -> None:
        self.input_edges = edges
        self.result = remap_edges(edges, annotated_nodes)

    # -- structural checks ---------------------------------------------------

    def test_returns_dataframe(self) -> None:
        assert isinstance(self.result, pl.DataFrame)

    def test_columns_preserved(self) -> None:
        assert self.result.columns == self.input_edges.columns

    def test_column_types_preserved(self) -> None:
        assert self.result.schema == self.input_edges.schema

    # -- superseded URLs are remapped ----------------------------------------

    def test_no_superseded_url_in_sub(self, annotated_nodes: pl.DataFrame) -> None:
        superseded = (
            annotated_nodes
            .unnest('meta')
            .explode('basicPropertyValues')
            .unnest('basicPropertyValues')
            .filter(pl.col('deprecated'), pl.col('pred') == _IAO_REPLACED_BY)
            ['id'].to_list()
        )
        remaining = self.result.filter(pl.col('sub').is_in(superseded))
        assert remaining.is_empty(), f"Superseded URLs still in 'sub': {remaining}"

    def test_no_superseded_url_in_obj(self, annotated_nodes: pl.DataFrame) -> None:
        superseded = (
            annotated_nodes
            .unnest('meta')
            .explode('basicPropertyValues')
            .unnest('basicPropertyValues')
            .filter(pl.col('deprecated'), pl.col('pred') == _IAO_REPLACED_BY)
            ['id'].to_list()
        )
        remaining = self.result.filter(pl.col('obj').is_in(superseded))
        assert remaining.is_empty(), f"Superseded URLs still in 'obj': {remaining}"

    def test_superseded_sub_replaced_with_canonical(self) -> None:
        """HP_001 in sub must become EFO_001."""
        # Original edge: HP_001 -> EFO_002; after remap: EFO_001 -> EFO_002
        remapped = self.result.filter(
            (pl.col('sub') == EFO_001) & (pl.col('obj') == EFO_002) & (pl.col('pred') == 'is_a')
        )
        assert remapped.height >= 1

    def test_superseded_obj_replaced_with_canonical(self) -> None:
        """HP_002 in obj must become EFO_002."""
        # Original edge: EFO_001 -> HP_002; after remap: EFO_001 -> EFO_002
        remapped = self.result.filter(
            (pl.col('sub') == EFO_001) & (pl.col('obj') == EFO_002) & (pl.col('pred') == 'is_a')
        )
        assert remapped.height >= 1

    # -- self-loops are dropped ----------------------------------------------

    def test_self_loops_removed(self) -> None:
        """HP_001 -> EFO_001 remaps to EFO_001 -> EFO_001; that self-loop must be dropped."""
        self_loops = self.result.filter(pl.col('sub') == pl.col('obj'))
        assert self_loops.is_empty()

    # -- deduplication -------------------------------------------------------

    def test_no_duplicate_edges(self) -> None:
        assert self.result.height == self.result.unique().height

    # -- unchanged edges remain ----------------------------------------------

    def test_canonical_only_edge_unchanged(self) -> None:
        """EFO_001 -> HP_003 has no deprecated endpoint and must pass through."""
        canonical_edge = self.result.filter(
            (pl.col('sub') == EFO_001) & (pl.col('obj') == HP_003)
        )
        assert canonical_edge.height == 1

    def test_canonical_obj_edge_unchanged(self) -> None:
        """HP_003 -> EFO_002 has no deprecated endpoint and must pass through."""
        canonical_edge = self.result.filter(
            (pl.col('sub') == HP_003) & (pl.col('obj') == EFO_002)
        )
        assert canonical_edge.height == 1


class TestRemapEdgesEdgeCases:
    """Edge-case tests for remap_edges."""

    def test_no_deprecated_nodes_returns_input_unchanged(self) -> None:
        """When n has no deprecated nodes, edges pass through unchanged."""
        n = _make_nodes(
            _make_node(EFO_001, 'disease a'),
            _make_node(EFO_002, 'disease b'),
        )
        e = _make_edges([(EFO_001, 'is_a', EFO_002)])
        result = remap_edges(e, n)
        assert result.equals(e)

    def test_empty_edges_returns_empty(self) -> None:
        n = annotate_name_duplicates(_make_nodes(
            _make_node(EFO_001, 'acidosis'),
            _make_node(HP_001, 'Acidosis'),
        ))
        e = _make_edges([])
        result = remap_edges(e, n)
        assert result.is_empty()
        assert result.columns == e.columns
