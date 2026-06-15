"""Tests for annotate_name_duplicates."""

from __future__ import annotations

import polars as pl
import pytest

from pts.schemas.ontology import node
from pts.transformers.disease import _IAO_REPLACED_BY, annotate_name_duplicates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIST_STR = pl.List(pl.String)


def _make_node(
    id_: str,
    lbl: str,
    *,
    deprecated: bool | None = None,
    bpv: list[dict] | None = None,
    type_: str = 'CLASS',
) -> dict:
    """Build a single node dict compatible with the ``node`` schema."""
    return {
        'id': id_,
        'lbl': lbl,
        'type': type_,
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


def _make_df(*nodes: dict) -> pl.DataFrame:
    return pl.from_dicts(list(nodes), schema=node)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_df() -> pl.DataFrame:
    """Minimal node table with three name-collision groups and one unique node.

    Collision groups
    ----------------
    - "acidosis"  : http://.../EFO_001 (efo, rank 1) vs http://.../HP_001 (hp, rank 100)
                    → EFO_001 canonical, HP_001 superseded
    - "fever"     : http://.../MONDO_001 (mondo, rank 2) vs http://.../HP_002 (hp, rank 100)
                    → MONDO_001 canonical, HP_002 superseded
    - "unique"    : http://.../EFO_002  — no collision, must be unchanged

    Also included: an already-deprecated node (http://.../HP_OLD) that must not be touched.
    """
    return _make_df(
        _make_node('http://www.ebi.ac.uk/efo/EFO_001', 'acidosis'),
        _make_node('http://purl.obolibrary.org/obo/HP_001', 'Acidosis'),   # same name, lower priority
        _make_node('http://purl.obolibrary.org/obo/MONDO_001', 'fever'),
        _make_node('http://purl.obolibrary.org/obo/HP_002', 'Fever'),       # same name, lower priority
        _make_node('http://www.ebi.ac.uk/efo/EFO_002', 'unique disease'),   # no collision
        _make_node('http://purl.obolibrary.org/obo/HP_OLD', 'old term', deprecated=True),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnnotateNameDuplicates:
    """Tests for annotate_name_duplicates."""

    @pytest.fixture(autouse=True)
    def _run(self, base_df: pl.DataFrame) -> None:
        self.input_df = base_df
        self.result = annotate_name_duplicates(base_df)

    # -- structural checks ---------------------------------------------------

    def test_returns_dataframe(self) -> None:
        assert isinstance(self.result, pl.DataFrame)

    def test_shape_preserved(self) -> None:
        assert self.result.shape == self.input_df.shape

    def test_schema_preserved(self) -> None:
        assert self.result.schema == self.input_df.schema

    # -- superseded nodes marked deprecated ----------------------------------

    @pytest.mark.parametrize('superseded_url', [
        'http://purl.obolibrary.org/obo/HP_001',
        'http://purl.obolibrary.org/obo/HP_002',
    ])
    def test_superseded_node_is_deprecated(self, superseded_url: str) -> None:
        deprecated = (
            self.result
            .filter(pl.col('id') == superseded_url)
            .select(pl.col('meta').struct['deprecated'])
            ['deprecated'][0]
        )
        assert deprecated is True

    # -- superseded nodes receive IAO entry pointing to canonical URL --------

    @pytest.mark.parametrize(('superseded_url', 'canonical_url'), [
        ('http://purl.obolibrary.org/obo/HP_001', 'http://www.ebi.ac.uk/efo/EFO_001'),
        ('http://purl.obolibrary.org/obo/HP_002', 'http://purl.obolibrary.org/obo/MONDO_001'),
    ])
    def test_superseded_node_has_iao_entry(self, superseded_url: str, canonical_url: str) -> None:
        iao_vals = (
            self.result
            .filter(pl.col('id') == superseded_url)
            .unnest('meta')
            .explode('basicPropertyValues')
            .unnest('basicPropertyValues')
            .filter(pl.col('pred') == _IAO_REPLACED_BY)
            ['val'].to_list()
        )
        assert canonical_url in iao_vals

    # -- canonical nodes are not touched -------------------------------------

    @pytest.mark.parametrize('canonical_url', [
        'http://www.ebi.ac.uk/efo/EFO_001',
        'http://purl.obolibrary.org/obo/MONDO_001',
    ])
    def test_canonical_node_not_deprecated(self, canonical_url: str) -> None:
        deprecated = (
            self.result
            .filter(pl.col('id') == canonical_url)
            .select(pl.col('meta').struct['deprecated'])
            ['deprecated'][0]
        )
        # must remain None (was never deprecated) — not True
        assert deprecated is not True

    # -- nodes without collisions are unchanged ------------------------------

    def test_unique_node_unchanged(self) -> None:
        url = 'http://www.ebi.ac.uk/efo/EFO_002'
        before = self.input_df.filter(pl.col('id') == url)
        after = self.result.filter(pl.col('id') == url)
        assert before.equals(after)

    # -- already-deprecated nodes are not touched ----------------------------

    def test_already_deprecated_node_unchanged(self) -> None:
        url = 'http://purl.obolibrary.org/obo/HP_OLD'
        before = self.input_df.filter(pl.col('id') == url)
        after = self.result.filter(pl.col('id') == url)
        assert before.equals(after)

    # -- n_clean filter removes superseded nodes automatically ---------------

    def test_n_clean_filter_removes_superseded(self) -> None:
        n_clean = self.result.filter(
            pl.col('type') == 'CLASS',
            ~pl.col('meta').struct['deprecated'] | pl.col('meta').struct['deprecated'].is_null(),
        )
        remaining_ids = n_clean['id'].to_list()
        assert 'http://purl.obolibrary.org/obo/HP_001' not in remaining_ids
        assert 'http://purl.obolibrary.org/obo/HP_002' not in remaining_ids

    def test_n_clean_filter_keeps_canonical(self) -> None:
        n_clean = self.result.filter(
            pl.col('type') == 'CLASS',
            ~pl.col('meta').struct['deprecated'] | pl.col('meta').struct['deprecated'].is_null(),
        )
        remaining_ids = n_clean['id'].to_list()
        assert 'http://www.ebi.ac.uk/efo/EFO_001' in remaining_ids
        assert 'http://purl.obolibrary.org/obo/MONDO_001' in remaining_ids


class TestAnnotateNameDuplicatesEdgeCases:
    """Edge-case tests for annotate_name_duplicates."""

    def test_no_collisions_returns_input_unchanged(self) -> None:
        df = _make_df(
            _make_node('http://www.ebi.ac.uk/efo/EFO_A', 'disease alpha'),
            _make_node('http://www.ebi.ac.uk/efo/EFO_B', 'disease beta'),
        )
        result = annotate_name_duplicates(df)
        assert result.equals(df)

    def test_non_class_nodes_ignored(self) -> None:
        """Nodes with type != CLASS must not be treated as collision candidates."""
        df = _make_df(
            _make_node('http://www.ebi.ac.uk/efo/EFO_A', 'disease alpha', type_='CLASS'),
            _make_node('http://purl.obolibrary.org/obo/HP_A', 'Disease Alpha', type_='PROPERTY'),
        )
        result = annotate_name_duplicates(df)
        # HP_A is not CLASS, so EFO_A should not be marked canonical over it
        hp_deprecated = result.filter(pl.col('id') == 'http://purl.obolibrary.org/obo/HP_A').select(pl.col('meta').struct['deprecated'])['deprecated'][0]
        assert hp_deprecated is not True

    def test_priority_efo_over_mondo(self) -> None:
        df = _make_df(
            _make_node('http://www.ebi.ac.uk/efo/EFO_X', 'shared name'),
            _make_node('http://purl.obolibrary.org/obo/MONDO_X', 'Shared Name'),
        )
        result = annotate_name_duplicates(df)
        efo_dep = result.filter(pl.col('id') == 'http://www.ebi.ac.uk/efo/EFO_X').select(pl.col('meta').struct['deprecated'])['deprecated'][0]
        mondo_dep = result.filter(pl.col('id') == 'http://purl.obolibrary.org/obo/MONDO_X').select(pl.col('meta').struct['deprecated'])['deprecated'][0]
        assert efo_dep is not True   # EFO wins
        assert mondo_dep is True     # MONDO is superseded

    def test_priority_mondo_over_hp(self) -> None:
        df = _make_df(
            _make_node('http://purl.obolibrary.org/obo/MONDO_Y', 'shared name'),
            _make_node('http://purl.obolibrary.org/obo/HP_Y', 'Shared Name'),
        )
        result = annotate_name_duplicates(df)
        mondo_dep = result.filter(pl.col('id') == 'http://purl.obolibrary.org/obo/MONDO_Y').select(pl.col('meta').struct['deprecated'])['deprecated'][0]
        hp_dep = result.filter(pl.col('id') == 'http://purl.obolibrary.org/obo/HP_Y').select(pl.col('meta').struct['deprecated'])['deprecated'][0]
        assert mondo_dep is not True  # MONDO wins
        assert hp_dep is True         # HP is superseded

    def test_unknown_prefix_superseded_by_hp(self) -> None:
        """Unknown prefix gets rank 99 — it beats HP (100) but loses to everything listed."""
        df = _make_df(
            _make_node('http://purl.obolibrary.org/obo/HP_Z', 'shared name'),
            _make_node('http://example.org/UNKNOWN_Z', 'Shared Name'),
        )
        result = annotate_name_duplicates(df)
        hp_dep = result.filter(pl.col('id') == 'http://purl.obolibrary.org/obo/HP_Z').select(pl.col('meta').struct['deprecated'])['deprecated'][0]
        unk_dep = result.filter(pl.col('id') == 'http://example.org/UNKNOWN_Z').select(pl.col('meta').struct['deprecated'])['deprecated'][0]
        assert unk_dep is not True  # unknown (rank 99) wins over HP (rank 100)
        assert hp_dep is True
