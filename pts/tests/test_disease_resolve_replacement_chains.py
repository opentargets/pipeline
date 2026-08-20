"""Tests for resolve_replacement_chains."""

from __future__ import annotations

import polars as pl
import pytest

from pts.schemas.ontology import node
from pts.transformers.disease.coalescing import _IAO_REPLACED_BY, resolve_replacement_chains

OBO = 'http://purl.obolibrary.org/obo/'
OTHER_PRED = 'http://www.geneontology.org/formats/oboInOwl#consider'


def _url(short_id: str) -> str:
    return f'{OBO}{short_id}'


def _make_node(
    short_id: str,
    *,
    deprecated: bool | None = None,
    replaced_by: list[str] | None = None,
    extra_bpv: list[dict] | None = None,
) -> dict:
    """Node carrying zero or more IAO_0100001 pointers."""
    bpv = [{'pred': _IAO_REPLACED_BY, 'val': _url(target)} for target in (replaced_by or [])]
    return {
        'id': _url(short_id),
        'lbl': short_id,
        'type': 'CLASS',
        'meta': {
            'basicPropertyValues': bpv + (extra_bpv or []),
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


def _pointers(df: pl.DataFrame, short_id: str) -> list[str]:
    bpv = df.filter(pl.col('id') == _url(short_id))['meta'].struct['basicPropertyValues'][0]
    return [entry['val'] for entry in bpv.to_list() if entry['pred'] == _IAO_REPLACED_BY]


class TestChainResolution:
    """A pointer must name a term that is still alive."""

    def test_two_hop_chain_resolves_to_the_live_term(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', deprecated=True, replaced_by=['C']),
            _make_node('C'),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('C')]

    def test_three_hop_chain_resolves_to_the_live_term(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', deprecated=True, replaced_by=['C']),
            _make_node('C', deprecated=True, replaced_by=['D']),
            _make_node('D'),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('D')]
        assert _pointers(result, 'B') == [_url('D')]

    def test_pointer_to_a_live_term_is_untouched(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B'),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('B')]

    def test_chain_ending_on_a_term_without_replacement_stops_there(self) -> None:
        """B is obsolete with nowhere to go, so A can do no better than B."""
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', deprecated=True),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('B')]

    def test_chain_running_into_a_dead_end_is_not_rewritten(self) -> None:
        """A -> B -> C with C obsolete and going nowhere.

        Rewriting A to C would swap one term n_clean drops for another, so the
        pointer is left alone rather than moved between two dead nodes.
        """
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', deprecated=True, replaced_by=['C']),
            _make_node('C', deprecated=True),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('B')]

    def test_chain_running_into_an_ambiguous_term_is_not_rewritten(self) -> None:
        """C names two successors, so the chain stops before it, not on it."""
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', deprecated=True, replaced_by=['C']),
            _make_node('C', deprecated=True, replaced_by=['D', 'E']),
            _make_node('D'),
            _make_node('E'),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('B')]


class TestAmbiguityAndCycles:
    """Hops that are not clearly correct are left alone."""

    def test_a_repeated_identical_pointer_is_not_ambiguity(self) -> None:
        """B names one successor, written twice.

        14 terms in the 26.06 source ontology carry the same replacement twice.
        Counting entries rather than distinct targets treats them as ambiguous
        and strands every chain running through them.
        """
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', deprecated=True, replaced_by=['C', 'C']),
            _make_node('C'),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('C')]

    def test_ambiguous_target_is_not_followed(self) -> None:
        """B names two replacements, so picking one would be a guess."""
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', deprecated=True, replaced_by=['C', 'D']),
            _make_node('C'),
            _make_node('D'),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('B')]

    def test_a_node_with_two_pointers_keeps_both(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B', 'C']),
            _make_node('B'),
            _make_node('C'),
        )
        result = resolve_replacement_chains(df)
        assert sorted(_pointers(result, 'A')) == [_url('B'), _url('C')]

    def test_cycle_is_left_as_it_stands(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', deprecated=True, replaced_by=['A']),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('B')]
        assert _pointers(result, 'B') == [_url('A')]

    def test_self_reference_is_left_as_it_stands(self) -> None:
        """Two terms do this in the 26.06 source ontology."""
        df = _make_df(_make_node('A', deprecated=True, replaced_by=['A']))
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('A')]


class TestStructure:
    """The output must stay a drop-in for the rest of the transformer."""

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return _make_df(
            _make_node('A', deprecated=True, replaced_by=['B'], extra_bpv=[{'pred': OTHER_PRED, 'val': 'keep me'}]),
            _make_node('B', deprecated=True, replaced_by=['C']),
            _make_node('C'),
            _make_node('LIVE'),
        )

    def test_shape_preserved(self, df: pl.DataFrame) -> None:
        assert resolve_replacement_chains(df).shape == df.shape

    def test_schema_preserved(self, df: pl.DataFrame) -> None:
        assert resolve_replacement_chains(df).schema == df.schema

    def test_other_predicates_are_untouched(self, df: pl.DataFrame) -> None:
        bpv = resolve_replacement_chains(df).filter(pl.col('id') == _url('A'))['meta'].struct[
            'basicPropertyValues'
        ][0]
        assert {'pred': OTHER_PRED, 'val': 'keep me'} in bpv.to_list()

    def test_node_without_pointers_is_untouched(self, df: pl.DataFrame) -> None:
        assert resolve_replacement_chains(df).filter(pl.col('id') == _url('LIVE'))['meta'].struct[
            'basicPropertyValues'
        ][0].to_list() == []

    def test_no_chains_returns_input_unchanged(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B'),
        )
        assert resolve_replacement_chains(df).equals(df)

    def test_no_pointers_returns_input_unchanged(self) -> None:
        df = _make_df(_make_node('A'), _make_node('B'))
        assert resolve_replacement_chains(df).equals(df)

    def test_a_pointer_from_a_live_node_is_ignored(self) -> None:
        """Only deprecated nodes assert a replacement, so a live one is not a hop."""
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B']),
            _make_node('B', replaced_by=['C']),
            _make_node('C'),
        )
        result = resolve_replacement_chains(df)
        assert _pointers(result, 'A') == [_url('B')]
