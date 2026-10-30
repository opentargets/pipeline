"""Tests for absorb_obsolete_locations and direct_location_ids."""

from __future__ import annotations

import polars as pl

from pts.schemas.ontology import node
from pts.transformers.disease.coalescing import _IAO_REPLACED_BY, LOCATED_IN, absorb_obsolete_locations
from pts.transformers.disease.disease import direct_location_ids

OBO = 'http://purl.obolibrary.org/obo/'


def _url(short_id: str) -> str:
    return f'{OBO}{short_id}'


def _make_node(
    short_id: str,
    *,
    node_type: str = 'CLASS',
    deprecated: bool | None = None,
    replaced_by: list[str] | None = None,
    locations: list[str] | None = None,
    null_bpv: bool = False,
) -> dict:
    """Node carrying zero or more IAO_0100001 pointers and OTAR_0000021 locations.

    ``null_bpv`` gives the node a null property list rather than an empty one,
    which is how the real ontology represents a term with no properties.
    """
    bpv = [{'pred': _IAO_REPLACED_BY, 'val': _url(target)} for target in (replaced_by or [])]
    bpv += [{'pred': LOCATED_IN, 'val': _url(location)} for location in (locations or [])]
    return {
        'id': _url(short_id),
        'lbl': short_id,
        'type': node_type,
        'meta': {
            'basicPropertyValues': None if null_bpv else bpv,
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


def _locations_of(df: pl.DataFrame, short_id: str) -> list[str]:
    row = direct_location_ids(df).filter(pl.col('id') == short_id)
    return row['directLocationIds'][0].to_list() if row.height else []


class TestDirectLocationIds:
    """OTAR_0000021 values become short ids, deduplicated and sorted."""

    def test_locations_are_shortened_and_sorted(self) -> None:
        df = _make_df(_make_node('A', locations=['UBERON_2', 'UBERON_1']))
        assert _locations_of(df, 'A') == ['UBERON_1', 'UBERON_2']

    def test_repeated_location_appears_once(self) -> None:
        df = _make_df(_make_node('A', locations=['UBERON_1', 'UBERON_1']))
        assert _locations_of(df, 'A') == ['UBERON_1']

    def test_other_properties_are_not_locations(self) -> None:
        df = _make_df(_make_node('A', deprecated=True, replaced_by=['B']), _make_node('B'))
        assert direct_location_ids(df).is_empty()

    def test_null_property_list_is_tolerated(self) -> None:
        df = _make_df(_make_node('A', null_bpv=True), _make_node('B', locations=['UBERON_1']))
        assert _locations_of(df, 'A') == []
        assert _locations_of(df, 'B') == ['UBERON_1']


class TestAbsorbObsoleteLocations:
    """An obsolete term hands its locations to its single live replacement."""

    def test_donor_locations_reach_the_survivor(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B'], locations=['UBERON_1']),
            _make_node('B', locations=['UBERON_2']),
        )
        assert _locations_of(absorb_obsolete_locations(df), 'B') == ['UBERON_1', 'UBERON_2']

    def test_a_survivor_with_null_properties_receives_the_donation(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B'], locations=['UBERON_1']),
            _make_node('B', null_bpv=True),
        )
        assert _locations_of(absorb_obsolete_locations(df), 'B') == ['UBERON_1']

    def test_a_location_the_survivor_holds_is_not_added_again(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B'], locations=['UBERON_1']),
            _make_node('B', locations=['UBERON_1']),
        )
        out = absorb_obsolete_locations(df)
        bpv = out.filter(pl.col('id') == _url('B'))['meta'].struct['basicPropertyValues'][0].to_list()
        assert [entry['val'] for entry in bpv if entry['pred'] == LOCATED_IN] == [_url('UBERON_1')]

    def test_several_donors_all_reach_the_same_survivor(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['C'], locations=['UBERON_1']),
            _make_node('B', deprecated=True, replaced_by=['C'], locations=['UBERON_1', 'UBERON_2']),
            _make_node('C'),
        )
        assert _locations_of(absorb_obsolete_locations(df), 'C') == ['UBERON_1', 'UBERON_2']

    def test_an_ambiguous_donor_gives_to_nobody(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B', 'C'], locations=['UBERON_1']),
            _make_node('B'),
            _make_node('C'),
        )
        out = absorb_obsolete_locations(df)
        assert _locations_of(out, 'B') == []
        assert _locations_of(out, 'C') == []

    def test_a_donor_naming_a_dead_target_is_skipped(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B'], locations=['UBERON_1']),
            _make_node('B', deprecated=True),
        )
        assert _locations_of(absorb_obsolete_locations(df), 'B') == []

    def test_a_non_class_donor_is_skipped(self) -> None:
        df = _make_df(
            _make_node('A', node_type='PROPERTY', deprecated=True, replaced_by=['B'], locations=['UBERON_1']),
            _make_node('B'),
        )
        assert _locations_of(absorb_obsolete_locations(df), 'B') == []

    def test_nothing_to_absorb_returns_the_input(self) -> None:
        df = _make_df(_make_node('A', locations=['UBERON_1']))
        assert absorb_obsolete_locations(df).equals(df)

    def test_schema_is_preserved(self) -> None:
        df = _make_df(
            _make_node('A', deprecated=True, replaced_by=['B'], locations=['UBERON_1']),
            _make_node('B'),
        )
        assert absorb_obsolete_locations(df).schema == df.schema
