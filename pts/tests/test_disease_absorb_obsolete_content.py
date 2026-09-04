"""Tests for absorb_obsolete_content."""

from __future__ import annotations

import polars as pl

from pts.schemas.ontology import node
from pts.transformers.disease.coalescing import _IAO_REPLACED_BY, absorb_obsolete_content

OBO = 'http://purl.obolibrary.org/obo/'

EXACT = 'hasExactSynonym'
RELATED = 'hasRelatedSynonym'
NARROW = 'hasNarrowSynonym'
BROAD = 'hasBroadSynonym'


def _url(short_id: str) -> str:
    return f'{OBO}{short_id}'


def _synonym(pred: str, val: str) -> dict:
    return {'pred': pred, 'synonymType': None, 'val': val, 'xrefs': []}


_UNSET = object()


def _make_node(
    short_id: str,
    *,
    lbl: str | object | None = _UNSET,
    node_type: str = 'CLASS',
    deprecated: bool | None = None,
    replaced_by: list[str] | None = None,
    synonyms: list[dict] | None = None,
    null_synonyms: bool = False,
) -> dict:
    """Node carrying zero or more IAO_0100001 pointers and its own synonyms.

    ``null_synonyms`` gives the node a null synonym list rather than an empty
    one, which is how the real ontology represents a term with no synonyms.
    """
    return {
        'id': _url(short_id),
        'lbl': short_id if lbl is _UNSET else lbl,
        'type': node_type,
        'meta': {
            'basicPropertyValues': [
                {'pred': _IAO_REPLACED_BY, 'val': _url(target)} for target in (replaced_by or [])
            ],
            'comments': [],
            'definition': {'val': None, 'xrefs': []},
            'deprecated': deprecated,
            'subsets': [],
            'synonyms': None if null_synonyms else (synonyms or []),
            'xrefs': [],
        },
    }


def _make_df(*nodes: dict) -> pl.DataFrame:
    return pl.from_dicts(list(nodes), schema=node)


def _synonyms_of(df: pl.DataFrame, short_id: str, pred: str) -> list[str]:
    entries = df.filter(pl.col('id') == _url(short_id))['meta'].struct['synonyms'][0]
    return [entry['val'] for entry in entries.to_list() if entry['pred'] == pred]


class TestDonation:
    """An obsolete term hands its synonyms and its label to its replacement."""

    def test_each_category_lands_on_the_matching_category(self) -> None:
        df = _make_df(
            _make_node(
                'A',
                synonyms=[
                    _synonym(EXACT, 'a exact'),
                    _synonym(RELATED, 'a related'),
                    _synonym(NARROW, 'a narrow'),
                    _synonym(BROAD, 'a broad'),
                ],
            ),
            _make_node(
                'B',
                lbl='b label',
                deprecated=True,
                replaced_by=['A'],
                synonyms=[
                    _synonym(EXACT, 'b exact'),
                    _synonym(RELATED, 'b related'),
                    _synonym(NARROW, 'b narrow'),
                    _synonym(BROAD, 'b broad'),
                ],
            ),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b exact', 'b label']
        assert _synonyms_of(result, 'A', RELATED) == ['a related', 'b related']
        assert _synonyms_of(result, 'A', NARROW) == ['a narrow', 'b narrow']
        assert _synonyms_of(result, 'A', BROAD) == ['a broad', 'b broad']

    def test_a_donor_with_no_synonyms_still_hands_over_its_label(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label']

    def test_several_donors_all_reach_the_same_survivor(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A']),
            _make_node('C', lbl='c label', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label', 'c label']

    def test_a_resolved_chain_donates_from_every_link(self) -> None:
        # resolve_replacement_chains rewrites C -> B -> A into C -> A and B -> A
        # before this runs, so both links name the same survivor.
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A']),
            _make_node('C', lbl='c label', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label', 'c label']

    def test_an_unlisted_synonym_predicate_is_not_donated(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node(
                'B',
                lbl='b label',
                deprecated=True,
                replaced_by=['A'],
                synonyms=[_synonym('hasAbbreviation', 'b abbrev')],
            ),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label']
        assert _synonyms_of(result, 'A', 'hasAbbreviation') == []


class TestDonorsThatMustBeSkipped:
    """Not every deprecated term with a pointer is a legitimate donor."""

    def test_an_ambiguous_donor_gives_to_nobody(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', synonyms=[_synonym(EXACT, 'b exact')]),
            _make_node(
                'C',
                lbl='c label',
                deprecated=True,
                replaced_by=['A', 'B'],
                synonyms=[_synonym(EXACT, 'c exact')],
            ),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact']
        assert _synonyms_of(result, 'B', EXACT) == ['b exact']

    def test_a_donor_repeating_one_pointer_is_not_ambiguous(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A', 'A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label']

    def test_a_non_class_donor_is_skipped(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node(
                'B',
                lbl='a property, not a disease',
                node_type='PROPERTY',
                deprecated=True,
                replaced_by=['A'],
                synonyms=[_synonym(EXACT, 'property synonym')],
            ),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact']

    def test_a_donor_naming_a_dead_target_is_skipped(self) -> None:
        df = _make_df(
            _make_node('A', lbl='a label', deprecated=True),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == []

    def test_a_donor_naming_an_id_the_graph_lacks_is_skipped(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['NOT_THERE']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact']

    def test_a_live_term_never_donates(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', replaced_by=['A'], synonyms=[_synonym(EXACT, 'b exact')]),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact']


class TestValuesThatMustBeFiltered:
    """A donated string is not always worth keeping."""

    def test_the_survivors_own_name_is_not_donated_back(self) -> None:
        # annotate_name_duplicates merges on a case-folded label, so the loser's
        # label is by construction the survivor's own name in another case.
        df = _make_df(
            _make_node('A', lbl='Acidosis', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='acidosis', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact']

    def test_a_donated_value_matching_an_existing_one_is_not_repeated(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'shared')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A'], synonyms=[_synonym(EXACT, 'Shared')]),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['shared', 'b label']

    def test_two_donors_offering_the_same_value_donate_it_once(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A'], synonyms=[_synonym(EXACT, 'shared')]),
            _make_node('C', lbl='c label', deprecated=True, replaced_by=['A'], synonyms=[_synonym(EXACT, 'Shared')]),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'shared', 'b label', 'c label']

    def test_an_obsoletion_prefixed_label_is_not_donated(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='obsolete anaemia of chronic disease', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact']

    def test_an_obsoletion_prefixed_synonym_is_not_donated(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node(
                'B',
                lbl='b label',
                deprecated=True,
                replaced_by=['A'],
                synonyms=[_synonym(EXACT, 'OBSOLETE. use A')],
            ),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label']

    def test_an_underscore_separated_obsoletion_marker_is_not_donated(self) -> None:
        # GO writes the marker as 'obsolete_<name>' rather than 'obsolete <name>'.
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node(
                'B',
                lbl='b label',
                deprecated=True,
                replaced_by=['A'],
                synonyms=[_synonym(EXACT, 'obsolete_inflammation')],
            ),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label']

    def test_the_bare_word_obsolete_is_not_donated(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='obsolete', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact']

    def test_obsolete_as_an_ordinary_word_is_kept(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='obsoletion syndrome', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'obsoletion syndrome']

    def test_whitespace_is_normalised_before_comparison(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'shared')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A'], synonyms=[_synonym(EXACT, ' shared\n')]),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['shared', 'b label']

    def test_a_null_synonym_value_is_dropped(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A'], synonyms=[_synonym(EXACT, None)]),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label']

    def test_a_donor_with_no_label_donates_only_its_synonyms(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl=None, deprecated=True, replaced_by=['A'], synonyms=[_synonym(EXACT, 'b exact')]),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b exact']


class TestNullSynonymLists:
    """A term with no synonyms carries a null list, not an empty one."""

    def test_a_survivor_with_null_synonyms_still_absorbs(self) -> None:
        # Concatenating onto a null list yields null in polars, which would drop
        # the donation entirely.  GO_0044691 'tooth eruption' is a real example.
        df = _make_df(
            _make_node('A', null_synonyms=True),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A'], synonyms=[_synonym(EXACT, 'teething')]),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['teething', 'b label']

    def test_a_donor_with_null_synonyms_still_hands_over_its_label(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A'], null_synonyms=True),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'A', EXACT) == ['a exact', 'b label']

    def test_a_survivor_with_null_synonyms_and_no_donor_keeps_null(self) -> None:
        df = _make_df(
            _make_node('A', null_synonyms=True),
            _make_node('B', synonyms=[_synonym(EXACT, 'b exact')]),
        )
        result = absorb_obsolete_content(df)
        assert result.equals(df)


class TestFrameIsOtherwiseUntouched:
    """The pass rewrites synonyms and nothing else."""

    def test_schema_and_height_are_preserved(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        assert result.schema == df.schema
        assert result.height == df.height
        assert result['id'].to_list() == df['id'].to_list()

    def test_a_frame_with_no_obsolete_terms_is_returned_unchanged(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', synonyms=[_synonym(EXACT, 'b exact')]),
        )
        result = absorb_obsolete_content(df)
        assert result.equals(df)

    def test_a_donor_keeps_its_own_synonyms(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A'], synonyms=[_synonym(EXACT, 'b exact')]),
        )
        result = absorb_obsolete_content(df)
        assert _synonyms_of(result, 'B', EXACT) == ['b exact']

    def test_other_meta_fields_survive(self) -> None:
        df = _make_df(
            _make_node('A', synonyms=[_synonym(EXACT, 'a exact')]),
            _make_node('B', lbl='b label', deprecated=True, replaced_by=['A']),
        )
        result = absorb_obsolete_content(df)
        meta = result.filter(pl.col('id') == _url('B'))['meta'][0]
        assert meta['deprecated'] is True
        assert [entry['val'] for entry in meta['basicPropertyValues']] == [_url('A')]
