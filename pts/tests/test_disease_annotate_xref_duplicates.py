"""Tests for annotate_xref_duplicates."""

from __future__ import annotations

import polars as pl
import pytest

from pts.schemas.ontology import node
from pts.transformers.disease.coalescing import _IAO_REPLACED_BY, annotate_xref_duplicates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EFO = 'http://www.ebi.ac.uk/efo/'
OBO = 'http://purl.obolibrary.org/obo/'
ORDO = 'http://www.orpha.net/ORDO/'


def _url(short_id: str) -> str:
    """Full URL for a short ontology id, using the namespace that ontology uses."""
    if short_id.startswith('EFO_'):
        return f'{EFO}{short_id}'
    if short_id.startswith('Orphanet_'):
        return f'{ORDO}{short_id}'
    return f'{OBO}{short_id}'


def _make_node(
    short_id: str,
    lbl: str,
    *,
    xrefs: list[str] | None = None,
    synonyms: list[str] | None = None,
    deprecated: bool | None = None,
    type_: str = 'CLASS',
) -> dict:
    """Build a single node dict compatible with the ``node`` schema.

    ``synonyms`` are added as hasExactSynonym, the only predicate the rule
    accepts as corroboration.
    """
    return {
        'id': _url(short_id),
        'lbl': lbl,
        'type': type_,
        'meta': {
            'basicPropertyValues': [],
            'comments': [],
            'definition': {'val': None, 'xrefs': []},
            'deprecated': deprecated,
            'subsets': [],
            'synonyms': [
                {'pred': 'hasExactSynonym', 'synonymType': None, 'val': s, 'xrefs': []} for s in (synonyms or [])
            ],
            'xrefs': [{'val': x} for x in (xrefs or [])],
        },
    }


def _make_df(*nodes: dict) -> pl.DataFrame:
    return pl.from_dicts(list(nodes), schema=node)


def _make_edges(*pairs: tuple[str, str]) -> pl.DataFrame:
    """Edge frame of is_a relations, given (child, parent) short-id pairs."""
    return pl.DataFrame(
        {
            'sub': [_url(child) for child, _ in pairs],
            'pred': ['is_a'] * len(pairs),
            'obj': [_url(parent) for _, parent in pairs],
        },
        schema={'sub': pl.String(), 'pred': pl.String(), 'obj': pl.String()},
    )


NO_EDGES = _make_edges()


def _is_deprecated(df: pl.DataFrame, short_id: str) -> bool:
    return df.filter(pl.col('id') == _url(short_id))['meta'].struct['deprecated'][0] is True


def _replacement(df: pl.DataFrame, short_id: str) -> str | None:
    """The IAO_0100001 target injected on a node, or None."""
    bpv = df.filter(pl.col('id') == _url(short_id))['meta'].struct['basicPropertyValues'][0]
    targets = [entry['val'] for entry in bpv.to_list() if entry['pred'] == _IAO_REPLACED_BY]
    return targets[0] if targets else None


def _merged(df: pl.DataFrame, superseded: str, canonical: str) -> bool:
    return _is_deprecated(df, superseded) and _replacement(df, superseded) == _url(canonical)


def _any_deprecated(df: pl.DataFrame, *short_ids: str) -> bool:
    return any(_is_deprecated(df, s) for s in short_ids)


# ---------------------------------------------------------------------------
# The direct cross-reference arm (B1)
# ---------------------------------------------------------------------------


class TestDirectXrefArm:
    """One term's dbXRefs names the other term's ontology id."""

    def test_merges_and_keeps_higher_priority_ontology(self) -> None:
        df = _make_df(
            _make_node('EFO_001', 'a disease', xrefs=['MONDO:001']),
            _make_node('MONDO_001', 'a disease by another name'),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _merged(result, 'MONDO_001', 'EFO_001')

    def test_mondo_wins_over_hp(self) -> None:
        df = _make_df(
            _make_node('MONDO_010', 'intervertebral disk degenerative disorder', xrefs=['HP:010']),
            _make_node('HP_010', 'Intervertebral disk degeneration'),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _merged(result, 'HP_010', 'MONDO_010')

    def test_direction_of_the_xref_does_not_matter(self) -> None:
        """The HP term pointing at MONDO must merge the same way round."""
        df = _make_df(
            _make_node('MONDO_010', 'intervertebral disk degenerative disorder'),
            _make_node('HP_010', 'Intervertebral disk degeneration', xrefs=['MONDO:010']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _merged(result, 'HP_010', 'MONDO_010')

    def test_orphanet_xref_spelling_is_matched(self) -> None:
        df = _make_df(
            _make_node('MONDO_020', 'Mobius syndrome', xrefs=['Orphanet:570']),
            _make_node('Orphanet_570', 'Moebius syndrome'),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _merged(result, 'Orphanet_570', 'MONDO_020')

    def test_no_merge_within_one_ontology(self) -> None:
        df = _make_df(
            _make_node('MONDO_001', 'a disease', xrefs=['MONDO:002']),
            _make_node('MONDO_002', 'a disease subtype'),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'MONDO_001', 'MONDO_002')

    def test_xref_to_absent_term_is_ignored(self) -> None:
        df = _make_df(_make_node('EFO_001', 'a disease', xrefs=['MONDO:999']))
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'EFO_001')


# ---------------------------------------------------------------------------
# The shared cross-reference + synonym arm (B2*)
# ---------------------------------------------------------------------------


class TestSharedXrefArm:
    """A shared fine-grained xref merges only with a corroborating synonym."""

    def test_merges_on_shared_fine_xref_and_shared_synonym(self) -> None:
        df = _make_df(
            _make_node('MONDO_100', 'obesity disorder', xrefs=['OMIM:601665'], synonyms=['obesity']),
            _make_node('HP_100', 'Obesity', xrefs=['OMIM:601665'], synonyms=['obesity']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _merged(result, 'HP_100', 'MONDO_100')

    def test_label_counts_as_the_shared_name(self) -> None:
        """A label matching the other term's synonym is corroboration enough."""
        df = _make_df(
            _make_node('MONDO_100', 'obesity', xrefs=['OMIM:601665']),
            _make_node('HP_100', 'Obesity disorder', xrefs=['OMIM:601665'], synonyms=['obesity']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _merged(result, 'HP_100', 'MONDO_100')

    def test_no_merge_without_a_shared_name(self) -> None:
        """This is the guard that stopped generic hub xrefs chaining terms."""
        df = _make_df(
            _make_node('MONDO_100', 'pancreatic carcinoma', xrefs=['NCIT:C3262']),
            _make_node('HP_100', 'Hepatic neoplasm', xrefs=['NCIT:C3262']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'MONDO_100', 'HP_100')

    def test_no_merge_on_coarse_xref(self) -> None:
        """ICD codes are many-to-one, so sharing one is not equivalence.

        Held by the allow-list rather than the coarse deny-list, since the arm
        only ever considers fine-grained namespaces. See the ontology_ids tests
        for the deny-list itself.
        """
        df = _make_df(
            _make_node('MONDO_100', 'obesity', xrefs=['ICD10:E66'], synonyms=['obesity']),
            _make_node('HP_100', 'Obesity', xrefs=['ICD10:E66'], synonyms=['obesity']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'MONDO_100', 'HP_100')

    def test_no_merge_on_excluded_namespace(self) -> None:
        """A shared publication is not a shared identity.

        As above, the allow-list is what stops this here.
        """
        df = _make_df(
            _make_node('MONDO_100', 'obesity', xrefs=['PMID:12345678'], synonyms=['obesity']),
            _make_node('HP_100', 'Obesity', xrefs=['PMID:12345678'], synonyms=['obesity']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'MONDO_100', 'HP_100')

    def test_snomed_spellings_are_folded(self) -> None:
        """MONDO writes SCTID, HP writes SNOMEDCT_US for the same code."""
        df = _make_df(
            _make_node('MONDO_100', 'obesity disorder', xrefs=['SCTID:414916001'], synonyms=['obesity']),
            _make_node('HP_100', 'Obesity', xrefs=['SNOMEDCT_US:414916001'], synonyms=['obesity']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _merged(result, 'HP_100', 'MONDO_100')

    def test_merges_a_direct_xref_even_across_a_hierarchy(self) -> None:
        """Arm 1 has no hierarchy guard, and that is deliberate.

        A direct cross-reference is a curator asserting the two terms are the
        same thing, which outranks an is_a edge between them. Ontologies do file
        an eponym under the disease it names, and merging those is right.
        """
        df = _make_df(
            _make_node('MONDO_010', 'a disease', xrefs=['HP:010']),
            _make_node('HP_010', 'the same disease by another name'),
        )
        edges = _make_edges(('HP_010', 'MONDO_010'))
        result = annotate_xref_duplicates(df, edges)
        assert _merged(result, 'HP_010', 'MONDO_010')

    def test_no_merge_between_ancestor_and_descendant(self) -> None:
        """Hierarchy collapse: the pair shares everything but one is a parent."""
        df = _make_df(
            _make_node('MONDO_200', 'cancer', xrefs=['UMLS:C0006826'], synonyms=['malignant neoplasm']),
            _make_node('HP_200', 'Malignant neoplasm', xrefs=['UMLS:C0006826'], synonyms=['malignant neoplasm']),
        )
        edges = _make_edges(('HP_200', 'MONDO_200'))
        result = annotate_xref_duplicates(df, edges)
        assert not _any_deprecated(result, 'MONDO_200', 'HP_200')

    def test_no_merge_across_an_indirect_ancestor(self) -> None:
        """The guard must follow the full is_a closure, not just direct parents."""
        df = _make_df(
            _make_node('MONDO_200', 'cancer', xrefs=['UMLS:C0006826'], synonyms=['malignant neoplasm']),
            _make_node('MONDO_201', 'carcinoma'),
            _make_node('HP_200', 'Malignant neoplasm', xrefs=['UMLS:C0006826'], synonyms=['malignant neoplasm']),
        )
        edges = _make_edges(('HP_200', 'MONDO_201'), ('MONDO_201', 'MONDO_200'))
        result = annotate_xref_duplicates(df, edges)
        assert not _any_deprecated(result, 'MONDO_200', 'HP_200')

    def test_no_merge_within_one_ontology(self) -> None:
        df = _make_df(
            _make_node('MONDO_100', 'obesity', xrefs=['OMIM:601665'], synonyms=['obesity']),
            _make_node('MONDO_101', 'obesity disorder', xrefs=['OMIM:601665'], synonyms=['obesity']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'MONDO_100', 'MONDO_101')


# ---------------------------------------------------------------------------
# Group-level gating
# ---------------------------------------------------------------------------


class TestGroupGating:
    """Groups bridging two terms of one ontology are dropped whole."""

    def test_group_bridging_two_same_ontology_terms_is_dropped(self) -> None:
        """HP_300 equates to two distinct MONDO terms, so the whole group goes."""
        df = _make_df(
            _make_node('MONDO_300', 'first disease'),
            _make_node('MONDO_301', 'second disease'),
            _make_node('HP_300', 'A phenotype', xrefs=['MONDO:300', 'MONDO:301']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'MONDO_300', 'MONDO_301', 'HP_300')

    def test_clean_three_ontology_group_is_kept(self) -> None:
        df = _make_df(
            _make_node('EFO_400', 'a disease'),
            _make_node('MONDO_400', 'a disease', xrefs=['EFO:400']),
            _make_node('HP_400', 'A disease', xrefs=['MONDO:400']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _merged(result, 'MONDO_400', 'EFO_400')
        assert _merged(result, 'HP_400', 'EFO_400')

    def test_transitive_group_points_every_member_at_one_canonical(self) -> None:
        """Every superseded member must name the surviving term, not its neighbour."""
        df = _make_df(
            _make_node('EFO_400', 'a disease'),
            _make_node('MONDO_400', 'a disease', xrefs=['EFO:400']),
            _make_node('HP_400', 'A disease', xrefs=['MONDO:400']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _replacement(result, 'HP_400') == _url('EFO_400')


# ---------------------------------------------------------------------------
# Participation rules
# ---------------------------------------------------------------------------


class TestParticipation:
    """Which nodes the rule is allowed to touch at all."""

    def test_go_and_mp_are_excluded(self) -> None:
        """A biological process and a mouse phenotype are not a disease duplicate."""
        df = _make_df(
            _make_node('GO_0006954', 'inflammatory response', xrefs=['MESH:D007249'], synonyms=['inflammation']),
            _make_node('MP_0001845', 'inflammation', xrefs=['MESH:D007249'], synonyms=['inflammation']),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'GO_0006954', 'MP_0001845')

    def test_already_deprecated_node_is_untouched(self) -> None:
        df = _make_df(
            _make_node('EFO_001', 'a disease', xrefs=['MONDO:001']),
            _make_node('MONDO_001', 'a disease', deprecated=True),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _replacement(result, 'MONDO_001') is None

    def test_non_class_nodes_ignored(self) -> None:
        df = _make_df(
            _make_node('EFO_001', 'a disease', xrefs=['MONDO:001']),
            _make_node('MONDO_001', 'a disease', type_='PROPERTY'),
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert not _any_deprecated(result, 'MONDO_001')


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------


class TestStructure:
    """The output must stay a drop-in for the rest of the transformer."""

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return _make_df(
            _make_node('EFO_001', 'a disease', xrefs=['MONDO:001']),
            _make_node('MONDO_001', 'a disease by another name'),
            _make_node('EFO_002', 'unrelated disease'),
        )

    def test_shape_preserved(self, df: pl.DataFrame) -> None:
        assert annotate_xref_duplicates(df, NO_EDGES).shape == df.shape

    def test_schema_preserved(self, df: pl.DataFrame) -> None:
        assert annotate_xref_duplicates(df, NO_EDGES).schema == df.schema

    def test_canonical_node_not_deprecated(self, df: pl.DataFrame) -> None:
        assert not _is_deprecated(annotate_xref_duplicates(df, NO_EDGES), 'EFO_001')

    def test_untouched_node_keeps_its_metadata(self, df: pl.DataFrame) -> None:
        result = annotate_xref_duplicates(df, NO_EDGES)
        assert _replacement(result, 'EFO_002') is None
        assert not _is_deprecated(result, 'EFO_002')

    def test_no_candidates_returns_input_unchanged(self) -> None:
        df = _make_df(
            _make_node('EFO_001', 'a disease'),
            _make_node('MONDO_001', 'another disease'),
        )
        assert annotate_xref_duplicates(df, NO_EDGES).equals(df)

    def test_existing_iao_entries_are_preserved(self) -> None:
        """A node already carrying basicPropertyValues must keep them."""
        df = _make_df(
            _make_node('EFO_001', 'a disease', xrefs=['MONDO:001']),
            _make_node('MONDO_001', 'a disease'),
        )
        df = df.with_columns(
            meta=pl.col('meta').struct.with_fields(
                basicPropertyValues=pl
                .when(pl.col('id') == _url('MONDO_001'))
                .then(pl.lit([{'pred': 'http://example.org/other', 'val': 'kept'}]))
                .otherwise(pl.col('meta').struct['basicPropertyValues'])
            )
        )
        result = annotate_xref_duplicates(df, NO_EDGES)
        preds = [
            entry['pred']
            for entry in result
            .filter(pl.col('id') == _url('MONDO_001'))['meta']
            .struct['basicPropertyValues'][0]
            .to_list()
        ]
        assert 'http://example.org/other' in preds
        assert _IAO_REPLACED_BY in preds

    def test_result_is_deterministic(self) -> None:
        """Union-find and group iteration must not depend on hash order."""
        df = _make_df(
            _make_node('EFO_500', 'a disease'),
            _make_node('MONDO_500', 'a disease', xrefs=['EFO:500']),
            _make_node('HP_500', 'A disease', xrefs=['MONDO:500']),
            _make_node('Orphanet_500', 'A disease', xrefs=['EFO:500']),
        )
        first = annotate_xref_duplicates(df, NO_EDGES)
        for _ in range(3):
            assert annotate_xref_duplicates(df, NO_EDGES).equals(first)
