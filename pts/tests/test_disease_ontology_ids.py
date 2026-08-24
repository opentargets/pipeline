"""Tests for the cross-ontology identifier helpers."""

from __future__ import annotations

import polars as pl

from pts.schemas.ontology import node
from pts.transformers.disease.ontology_ids import name_bags, normalised_xrefs

OBO = 'http://purl.obolibrary.org/obo/'


def _make_node(
    short_id: str,
    lbl: str = 'a term',
    *,
    xrefs: list[str] | None = None,
    synonyms: list[str] | None = None,
) -> dict:
    return {
        'id': f'{OBO}{short_id}',
        'lbl': lbl,
        'type': 'CLASS',
        'meta': {
            'basicPropertyValues': [],
            'comments': [],
            'definition': {'val': None, 'xrefs': []},
            'deprecated': None,
            'subsets': [],
            'synonyms': [
                {'pred': 'hasExactSynonym', 'synonymType': None, 'val': s, 'xrefs': []} for s in (synonyms or [])
            ],
            'xrefs': [{'val': x} for x in (xrefs or [])],
        },
    }


def _active(*nodes: dict) -> pl.DataFrame:
    """The frame shape the helpers are handed, carrying short_id."""
    return pl.from_dicts(list(nodes), schema=node).with_columns(
        short_id=pl.col('id').str.split('/').list.last(),
    )


def _canonicals(df: pl.DataFrame) -> set[str]:
    return set(normalised_xrefs(df)['canonical'].to_list())


class TestNormalisedXrefs:
    """What survives normalisation, and in what form."""

    def test_namespace_is_uppercased(self) -> None:
        assert _canonicals(_active(_make_node('MONDO_1', xrefs=['omim:123']))) == {'OMIM:123'}

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert _canonicals(_active(_make_node('MONDO_1', xrefs=[' OMIM : 123 ']))) == {'OMIM:123'}

    def test_snomed_spellings_fold_together(self) -> None:
        df = _active(
            _make_node('MONDO_1', xrefs=['SCTID:99']),
            _make_node('HP_1', xrefs=['SNOMEDCT_US:99']),
        )
        assert _canonicals(df) == {'SNOMED:99'}

    def test_orphanet_alias_folds(self) -> None:
        assert _canonicals(_active(_make_node('MONDO_1', xrefs=['ORPHA:570']))) == {'ORPHANET:570'}

    def test_publication_references_are_dropped(self) -> None:
        """A shared paper is not a shared identity, whoever asks."""
        assert _canonicals(_active(_make_node('MONDO_1', xrefs=['PMID:12345678']))) == set()

    def test_coarse_classification_codes_are_dropped(self) -> None:
        """Many-to-one by design, so sharing one says nothing about identity."""
        df = _active(_make_node('MONDO_1', xrefs=['ICD10:E66', 'ICD10CM:E66', 'MedDRA:10029883']))
        assert _canonicals(df) == set()

    def test_a_value_without_a_colon_is_dropped(self) -> None:
        assert _canonicals(_active(_make_node('MONDO_1', xrefs=['nonsense']))) == set()

    def test_a_repeated_reference_appears_once(self) -> None:
        df = _active(_make_node('MONDO_1', xrefs=['OMIM:123', 'omim:123']))
        assert normalised_xrefs(df).height == 1


class TestNameBags:
    """The names a term answers to, for corroborating a shared reference."""

    def test_label_and_exact_synonyms_are_collected(self) -> None:
        bags = name_bags(_active(_make_node('MONDO_1', 'Obesity', synonyms=['adiposity'])))
        assert bags['MONDO_1'] == {'obesity', 'adiposity'}

    def test_only_exact_synonyms_count(self) -> None:
        """A related or broad synonym is how a term ends up under a name it is not."""
        df = pl.from_dicts(
            [
                {
                    **_make_node('MONDO_1', 'Obesity'),
                    'meta': {
                        'basicPropertyValues': [],
                        'comments': [],
                        'definition': {'val': None, 'xrefs': []},
                        'deprecated': None,
                        'subsets': [],
                        'synonyms': [
                            {'pred': 'hasRelatedSynonym', 'synonymType': None, 'val': 'fatness', 'xrefs': []},
                        ],
                        'xrefs': [],
                    },
                }
            ],
            schema=node,
        ).with_columns(short_id=pl.col('id').str.split('/').list.last())
        assert name_bags(df)['MONDO_1'] == {'obesity'}

    def test_embedded_newlines_do_not_split_a_name(self) -> None:
        """The index strips newlines from synonyms, so comparison must too."""
        df = _active(_make_node('MONDO_1', 'a disease', synonyms=['multiple\nsclerosis']))
        assert name_bags(df)['MONDO_1'] == {'a disease', 'multiplesclerosis'}
