"""End-to-end tests for absorbing an obsolete term's synonyms into its replacement.

disease() now folds an obsolete term's own synonyms into the surviving term that
replaces it: each of the 4 synonym categories unions into the matching category on
the survivor (exact into exact, related into related, and so on), and the obsolete
term's own name/label is added to the survivor's exactSynonyms. See
disease.py's obsolete_synonyms/obsolete_names blocks, and
test_disease_obsolete_terms.py for the pre-existing obsoleteTerms bookkeeping this
sits alongside.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.transformers.disease.disease import disease

OBO = 'http://purl.obolibrary.org/obo/'
_IAO_REPLACED_BY = f'{OBO}IAO_0100001'


def _url(short_id: str) -> str:
    return f'{OBO}{short_id}'


def _synonym(pred: str, val: str) -> dict[str, Any]:
    return {'pred': pred, 'synonymType': None, 'val': val, 'xrefs': []}


def _node(
    short_id: str,
    lbl: str,
    *,
    deprecated: bool = False,
    replaced_by: str | None = None,
    exact: list[str] | None = None,
    related: list[str] | None = None,
    narrow: list[str] | None = None,
    broad: list[str] | None = None,
) -> dict[str, Any]:
    synonyms = (
        [_synonym('hasExactSynonym', v) for v in (exact or [])]
        + [_synonym('hasRelatedSynonym', v) for v in (related or [])]
        + [_synonym('hasNarrowSynonym', v) for v in (narrow or [])]
        + [_synonym('hasBroadSynonym', v) for v in (broad or [])]
    )
    basic_property_values = [{'pred': _IAO_REPLACED_BY, 'val': _url(replaced_by)}] if replaced_by else []
    return {
        'id': _url(short_id),
        'lbl': lbl,
        'type': 'CLASS',
        'meta': {
            'basicPropertyValues': basic_property_values,
            'comments': [],
            'definition': {'val': None, 'xrefs': []},
            'deprecated': deprecated,
            'subsets': [],
            'synonyms': synonyms,
            'xrefs': [],
        },
    }


def _run_disease(tmp_path: Path, nodes: list[dict[str, Any]]) -> pl.DataFrame:
    """Run the full disease() transform against a minimal OBO-graph JSON fixture.

    The single edge is unrelated filler: with zero edges the edges frame has no
    columns at all (there is nothing to infer a schema from), which breaks the
    is_a/location joins before the code under test is even reached.

    Fixtures below always give every synonym category (exact/related/narrow/broad)
    at least one value somewhere among the nodes passed in. disease.py's synonym
    pivot only creates a column for a predicate that appears in the data (see
    disease.py:151-156, mirrored for obsolete terms), so a fixture missing a whole
    category anywhere would raise ColumnNotFoundError before reaching the behaviour
    under test -- a pre-existing fragility, not something these tests are about.
    """
    source = {
        'graphs': [
            {
                'id': _url('test.owl'),
                'meta': {'basicPropertyValues': [], 'version': 'test'},
                'logicalDefinitionAxioms': [],
                'domainRangeAxioms': [],
                'nodes': nodes,
                'edges': [
                    {
                        'sub': nodes[0]['id'],
                        'pred': 'unrelated',
                        'obj': nodes[0]['id'],
                        'meta': {'xrefs': [], 'basicPropertyValues': []},
                    }
                ],
            }
        ]
    }
    source_path = tmp_path / 'source.json'
    source_path.write_text(json.dumps(source))
    destination = tmp_path / 'out'

    disease(source=str(source_path), destination=str(destination), settings={}, config=None)  # type: ignore[arg-type]

    return pl.read_parquet(f'{destination}/*.parquet')


def _row(df: pl.DataFrame, short_id: str) -> dict[str, Any]:
    return df.filter(pl.col('id') == short_id).to_dicts()[0]


class TestObsoleteSynonymAbsorption:
    @pytest.fixture
    def result(self, tmp_path: Path) -> pl.DataFrame:
        nodes = [
            _node(
                'MONDO_100',
                'neutropenia',
                exact=['neutropenia'],
                related=['existing related'],
                narrow=['existing narrow'],
                broad=['existing broad'],
            ),
            _node(
                'HP_100',
                'Decreased total neutrophil count',
                deprecated=True,
                replaced_by='MONDO_100',
                exact=['Low neutrophil count'],
                related=['obsolete related'],
                narrow=['obsolete narrow'],
                broad=['obsolete broad'],
            ),
        ]
        return _run_disease(tmp_path, nodes)

    def test_each_category_unions_with_its_own_category(self, result: pl.DataFrame) -> None:
        row = _row(result, 'MONDO_100')
        assert set(row['exactSynonyms']) >= {'neutropenia', 'Low neutrophil count'}
        assert set(row['relatedSynonyms']) == {'existing related', 'obsolete related'}
        assert set(row['narrowSynonyms']) == {'existing narrow', 'obsolete narrow'}
        assert set(row['broadSynonyms']) == {'existing broad', 'obsolete broad'}

    def test_obsolete_synonyms_do_not_cross_into_other_categories(self, result: pl.DataFrame) -> None:
        row = _row(result, 'MONDO_100')
        assert 'obsolete related' not in row['exactSynonyms']
        assert 'obsolete narrow' not in row['relatedSynonyms']
        assert 'obsolete broad' not in row['narrowSynonyms']

    def test_obsolete_name_is_added_as_an_exact_synonym(self, result: pl.DataFrame) -> None:
        assert 'Decreased total neutrophil count' in _row(result, 'MONDO_100')['exactSynonyms']

    def test_a_synonym_shared_by_both_terms_is_not_duplicated(self, tmp_path: Path) -> None:
        nodes = [
            _node(
                'MONDO_100',
                'neutropenia',
                exact=['neutropenia', 'low neutrophil count'],
                related=['x'],
                narrow=['x'],
                broad=['x'],
            ),
            _node(
                'HP_100',
                'Decreased total neutrophil count',
                deprecated=True,
                replaced_by='MONDO_100',
                exact=['low neutrophil count'],
                related=['x'],
                narrow=['x'],
                broad=['x'],
            ),
        ]
        result = _run_disease(tmp_path, nodes)
        row = _row(result, 'MONDO_100')
        assert sorted(row['exactSynonyms']).count('low neutrophil count') == 1

    def test_two_obsolete_terms_replaced_by_the_same_survivor_both_contribute_their_synonyms(
        self, tmp_path: Path
    ) -> None:
        nodes = [
            _node(
                'MONDO_100',
                'neutropenia',
                exact=['neutropenia'],
                related=['x'],
                narrow=['x'],
                broad=['x'],
            ),
            _node(
                'HP_100',
                'Decreased total neutrophil count',
                deprecated=True,
                replaced_by='MONDO_100',
                exact=['Low neutrophil count'],
                related=['x'],
                narrow=['x'],
                broad=['x'],
            ),
            _node(
                'Orphanet_100',
                'Neutropenia disorder',
                deprecated=True,
                replaced_by='MONDO_100',
                exact=['Peripheral neutropenia'],
            ),
        ]
        result = _run_disease(tmp_path, nodes)
        row = _row(result, 'MONDO_100')
        assert set(row['exactSynonyms']) >= {
            'neutropenia',
            'Low neutrophil count',
            'Decreased total neutrophil count',
            'Peripheral neutropenia',
            'Neutropenia disorder',
        }


class TestObsoleteSynonymAbsorptionEdgeCases:
    def test_a_term_with_no_obsolete_terms_keeps_only_its_own_synonyms(self, tmp_path: Path) -> None:
        nodes = [
            _node(
                'MONDO_100',
                'neutropenia',
                exact=['neutropenia'],
                related=['existing related'],
                narrow=['existing narrow'],
                broad=['existing broad'],
            ),
            _node(
                'HP_100',
                'Decreased total neutrophil count',
                deprecated=True,
                replaced_by='MONDO_100',
                exact=['Low neutrophil count'],
                related=['x'],
                narrow=['x'],
                broad=['x'],
            ),
            _node('EFO_200', 'an unrelated disease', exact=['unrelated exact synonym']),
        ]
        result = _run_disease(tmp_path, nodes)
        row = _row(result, 'EFO_200')
        assert row['exactSynonyms'] == ['unrelated exact synonym']
        assert row['relatedSynonyms'] == []
