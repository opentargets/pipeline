"""End-to-end tests for how disease() tracks obsolete terms.

Covers the pre-existing obsoleteTerms bookkeeping (an obsolete term's short id is
listed on its replacement, and the obsolete term itself is dropped from the index),
run through the full transform to confirm it still holds once obsolete terms also
contribute synonyms (see test_disease_obsolete_synonym_absorption.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

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


# disease.py's synonym pivot only creates a column for a predicate that appears
# somewhere in the data (see disease.py:151-156, mirrored for obsolete terms), so
# every fixture below covers all 4 categories on both the survivor and each
# obsolete node to avoid a spurious ColumnNotFoundError -- a pre-existing
# fragility unrelated to what these tests check.
def _survivor() -> dict[str, Any]:
    return _node(
        'MONDO_100',
        'neutropenia',
        exact=['neutropenia'],
        related=['x'],
        narrow=['x'],
        broad=['x'],
    )


def _obsolete() -> dict[str, Any]:
    return _node(
        'HP_100',
        'Decreased total neutrophil count',
        deprecated=True,
        replaced_by='MONDO_100',
        exact=['Low neutrophil count'],
        related=['x'],
        narrow=['x'],
        broad=['x'],
    )


class TestObsoleteTerms:
    def test_obsolete_terms_field_lists_the_obsolete_id(self, tmp_path: Path) -> None:
        nodes = [_survivor(), _obsolete()]
        result = _run_disease(tmp_path, nodes)
        assert _row(result, 'MONDO_100')['obsoleteTerms'] == ['HP_100']

    def test_obsolete_term_itself_is_absent_from_the_index(self, tmp_path: Path) -> None:
        nodes = [_survivor(), _obsolete()]
        result = _run_disease(tmp_path, nodes)
        assert result.filter(pl.col('id') == 'HP_100').height == 0

    def test_two_obsolete_terms_pointing_at_the_same_survivor_are_both_listed(self, tmp_path: Path) -> None:
        nodes = [
            _survivor(),
            _obsolete(),
            _node(
                'Orphanet_100',
                'Neutropenia disorder',
                deprecated=True,
                replaced_by='MONDO_100',
                exact=['Peripheral neutropenia'],
            ),
        ]
        result = _run_disease(tmp_path, nodes)
        assert set(_row(result, 'MONDO_100')['obsoleteTerms']) == {'HP_100', 'Orphanet_100'}

    def test_a_term_with_no_obsolete_terms_has_an_empty_list(self, tmp_path: Path) -> None:
        nodes = [
            _survivor(),
            _obsolete(),
            _node('EFO_200', 'an unrelated disease', exact=['unrelated exact synonym']),
        ]
        result = _run_disease(tmp_path, nodes)
        assert _row(result, 'EFO_200')['obsoleteTerms'] == []
