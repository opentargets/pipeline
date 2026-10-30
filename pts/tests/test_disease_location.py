"""Tests for the disease_location transformer."""

from __future__ import annotations

import io
import json

import polars as pl

from pts.schemas import ontology
from pts.transformers.disease_location import stated_locations

OBO = 'http://purl.obolibrary.org/obo/'
LOCATED_IN = f'{OBO}RO_0004026'

_AXIOM_SCHEMA = pl.Schema({
    'definedClassId': pl.String(),
    'genusIds': pl.List(pl.String()),
    'restrictions': pl.List(pl.Struct({'propertyId': pl.String(), 'fillerId': pl.String()})),
})


def _edge(sub: str, pred: str, obj: str) -> dict:
    return {'sub': f'{OBO}{sub}', 'pred': pred, 'obj': f'{OBO}{obj}', 'meta': None}


def _edges(*edges: dict) -> pl.DataFrame:
    return pl.from_dicts(list(edges), schema=ontology.edge)


def _axiom(defined: str, restrictions: list[tuple[str, str]] | None) -> dict:
    return {
        'definedClassId': f'{OBO}{defined}',
        'genusIds': [],
        'restrictions': None
        if restrictions is None
        else [{'propertyId': prop, 'fillerId': filler} for prop, filler in restrictions],
    }


def _axioms(*axioms: dict) -> pl.DataFrame:
    return pl.from_dicts(list(axioms), schema=_AXIOM_SCHEMA)


def _pairs(df: pl.DataFrame) -> list[tuple[str, str]]:
    return sorted(df.select('diseaseId', 'biosampleId').rows())


class TestStatedLocations:
    """MONDO's disease-has-location assertions become short-id pairs."""

    def test_edge_with_location_predicate_is_stated(self) -> None:
        out = stated_locations(_edges(_edge('MONDO_1', LOCATED_IN, 'UBERON_1')), _axioms())
        assert _pairs(out) == [('MONDO_1', 'UBERON_1')]

    def test_other_edge_predicates_are_ignored(self) -> None:
        edges = _edges(
            _edge('MONDO_1', 'is_a', 'MONDO_2'),
            _edge('MONDO_1', f'{OBO}BFO_0000050', 'UBERON_1'),
        )
        assert stated_locations(edges, _axioms()).is_empty()

    def test_axiom_restriction_is_stated(self) -> None:
        axioms = _axioms(
            _axiom('MONDO_2', [(LOCATED_IN, f'{OBO}UBERON_2'), (f'{OBO}RO_0000053', f'{OBO}MONDO_9')]),
        )
        assert _pairs(stated_locations(_edges(), axioms)) == [('MONDO_2', 'UBERON_2')]

    def test_axiom_with_null_restrictions_is_tolerated(self) -> None:
        assert stated_locations(_edges(), _axioms(_axiom('MONDO_2', None))).is_empty()

    def test_edge_and_axiom_duplicates_collapse(self) -> None:
        out = stated_locations(
            _edges(_edge('MONDO_1', LOCATED_IN, 'UBERON_1')),
            _axioms(_axiom('MONDO_1', [(LOCATED_IN, f'{OBO}UBERON_1')])),
        )
        assert _pairs(out) == [('MONDO_1', 'UBERON_1')]

    def test_non_obo_iri_is_shortened(self) -> None:
        axioms = _axioms(_axiom('MONDO_3', [(LOCATED_IN, 'http://www.ebi.ac.uk/efo/EFO_0000001')]))
        assert _pairs(stated_locations(_edges(), axioms)) == [('MONDO_3', 'EFO_0000001')]

    def test_ontology_schema_keeps_restrictions(self) -> None:
        doc = {
            'graphs': [
                {
                    'id': 'mondo',
                    'meta': None,
                    'nodes': [],
                    'edges': [],
                    'logicalDefinitionAxioms': [
                        {
                            'definedClassId': f'{OBO}MONDO_4',
                            'genusIds': [],
                            'restrictions': [{'propertyId': LOCATED_IN, 'fillerId': f'{OBO}UBERON_4'}],
                        },
                    ],
                    'domainRangeAxioms': [],
                },
            ],
        }
        df = pl.read_json(io.BytesIO(json.dumps(doc).encode()), schema=ontology.schema)
        axiom = df['graphs'][0][0]['logicalDefinitionAxioms'][0]
        assert axiom['restrictions'][0]['fillerId'] == f'{OBO}UBERON_4'
