"""Tests for the disease_location transformer."""

from __future__ import annotations

import io
import json
from pathlib import Path

import polars as pl

from pts.schemas import ontology
from pts.schemas.disease_location import disease_location_schema
from pts.transformers.disease_location import (
    disease_location,
    inherit_locations,
    most_specific_locations,
    stated_locations,
)
from pts.transformers.utils.dataset import scan_dataset, write_dataset

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


def _stated(*pairs: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(list(pairs), schema=['diseaseId', 'biosampleId'], orient='row')


def _diseases(*rows: tuple[str, list[str] | None]) -> pl.DataFrame:
    return pl.DataFrame(
        list(rows),
        schema={'id': pl.String(), 'ancestors': pl.List(pl.String())},
        orient='row',
    )


class TestInheritLocations:
    """A disease takes the stated locations of itself and of every ancestor."""

    def test_disease_gets_its_own_locations(self) -> None:
        out = inherit_locations(_stated(('MONDO_A', 'UBERON_1')), _diseases(('MONDO_A', [])))
        assert _pairs(out) == [('MONDO_A', 'UBERON_1')]

    def test_disease_gets_ancestor_locations(self) -> None:
        out = inherit_locations(
            _stated(('MONDO_A', 'UBERON_1')),
            _diseases(('MONDO_A', []), ('ORPHA_1', ['MONDO_A'])),
        )
        assert ('ORPHA_1', 'UBERON_1') in _pairs(out)

    def test_unrelated_disease_gets_nothing(self) -> None:
        out = inherit_locations(
            _stated(('MONDO_A', 'UBERON_1')),
            _diseases(('MONDO_A', []), ('MONDO_Z', [])),
        )
        assert [pair for pair in _pairs(out) if pair[0] == 'MONDO_Z'] == []

    def test_stated_term_absent_from_index_reaches_no_one(self) -> None:
        out = inherit_locations(_stated(('MONDO_GONE', 'UBERON_1')), _diseases(('MONDO_A', ['MONDO_B'])))
        assert out.is_empty()

    def test_null_ancestors_keeps_own_locations(self) -> None:
        out = inherit_locations(_stated(('MONDO_A', 'UBERON_1')), _diseases(('MONDO_A', None)))
        assert _pairs(out) == [('MONDO_A', 'UBERON_1')]

    def test_location_from_two_ancestors_appears_once(self) -> None:
        out = inherit_locations(
            _stated(('MONDO_A', 'UBERON_1'), ('MONDO_B', 'UBERON_1')),
            _diseases(('MONDO_A', []), ('MONDO_B', []), ('MONDO_C', ['MONDO_A', 'MONDO_B'])),
        )
        assert [pair for pair in _pairs(out) if pair[0] == 'MONDO_C'] == [('MONDO_C', 'UBERON_1')]


COLON, LARGE_INTESTINE, INTESTINE, IMMUNE = 'UBERON_0001155', 'UBERON_0000059', 'UBERON_0000160', 'UBERON_0002405'


def _biosample() -> pl.DataFrame:
    return pl.DataFrame(
        [
            (COLON, [LARGE_INTESTINE, INTESTINE]),
            (LARGE_INTESTINE, [INTESTINE]),
            (INTESTINE, []),
            (IMMUNE, None),
        ],
        schema={'biosampleId': pl.String(), 'ancestors': pl.List(pl.String())},
        orient='row',
    )


def _locations_on(disease: str, *locations: str) -> pl.DataFrame:
    return _stated(*((disease, location) for location in locations))


class TestMostSpecificLocations:
    """Only locations the biosample index knows survive, and only the most specific of them."""

    def test_location_missing_from_biosample_is_dropped(self) -> None:
        assert most_specific_locations(_locations_on('D1', 'GO_0000781'), _biosample()).is_empty()

    def test_broader_location_is_dropped(self) -> None:
        out = most_specific_locations(_locations_on('D1', COLON, LARGE_INTESTINE, INTESTINE), _biosample())
        assert _pairs(out) == [('D1', COLON)]

    def test_unrelated_locations_are_both_kept(self) -> None:
        out = most_specific_locations(_locations_on('D1', COLON, IMMUNE), _biosample())
        assert _pairs(out) == [('D1', COLON), ('D1', IMMUNE)]

    def test_pruning_is_per_disease(self) -> None:
        pairs = pl.concat([_locations_on('D1', COLON), _locations_on('D2', INTESTINE)])
        assert _pairs(most_specific_locations(pairs, _biosample())) == [('D1', COLON), ('D2', INTESTINE)]

    def test_location_with_null_ancestors_is_kept(self) -> None:
        assert _pairs(most_specific_locations(_locations_on('D1', IMMUNE), _biosample())) == [('D1', IMMUNE)]

    def test_uc_shape(self) -> None:
        pairs = _locations_on('UC', COLON, LARGE_INTESTINE, INTESTINE, IMMUNE)
        assert _pairs(most_specific_locations(pairs, _biosample())) == [('UC', COLON), ('UC', IMMUNE)]


def _write_mondo(path: Path) -> None:
    doc = {
        'graphs': [
            {
                'id': 'mondo',
                'meta': None,
                'nodes': [],
                'edges': [{'sub': f'{OBO}MONDO_A', 'pred': LOCATED_IN, 'obj': f'{OBO}{COLON}', 'meta': None}],
                'logicalDefinitionAxioms': [
                    {
                        'definedClassId': f'{OBO}MONDO_B',
                        'genusIds': [],
                        'restrictions': [
                            {'propertyId': LOCATED_IN, 'fillerId': f'{OBO}{IMMUNE}'},
                            {'propertyId': LOCATED_IN, 'fillerId': f'{OBO}GO_0000781'},
                        ],
                    },
                ],
                'domainRangeAxioms': [],
            },
        ],
    }
    path.write_text(json.dumps(doc))


def test_disease_location_end_to_end(tmp_path: Path) -> None:
    _write_mondo(tmp_path / 'mondo.json')
    write_dataset(
        _diseases(
            ('MONDO_A', []),
            ('MONDO_B', []),
            ('MONDO_C', ['MONDO_A', 'MONDO_B']),
            ('ORPHA_1', ['MONDO_C', 'MONDO_A', 'MONDO_B']),
            ('MONDO_Z', []),
        ),
        str(tmp_path / 'disease'),
    )
    write_dataset(_biosample().filter(pl.col('biosampleId') != LARGE_INTESTINE), str(tmp_path / 'biosample'))
    (tmp_path / 'biosample' / '_SUCCESS').touch()

    disease_location(
        {
            'mondo': str(tmp_path / 'mondo.json'),
            'disease': str(tmp_path / 'disease'),
            'biosample': str(tmp_path / 'biosample'),
        },
        str(tmp_path / 'out'),
        {},
        None,
    )

    out = scan_dataset(str(tmp_path / 'out')).collect()
    assert out.schema == disease_location_schema
    assert out.rows() == sorted(out.rows())
    assert out.rows() == [
        ('MONDO_A', COLON),
        ('MONDO_B', IMMUNE),
        ('MONDO_C', COLON),
        ('MONDO_C', IMMUNE),
        ('ORPHA_1', COLON),
        ('ORPHA_1', IMMUNE),
    ]
