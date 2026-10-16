"""The ChEMBL protein classification builder."""

from __future__ import annotations

import polars as pl
import pytest
from loguru import logger

from pts.pyspark.target import _build_protein_classification, _flatten_protein_classification
from pts.target.chembl_class import TABLES, build_protein_classification, flatten_protein_classification
from tests.target.parity import rows

SCHEMAS = {
    'target_dictionary': 'tid int, chembl_id string, pref_name string, target_type string',
    'target_components': 'targcomp_id int, tid int, component_id int',
    'component_sequences': 'component_id int, accession string',
    'component_class': 'comp_class_id int, component_id int, protein_class_id int',
    'protein_classification': 'protein_class_id int, parent_id int, pref_name string, class_level int',
}
DTYPES = {
    'target_dictionary': {'tid': pl.Int32, 'chembl_id': pl.String, 'pref_name': pl.String, 'target_type': pl.String},
    'target_components': {'targcomp_id': pl.Int32, 'tid': pl.Int32, 'component_id': pl.Int32},
    'component_sequences': {'component_id': pl.Int32, 'accession': pl.String},
    'component_class': {'comp_class_id': pl.Int32, 'component_id': pl.Int32, 'protein_class_id': pl.Int32},
    'protein_classification': {
        'protein_class_id': pl.Int32,
        'parent_id': pl.Int32,
        'pref_name': pl.String,
        'class_level': pl.Int32,
    },
}
# A full six-level chain plus two more level-1 classes. Component 1 carries three classes listed
# so the lowest id (6) is not the first row; components 2 and 3 belong only to the two-component
# target 102; component 4 has no accession; component 1 sits under two single-component targets.
DATA = {
    'protein_classification': [
        (1, None, 'Enzyme', 1),
        (2, 1, 'Kinase', 2),
        (3, 2, 'Protein Kinase', 3),
        (4, 3, 'TK', 4),
        (5, 4, 'TK group', 5),
        (6, 5, 'TK family', 6),
        (20, None, 'Transporter', 1),
        (21, None, 'Ion channel', 1),
    ],
    'component_class': [(1, 1, 20), (2, 1, 6), (3, 2, 20), (4, 3, 6), (5, 1, 21), (6, 4, 20)],
    'component_sequences': [(1, 'P00001'), (2, 'P00002'), (3, 'P00003'), (4, None)],
    'target_components': [(1, 100, 1), (2, 101, 1), (3, 102, 2), (4, 102, 3), (5, 103, 4)],
    'target_dictionary': [
        (100, 'CHEMBL_T1', 'A', 'SINGLE PROTEIN'),
        (101, 'CHEMBL_T2', 'B', 'SINGLE PROTEIN'),
        (102, 'CHEMBL_T3', 'C', 'PROTEIN COMPLEX'),
        (103, 'CHEMBL_T4', 'D', 'SINGLE PROTEIN'),
    ],
}


@pytest.fixture
def chembl() -> dict[str, pl.DataFrame]:
    return {name: pl.DataFrame(DATA[name], schema=DTYPES[name], orient='row') for name in TABLES}


def _flat(chembl: dict[str, pl.DataFrame]) -> dict:
    return {r['leaf_id']: r for r in flatten_protein_classification(chembl['protein_classification']).to_dicts()}


def _by_accession(chembl: dict[str, pl.DataFrame]) -> dict:
    return {r['accession']: r['targetClass'] for r in build_protein_classification(**chembl).to_dicts()}


def test_full_six_level_chain_is_flattened(chembl) -> None:
    deep = _flat(chembl)[6]
    assert [deep[f'l{i}'] for i in range(1, 7)] == ['Enzyme', 'Kinase', 'Protein Kinase', 'TK', 'TK group', 'TK family']


def test_labels_land_at_their_own_level_not_their_depth(chembl) -> None:
    flat = _flat(chembl)
    assert flat[20]['l1'] == 'Transporter'
    assert flat[20]['l2'] is None
    assert flat[20]['l6'] is None


def test_only_the_lowest_class_of_a_component_survives(chembl) -> None:
    assert {c['id'] for c in _by_accession(chembl)['P00001']} == {6}


def test_every_ancestor_becomes_its_own_class_entry(chembl) -> None:
    assert _by_accession(chembl)['P00001'] == [
        {'id': 6, 'label': 'Enzyme', 'level': 'l1'},
        {'id': 6, 'label': 'Kinase', 'level': 'l2'},
        {'id': 6, 'label': 'Protein Kinase', 'level': 'l3'},
        {'id': 6, 'label': 'TK', 'level': 'l4'},
        {'id': 6, 'label': 'TK group', 'level': 'l5'},
        {'id': 6, 'label': 'TK family', 'level': 'l6'},
    ]


def test_multi_component_targets_and_null_accessions_contribute_nothing(chembl) -> None:
    assert set(_by_accession(chembl)) == {'P00001'}


def test_class_id_is_long(chembl) -> None:
    assert build_protein_classification(**chembl).schema['targetClass'] == pl.List(
        pl.Struct({'id': pl.Int64, 'label': pl.String, 'level': pl.String})
    )


def test_class_level_above_max_triggers_a_warning() -> None:
    classes = pl.DataFrame(
        [(1, None, 'Enzyme', 1), (2, 1, 'Kinase', 7)], schema=DTYPES['protein_classification'], orient='row'
    )
    messages: list[str] = []
    sink_id = logger.add(messages.append, level='WARNING')
    try:
        flatten_protein_classification(classes)
    finally:
        logger.remove(sink_id)
    assert any('7' in m for m in messages)


def test_class_level_at_max_does_not_warn(chembl) -> None:
    messages: list[str] = []
    sink_id = logger.add(messages.append, level='WARNING')
    try:
        flatten_protein_classification(chembl['protein_classification'])
    finally:
        logger.remove(sink_id)
    assert messages == []


def test_matches_spark(spark, chembl) -> None:
    frames = {name: spark.createDataFrame(DATA[name], SCHEMAS[name]) for name in TABLES}
    assert rows(flatten_protein_classification(chembl['protein_classification']), 'leaf_id') == rows(
        _flatten_protein_classification(frames['protein_classification']), 'leaf_id'
    )
    assert rows(build_protein_classification(**chembl), 'accession', ('targetClass',)) == rows(
        _build_protein_classification(**frames), 'accession', ('targetClass',)
    )
