"""Tests for the mouse phenotype transformer."""

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from pts.pyspark.mouse_phenotype import _create_mp_classification
from pts.transformers.mouse_phenotype import MP_ROOT, mp_classes, normalise_ontology_id, read_mp_classes

OBO = 'http://purl.obolibrary.org/obo/'
COLUMNS = ['modelPhenotypeId', 'modelPhenotypeClassId', 'modelPhenotypeClassLabel']


def _node(i: int | str, label: str | None = '') -> dict:
    node_id = f'{OBO}MP_{i:07d}' if isinstance(i, int) else i
    return {'id': node_id, 'lbl': label}


def _edge(sub: int, obj: int, pred: str = 'is_a') -> dict:
    return {'sub': f'{OBO}MP_{sub:07d}', 'pred': pred, 'obj': f'{OBO}MP_{obj:07d}'}


def _ontology(nodes: list[dict], edges: list[dict]) -> dict:
    return {'graphs': [{'nodes': nodes, 'edges': edges}]}


def _rows(df: pl.DataFrame) -> set[tuple]:
    return set(df.select(COLUMNS).iter_rows())


def _id(i: int) -> str:
    return f'MP:{i:07d}'


@pytest.fixture
def diamond() -> dict:
    # 1 -> {2, 3} -> 4, and 5 under 2 only
    return _ontology(
        [_node(i, f'term {i}') for i in range(1, 6)],
        [_edge(2, 1), _edge(3, 1), _edge(4, 2), _edge(4, 3), _edge(5, 2)],
    )


def test_normalise_ontology_id():
    assert normalise_ontology_id(f'{OBO}MP_0000001') == 'MP:0000001'
    assert normalise_ontology_id(f'{OBO}UBERON_0003102') == 'UBERON_0003102'
    assert normalise_ontology_id('MP:0000001') == 'MP:0000001'
    assert normalise_ontology_id(f'{OBO}MP_1#x') == 'MP:1'
    assert normalise_ontology_id('foo') == 'foo'


def test_term_under_two_classes_gets_both(diamond):
    got = _rows(mp_classes(diamond))
    assert (_id(4), _id(2), 'term 2') in got
    assert (_id(4), _id(3), 'term 3') in got


def test_class_includes_itself(diamond):
    assert (_id(2), _id(2), 'term 2') in _rows(mp_classes(diamond))


def test_root_is_not_a_class(diamond):
    df = mp_classes(diamond)
    assert MP_ROOT not in df['modelPhenotypeClassId'].to_list()
    assert MP_ROOT not in df['modelPhenotypeId'].to_list()


def test_label_keeps_first_and_only_replaces_empty():
    ontology = _ontology(
        [_node(1, 'root'), _node(2, ''), _node(2, 'late'), _node(3, 'first'), _node(3, 'second')],
        [_edge(2, 1), _edge(3, 1)],
    )
    got = _rows(mp_classes(ontology))
    assert (_id(2), _id(2), 'late') in got
    assert (_id(3), _id(3), 'first') in got


def test_label_falls_back_to_id_for_unlabelled_class():
    ontology = _ontology([_node(1, 'root')], [_edge(2, 1)])
    assert _rows(mp_classes(ontology)) == {(_id(2), _id(2), _id(2))}


def test_missing_root_raises():
    with pytest.raises(ValueError, match='MP:0000001'):
        mp_classes(_ontology([_node(2, 'x')], [_edge(2, 1)]))


def test_non_is_a_edges_ignored():
    ontology = _ontology(
        [_node(1, 'root'), _node(2, 'a'), _node(3, 'b')],
        [_edge(2, 1), _edge(3, 2, pred='part_of')],
    )
    assert _rows(mp_classes(ontology)) == {(_id(2), _id(2), 'a')}


def test_output_is_sorted_strings(diamond):
    df = mp_classes(diamond)
    assert df.columns == COLUMNS
    assert all(dtype == pl.String for dtype in df.schema.values())
    assert df.equals(df.sort(COLUMNS))


def test_read_mp_classes(diamond, tmp_path: Path):
    path = tmp_path / 'mp.json'
    path.write_text(json.dumps(diamond))
    assert read_mp_classes(str(path)).equals(mp_classes(diamond))


def test_unlabelled_class_label_is_null():
    ontology = _ontology([_node(1, 'root'), _node(2, None)], [_edge(2, 1)])
    assert _rows(mp_classes(ontology)) == {(_id(2), _id(2), None)}


def test_mp_classes_match_spark(spark, diamond):
    graph = diamond['graphs'][0]
    short = {'pred': 'is_a', 'obj': 'MP:0000001'}
    ontology = _ontology(
        graph['nodes']
        + [
            _node(f'{OBO}UBERON_1', 'other'),
            _node(6, None),
            _node(6, 'six'),
            _node(7, None),
            _node(8, 'eight'),
            {'id': f'{OBO}MP_0000009'},
            _node('MP:0000010', 'ten'),
        ],
        graph['edges']
        + [
            _edge(6, 5),
            _edge(6, 1),
            {'sub': f'{OBO}UBERON_1', 'pred': 'is_a', 'obj': f'{OBO}MP_0000003'},
            # root and a non-root term written both as short ids and as IRIs
            {**short, 'sub': 'MP:0000007'},
            {**short, 'sub': f'{OBO}MP_0000009'},
            {**short, 'sub': 'MP:0000011'},
            {'sub': 'MP:0000008', 'pred': 'is_a', 'obj': f'{OBO}MP_0000001'},
            {'sub': 'MP:0000010', 'pred': 'is_a', 'obj': f'{OBO}MP_0000002'},
            {'sub': f'{OBO}MP_0000010', 'pred': 'is_a', 'obj': 'MP:0000003'},
            {'sub': f'{OBO}MP_0000012', 'pred': 'is_a', 'obj': 'MP:0000010'},
        ],
    )
    old = _create_mp_classification(ontology, SimpleNamespace(spark=spark))
    expected = sorted(tuple(r) for r in old.collect())
    got = sorted(mp_classes(ontology).iter_rows())
    assert got == expected
    assert (_id(7), _id(7), None) in got
    assert (_id(9), _id(9), '') in got
