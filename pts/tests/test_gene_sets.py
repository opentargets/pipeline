"""Tests for the gene_sets transformer."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from otter.config.model import Config

from pts.schemas.gene_sets import gene_sets_schema
from pts.transformers import gene_sets as gene_sets_module
from pts.transformers.gene_sets import gene_sets
from pts.transformers.utils.dataset import scan_dataset, write_dataset

TARGET = pl.DataFrame(
    {
        'id': ['T1', 'T2', 'T3', 'T4'],
        'go': [[{'id': 'GO:3'}], [{'id': 'GO:2'}], [{'id': 'GO:9'}], [{'id': 'GO:5'}]],
        'pathways': [[{'pathwayId': 'R4'}], [], None, []],
        'targetClass': [
            [{'id': 2, 'label': 'kinase', 'level': 'l2'}, {'id': 2, 'label': 'enzyme', 'level': 'l1'}],
            [],
            [],
            [],
        ],
    },
    schema_overrides={'pathways': pl.List(pl.Struct({'pathwayId': pl.String}))},
)

GO = pl.DataFrame({
    'id': ['GO:1', 'GO:2', 'GO:3', 'GO:5', 'GO:9'],
    'label': ['root', 'parent', 'child', 'function', 'obsolete'],
    'namespace': ['biological_process'] * 3 + ['molecular_function', 'biological_process'],
    'isA': [[], ['GO:1'], [], [], []],
    'partOf': [[], [], ['GO:2'], ['GO:3'], []],
    'isObsolete': [False, False, False, False, True],
    # as computed by the go step: GO:5's part_of GO:3 crosses namespaces, so it is not an ancestor
    'ancestors': [[], ['GO:1'], ['GO:1', 'GO:2'], [], []],
})

REACTOME = pl.DataFrame({
    'id': ['R1', 'R2', 'R3', 'R4'],
    'label': ['top', 'left', 'right', 'bottom'],
    'parents': [[], ['R1'], ['R1'], ['R2', 'R3']],
    'ancestors': [[], ['R1'], ['R1'], ['R1', 'R2', 'R3']],
})

CLASSES = pl.DataFrame({
    'protein_class_id': [0, 1, 2],
    'parent_id': [None, 0, 1],
    'pref_name': ['Protein', 'enzyme', 'kinase'],
    'class_level': [0, 1, 2],
})


#: the arguments the step passed to read_dump_tables
DUMP_CALLS: list[tuple[tuple, dict]] = []


def _read_dump_tables(*args: object, **kwargs: object) -> dict[str, pl.DataFrame]:
    DUMP_CALLS.append((args, kwargs))
    return {'protein_classification': CLASSES}


@pytest.fixture(scope='module')
def output(tmp_path_factory: pytest.TempPathFactory) -> pl.DataFrame:
    path = tmp_path_factory.mktemp('gene_sets')
    source = {'chembl': str(path / 'chembl.tar.gz')}
    for name, df in {'targets': TARGET, 'go': GO, 'reactome': REACTOME}.items():
        source[name] = str(path / name)
        write_dataset(df, source[name])
    config = Config(step='gene_sets', steps=['gene_sets'], work_path=path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gene_sets_module, 'read_dump_tables', _read_dump_tables)
        gene_sets(source, str(path / 'output'), {}, config)
    return scan_dataset(str(path / 'output')).collect()


@pytest.fixture(scope='module')
def result(output: pl.DataFrame) -> dict[tuple[str, str], dict]:
    return {(row['source'], row['id']): row for row in output.iter_rows(named=True)}


def test_protein_classification_is_restored_from_the_chembl_dump(output: pl.DataFrame) -> None:
    [(args, kwargs)] = DUMP_CALLS
    assert args[0].endswith('chembl.tar.gz')
    assert args[1] == {'protein_classification': ['protein_class_id', 'parent_id', 'pref_name', 'class_level']}
    assert kwargs['schema_name'] == 'public'
    assert str(kwargs['scratch_root']) == str(Path(args[0]).parent)


def test_output_matches_the_schema(output: pl.DataFrame) -> None:
    assert output.schema == pl.Schema(gene_sets_schema)


def test_go_targets_reach_every_ancestor_including_the_root(result: dict) -> None:
    assert result['go', 'GO:3']['entityIds'] == ['T1']
    assert result['go', 'GO:2']['entityIds'] == ['T1', 'T2']
    assert result['go', 'GO:1']['entityIds'] == ['T1', 'T2']


def test_go_category_and_parents_come_from_the_term(result: dict) -> None:
    assert result['go', 'GO:3']['category'] == 'GO:BP'
    assert result['go', 'GO:3']['parentIds'] == ['GO:2']


def test_go_parents_from_another_namespace_are_dropped(result: dict) -> None:
    assert result['go', 'GO:5']['category'] == 'GO:MF'
    assert result['go', 'GO:5']['parentIds'] == []
    assert result['go', 'GO:5']['entityIds'] == ['T4']
    assert 'T4' not in result['go', 'GO:3']['entityIds']


def test_obsolete_go_terms_are_dropped(result: dict) -> None:
    assert ('go', 'GO:9') not in result


def test_reactome_diamond_counts_each_target_once(result: dict) -> None:
    assert result['reactome', 'R1']['entityIds'] == ['T1']
    assert result['reactome', 'R2']['entityIds'] == ['T1']
    assert result['reactome', 'R4']['parentIds'] == ['R2', 'R3']


def test_chembl_classes_use_ids_and_drop_the_root(result: dict) -> None:
    assert result['chembl_target_class', '2']['label'] == 'kinase'
    assert result['chembl_target_class', '2']['parentIds'] == ['1']
    assert result['chembl_target_class', '1']['entityIds'] == ['T1']
    assert result['chembl_target_class', '1']['parentIds'] == []
    assert ('chembl_target_class', '0') not in result


def test_rows_are_sorted_by_source_and_id(result: dict) -> None:
    assert list(result) == sorted(result)
