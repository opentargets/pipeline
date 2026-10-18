"""Tests for the mouse phenotype transformer."""

import json
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.mouse_phenotype import MOUSE_PHENOTYPE
from pts.transformers.mouse_phenotype import (
    GROUP_KEYS,
    MP_ROOT,
    mouse_phenotype,
    mp_classes,
    normalise_ontology_id,
    read_mp_classes,
    split_by_target,
)
from tests.mouse_phenotype_spark_rows import CLASSES, EXCLUDED, OUTPUT

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


def test_mp_classes_match_spark_golden(diamond):
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
    got = sorted(mp_classes(ontology).iter_rows())
    assert got == CLASSES
    assert (_id(7), _id(7), None) in got
    assert (_id(9), _id(9), '') in got


SUMMARY_HEADER = [
    'model_id',
    'model_genetic_background',
    'model_description',
    'disease_id',
    'disease_term',
    'disease_model_avg_norm',
    'disease_model_max_norm',
    'marker_id',
]
GENE_MODEL_HEADER = [
    '1. MGI accession id',
    '2. marker type',
    '3. marker symbol',
    '4. marker name',
    '5. genome build',
    '6. Entrez gene id',
    '7. NCBI gene chromosome',
    '8. NCBI gene start',
    '9. NCBI gene end',
    '10. NCBI gene strand',
    '11. Ensembl gene id',
    '12. Ensembl gene chromosome',
    '13. Ensembl gene start',
    '14. Ensembl gene end',
    '15. Ensembl gene strand',
]


def _summary(model: str, marker: str, description: str = 'A<tm1>/A<tm1>', background: str = 'involves: 129') -> list:
    return [model, background, description, 'OMIM:1', 'disease', '64.65', '60.0', marker]


def _gene_model(mgi: str, symbol: str, ensembl: str | None) -> list:
    return [mgi, 'Gene', symbol, 'name', 'GRCm39', '1', '6', '1', '2', '+', ensembl, '6', '1', '2', '+', None]


def _pubmed(mp: str, pmids: str | None, markers: str | None) -> list:
    return ['A<tm1>/A<tm1>', 'A<tm1>', 'involves: 129', mp, pmids, markers, 'MGI:9']


# 10 and 20 are the classes; 2 sits under both
MP_JSON = _ontology(
    [_node(i, f'term {i}') for i in [1, 2, 3, 4, 10, 20]],
    [_edge(10, 1), _edge(20, 1), _edge(2, 10), _edge(2, 20), _edge(3, 10), _edge(4, 20)],
)

TABLES: dict[str, tuple[list[str] | None, list[list]]] = {
    'solr_disease_model_summary': (
        SUMMARY_HEADER,
        [
            _summary('MGI:1#hom#early', 'MGI:101'),
            # the same model and gene once the disease columns are dropped
            [*_summary('MGI:1#hom#early', 'MGI:101')[:3], 'OMIM:2', 'other', '50', '50', 'MGI:101'],
            _summary('MGI:1#hom#early', 'MGI:102', 'B<tm1>/B<tm1>'),
            _summary('MGI:2#het', 'MGI:101', 'A<tm2>/+', 'C57BL/6'),
            # non-MGI ids clean to null and their structs collapse
            _summary('IMPC-CURATE-1#hom', 'MGI:102', 'B<em1>/B<em1>', 'C57BL/6N'),
            _summary('IMPC-CURATE-2', 'MGI:102', 'B<em1>/B<em1>', 'C57BL/6N'),
            _summary('MGI:3', 'MGI:101'),
            _summary('MGI:4', 'MGI:101'),
            _summary('MGI:5', 'MGI:102'),
            _summary('MGI:6', 'MGI:103'),
            _summary('MGI:7', 'MGI:104'),
            _summary('MGI:8', 'MGI:999'),
        ],
    ),
    'solr_mouse_model': (
        ['model_id', 'model_phenotypes'],
        [
            ['MGI:1#hom#early', 'MP:0000002 two,MP:0000003 three'],
            ['MGI:2#het', 'MP:0000002 two'],
            ['IMPC-CURATE-1#hom', 'MP:0000002 two'],
            ['IMPC-CURATE-2', 'MP:0000002 two'],
            ['MGI:3', None],
            ['MGI:4', 'MP:\u0660\u0661 arabic digits,no id'],
            ['MGI:5', 'MP:0000099 unclassified,MP:0000098 no term'],
            ['MGI:6', 'MP:0000002 two,MP:0000004 four'],
            ['MGI:7', 'MP:0000004 four'],
            ['MGI:8', 'MP:0000002 two'],
        ],
    ),
    'solr_ontology': (
        ['ontology', 'phenotype_id', 'phenotype_term'],
        [
            ['MP', 'MP:0000002', 'two'],
            ['MP', 'MP:0000003', 'three'],
            ['MP', 'MP:0000004', 'four'],
            ['MP', 'MP:0000004', 'four again'],
            ['MP', 'MP:0000099', 'unclassified'],
            ['HP', 'HP:0000001', 'hp one'],
        ],
    ),
    'solr_gene_gene': (
        ['gene_id', 'hgnc_gene_id'],
        [*([f'MGI:10{i}', f'HGNC:{i}'] for i in range(1, 5)), ['MGI:104', 'HGNC:5']],
    ),
    'hgnc_gene_mappings': (
        ['hgnc_id', 'symbol', 'ensembl_gene_id'],
        [[f'HGNC:{i}', f'S{i}', f'ENSG{i}'] for i in range(1, 6)],
    ),
    'mouse_gene_mappings': (
        GENE_MODEL_HEADER,
        [
            _gene_model('MGI:101', 'A', 'ENSMUSG1'),
            _gene_model('MGI:102', 'B', 'ENSMUSG2'),
            _gene_model('MGI:103', 'C', 'ENSMUSG3'),
            _gene_model('MGI:104', 'D', 'ENSMUSG4'),
            _gene_model('MGI:105', 'F', None),
        ],
    ),
    'mouse_pubmed_refs': (
        None,
        [
            _pubmed('MP:0000002', '222|111', 'MGI:101'),
            _pubmed('MP:0000003', '333', 'MGI:101|MGI:102'),
            _pubmed('MP:0000004', '444', 'MGI:104'),
        ],
    ),
}
TAB_SEPARATED = {'hgnc_gene_mappings', 'mouse_gene_mappings', 'mouse_pubmed_refs'}
TARGETS = ['ENSG1', 'ENSG2', 'ENSG4', 'ENSG5']


def _render(value: str | None, sep: str) -> str:
    if value is None:
        return ''
    if sep in value or '"' in value:
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return value


def _sources(root: Path, targets: list[str] = TARGETS, mp: dict = MP_JSON, **tables: list[list]) -> dict[str, str]:
    root.mkdir(parents=True)
    source = {}
    for key, (header, rows) in TABLES.items():
        sep = '\t' if key in TAB_SEPARATED else ','
        lines = [] if header is None else [sep.join(header)]
        lines += [sep.join(_render(v, sep) for v in row) for row in tables.get(key, rows)]
        (root / key).write_text('\n'.join(lines) + '\n')
        source[key] = str(root / key)
    (root / 'mp.json').write_text(json.dumps(mp))
    (root / 'target').mkdir()
    pl.DataFrame({'id': targets}, schema={'id': pl.String}).write_parquet(root / 'target' / 'part.parquet')
    return source | {'mp_ontology': str(root / 'mp.json'), 'target': str(root / 'target')}


def _destination(root: Path) -> dict[str, str]:
    return {'output': str(root / 'output'), 'excluded': str(root / 'excluded')}


def _run(tmp_path: Path, **kwargs) -> tuple[pl.DataFrame, pl.DataFrame]:
    destination = _destination(tmp_path / 'out')
    mouse_phenotype(_sources(tmp_path / 'src', **kwargs), destination, {}, None)
    return pl.read_parquet(destination['output']), pl.read_parquet(destination['excluded'])


def _key(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _normalise(row: dict) -> dict:
    models = [
        {**m, 'literature': None if m['literature'] is None else sorted(m['literature'])}
        for m in row['biologicalModels']
    ]
    return {
        **row,
        'biologicalModels': sorted(models, key=_key),
        'modelPhenotypeClasses': sorted(row['modelPhenotypeClasses'], key=_key),
    }


def _polars_rows(df: pl.DataFrame) -> tuple[list[str], list[dict]]:
    return df.columns, sorted((_normalise(r) for r in df.to_dicts()), key=_key)


def test_step_matches_spark_golden(tmp_path: Path):
    destination = _destination(tmp_path / 'out')
    mouse_phenotype(_sources(tmp_path / 'src'), destination, {}, None)
    output = pl.read_parquet(destination['output'])
    excluded = pl.read_parquet(destination['excluded'])

    assert _polars_rows(output) == OUTPUT
    assert _polars_rows(excluded) == EXCLUDED
    # the fixture reaches every case it is built for
    assert {m['id'] for models in output['biologicalModels'] for m in models} == {'MGI:1', 'MGI:2', 'MGI:7', None}
    assert set(output['modelPhenotypeLabel']) == {'two', 'three', 'four', 'four again'}
    assert sorted(output.filter(pl.col('targetInModelMgiId') == 'MGI:104')['targetFromSourceId']) == [
        'ENSG4',
        'ENSG4',
        'ENSG5',
        'ENSG5',
    ]
    assert set(excluded['targetFromSourceId']) == {'ENSG3'}


def test_lists_sorted(tmp_path: Path):
    output, _ = _run(tmp_path)
    for column in ['biologicalModels', 'modelPhenotypeClasses']:
        assert output[column].to_list() == output[column].list.sort().to_list(), column
    literature = [m['literature'] for models in output['biologicalModels'] for m in models]
    assert any(len(refs or []) > 1 for refs in literature)
    assert all(refs is None or refs == sorted(refs) for refs in literature)


def test_rows_sorted(tmp_path: Path):
    output, excluded = _run(tmp_path)
    for df in [output, excluded]:
        assert df.height > 1
        assert df.equals(df.sort(GROUP_KEYS))


def test_schema_is_published(tmp_path: Path):
    output, excluded = _run(tmp_path)
    published = pl.Schema({
        'targetInModel': pl.String,
        'targetInModelMgiId': pl.String,
        'targetInModelEnsemblId': pl.String,
        'targetFromSourceId': pl.String,
        'modelPhenotypeId': pl.String,
        'modelPhenotypeLabel': pl.String,
        'biologicalModels': pl.List(
            pl.Struct({
                'allelicComposition': pl.String,
                'geneticBackground': pl.String,
                'id': pl.String,
                'literature': pl.List(pl.String),
            })
        ),
        'modelPhenotypeClasses': pl.List(pl.Struct({'id': pl.String, 'label': pl.String})),
    })
    assert published == MOUSE_PHENOTYPE
    assert output.schema == MOUSE_PHENOTYPE
    assert excluded.columns == ['targetFromSourceId', *(c for c in MOUSE_PHENOTYPE if c != 'targetFromSourceId')]
    assert excluded.schema == pl.Schema({c: MOUSE_PHENOTYPE[c] for c in excluded.columns})


def test_empty_inputs_write_both_outputs(tmp_path: Path):
    _, full_excluded = _run(tmp_path / 'full')
    output, excluded = _run(
        tmp_path / 'empty', targets=[], mp=_ontology([_node(1, 'root')], []), **{key: [] for key in TABLES}
    )
    assert (output.height, excluded.height) == (0, 0)
    assert output.schema == MOUSE_PHENOTYPE
    assert excluded.schema == full_excluded.schema


def test_excluded_holds_every_row_of_an_unknown_target(tmp_path: Path):
    output, excluded = _run(tmp_path)
    assert 'ENSG3' not in output['targetFromSourceId'].to_list()
    assert excluded.select('targetFromSourceId', 'modelPhenotypeId', 'modelPhenotypeLabel').rows() == [
        ('ENSG3', 'MP:0000002', 'two'),
        ('ENSG3', 'MP:0000004', 'four'),
        ('ENSG3', 'MP:0000004', 'four again'),
    ]


def test_split_by_target():
    df = pl.DataFrame(
        {name: ['x', 'y', 'z'] if name == 'targetFromSourceId' else [None] * 3 for name in MOUSE_PHENOTYPE},
        schema=MOUSE_PHENOTYPE,
    )
    known, unknown = split_by_target(df, pl.Series(['y', 'y', 'w']))
    assert known.columns == list(MOUSE_PHENOTYPE)
    assert known['targetFromSourceId'].to_list() == ['y']
    assert unknown['targetFromSourceId'].to_list() == ['x', 'z']
    assert unknown.columns[0] == 'targetFromSourceId'


def test_model_without_references_has_null_literature(tmp_path: Path):
    output, excluded = _run(tmp_path)
    literature = {
        (row['targetInModelMgiId'], row['modelPhenotypeId'], model['id']): model['literature']
        for row in pl.concat([output, excluded.select(output.columns)]).to_dicts()
        for model in row['biologicalModels']
    }
    assert literature[('MGI:103', 'MP:0000002', 'MGI:6')] is None
    assert literature[('MGI:102', 'MP:0000002', None)] is None
    assert literature[('MGI:101', 'MP:0000002', 'MGI:2')] == ['111', '222']
    assert literature[('MGI:101', 'MP:0000002', 'MGI:1')] == ['111', '222', '333']
