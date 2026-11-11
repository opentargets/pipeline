"""Tests for the polars port of `OnToma.map_entities` and of `add_efo_mapping`."""

from pathlib import Path

import polars as pl
import pytest

from pts.schemas.ontoma_lut import ENTITY_LUT, RAW_ENTITY_LUT
from pts.transformers.evidence.utils.ontology import add_efo_mapping
from pts.transformers.ontoma_lut.build import build_lut
from pts.transformers.utils.disease_mapping import map_entities
from pts.transformers.utils.disease_mapping.mapping import non_acronym_ids
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol, normalise_term

LUT_SCHEMA = {
    'entityKind': pl.String,
    'entityType': pl.String,
    'entityLabelNormalised': pl.String,
    'entityIds': pl.List(pl.Struct({'entityId': pl.String, 'entitySource': pl.String, 'isAcronym': pl.Boolean})),
}


def _lut(rows: list[tuple[str, str, str, list[tuple[str, str]]]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            (kind, type_, label, [{'entityId': i, 'entitySource': s, 'isAcronym': False} for i, s in ids])
            for kind, type_, label, ids in rows
        ],
        schema=LUT_SCHEMA,
        orient='row',
    )


LABEL_LUT = _lut([
    ('label', 'DS', normalise_term('cystic fibrosis'), [('MONDO_0009061', 'name')]),
    # both spellings of an accented label are tried, and what each finds is merged
    ('label', 'DS', normalise_term('Papillon-Lefevre syndrome'), [('MONDO_0009859', 'name')]),
    ('label', 'DS', normalise_term('Papillon-Lefèvre syndrome'), [('EFO_0000001', 'exact_synonym')]),
    ('label', 'DS', normalise_symbol('ALS'), [('MONDO_0004976', 'exact_synonym'), ('EFO_0000253', 'name')]),
    ('label', 'CD', normalise_symbol('aspirin'), [('CHEMBL25', 'name')]),
])
ID_LUT = _lut([
    ('id', 'DS', normalise_symbol('ORPHANET:558'), [('MONDO_0007947', 'crossref')]),
    ('id', 'DS', normalise_symbol('MONDO:0009061'), [('MONDO_0009061', 'id')]),
])


def _map(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    kwargs.setdefault('type_col', pl.lit('DS'))
    return map_entities(df.lazy(), LABEL_LUT.lazy(), 'ids', 'label', 'label', **kwargs).collect()


def test_maps_labels_and_leaves_unmapped_null() -> None:
    df = pl.DataFrame({'label': ['Cystic Fibrosis', 'unknown disease', None, ''], 'n': [1, 2, 3, 4]})
    assert _map(df).rows() == [
        ('Cystic Fibrosis', 1, ['MONDO_0009061']),
        ('unknown disease', 2, None),
        (None, 3, None),
        ('', 4, None),
    ]


def test_merges_what_both_spellings_find() -> None:
    df = pl.DataFrame({'label': ['Papillon-Lefèvre syndrome']})
    assert _map(df)['ids'].to_list() == [['EFO_0000001', 'MONDO_0009859']]


def test_acronym_goes_through_the_symbol_track() -> None:
    assert _map(pl.DataFrame({'label': ['ALS']}))['ids'].to_list() == [['EFO_0000253', 'MONDO_0004976']]


def test_identical_rows_collapse_as_in_ontoma() -> None:
    df = pl.DataFrame({'label': ['cystic fibrosis', 'cystic fibrosis', 'cystic fibrosis'], 'n': [1, 1, 2]})
    assert _map(df).rows() == [('cystic fibrosis', 1, ['MONDO_0009061']), ('cystic fibrosis', 2, ['MONDO_0009061'])]


def test_unordered_input_maps_each_row_to_its_own_label() -> None:
    labels = [f'disease{i}' for i in range(2000)]
    lut = _lut([('label', 'DS', normalise_term(label), [(f'ID_{label}', 'name')]) for label in labels])
    lf = pl.LazyFrame({'label': labels * 2, 'n': 1}).group_by('label').agg(pl.col('n').sum())
    out = map_entities(lf, lut.lazy(), 'ids', 'label', 'label', type_col=pl.lit('DS')).collect()
    assert out.height == len(labels)
    assert out['ids'].to_list() == [[f'ID_{label}'] for label in out['label']]


def test_type_column_selects_lut_rows_and_track() -> None:
    df = pl.DataFrame({
        'label': ['aspirin', 'aspirin', 'cystic fibrosis', 'cystic fibrosis'],
        'type': ['CD', 'DS', 'DS', None],
    })
    out = _map(df, type_col=None, type_col_name='type')
    assert out.rows() == [
        ('aspirin', 'CD', ['CHEMBL25']),
        ('aspirin', 'DS', None),
        ('cystic fibrosis', 'DS', ['MONDO_0009061']),
        ('cystic fibrosis', None, None),
    ]


def test_include_flags() -> None:
    df = pl.DataFrame({'label': ['Papillon-Lefèvre syndrome']})
    out = _map(df, include_normalised_entities=True, include_entity_source=True)
    assert out.columns == ['label', 'entityLabelNormalised', 'ids']
    assert sorted(out.select('entityLabelNormalised', 'ids').rows()) == sorted([
        (
            normalise_term('Papillon-Lefevre syndrome'),
            [{'entityId': 'MONDO_0009859', 'entitySource': 'name', 'isAcronym': False}],
        ),
        (
            normalise_term('Papillon-Lefèvre syndrome'),
            [{'entityId': 'EFO_0000001', 'entitySource': 'exact_synonym', 'isAcronym': False}],
        ),
    ])


def test_maps_ids_after_formatting_them() -> None:
    df = pl.DataFrame({
        'id': [
            'Orphanet_558',
            ' ordo:558 ',
            'efo:Orphanet_558',
            # unnesting takes all after the FIRST colon, here `//WWW...`, so a URL does not resolve
            'http://www.orpha.net/ORDO/Orphanet_558',
            'MONDO_0009061',
            None,
        ]
    })
    out = map_entities(df.lazy(), ID_LUT.lazy(), 'ids', 'id', 'id', type_col=pl.lit('DS')).collect()
    assert out['ids'].to_list() == [
        ['MONDO_0007947'],
        ['MONDO_0007947'],
        ['MONDO_0007947'],
        None,
        ['MONDO_0009061'],
        None,
    ]


@pytest.mark.parametrize('type_args', [{}, {'type_col': pl.lit('DS'), 'type_col_name': 'type'}])
def test_requires_exactly_one_type_argument(type_args: dict) -> None:
    with pytest.raises(ValueError, match='Exactly one'):
        map_entities(
            pl.LazyFrame({'label': ['x'], 'type': ['DS']}), LABEL_LUT.lazy(), 'ids', 'label', 'label', **type_args
        )


def test_rejects_an_output_column_already_present() -> None:
    with pytest.raises(ValueError, match='already present'):
        _map(pl.DataFrame({'label': ['x'], 'ids': ['y']}))


def test_input_columns_are_kept_untouched() -> None:
    # OnToma would overwrite these working-column names; the port keeps every input column as is
    df = pl.DataFrame({
        'entityLabel': ['#MS ', 'cystic fibrosis'],
        'entityType': ['GP', 'CD'],
        'entityKind': ['x', 'y'],
        'nlpPipelineTrack': ['a', 'b'],
        'entityLabelNormalised': ['p', 'q'],
        'entityIds': [1, 2],
        '__disease_mapping_row': [3, 4],
    })
    out = map_entities(df.lazy(), LABEL_LUT.lazy(), 'ids', 'entityLabel', 'label', type_col=pl.lit('DS')).collect()
    assert out.drop('ids').equals(df)
    assert out['ids'].to_list() == [None, ['MONDO_0009061']]


def test_no_working_column_is_left_behind() -> None:
    df = pl.DataFrame({'label': ['cystic fibrosis'], 'type': ['DS']})
    out = _map(df, type_col=None, type_col_name='type', include_normalised_entities=True)
    assert out.columns == ['label', 'type', 'entityLabelNormalised', 'ids']


@pytest.fixture
def lut_paths(tmp_path: Path) -> tuple[str, str]:
    label, ids = tmp_path / 'label_lut', tmp_path / 'id_lut'
    label.mkdir()
    ids.mkdir()
    LABEL_LUT.write_parquet(label / 'part-0.parquet')
    ID_LUT.write_parquet(ids / 'part-0.parquet')
    return str(label), str(ids)


def test_add_efo_mapping_merges_label_and_id_mappings(lut_paths: tuple[str, str]) -> None:
    label_lut, id_lut = lut_paths
    evidence = pl.LazyFrame({
        'diseaseFromSource': ['cystic fibrosis', 'cystic fibrosis', 'unknown', 'unknown'],
        'diseaseFromSourceId': ['MONDO:0009061', 'Orphanet_558', 'Orphanet_558', 'unknown'],
    })
    out = add_efo_mapping(evidence, label_lut, disease_id_lut_path=id_lut).collect()
    assert out.rows() == [
        ('cystic fibrosis', 'MONDO:0009061', 'MONDO_0009061'),
        ('cystic fibrosis', 'Orphanet_558', 'MONDO_0007947'),
        ('cystic fibrosis', 'Orphanet_558', 'MONDO_0009061'),
        ('unknown', 'Orphanet_558', 'MONDO_0007947'),
        # unmapped evidence is kept, with a null mapping
        ('unknown', 'unknown', None),
    ]


def test_add_efo_mapping_by_label_only(lut_paths: tuple[str, str]) -> None:
    label_lut, _ = lut_paths
    evidence = pl.LazyFrame({'diseaseFromSource': ['ALS', 'unknown'], 'diseaseFromSourceId': ['x', 'y']})
    out = add_efo_mapping(evidence, label_lut, id_col_name=None).collect()
    assert out.rows() == [('ALS', 'x', 'EFO_0000253'), ('ALS', 'x', 'MONDO_0004976'), ('unknown', 'y', None)]


@pytest.mark.parametrize('with_id_lut', [False, True])
def test_add_efo_mapping_keeps_evidence_columns(lut_paths: tuple[str, str], with_id_lut: bool) -> None:
    label_lut, id_lut = lut_paths
    # named like the spark implementation's intermediate columns
    evidence = pl.LazyFrame({
        'diseaseFromSource': ['ALS'],
        'diseaseFromSourceId': ['x'],
        'labelMappedResult': [['kept']],
        'idMappedResult': [['kept too']],
    })
    out = add_efo_mapping(evidence, label_lut, disease_id_lut_path=id_lut if with_id_lut else None).collect()
    assert out.columns == [*evidence.collect_schema().names(), 'diseaseFromSourceMappedId']
    assert out['labelMappedResult'].to_list() == [['kept'], ['kept']]
    assert out['idMappedResult'].to_list() == [['kept too'], ['kept too']]


def _acronym_lut() -> pl.LazyFrame:
    raw = pl.DataFrame(
        [
            ('MONDO_1', 'CF', 0.999, 'symbol', 'exact_synonym', 'DS', 'label'),
            ('MONDO_1', 'Cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
            ('MONDO_1', 'MONDO:1', 1.0, 'symbol', 'id', 'DS', 'id'),
        ],
        schema=RAW_ENTITY_LUT,
        orient='row',
    )
    return build_lut(
        [raw], acronym_allowlist=pl.DataFrame(schema={'acronym': pl.String, 'diseaseId': pl.String})
    ).lazy()


def _structs(label: str, kind: str = 'label') -> list[dict] | None:
    lf = pl.LazyFrame({'x': [label]})
    out = map_entities(
        lf, _acronym_lut(), 'ids', 'x', kind, type_col=pl.lit('DS'), include_entity_source=True
    ).collect()
    return out['ids'].to_list()[0]


@pytest.mark.parametrize('label', ['CF', 'C.F.', '(CF)', '#CF', ' cf '])
def test_spellings_of_an_acronym_hit_the_flagged_entry(label: str) -> None:
    assert _structs(label) == [{'entityId': 'MONDO_1', 'entitySource': 'exact_synonym', 'isAcronym': True}]


def test_full_name_is_not_flagged() -> None:
    assert _structs('cystic fibrosis') == [{'entityId': 'MONDO_1', 'entitySource': 'name', 'isAcronym': False}]


def test_id_queries_never_flagged() -> None:
    found = _structs('MONDO:1', kind='id')
    assert found
    assert all(not entry['isAcronym'] for entry in found)


def test_without_entity_source_ids_are_bare() -> None:
    lf = pl.LazyFrame({'x': ['CF']})
    out = map_entities(lf, _acronym_lut(), 'ids', 'x', 'label', type_col=pl.lit('DS')).collect()
    assert out['ids'].to_list() == [['MONDO_1']]


def test_non_acronym_ids() -> None:
    def entry(i: str, source: str, flag: bool | None) -> dict:
        return {'entityId': i, 'entitySource': source, 'isAcronym': flag}

    structs = pl.Series(
        [
            [entry('A', 's', True), entry('B', 's', False), entry('B', 't', False)],
            [entry('A', 's', True)],
            [entry('C', 's', None), entry('A', 's', True)],
            None,
            [],
        ],
        dtype=ENTITY_LUT['entityIds'],
    )
    out = pl.DataFrame({'s': structs}).select(non_acronym_ids(pl.col('s')))['s'].to_list()
    assert out == [['B'], None, ['C'], None, None]
