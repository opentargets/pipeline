"""Tests for IMPC evidence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from loguru import logger
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.impc import (
    build_gene_mapping,
    cleanup_model_identifier,
    process_literature_references,
    process_ontology_terms,
)
from pts.pyspark.common.session import Session
from pts.pyspark.impc import (
    _finalise_evidence_strings,
    _load_impc_datasets_for_evidence,
    _prepare_impc_datasets_for_evidence,
    generate_impc_evidence_strings,
)
from pts.transformers.evidence.impc import (
    DIRECTION_ON_TARGET,
    DIRECTION_ON_TRAIT,
    IMPC,
    MODEL_ID,
    SOURCES,
    TIE_BREAK,
    UNIQUE_FIELDS,
    biological_model_id,
    deduplicate,
    impc,
    parse_impc,
)
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol, normalise_term
from tests.evidence.fixtures import write_disease, write_literature, write_ontoma_lut, write_target

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


def summary(
    model: str = 'MGI:1#hom#early',
    background: str = 'involves: 129',
    description: str = 'Cftr<tm1>/Cftr<tm1>',
    disease: str | None = 'OMIM:219700',
    term: str | None = 'Cystic fibrosis',
    score: str | None = '64.65',
    marker: str | None = 'MGI:101',
) -> list[str | None]:
    return [model, background, description, disease, term, score, '60.0', marker]


def gene_model(mgi: str = 'MGI:101', symbol: str = 'Cftr', ensembl: str | None = 'ENSMUSG1') -> list[str | None]:
    return [mgi, 'Gene', symbol, 'name', 'GRCm39', '1', '6', '1', '2', '+', ensembl, '6', '1', '2', '+', None]


def pubmed(mp: str = 'MP:0000001', pmids: str | None = '111', markers: str | None = 'MGI:101') -> list[str | None]:
    return ['Cftr<tm1>/Cftr<tm1>', 'Cftr<tm1>', 'involves: 129', mp, pmids, markers, 'MGI:9']


DEFAULTS: dict[str, tuple[list[str] | None, list[list[str | None]]]] = {
    'solr_disease_model_summary': (SUMMARY_HEADER, [summary()]),
    'solr_mouse_model': (['model_id', 'model_phenotypes'], [['MGI:1#hom#early', 'MP:0000001 one,MP:0000002 two']]),
    'solr_disease': (['disease_id', 'disease_phenotypes'], [['OMIM:219700', 'HP:0000001 x,HP:0000002 y']]),
    'solr_ontology_ontology': (['mp_id', 'hp_id'], [['MP:0000001', 'HP:0000001'], ['MP:0000002', 'HP:0000003']]),
    'solr_ontology': (
        ['ontology', 'phenotype_id', 'phenotype_term'],
        [
            ['MP', 'MP:0000001', 'mp one'],
            ['MP', 'MP:0000002', 'mp two'],
            ['HP', 'HP:0000001', 'hp one'],
            ['HP', 'HP:0000003', 'hp three'],
        ],
    ),
    'solr_gene_gene': (['gene_id', 'hgnc_gene_id'], [['MGI:101', 'HGNC:1']]),
    'hgnc_gene_mappings': (['hgnc_id', 'symbol', 'ensembl_gene_id'], [['HGNC:1', 'CFTR', 'ENSG1']]),
    'mouse_gene_mappings': (GENE_MODEL_HEADER, [gene_model()]),
    'mouse_pubmed_refs': (None, [pubmed()]),
}


def _render(value: str | None, sep: str) -> str:
    if value is None:
        return ''
    if sep in value or '"' in value:
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return value


def _write_table(path: Path, header: list[str] | None, rows: list[list[str | None]], sep: str) -> str:
    lines = [] if header is None else [sep.join(header)]
    lines += [sep.join(_render(v, sep) for v in row) for row in rows]
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def _inputs(root: Path, **tables: list[list[str | None]]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    unknown = set(tables) - set(DEFAULTS)
    assert not unknown, unknown
    source = {}
    for key, (header, rows) in DEFAULTS.items():
        sep = '\t' if key in {'hgnc_gene_mappings', 'mouse_gene_mappings', 'mouse_pubmed_refs'} else ','
        source[key] = _write_table(root / key, header, tables.get(key, rows), sep)
    return source


def _parse(tmp: Path, score_cutoff: float = 41, **tables: list[list[str | None]]) -> pl.DataFrame:
    return parse_impc(_inputs(tmp / 'in', **tables), score_cutoff).collect()


def test_sources_drop_mp_ontology() -> None:
    assert SOURCES == [
        'solr_gene_gene',
        'solr_ontology_ontology',
        'solr_mouse_model',
        'solr_disease',
        'solr_disease_model_summary',
        'solr_ontology',
        'hgnc_gene_mappings',
        'mouse_gene_mappings',
        'mouse_pubmed_refs',
    ]


def test_evidence_columns_and_values(tmp_path: Path) -> None:
    raw = _parse(tmp_path)
    assert raw.drop(MODEL_ID).rows(named=True) == [
        {
            'targetInModelMgiId': 'MGI:101',
            'diseaseFromSourceId': 'OMIM:219700',
            'biologicalModelGeneticBackground': 'involves: 129',
            'biologicalModelAllelicComposition': 'Cftr<tm1>/Cftr<tm1>',
            'diseaseFromSource': 'Cystic fibrosis',
            'resourceScore': 64.65,
            'targetInModel': 'Cftr',
            'targetInModelEnsemblId': 'ENSMUSG1',
            'targetFromSourceId': 'ENSG1',
            'diseaseModelAssociatedModelPhenotypes': [
                {'id': 'MP:0000001', 'label': 'mp one'},
                {'id': 'MP:0000002', 'label': 'mp two'},
            ],
            'diseaseModelAssociatedHumanPhenotypes': [{'id': 'HP:0000001', 'label': 'hp one'}],
            'literature': ['111'],
            'biologicalModelId': 'MGI:1',
        }
    ]
    assert raw[MODEL_ID].to_list() == ['MGI:1#hom#early']


def test_model_without_mp_terms_has_no_model_phenotypes(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        solr_mouse_model=[['MGI:1#hom#early', 'MP:0009999 unknown'], ['MGI:2', 'MP:0000001 one']],
    )
    assert raw.height == 1
    assert raw['diseaseModelAssociatedModelPhenotypes'].to_list() == [None]
    assert raw['diseaseModelAssociatedHumanPhenotypes'].to_list() == [None]
    assert raw['literature'].to_list() == [None]


def test_hp_terms_distinct(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        solr_mouse_model=[['MGI:1#hom#early', 'MP:0000001 one,MP:0000002 two,MP:0000001 again']],
        solr_disease=[['OMIM:219700', 'HP:0000003 c,HP:0000001 a,HP:0000003 c'], ['OMIM:219700', 'HP:0000001 a']],
        solr_ontology_ontology=[
            ['MP:0000001', 'HP:0000001'],
            ['MP:0000002', 'HP:0000001'],
            ['MP:0000002', 'HP:0000003'],
        ],
        solr_ontology=[
            ['MP', 'MP:0000001', 'mp one'],
            ['MP', 'MP:0000001', 'mp one'],
            ['MP', 'MP:0000002', 'mp two'],
            ['HP', 'HP:0000003', 'hp three'],
            ['HP', 'HP:0000001', 'hp one'],
            ['HP', 'HP:0000001', 'hp one'],
            ['HP', 'HP:0000001', 'hp one alias'],
        ],
    )
    assert raw['diseaseModelAssociatedHumanPhenotypes'].to_list() == [
        [
            {'id': 'HP:0000001', 'label': 'hp one'},
            {'id': 'HP:0000001', 'label': 'hp one alias'},
            {'id': 'HP:0000003', 'label': 'hp three'},
        ]
    ]
    assert raw['diseaseModelAssociatedModelPhenotypes'].to_list() == [
        [{'id': 'MP:0000001', 'label': 'mp one'}, {'id': 'MP:0000002', 'label': 'mp two'}]
    ]


def test_phenotype_ids_need_ascii_digits(tmp_path: Path) -> None:
    arabic_indic = '\u0660' * 6 + '\u0661'
    raw = _parse(tmp_path, solr_mouse_model=[['MGI:1#hom#early', f'MP:{arabic_indic} one,MP:0000002 two']])
    assert raw['diseaseModelAssociatedModelPhenotypes'].to_list() == [[{'id': 'MP:0000002', 'label': 'mp two'}]]


def test_score_cutoff_drops_below_and_null(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        solr_disease_model_summary=[
            summary(model='MGI:1', score='41'),
            summary(model='MGI:2', score='40.99'),
            summary(model='MGI:3', score=None),
            summary(model='MGI:4', score='not a number'),
            summary(model='MGI:5', score='100'),
        ],
    )
    assert raw.select(MODEL_ID, 'resourceScore').rows() == [('MGI:1', 41.0), ('MGI:5', 100.0)]


def test_gene_mapping_inner_joins_multiply(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        solr_disease_model_summary=[
            summary(),
            summary(model='MGI:2', marker='MGI:102'),
            summary(model='MGI:3', marker='MGI:103'),
        ],
        solr_gene_gene=[['MGI:101', 'HGNC:1'], ['MGI:101', 'HGNC:2'], ['MGI:102', 'HGNC:3'], ['MGI:103', 'HGNC:1']],
        hgnc_gene_mappings=[['HGNC:1', 'CFTR', 'ENSG1'], ['HGNC:2', 'CFTR2', 'ENSG2'], ['HGNC:3', 'X', None]],
        mouse_gene_mappings=[gene_model(), gene_model('MGI:102', 'Other'), gene_model('MGI:103', 'Nope', None)],
    )
    assert raw.select(MODEL_ID, 'targetInModel', 'targetFromSourceId').sort('targetFromSourceId').rows() == [
        ('MGI:1#hom#early', 'Cftr', 'ENSG1'),
        ('MGI:1#hom#early', 'Cftr', 'ENSG2'),
    ]


def test_literature_cross_product_distinct(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        solr_disease_model_summary=[summary(), summary(model='MGI:1#hom#early', marker='MGI:102')],
        solr_gene_gene=[['MGI:101', 'HGNC:1'], ['MGI:102', 'HGNC:1']],
        mouse_gene_mappings=[gene_model(), gene_model('MGI:102', 'Cftr2')],
        mouse_pubmed_refs=[
            pubmed(pmids='333|111', markers='MGI:101|MGI:102'),
            pubmed(mp='MP:0000002', pmids='111|', markers='MGI:101'),
            pubmed(mp='MP:0000002', pmids=None, markers='MGI:102'),
            pubmed(mp='MP:0000003', pmids='999', markers='MGI:101'),
        ],
    )
    assert raw.select('targetInModelMgiId', 'literature').sort('targetInModelMgiId').rows() == [
        ('MGI:101', ['', '111', '333']),
        ('MGI:102', ['111', '333']),
    ]


@pytest.mark.parametrize(
    ('model', 'expected'),
    [
        ('MGI:1#hom#early', 'MGI:1'),
        ('MGI:123', 'MGI:123'),
        ('MGI:12a#hom', None),
        ('IMPC:1#hom', None),
        ('#MGI:1', None),
        ('MGI:', None),
        ('MGI:\u0661\u0662', None),
        ('MGI:1\u2028', 'MGI:1\u2028'),
        ('MGI:1\u2028x', None),
        ('', None),
        (None, None),
    ],
)
def test_biological_model_id_only_for_mgi(model: str | None, expected: str | None) -> None:
    df = pl.DataFrame({'m': [model]}, schema={'m': pl.String})
    assert df.select(biological_model_id(pl.col('m'))).item() == expected


def test_biological_model_id_matches_spark(spark: SparkSession) -> None:
    models = [
        'MGI:1#hom#early',
        'MGI:12a#hom',
        '#MGI:1',
        'MGI:\u0661\u0662',
        'MGI:1\u2028',
        'MGI:1\u0085',
        'MGI:1\r\n',
        'MGI:1\n\n',
        '',
        None,
    ]
    expected = [
        r[0]
        for r in spark
        .createDataFrame([(m,) for m in models], 'm string')
        .select(cleanup_model_identifier(f.col('m')))
        .collect()
    ]
    df = pl.DataFrame({'m': models}, schema={'m': pl.String})
    assert df.select(biological_model_id(pl.col('m'))).to_series().to_list() == expected


def _frame(rows: list[dict]) -> pl.DataFrame:
    base = dict.fromkeys(UNIQUE_FIELDS, 'x') | dict.fromkeys(TIE_BREAK, 'x')
    return pl.DataFrame([base | row for row in rows], schema_overrides={'resourceScore': pl.Float64})


def test_dedup_keeps_top_score() -> None:
    out = deduplicate(
        _frame([
            {'resourceScore': 50.0, 'tag': 'low'},
            {'resourceScore': 70.0, 'tag': 'top'},
            {'resourceScore': 60.0, 'tag': 'mid'},
            {'resourceScore': 55.0, 'tag': 'other', 'targetFromSourceId': 'y'},
        ])
    )
    assert sorted(out['tag'].to_list()) == ['other', 'top']
    assert out['datasourceId'].to_list() == ['impc'] * 2
    assert out['datatypeId'].to_list() == ['animal_model'] * 2
    assert MODEL_ID not in out.columns


def test_dedup_partitions_null_keys_together() -> None:
    out = deduplicate(
        _frame([
            {'resourceScore': 50.0, 'tag': 'a', 'diseaseFromSourceMappedId': None},
            {'resourceScore': 60.0, 'tag': 'b', 'diseaseFromSourceMappedId': None},
        ])
    )
    assert out['tag'].to_list() == ['b']


def test_dedup_tie_is_deterministic() -> None:
    rows = [
        {'resourceScore': 70.0, 'tag': 'b', MODEL_ID: 'MGI:2#hom'},
        {'resourceScore': 70.0, 'tag': 'a', MODEL_ID: 'MGI:1#hom'},
        {'resourceScore': 70.0, 'tag': 'c', MODEL_ID: 'MGI:1#hom', 'diseaseFromSourceId': 'z'},
        {'resourceScore': 70.0, 'tag': 'd', MODEL_ID: None},
        {'resourceScore': 10.0, 'tag': 'e', MODEL_ID: 'MGI:0'},
    ]
    first = deduplicate(_frame(rows))
    second = deduplicate(_frame(rows[::-1]))
    assert first['tag'].to_list() == second['tag'].to_list() == ['a']
    assert first.equals(second)


def test_tie_break_order() -> None:
    assert TIE_BREAK == [MODEL_ID, 'diseaseFromSourceId', 'targetInModelMgiId', 'targetInModelEnsemblId']


def test_dedup_logs_tied_partitions() -> None:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), format='{message}')
    try:
        deduplicate(
            _frame([
                {'resourceScore': 70.0, MODEL_ID: 'MGI:1'},
                {'resourceScore': 70.0, MODEL_ID: 'MGI:2', 'tag': 'differs'},
                {'resourceScore': 60.0, 'targetFromSourceId': 'y', MODEL_ID: 'MGI:1'},
                {'resourceScore': 60.0, 'targetFromSourceId': 'y', MODEL_ID: 'MGI:2'},
                {'resourceScore': 80.0, 'targetFromSourceId': 'z', MODEL_ID: 'MGI:1'},
                {'resourceScore': 70.0, 'targetFromSourceId': 'z', MODEL_ID: 'MGI:2', 'tag': 'differs'},
            ])
        )
    finally:
        logger.remove(sink)
    assert '1 of 3 impc evidence partitions tie on the top resourceScore' in '\n'.join(messages)


def test_dedup_matches_spark_without_ties(spark: SparkSession, tmp_path: Path) -> None:
    frame = _frame([
        {'resourceScore': 50.0, 'tag': 'low'},
        {'resourceScore': 70.0, 'tag': 'top'},
        {'resourceScore': 55.0, 'tag': 'other', 'targetFromSourceId': 'y'},
        {'resourceScore': 55.0, 'tag': 'null', 'diseaseFromSourceMappedId': None},
        {'resourceScore': 45.0, 'tag': 'null2', 'diseaseFromSourceMappedId': None},
    ])
    frame.drop(MODEL_ID).write_parquet(tmp_path / 'mapped.parquet')
    expected = _finalise_evidence_strings(spark.read.parquet(str(tmp_path / 'mapped.parquet')))
    out = deduplicate(frame)
    assert out.columns == expected.columns
    assert sorted(map(repr, out.rows())) == sorted(repr(tuple(r)) for r in expected.collect())


def test_score_is_resource_score_over_100_ieee() -> None:
    values = [41.0, 64.65, 99.99, 70.3, 100.0, None]
    df = pl.DataFrame({'resourceScore': values}, schema={'resourceScore': pl.Float64})
    expected = [None if v is None else float(np.float64(v) / np.float64(100)) for v in values]
    assert df.select(IMPC.score).to_series().to_list() == expected
    assert df.select(IMPC.score).item(0, 0) == 41 / 100


def test_direction_risk() -> None:
    df = pl.DataFrame({'diseaseId': ['EFO_1', None]})
    assert df.select(DIRECTION_ON_TRAIT).to_series().to_list() == ['risk', None]


def test_direction_lof_on_target() -> None:
    df = pl.DataFrame({'diseaseId': ['EFO_1', None]})
    assert df.select(DIRECTION_ON_TARGET).to_series().to_list() == ['LoF', None]


def test_unique_fields_follow_the_spark_config() -> None:
    assert IMPC.datasource_id == 'impc'
    assert IMPC.unique_fields == [
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'biologicalModelGeneticBackground',
        'targetInModel',
        'biologicalModelAllelicComposition',
        'diseaseFromSource',
    ]
    assert UNIQUE_FIELDS == [
        'diseaseFromSource',
        'targetInModel',
        'biologicalModelAllelicComposition',
        'biologicalModelGeneticBackground',
        'diseaseFromSourceMappedId',
        'targetFromSourceId',
    ]


def _normalise(value: object) -> object:
    if isinstance(value, list):
        return sorted((_normalise(v) for v in value), key=repr)
    if isinstance(value, Row):
        return value.asDict(recursive=True)
    return value


def _session(spark: SparkSession) -> SimpleNamespace:
    return SimpleNamespace(
        load_data=lambda *args, **kwargs: Session.load_data(SimpleNamespace(spark=spark), *args, **kwargs)
    )


def _spark_parse(spark: SparkSession, source: dict[str, str]) -> tuple[list[str], list[str]]:
    datasets = _prepare_impc_datasets_for_evidence(_load_impc_datasets_for_evidence(_session(spark), source))
    mp_terms, hp_terms = process_ontology_terms(datasets['ontology'])
    gene_mapping = build_gene_mapping(
        datasets['mgi_gene_id_to_ensembl_mouse_gene_id'],
        datasets['mouse_to_human_gene'],
        datasets['hgnc_gene_id_to_ensembl_human_gene_id'],
    )
    literature = process_literature_references(
        datasets['mgi_pubmed'],
        datasets['disease_model_summary_transformed'],
        datasets['model_mouse_phenotypes_transformed'],
    )
    out = generate_impc_evidence_strings(
        datasets['model_mouse_phenotypes_transformed'],
        datasets['mouse_to_human_phenotype'],
        mp_terms,
        datasets['disease_model_summary_transformed'],
        datasets['disease_human_phenotypes_transformed'],
        hp_terms,
        gene_mapping,
        literature,
        41.0,
    )
    return out.columns, sorted(repr(tuple(_normalise(v) for v in row)) for row in out.collect())


def _polars_rows(raw: pl.DataFrame) -> list[str]:
    return sorted(repr(tuple(_normalise(v) for v in row)) for row in raw.drop(MODEL_ID).rows())


def test_parse_matches_spark_on_mixed_inputs(spark: SparkSession, tmp_path: Path) -> None:
    source = _inputs(
        tmp_path,
        solr_disease_model_summary=[
            summary(),
            summary(),
            summary(model='MGI:2#het#late', description='Cftr<tm1>/+', score='41', term='Say "hi", disease'),
            summary(model='MGI:3', disease='OMIM:1', term='back\\slash', score='70.3'),
            summary(model='MGI:4', score='40', marker='MGI:102'),
            summary(model='MGI:5', score=None),
            summary(model='MGI:6', disease=None, term=None),
            summary(model='MGI:7', marker=None),
            summary(model='IMPC:8#hom'),
        ],
        solr_mouse_model=[
            ['MGI:1#hom#early', 'MP:0000001 one,MP:0000002 two'],
            ['MGI:1#hom#early', 'MP:0000001 one'],
            ['MGI:2#het#late', 'MP:0000002 two,MP:0009999 missing'],
            ['MGI:3', 'no phenotypes'],
            ['MGI:6', 'MP:0000001 one'],
            ['IMPC:8#hom', 'MP:0000001 one,MP:\u0660\u0660\u0660\u0660\u0660\u0660\u0662 two'],
            ['MGI:9', None],
        ],
        solr_disease=[
            ['OMIM:219700', 'HP:0000001 x,HP:0000003 y'],
            ['OMIM:219700', 'HP:0000001 x'],
            ['OMIM:1', None],
        ],
        solr_gene_gene=[['MGI:101', 'HGNC:1'], ['MGI:101', 'HGNC:2'], ['MGI:102', 'HGNC:3'], ['MGI:103', 'HGNC:1']],
        hgnc_gene_mappings=[['HGNC:1', 'CFTR', 'ENSG1'], ['HGNC:2', 'CFTR2', 'ENSG2'], ['HGNC:3', 'X', 'null']],
        mouse_gene_mappings=[gene_model(), gene_model('MGI:102', 'Other'), gene_model('MGI:103', 'Third', 'null')],
        mouse_pubmed_refs=[
            pubmed(pmids='333|111', markers='MGI:101|MGI:102'),
            pubmed(mp='MP:0000002', pmids='111|', markers='MGI:101'),
            pubmed(mp='MP:0000002', pmids='null', markers='MGI:101'),
            pubmed(mp='MP:0000001', pmids='', markers='MGI:101'),
        ],
    )
    raw = parse_impc(source, 41).collect()
    columns, expected = _spark_parse(spark, source)
    assert raw.drop(MODEL_ID).columns == columns
    assert _polars_rows(raw) == expected
    assert raw.height == 10


def _sources(root: Path, **tables: list[list[str | None]]) -> dict[str, str]:
    return _inputs(root, **tables) | {
        'ontoma_disease_label_lut': write_ontoma_lut(
            root / 'label_lut', 'label', normalise_term('Cystic fibrosis'), 'MONDO_0009061'
        ),
        'ontoma_disease_id_lut': write_ontoma_lut(
            root / 'id_lut', 'id', normalise_symbol('OMIM:154700'), 'MONDO_0007947'
        ),
        'disease': write_disease(root / 'disease', ['MONDO_0009061', 'MONDO_0007947']),
        'target': write_target(root / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'CFTR', 'ENSG2': 'FBN1'}),
        'literature': write_literature(root / 'literature', {'111': '2020-05-01'}),
    }


def _run(tmp_path: Path, **tables: list[list[str | None]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    destination = {'evidence': str(tmp_path / 'out' / 'valid'), 'failed_evidence': str(tmp_path / 'out' / 'failed')}
    impc(_sources(tmp_path / 'src', **tables), destination, {'score_cutoff': 41}, None)
    return pl.read_parquet(destination['evidence']), pl.read_parquet(destination['failed_evidence'])


def test_maps_dedups_scores_and_dates(tmp_path: Path) -> None:
    valid, failed = _run(
        tmp_path,
        solr_disease_model_summary=[
            summary(score='64.65'),
            summary(model='MGI:2#hom', score='80'),
            summary(model='MGI:3', disease='OMIM:154700', term='Marfan', score='50', marker='MGI:102'),
        ],
        solr_gene_gene=[['MGI:101', 'HGNC:1'], ['MGI:102', 'HGNC:2']],
        hgnc_gene_mappings=[['HGNC:1', 'CFTR', 'ENSG1'], ['HGNC:2', 'FBN1', 'ENSG2']],
        mouse_gene_mappings=[gene_model(), gene_model('MGI:102', 'Fbn1')],
    )
    assert valid.sort('targetId').select(
        'targetId', 'diseaseId', 'score', 'biologicalModelId', 'directionOnTrait', 'directionOnTarget', 'evidenceDate'
    ).rows() == [
        ('ENSG1', 'MONDO_0009061', 0.8, 'MGI:2', 'risk', 'LoF', None),
        ('ENSG2', 'MONDO_0007947', 0.5, 'MGI:3', 'risk', 'LoF', None),
    ]
    assert valid['datasourceId'].to_list() == ['impc'] * 2
    assert valid['datatypeId'].to_list() == ['animal_model'] * 2
    assert valid['id'].n_unique() == 2
    assert MODEL_ID not in valid.columns
    assert failed.height == 0


def test_literature_dates_evidence(tmp_path: Path) -> None:
    valid, _ = _run(tmp_path)
    assert valid.select('literature', 'evidenceDate').rows() == [(['111'], '2020-05-01')]


def test_unmapped_disease_lands_in_failed_evidence(tmp_path: Path) -> None:
    valid, failed = _run(tmp_path, solr_disease_model_summary=[summary(disease='OMIM:9', term='nothing at all')])
    assert failed['diseaseFromSource'].to_list() == ['nothing at all']
    assert failed['diseaseFromSourceMappedId'].to_list() == [None]
    assert failed['targetId'].to_list() == ['ENSG1']
    assert valid.height == 0


def test_empty_inputs_write_empty_outputs(tmp_path: Path) -> None:
    valid, failed = _run(tmp_path, **{key: [] for key in DEFAULTS})
    assert valid.height == 0
    assert failed.height == 0
