"""Tests for CRISPR Brain screens (crispr_screen) evidence."""

from __future__ import annotations

import gzip
import json
from math import nan
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pts.transformers.evidence.crispr_screens import (
    CRISPR_SCREEN,
    GENES_SCHEMA,
    crispr_screens,
    generate,
    parsing_experiment,
)
from tests.evidence.fixtures import write_disease, write_literature, write_target

GENES_HEADER = list(GENES_SCHEMA)
RNA_SEQ_HEADER = ['Gene', 'Log2CPM', 'Log2FC', 'P Value', 'FDR', 'name', 'P Value (Signed -Log10)']
MAPPING_HEADER = ['studyId', 'diseaseFromSource', 'diseaseFromSourceMappedId', 'contrast']

RAW_SCHEMA = {
    'studyId': pl.String,
    'datatypeId': pl.String,
    'datasourceId': pl.String,
    'projectId': pl.String,
    'crisprScreenLibrary': pl.String,
    'studyOverview': pl.String,
    'cellType': pl.String,
    'geneticBackground': pl.String,
    'literature': pl.List(pl.String),
    'diseaseFromSource': pl.String,
    'diseaseFromSourceMappedId': pl.String,
    'contrast': pl.String,
    'targetFromSourceId': pl.String,
    'resourceScore': pl.Float32,
    'statisticalTestTail': pl.String,
    'log2FoldChangeValue': pl.Float64,
}


def _screen(name: str, link: str = '', genotype: str = 'WT', phenotype: str = 'Survival') -> dict:
    return {
        'download_url': f'https://example.org/{name}.csv.gz',
        'metadata': {
            'Screen Name': name,
            'Cell Type': 'Astrocyte',
            'Genotype': genotype,
            'Libraries Screened': 'H1',
            'Phenotype': phenotype,
            'Description': f'## {name}\n## Experiment\nThe experiment.\n## Analysis\nThe analysis.',
            'Reference Link': link,
        },
    }


def _hit(gene: str, phenotype: str, p_value: str, hit_class: str) -> list[str]:
    return [gene, 'tss', phenotype, p_value, '1', hit_class, '1', '1', '1', '1', '1', '1']


def _write_study(directory: Path, name: str, header: list[str], rows: list[list[str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(','.join(row) for row in [header, *rows]) + '\n'
    (directory / f'{name}.csv.gz').write_bytes(gzip.compress(text.encode()))


def _write_inputs(
    src: Path, screens: list[dict], studies: dict[str, list[list[str]]], mapping: list[list[str]]
) -> dict:
    src.mkdir(parents=True, exist_ok=True)
    payload = {'count': len(screens), **{s['metadata']['Screen Name']: s for s in screens}}
    (src / 'screens.json.gz').write_bytes(gzip.compress(json.dumps(payload).encode()))
    for name, rows in studies.items():
        _write_study(src / 'studies', name, GENES_HEADER, rows)
    (src / 'mapping.tsv').write_text('\n'.join('\t'.join(row) for row in [MAPPING_HEADER, *mapping]) + '\n')
    return {
        'screens': str(src / 'screens.json.gz'),
        'studies_dir': str(src / 'studies'),
        'disease_mapping': str(src / 'mapping.tsv'),
    }


def _sources(src: Path) -> dict:
    return _write_inputs(
        src,
        screens=[
            _screen('iAstrocyte-A', 'https://pubmed.ncbi.nlm.nih.gov/30297964/', phenotype='Lysosome'),
            _screen('Neuron B', 'https://unknown.org', genotype='PSAP KO'),
            _screen('Unmapped-C'),
        ],
        studies={
            'iAstrocyte-A': [
                _hit('GENE1', '-2.5', '0.01', 'Negative Hit'),
                _hit('GENE2', '1.5', '0.2', 'Positive Hit'),
                _hit('GENE3', '0.1', '0.9', 'Non-Hit'),
                _hit('GENE2', '1.5', '0.2', 'Positive Hit'),
            ],
            'Neuron B': [_hit('GENE4', '3.0', '0.001', 'Positive Hit')],
            'Unmapped-C': [_hit('GENE5', '3.0', '0.001', 'Positive Hit')],
        },
        mapping=[
            ['iAstrocyte-A', 'neurodegenerative disease, AD', 'MONDO_1, MONDO_2', ''],
            ['Neuron B', 'ageing', 'MONDO_3', 'Iron'],
            ['Unmapped-C', 'x', '', 'x'],
        ],
    )


def _study(study_id: str, library: str = 'H1', cell_type: str = 'Astrocyte') -> dict:
    return {
        'studyId': study_id,
        'datatypeId': 'affected_pathway',
        'datasourceId': 'crispr_screen',
        'projectId': 'crispr_brain',
        'crisprScreenLibrary': library,
        'studyOverview': 'Description',
        'cellType': cell_type,
    }


def test_generate_joins_screens_to_hits_and_explodes_mapped_diseases(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path)).collect()

    assert raw.schema == pl.Schema(RAW_SCHEMA)
    a = {
        **_study('iAstrocyte-A'),
        'geneticBackground': None,
        'literature': ['30297964'],
        'diseaseFromSource': 'neurodegenerative disease, AD',
        'contrast': 'Lysosome',
    }
    b = {
        **_study('Neuron B'),
        'geneticBackground': 'PSAP KO',
        'literature': None,
        'diseaseFromSource': 'ageing',
        'contrast': 'Iron',
    }
    lower = {'statisticalTestTail': 'lower tail'}
    upper = {'statisticalTestTail': 'upper tail'}
    expected = [
        {**b, 'diseaseFromSourceMappedId': 'MONDO_3', 'targetFromSourceId': 'GENE4', 'resourceScore': 0.001, **upper}
        | {'log2FoldChangeValue': 3.0}
    ]
    expected += [
        {**a, 'diseaseFromSourceMappedId': d, 'targetFromSourceId': 'GENE1', 'resourceScore': 0.01, **lower}
        | {'log2FoldChangeValue': -2.5}
        for d in ['MONDO_1', 'MONDO_2']
    ]
    expected += [
        {**a, 'diseaseFromSourceMappedId': d, 'targetFromSourceId': 'GENE2', 'resourceScore': 0.2, **upper}
        | {'log2FoldChangeValue': 1.5}
        for d in ['MONDO_1', 'MONDO_2']
    ]
    got = raw.sort('studyId', 'targetFromSourceId', 'diseaseFromSourceMappedId').to_dicts()
    assert got == [{**row, 'resourceScore': pytest.approx(row['resourceScore'], rel=1e-7)} for row in expected]
    assert raw.columns == list(RAW_SCHEMA)


def test_generate_reads_p_values_as_float32(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path)).collect()

    got = raw.filter(pl.col('targetFromSourceId') == 'GENE2')['resourceScore'].to_list()
    assert got == [float(np.float32(0.2))] * 2


def test_generate_drops_studies_without_a_mapped_disease_or_hits(tmp_path: Path) -> None:
    sources = _write_inputs(
        tmp_path,
        screens=[_screen('Mapped'), _screen('Null-Mapping'), _screen('No-Mapping'), _screen('No-Hits')],
        studies={
            'Mapped': [_hit('GENE1', '1', '0.1', 'Positive Hit')],
            'Null-Mapping': [_hit('GENE2', '1', '0.1', 'Positive Hit')],
            'No-Mapping': [_hit('GENE3', '1', '0.1', 'Positive Hit')],
            'No-Hits': [_hit('GENE4', '1', '0.1', 'Non-Hit'), _hit('GENE5', '1', '0.1', '')],
        },
        mapping=[
            ['Mapped', 'x', 'MONDO_1', ''],
            ['Null-Mapping', 'x', '', ''],
            ['No-Hits', 'x', 'MONDO_4', ''],
        ],
    )

    raw = generate(**sources).collect()

    assert raw.select('studyId', 'targetFromSourceId', 'diseaseFromSourceMappedId').rows() == [
        ('Mapped', 'GENE1', 'MONDO_1'),
    ]


def test_generate_takes_the_study_id_from_the_file_name(tmp_path: Path) -> None:
    sources = _write_inputs(
        tmp_path,
        screens=[_screen('iTF Microglia-Day 8'), _screen('Encoded Name')],
        studies={
            'iTF Microglia-Day 8': [_hit('GENE1', '1', '0.1', 'Positive Hit')],
            'Encoded%20Name': [_hit('GENE2', '1', '0.1', 'Positive Hit')],
        },
        mapping=[['iTF Microglia-Day 8', 'x', 'MONDO_1', ''], ['Encoded Name', 'x', 'MONDO_2', '']],
    )

    raw = generate(**sources).collect()

    assert raw.select('studyId', 'targetFromSourceId').sort('studyId').rows() == [
        ('Encoded Name', 'GENE2'),
        ('iTF Microglia-Day 8', 'GENE1'),
    ]


def test_generate_reads_study_columns_by_position(tmp_path: Path) -> None:
    sources = _write_inputs(
        tmp_path,
        screens=[_screen('RNA-Seq')],
        studies={},
        mapping=[['RNA-Seq', 'x', 'MONDO_1', '']],
    )
    _write_study(
        tmp_path / 'studies',
        'RNA-Seq',
        RNA_SEQ_HEADER,
        [
            ['GENE1', '5.0', '-0.5', '0.03', '0.1', 'Negative Hit', '1.5'],
            ['GENE2', '5.0', '1', '0.03', '0.1', 'GENE2', '1'],
        ],
    )

    raw = generate(**sources).collect()

    assert raw.select('targetFromSourceId', 'log2FoldChangeValue', 'statisticalTestTail').rows() == [
        ('GENE1', -0.5, 'lower tail'),
    ]
    assert raw['resourceScore'].to_list() == [float(np.float32(0.03))]


def test_generate_casts_unparseable_numbers_to_null(tmp_path: Path) -> None:
    sources = _write_inputs(
        tmp_path,
        screens=[_screen('S')],
        studies={'S': [_hit('GENE1', 'abc', ' 0.5 ', 'Positive Hit'), _hit('GENE2', '1', 'NA', 'Negative Hit')]},
        mapping=[['S', 'x', 'MONDO_1', '']],
    )

    raw = generate(**sources).collect().sort('targetFromSourceId')

    assert raw.select('targetFromSourceId', 'resourceScore', 'log2FoldChangeValue').rows() == [
        ('GENE1', 0.5, None),
        ('GENE2', None, 1.0),
    ]


def test_parsing_experiment_splits_title_experiment_and_analysis() -> None:
    description = (
        '## iAstrocyte-LAMP1-Veh-CRISPRi\n\n'
        'CRISPRi FACS screen for cell-surface LAMP1 (marker of lysosome\n'
        'exocytosis) in human iPSC-derived astrocytes.\n\n'
        '## Experiment\n\n'
        'To screen against the **druggable** genome or genes involved in vesicular\n'
        'trafficking, we used the H1 and H4 sub-libraries.\n\n'
        '## Analysis\n\n'
        'We analyzed the data from the pooled CRISPRi screens using the\n'
        'MAGeCK-iNC bioinformatic pipeline previously described.\n'
    )
    assert parsing_experiment(description) == {
        'title': 'iAstrocyte-LAMP1-Veh-CRISPRi',
        'experiment': 'To screen against the druggable genome or genes involved in vesicular',
        'analysis': 'We analyzed the data from the pooled CRISPRi screens using the',
    }


@pytest.mark.parametrize(
    ('description', 'expected'),
    [
        (None, {'title': '', 'experiment': None, 'analysis': None}),
        ('', {'title': '', 'experiment': None, 'analysis': None}),
        ('Description', {'title': 'Description', 'experiment': None, 'analysis': None}),
        ('## T\n## Experiment', {'title': 'T', 'experiment': None, 'analysis': None}),
        ('# Té\n## Analysis\nAé', {'title': 'T', 'experiment': None, 'analysis': 'A'}),
    ],
)
def test_parsing_experiment_edge_cases(description: str | None, expected: dict) -> None:
    assert parsing_experiment(description) == expected


@pytest.mark.parametrize(
    ('resource_score', 'expected'),
    [(0.5, 0.0), (0.005, 1.0), (0.0, 1.0), (-1.0, 1.0), (None, None), (nan, None)],
)
def test_score_matches_spark(resource_score: float | None, expected: float | None) -> None:
    # out_min=0.0, weak_ref=0.5, strong_ref=0.005 -- measured against spark.
    df = pl.DataFrame({'resourceScore': [resource_score]}, schema={'resourceScore': pl.Float64})
    got = df.select(CRISPR_SCREEN.score.alias('out'))['out'][0]
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_crispr_screen_unique_fields_are_the_config_ones() -> None:
    assert CRISPR_SCREEN.datasource_id == 'crispr_screen'
    assert CRISPR_SCREEN.unique_fields == ['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'studyId']


def test_crispr_screens_step_scores_and_splits(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    sources = _sources(src)
    dst = tmp_path / 'out'

    crispr_screens(
        {
            **sources,
            'disease': write_disease(src / 'disease', ['MONDO_1', 'MONDO_3']),
            'target': write_target(
                src / 'target', ['ENSG1', 'ENSG2', 'ENSG4'], {'ENSG1': 'GENE1', 'ENSG2': 'GENE2', 'ENSG4': 'GENE4'}
            ),
            'literature': write_literature(src / 'literature', {'30297964': '2018-10-08'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid').sort('targetId')
    failed = pl.read_parquet(dst / 'failed').sort('targetId')
    assert valid.select('targetId', 'diseaseId', 'studyId').rows() == [
        ('ENSG1', 'MONDO_1', 'iAstrocyte-A'),
        ('ENSG2', 'MONDO_1', 'iAstrocyte-A'),
        ('ENSG4', 'MONDO_3', 'Neuron B'),
    ]
    assert valid['score'].to_list() == [
        pytest.approx(np.log10(0.01 / 0.5) / np.log10(0.01), rel=1e-6),
        pytest.approx(np.log10(float(np.float32(0.2)) / 0.5) / np.log10(0.01)),
        1.0,
    ]
    assert valid['publicationDate'].to_list() == ['2018-10-08', '2018-10-08', None]
    assert failed['diseaseFromSourceMappedId'].to_list() == ['MONDO_2', 'MONDO_2']
