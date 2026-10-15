"""Tests for intOGen evidence."""

from __future__ import annotations

from math import nan
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.intogen import (
    DIRECTION_ON_TARGET,
    DIRECTION_ON_TRAIT,
    INTOGEN,
    generate,
    intogen,
    parse_source,
)
from tests.evidence.fixtures import write_disease, write_literature, write_target

GENES_HEADER = ['SYMBOL', 'COHORT', 'CANCER_TYPE', 'METHODS', 'MUTATIONS', 'SAMPLES', 'QVALUE_COMBINATION', 'ROLE']
COHORTS_HEADER = ['COHORT', 'CANCER', 'CANCER_NAME', 'REFERENCE', 'SAMPLES', 'MUTATIONS', 'COHORT_NICK', 'COHORT_NAME']
MAPPING_HEADER = ['Cancer_type_acronym', 'Cancer_type_name', 'EFO_id', 'EFO_name', 'Source']

RAW_SCHEMA = {
    'datatypeId': pl.String,
    'datasourceId': pl.String,
    'targetFromSourceId': pl.String,
    'cohortId': pl.String,
    'cohortShortName': pl.String,
    'cohortDescription': pl.String,
    'significantDriverMethods': pl.List(pl.String),
    'mutatedSamples': pl.List(
        pl.Struct({
            'functionalConsequenceId': pl.String,
            'numberSamplesTested': pl.Float64,
            'numberMutatedSamples': pl.Float64,
        })
    ),
    'diseaseFromSource': pl.String,
    'resourceScore': pl.Float64,
    'literature': pl.List(pl.String),
    'diseaseFromSourceMappedId': pl.String,
}


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join('\t'.join(row) for row in [header, *rows]) + '\n')
    return str(path)


def _write_inputs(src: Path, genes: list[list[str]], cohorts: list[list[str]], mapping: list[list[str]]) -> dict:
    return {
        'genes': _write_tsv(src / 'genes.tsv', GENES_HEADER, genes),
        'cohorts': _write_tsv(src / 'cohorts.tsv', COHORTS_HEADER, cohorts),
        'disease_mapping': _write_tsv(src / 'mapping.tsv', MAPPING_HEADER, mapping),
    }


def _sources(src: Path) -> dict:
    return _write_inputs(
        src,
        genes=[
            ['A1', 'C1', 'BRCA', 'clustl,hotmaps', '9', '4', '1e-5', 'Act'],
            ['B1', 'C1', 'BRCA', 'combination', '3', '2.0', '0.5', 'LoF'],
            ['A1', 'C2', 'XX', 'combination', '1', '7', '0.0', 'ambiguous'],
            ['B1', 'NOPE', 'BRCA', 'combination', '1', '1', '0.01', 'LoF'],
        ],
        cohorts=[
            ['C1', 'BRCA', 'Breast cancer', 'DOI: 10.1038/ng.2529', '100', '5', 'C1_NICK', 'Breast cohort'],
            ['C2', 'XX', 'Unmapped cancer', 'WEB: https://example.org', '10', '1', 'C2_NICK', 'Other cohort'],
        ],
        mapping=[['BRCA', 'Breast cancer', ' EFO_1 ', 'breast carcinoma', 'intOGen']],
    )


@pytest.mark.parametrize(
    ('reference', 'expected'),
    [
        ('PMID: 23334666', ['23334666']),
        ('DOI: 10.1038/ng.2529', ['23334666']),
        ('DOI:10.1101/162784 ', ['32025007']),
        ('DOI: 10.1038/unmapped', ['10.1038/unmapped']),
        ('DOI: 10.1038/NG.2529', ['10.1038/NG.2529']),
        ('WEB: https://example.org', None),
        ('web: x', ['x']),
        ('no colon', None),
        ('a: b: c', ['b']),
        ('empty:', ['']),
        ('', None),
        (None, None),
        ('PMID:\t1\t', ['\t1\t']),
    ],
)
def test_parse_source_matches_spark(reference: str | None, expected: list[str] | None) -> None:
    df = pl.DataFrame({'REFERENCE': [reference]}, schema={'REFERENCE': pl.String})
    assert df.select(parse_source(pl.col('REFERENCE')).alias('out'))['out'].to_list() == [expected]


def test_generate_joins_genes_to_cohorts_and_maps_diseases(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path)).collect()

    assert raw.schema == pl.Schema(RAW_SCHEMA)
    assert raw.sort('targetFromSourceId', 'cohortId').to_dicts() == [
        {
            'datatypeId': 'somatic_mutation',
            'datasourceId': 'intogen',
            'targetFromSourceId': 'A1',
            'cohortId': 'C1',
            'cohortShortName': 'C1_NICK',
            'cohortDescription': 'Breast cohort',
            'significantDriverMethods': ['clustl', 'hotmaps'],
            'mutatedSamples': [
                {'functionalConsequenceId': 'SO_0002053', 'numberSamplesTested': 4.0, 'numberMutatedSamples': 4.0}
            ],
            'diseaseFromSource': 'Breast cancer',
            'resourceScore': 1e-5,
            'literature': ['23334666'],
            'diseaseFromSourceMappedId': 'EFO_1',
        },
        {
            'datatypeId': 'somatic_mutation',
            'datasourceId': 'intogen',
            'targetFromSourceId': 'A1',
            'cohortId': 'C2',
            'cohortShortName': 'C2_NICK',
            'cohortDescription': 'Other cohort',
            'significantDriverMethods': ['combination'],
            'mutatedSamples': [
                {'functionalConsequenceId': None, 'numberSamplesTested': 7.0, 'numberMutatedSamples': 7.0}
            ],
            'diseaseFromSource': 'Unmapped cancer',
            'resourceScore': 0.0,
            'literature': None,
            'diseaseFromSourceMappedId': None,
        },
        {
            'datatypeId': 'somatic_mutation',
            'datasourceId': 'intogen',
            'targetFromSourceId': 'B1',
            'cohortId': 'C1',
            'cohortShortName': 'C1_NICK',
            'cohortDescription': 'Breast cohort',
            'significantDriverMethods': ['combination'],
            'mutatedSamples': [
                {'functionalConsequenceId': 'SO_0002054', 'numberSamplesTested': 2.0, 'numberMutatedSamples': 2.0}
            ],
            'diseaseFromSource': 'Breast cancer',
            'resourceScore': 0.5,
            'literature': ['23334666'],
            'diseaseFromSourceMappedId': 'EFO_1',
        },
    ]


def test_generate_keeps_one_mutated_samples_element_when_its_fields_are_empty(tmp_path: Path) -> None:
    sources = _write_inputs(
        tmp_path,
        genes=[['A1', 'C1', 'BRCA', '', '', '', '', '']],
        cohorts=[['C1', 'BRCA', 'Breast cancer', '', '', '', 'C1_NICK', 'Breast cohort']],
        mapping=[],
    )

    raw = generate(**sources).collect()

    assert raw.select('significantDriverMethods', 'mutatedSamples', 'resourceScore', 'literature').row(0) == (
        None,
        [{'functionalConsequenceId': None, 'numberSamplesTested': None, 'numberMutatedSamples': None}],
        None,
        None,
    )


def test_generate_keeps_every_mapping_of_a_cancer_type(tmp_path: Path) -> None:
    sources = _write_inputs(
        tmp_path,
        genes=[['A1', 'C1', 'BRCA', 'combination', '1', '1', '0.1', 'Act']],
        cohorts=[['C1', 'BRCA', 'Breast cancer', 'PMID: 1', '1', '1', 'C1_NICK', 'Breast cohort']],
        mapping=[['BRCA', 'x', 'EFO_1', 'x', 'x'], ['BRCA', 'x', 'EFO_2', 'x', 'x']],
    )

    raw = generate(**sources).collect()

    assert raw['diseaseFromSourceMappedId'].to_list() == ['EFO_1', 'EFO_2']


@pytest.mark.parametrize(
    ('resource_score', 'expected'),
    [(0.1, 0.25), (1e-10, 1.0), (0.0, 1.0), (None, None), (nan, None)],
)
def test_score_matches_spark(resource_score: float | None, expected: float | None) -> None:
    df = pl.DataFrame({'resourceScore': [resource_score]}, schema={'resourceScore': pl.Float64})
    got = df.select(INTOGEN.score.alias('out'))['out'][0]
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_direction_on_trait_is_risk_when_the_disease_is_present() -> None:
    df = pl.DataFrame({'diseaseId': ['MONDO:1', None]})
    assert df.select(DIRECTION_ON_TRAIT.alias('d'))['d'].to_list() == ['risk', None]


def test_intogen_direction_on_target_needs_a_matching_consequence() -> None:
    df = pl.DataFrame({
        'mutatedSamples': [
            [{'functionalConsequenceId': 'SO_0002054'}],
            [{'functionalConsequenceId': 'SO_0002053'}],
            [{'functionalConsequenceId': None}],
            None,
            [],
        ],
    })
    got = df.with_columns(DIRECTION_ON_TARGET.alias('d'))['d'].to_list()
    assert got == ['LoF', 'GoF', None, None, None]


def test_intogen_step_scores_and_assigns_directions(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    sources = _sources(src)
    dst = tmp_path / 'out'

    intogen(
        {
            **sources,
            'disease': write_disease(src / 'disease', ['EFO_1']),
            'target': write_target(src / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'A1', 'ENSG2': 'B1'}),
            'literature': write_literature(src / 'literature', {'23334666': '2013-01-20'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid').sort('targetId')
    failed = pl.read_parquet(dst / 'failed')
    assert valid.select('targetId', 'diseaseId', 'directionOnTrait', 'directionOnTarget').rows() == [
        ('ENSG1', 'EFO_1', 'risk', 'GoF'),
        ('ENSG2', 'EFO_1', 'risk', 'LoF'),
    ]
    assert valid['score'].to_list() == [pytest.approx(0.25 + 0.75 * 4 / 9), 0.25]
    assert valid['publicationDate'].to_list() == ['2013-01-20', '2013-01-20']
    assert failed['cohortId'].to_list() == ['C2']
