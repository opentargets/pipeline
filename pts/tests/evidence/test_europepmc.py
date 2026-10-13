"""Tests for Europe PMC literature evidence."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.transformers.evidence.europepmc import EUROPEPMC, compute_evidence, europepmc, generate
from tests.evidence.fixtures import write_disease, write_literature, write_target

nan = float('nan')

COOCCURRENCE_SCHEMA = {
    'pmid': pl.String,
    'pmcid': pl.String,
    'section': pl.String,
    'type': pl.String,
    'isMapped': pl.Boolean,
    'text': pl.String,
    'label1': pl.String,
    'mappedId1': pl.String,
    'mappedId2': pl.String,
    'start1': pl.Int64,
    'end1': pl.Int64,
    'start2': pl.Int64,
    'end2': pl.Int64,
    'evidenceScore': pl.Float64,
    'year': pl.Int32,
}


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        'pmid': '12345',
        'pmcid': 'PMC001',
        'section': 'abstract',
        'type': 'GP-DS',
        'isMapped': True,
        'text': 'short text',
        'label1': 'BRCA1',
        'mappedId1': 'ENSG001',
        'mappedId2': 'EFO_0000311',
        'start1': 0,
        'end1': 5,
        'start2': 10,
        'end2': 15,
        'evidenceScore': 0.8,
        'year': 2020,
    }
    return row | overrides


def _evidence(*rows: dict[str, Any]) -> pl.DataFrame:
    result = compute_evidence(pl.LazyFrame(list(rows), schema=COOCCURRENCE_SCHEMA))
    assert isinstance(result, pl.LazyFrame)
    return result.collect()


# score


def test_score_divides_resource_score_by_100_capped_at_one() -> None:
    df = pl.DataFrame({'resourceScore': [50.0, 150.0, 0.0, None, nan]})
    got = df.select(EUROPEPMC.score.alias('s'))['s'].to_list()
    assert got == pytest.approx([0.5, 1.0, 0.0, 1.0, 1.0])


def test_score_divides_exactly() -> None:
    rng = random.Random(0)
    values = [rng.uniform(0.0, 100.0) for _ in range(5000)]
    got = pl.DataFrame({'resourceScore': values}).select(EUROPEPMC.score.alias('s'))['s'].to_list()
    assert got == [min(v / 100, 1.0) for v in values]


# filtering


def test_keeps_only_gene_disease_cooccurrences() -> None:
    result = _evidence(
        _row(evidenceScore=1.5),
        _row(pmid='99999', mappedId1='ENSG002', evidenceScore=1.5),
        _row(pmid='99998', mappedId1='ENSG003', type='DS-CD', evidenceScore=1.5),
    )
    assert sorted(result['targetFromSourceId']) == ['ENSG001', 'ENSG002']


def test_drops_excluded_target_terms() -> None:
    result = _evidence(
        _row(label1='TEC', evidenceScore=1.5),
        _row(label1='BRCA1', pmid='99999', mappedId1='ENSG002', evidenceScore=1.5),
        _row(label1='BRCA1', pmid='99998', mappedId1='ENSG003', evidenceScore=1.5),
    )
    assert sorted(result['targetFromSourceId']) == ['ENSG002', 'ENSG003']


def test_drops_text_of_600_characters_or_more() -> None:
    result = _evidence(
        _row(text='x' * 600, evidenceScore=1.5),
        _row(text='é' * 599, pmid='99999', mappedId1='ENSG002', evidenceScore=1.5),
        _row(text='short2', pmid='99998', mappedId1='ENSG003', evidenceScore=1.5),
    )
    assert sorted(result['targetFromSourceId']) == ['ENSG002', 'ENSG003']


def test_drops_unmapped() -> None:
    result = _evidence(
        _row(isMapped=False, evidenceScore=1.5),
        _row(isMapped=True, pmid='99999', mappedId1='ENSG002', evidenceScore=1.5),
        _row(isMapped=True, pmid='99998', mappedId1='ENSG003', evidenceScore=1.5),
    )
    assert result.height == 2


def test_drops_sections_not_of_interest() -> None:
    result = _evidence(
        _row(section='methods', evidenceScore=1.5),
        _row(section='concl', pmid='99999', mappedId1='ENSG002', evidenceScore=1.5),
    )
    assert result['targetFromSourceId'].to_list() == ['ENSG002']


def test_drops_null_filter_columns_as_a_spark_filter_does() -> None:
    result = _evidence(
        _row(isMapped=None, evidenceScore=1.5),
        _row(label1=None, pmid='2', evidenceScore=1.5),
        _row(text=None, pmid='3', evidenceScore=1.5),
        _row(type=None, pmid='4', evidenceScore=1.5),
        _row(section=None, pmid='5', evidenceScore=1.5),
        _row(pmid=None, pmcid=None, evidenceScore=1.5),
        _row(pmid='6', evidenceScore=1.5),
    )
    assert result['literature'].to_list() == [['6']]


def test_keeps_only_resource_score_above_one() -> None:
    result = _evidence(
        _row(evidenceScore=0.5),
        _row(evidenceScore=0.5, text='other'),
        _row(pmid='2', evidenceScore=1.0),
        _row(pmid='3', evidenceScore=None),
        _row(pmid='4', evidenceScore=1.0000001),
    )
    assert result['literature'].to_list() == [['4']]


# aggregation


def test_aggregates_by_publication_and_sums_the_score() -> None:
    result = _evidence(
        _row(section='title', evidenceScore=1.0, text='sentence 1'),
        _row(section='abstract', evidenceScore=1.0, text='sentence 2'),
    )
    assert result.height == 1
    assert result['resourceScore'].to_list() == pytest.approx([2.0])


def test_sum_skips_a_null_score() -> None:
    result = _evidence(
        _row(evidenceScore=1.5),
        _row(evidenceScore=None, text='other'),
        _row(evidenceScore=0.25, text='x'),
    )
    assert result['resourceScore'].to_list() == [1.75]


def test_groups_by_year_too() -> None:
    result = _evidence(_row(evidenceScore=1.5, year=2020), _row(evidenceScore=1.5, year=None))
    assert sorted(result['publicationYear'].to_list(), key=str) == [2020, None]


def test_output_columns_and_dtypes() -> None:
    result = _evidence(_row(evidenceScore=1.5), _row(evidenceScore=1.0, text='another'))
    assert result.schema == pl.Schema({
        'datasourceId': pl.String,
        'datatypeId': pl.String,
        'targetFromSourceId': pl.String,
        'diseaseFromSourceMappedId': pl.String,
        'resourceScore': pl.Float64,
        'literature': pl.List(pl.String),
        'textMiningSentences': pl.List(
            pl.Struct({
                'text': pl.String,
                'tStart': pl.Int64,
                'tEnd': pl.Int64,
                'dStart': pl.Int64,
                'dEnd': pl.Int64,
                'section': pl.String,
            })
        ),
        'pmcIds': pl.List(pl.String),
        'publicationYear': pl.Int32,
    })
    row = result.row(0, named=True)
    assert row['datasourceId'] == 'europepmc'
    assert row['datatypeId'] == 'literature'
    assert row['diseaseFromSourceMappedId'] == 'EFO_0000311'


def test_text_mining_sentences_are_a_sorted_set() -> None:
    result = _evidence(
        _row(evidenceScore=1.0, text='s2', start1=20, end1=25, start2=30, end2=35),
        _row(evidenceScore=1.5, text='s1', start1=0, end1=5, start2=10, end2=15),
        _row(evidenceScore=0.5, text='s2', start1=20, end1=25, start2=30, end2=35),
    )
    assert result['textMiningSentences'].to_list() == [
        [
            {'text': 's1', 'tStart': 0, 'tEnd': 5, 'dStart': 10, 'dEnd': 15, 'section': 'abstract'},
            {'text': 's2', 'tStart': 20, 'tEnd': 25, 'dStart': 30, 'dEnd': 35, 'section': 'abstract'},
        ]
    ]
    assert result['resourceScore'].to_list() == pytest.approx([3.0])


def test_falls_back_to_pmcid_and_nulls_an_empty_pmcid_set() -> None:
    result = _evidence(
        _row(pmid=None, pmcid='PMC9', evidenceScore=1.5),
        _row(pmid='7', pmcid=None, mappedId1='ENSG002', evidenceScore=1.5),
    ).sort('targetFromSourceId')
    assert result['literature'].to_list() == [[], ['7']]
    assert result['pmcIds'].to_list() == [['PMC9'], None]


def test_trims_spaces_but_not_other_whitespace_from_pmid() -> None:
    result = _evidence(
        _row(pmid=' 7 ', evidenceScore=1.5),
        _row(pmid='7', evidenceScore=1.5, text='other'),
        _row(pmid='\t8', mappedId1='ENSG002', evidenceScore=1.5),
    ).sort('targetFromSourceId')
    assert result['literature'].to_list() == [['7'], ['\t8']]
    assert result['resourceScore'].to_list() == pytest.approx([3.0, 1.5])


def test_score_is_summed_in_ascending_order_whatever_the_row_order() -> None:
    rng = random.Random(0)
    scores = [rng.uniform(0.0, 1.0) * 10 ** rng.randint(-8, 3) for _ in range(50)]
    expected = 0.0
    for score in sorted(scores):
        expected += score
    rows = [_row(text=f't{i}', evidenceScore=s) for i, s in enumerate(scores)]
    for _ in range(5):
        rng.shuffle(rows)
        assert _evidence(*rows)['resourceScore'].to_list() == [expected]


def test_generate_reads_the_cooccurrence_dataset(tmp_path: Path) -> None:
    (tmp_path / 'cooccurrence').mkdir()
    pl.DataFrame([_row(evidenceScore=1.5)], schema=COOCCURRENCE_SCHEMA).write_parquet(
        tmp_path / 'cooccurrence' / 'part.parquet'
    )
    result = generate(str(tmp_path / 'cooccurrence'))
    assert isinstance(result, pl.LazyFrame)
    assert result.collect()['targetFromSourceId'].to_list() == ['ENSG001']


# step


def test_step_generates_scores_and_splits(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    (src / 'cooccurrence').mkdir(parents=True)
    pl.DataFrame(
        [
            _row(pmid='1', evidenceScore=150.0),
            _row(pmid='1', mappedId2='EFO_9', evidenceScore=50.0),
        ],
        schema=COOCCURRENCE_SCHEMA,
    ).write_parquet(src / 'cooccurrence' / 'part.parquet')
    dst = tmp_path / 'out'

    europepmc(
        {
            'cooccurrence': str(src / 'cooccurrence'),
            'disease': write_disease(src / 'disease', ['EFO_0000311']),
            'target': write_target(src / 'target', ['ENSG001']),
            'literature': write_literature(src / 'literature', {'1': '2020-01-01'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid')
    failed = pl.read_parquet(dst / 'failed')
    assert valid['diseaseId'].to_list() == ['EFO_0000311']
    assert valid['score'].to_list() == [1.0]
    assert valid['publicationDate'].to_list() == ['2020-01-01']
    assert failed['diseaseFromSourceMappedId'].to_list() == ['EFO_9']
