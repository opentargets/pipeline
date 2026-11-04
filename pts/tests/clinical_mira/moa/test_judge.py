import polars as pl
import pytest

from pts.transformers.clinical_mira.moa.benchmark import summarise
from pts.transformers.clinical_mira.moa.judge import (
    adjust,
    calibration,
    handcheck,
    judge_input,
    request_id,
    select_for_judging,
)


def _scored(rows: list[dict]) -> pl.DataFrame:
    base = {
        'kept': True,
        'isHumanTarget': True,
        'grounding': 'source',
        'class_level': False,
        'benchmarked': True,
        'in_headline': True,
        'correct': True,
        'targetFromSource': 'T',
        'evidence': [{'nctId': 'n1', 'quote': 'q T'}],
    }
    return pl.DataFrame(
        [base | r for r in rows],
        schema={
            'pool_id': pl.String,
            'mechanism_idx': pl.UInt32,
            'kept': pl.Boolean,
            'isHumanTarget': pl.Boolean,
            'grounding': pl.String,
            'class_level': pl.Boolean,
            'benchmarked': pl.Boolean,
            'in_headline': pl.Boolean,
            'correct': pl.Boolean,
            'targetFromSource': pl.String,
            'evidence': pl.List(pl.Struct({'nctId': pl.String, 'quote': pl.String})),
        },
    )


def _keys(df: pl.DataFrame) -> list[tuple[str, int]]:
    return sorted(df.select('pool_id', 'mechanism_idx').iter_rows())


def test_selection_includes_all_misses_and_unbenchmarked():
    scored = _scored([
        {'pool_id': 'miss', 'mechanism_idx': 0, 'correct': False},
        {'pool_id': 'free', 'mechanism_idx': 0, 'benchmarked': False, 'in_headline': False, 'correct': None},
        {'pool_id': 'drop', 'mechanism_idx': 0, 'benchmarked': False, 'in_headline': False, 'kept': False},
        {
            'pool_id': 'unres',
            'mechanism_idx': 0,
            'benchmarked': False,
            'in_headline': False,
            'grounding': 'unresolved',
            'correct': None,
        },
        {'pool_id': 'hit', 'mechanism_idx': 0},
    ])

    assert _keys(select_for_judging(scored, n_matches=0)) == [('free', 0), ('miss', 0)]


def test_match_sample_is_seeded_and_capped():
    scored = _scored([{'pool_id': f'p{i}', 'mechanism_idx': 0} for i in range(10)])

    first = select_for_judging(scored, n_matches=3, seed=1)
    second = select_for_judging(scored, n_matches=3, seed=1)

    assert first.height == 3
    assert _keys(first) == _keys(second)
    assert select_for_judging(scored, n_matches=50).height == 10


def _verdicts(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'pool_id': [r[0] for r in rows],
            'mechanism_idx': [0] * len(rows),
            'targetSupported': [r[1] for r in rows],
            'actionSupported': ['yes'] * len(rows),
            'primaryMechanism': [r[2] for r in rows],
            'rationale': ['because'] * len(rows),
        },
        schema_overrides={'mechanism_idx': pl.UInt32},
    )


def test_adjusted_flips_only_judge_supported_misses():
    scored = _scored([
        {'pool_id': 'a', 'mechanism_idx': 0, 'correct': False},
        {'pool_id': 'b', 'mechanism_idx': 0, 'correct': False},
        {'pool_id': 'c', 'mechanism_idx': 0, 'correct': True},
        {'pool_id': 'd', 'mechanism_idx': 0, 'correct': False},
    ])
    verdicts = _verdicts([('a', 'yes', 'yes'), ('b', 'yes', 'unclear'), ('c', 'no', 'no')])

    out = adjust(scored, verdicts).sort('pool_id')

    assert out['judge_correct'].to_list() == [True, False, False, None]
    assert out['adjusted_correct'].to_list() == [True, False, True, False]


def test_calibration_agreement():
    scored = _scored([
        {'pool_id': 'a', 'mechanism_idx': 0, 'correct': True},
        {'pool_id': 'b', 'mechanism_idx': 0, 'correct': True},
        {'pool_id': 'c', 'mechanism_idx': 0, 'correct': False},
        {'pool_id': 'd', 'mechanism_idx': 0, 'correct': False},
    ])
    verdicts = _verdicts([('a', 'yes', 'yes'), ('b', 'no', 'yes'), ('c', 'no', 'no'), ('d', 'no', 'unclear')])

    table = calibration(adjust(scored, verdicts))

    assert table['agreement'][0] == pytest.approx(0.75)
    assert table.filter(pl.col('correct') & ~pl.col('judge_correct'))['n'].to_list() == [1]


def test_judge_input_mentions_no_chembl_when_absent():
    row = {
        'targetFromSource': 'KRAS G12C',
        'targetType': 'single protein',
        'actionType': 'INHIBITOR',
        'mechanismOfAction': 'KRAS G12C inhibitor',
        'evidence': [{'nctId': 'n1', 'quote': 'a KRAS G12C inhibitor'}],
    }

    text = judge_input(row, 'POOL TEXT', [])

    assert 'TRIAL TEXTS\nPOOL TEXT' in text
    assert '[n1] "a KRAS G12C inhibitor"' in text
    assert 'CHEMBL MECHANISMS\nnone recorded' in text


def test_judge_input_lists_chembl_rows():
    row = {
        'targetFromSource': 'T',
        'targetType': 'single protein',
        'actionType': 'INHIBITOR',
        'mechanismOfAction': 'T inhibitor',
        'evidence': [],
    }

    text = judge_input(row, 'x', [{'mechanismOfAction': 'Foo inhibitor', 'actionType': 'INHIBITOR'}])

    assert '- Foo inhibitor (INHIBITOR)' in text


def test_request_id_round_trips():
    assert request_id('p1', 3) == 'p1:3'


def test_handcheck_is_seeded_sample_of_judged_rows():
    scored = _scored([{'pool_id': f'p{i}', 'mechanism_idx': 0} for i in range(5)])
    verdicts = _verdicts([(f'p{i}', 'yes', 'yes') for i in range(3)])
    pools = pl.DataFrame({'pool_id': [f'p{i}' for i in range(5)], 'drug_name': [f'd{i}' for i in range(5)]})

    sample = handcheck(adjust(scored, verdicts), pools, n=2, seed=0)

    assert sample.height == 2
    assert {'drug_name', 'targetFromSource', 'quotes', 'correct', 'targetSupported', 'rationale'} <= set(sample.columns)
    assert set(sample['drug_name']) <= {'d0', 'd1', 'd2'}


def test_adjusted_summary_counts_judged_unbenchmarked_mechanisms():
    scored = _scored([
        {'pool_id': 'hit', 'mechanism_idx': 0},
        {'pool_id': 'free', 'mechanism_idx': 0, 'benchmarked': False, 'in_headline': False, 'correct': None},
    ]).with_columns(action_match=pl.lit(None, dtype=pl.Boolean), actionType=pl.lit('INHIBITOR'))
    verdicts = _verdicts([('free', 'yes', 'yes')])
    links = pl.DataFrame({
        'pool_id': ['hit', 'free'],
        'drugId': ['CHEMBL1', None],
        'link': ['linked', 'unlinked'],
        'phase_group': ['early', 'early'],
    })
    statuses = pl.DataFrame({'pool_id': ['hit', 'free'], 'extracted': [True, True], 'not_stated': [False, False]})
    bench = pl.DataFrame({'drugId': ['CHEMBL1']})

    strict = summarise(scored, links, statuses, bench, by=['phase_group']).row(0, named=True)
    judged = summarise(adjust(scored, verdicts), links, statuses, bench, by=['phase_group'], adjusted=True)

    assert (strict['scored'], strict['correct']) == (1, 1)
    assert judged.select('scored', 'correct').row(0) == (2, 2)


def test_judge_input_is_blind_when_chembl_is_not_given():
    row = {
        'targetFromSource': 'T',
        'targetType': 'single protein',
        'actionType': 'INHIBITOR',
        'mechanismOfAction': 'T inhibitor',
        'evidence': [],
    }

    assert 'CHEMBL' not in judge_input(row, 'x', None)
