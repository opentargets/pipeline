import polars as pl
import pytest

from pts.transformers.clinical_mira.moa.benchmark import (
    benchmark_targets,
    link_pools,
    precision_by_grounding,
    score,
    summarise,
    wilson,
)

DRUGS = pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String}))


def _report(rows: list[tuple[str, list[str | None], str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'id': [r[0] for r in rows],
            'source': ['ClinicalTrials.gov'] * len(rows),
            'drugs': [[{'drugFromSource': 'x', 'drugId': d} for d in r[1]] for r in rows],
            'clinicalStage': [r[2] for r in rows],
        },
        schema={'id': pl.String, 'source': pl.String, 'drugs': DRUGS, 'clinicalStage': pl.String},
    )


MOLECULE = pl.DataFrame({
    'id': ['CHEMBL1', 'CHEMBL2'],
    'maximumClinicalStage': ['PHASE_1', 'APPROVAL'],
    'drugType': ['Small molecule', 'Antibody'],
})


def test_link_majority_over_80_percent():
    pools = pl.DataFrame({'pool_id': ['p1'], 'nct_ids': [['n1', 'n2', 'n3', 'n4', 'n5']]})
    report = _report([
        ('n1', ['CHEMBL1'], 'PHASE_1'),
        ('n2', ['CHEMBL1'], 'PHASE_1'),
        ('n3', ['CHEMBL1'], 'PHASE_1'),
        ('n4', ['CHEMBL1'], 'PHASE_1'),
        ('n5', ['CHEMBL2'], 'PHASE_1'),
    ])

    row = link_pools(pools, report, MOLECULE).row(0, named=True)

    assert row['drugId'] == 'CHEMBL1'
    assert row['link'] == 'linked'
    assert row['maximumClinicalStage'] == 'PHASE_1'
    assert row['drugType'] == 'Small molecule'
    assert row['phase_group'] == 'early'


def test_link_split_is_ambiguous_and_falls_back_to_trial_phase():
    pools = pl.DataFrame({'pool_id': ['p1'], 'nct_ids': [['n1', 'n2', 'n3', 'n4', 'n5']]})
    report = _report([
        ('n1', ['CHEMBL1'], 'PHASE_1'),
        ('n2', ['CHEMBL1'], 'PHASE_2'),
        ('n3', ['CHEMBL1'], 'PHASE_1'),
        ('n4', ['CHEMBL2'], 'PHASE_4'),
        ('n5', ['CHEMBL2'], 'UNKNOWN'),
    ])

    row = link_pools(pools, report, MOLECULE).row(0, named=True)

    assert row['link'] == 'ambiguous'
    assert row['drugId'] is None
    assert row['maximumClinicalStage'] == 'APPROVAL'
    assert row['phase_group'] == 'late'


def test_multi_drug_or_unmapped_trials_leave_pool_unlinked():
    pools = pl.DataFrame({'pool_id': ['p1'], 'nct_ids': [['n1', 'n2']]})
    report = _report([('n1', ['CHEMBL1', 'CHEMBL2'], 'PHASE_2'), ('n2', [None], 'PHASE_1')])

    row = link_pools(pools, report, MOLECULE).row(0, named=True)

    assert row['link'] == 'unlinked'
    assert row['maximumClinicalStage'] == 'PHASE_2'
    assert row['phase_group'] == 'early'


def _moa() -> pl.DataFrame:
    return pl.DataFrame({
        'chemblIds': [['CHEMBL1'], ['CHEMBL1'], ['CHEMBL3']],
        'actionType': ['INHIBITOR', 'ANTAGONIST', 'INHIBITOR'],
        'targets': [['G1', 'G2'], ['G5'], []],
    })


def test_benchmark_keeps_drugs_with_human_targets():
    bench = benchmark_targets(_moa())

    assert bench['drugId'].to_list() == ['CHEMBL1']
    assert sorted(bench['targetIds'].to_list()[0]) == ['G1', 'G2', 'G5']


LINKS = pl.DataFrame({
    'pool_id': ['p1', 'p2'],
    'drugId': ['CHEMBL1', None],
    'link': ['linked', 'unlinked'],
    'maximumClinicalStage': ['PHASE_1', 'PHASE_2'],
    'drugType': ['Small molecule', 'Antibody'],
    'phase_group': ['early', 'early'],
})


def _mechanisms(rows: list[dict]) -> pl.DataFrame:
    base = {'kept': True, 'isHumanTarget': True, 'grounding': 'source', 'class_level': False, 'actionType': 'INHIBITOR'}
    return pl.DataFrame(
        [base | r for r in rows],
        schema={
            'pool_id': pl.String,
            'mechanism_idx': pl.UInt32,
            'targetIds': pl.List(pl.String),
            'kept': pl.Boolean,
            'isHumanTarget': pl.Boolean,
            'grounding': pl.String,
            'class_level': pl.Boolean,
            'actionType': pl.String,
        },
    )


def test_overlap_with_any_moa_gene_is_correct():
    scored = score(
        _mechanisms([
            {'pool_id': 'p1', 'mechanism_idx': 0, 'targetIds': ['G2']},
            {'pool_id': 'p1', 'mechanism_idx': 1, 'targetIds': ['G9'], 'actionType': 'AGONIST'},
            {'pool_id': 'p1', 'mechanism_idx': 2, 'targetIds': ['G5'], 'actionType': 'INHIBITOR'},
        ]),
        LINKS,
        benchmark_targets(_moa()),
    )

    assert scored['correct'].to_list() == [True, False, True]
    assert scored['action_match'].to_list() == [True, None, False]


def test_class_level_and_unbenchmarked_excluded_from_headline():
    scored = score(
        _mechanisms([
            {'pool_id': 'p1', 'mechanism_idx': 0, 'targetIds': ['G1'], 'class_level': True},
            {'pool_id': 'p1', 'mechanism_idx': 1, 'targetIds': [], 'grounding': 'unresolved'},
            {'pool_id': 'p2', 'mechanism_idx': 0, 'targetIds': ['G1']},
        ]),
        LINKS,
        benchmark_targets(_moa()),
    )

    assert scored['in_headline'].to_list() == [False, False, False]
    assert scored['benchmarked'].to_list() == [True, True, False]
    assert scored['correct'].to_list() == [None, None, None]


def test_wilson_bounds():
    low, high = wilson(9, 10)

    assert low == pytest.approx(0.596, abs=1e-3)
    assert high == pytest.approx(0.982, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)


def test_summary_precision_by_phase_group():
    links = pl.DataFrame({
        'pool_id': [f'p{i}' for i in range(12)],
        'drugId': [f'CHEMBL{i}' for i in range(12)],
        'link': ['linked'] * 12,
        'maximumClinicalStage': ['PHASE_1'] * 12,
        'drugType': ['Small molecule'] * 12,
        'phase_group': ['early'] * 12,
    })
    moa = pl.DataFrame({
        'chemblIds': [[f'CHEMBL{i}'] for i in range(12)],
        'actionType': ['INHIBITOR'] * 12,
        'targets': [['G1']] * 12,
    })
    mechanisms = _mechanisms([
        {'pool_id': f'p{i}', 'mechanism_idx': 0, 'targetIds': ['G1'] if i < 9 else ['G9']} for i in range(10)
    ])
    statuses = pl.DataFrame({
        'pool_id': [f'p{i}' for i in range(12)],
        'extracted': [True] * 12,
        'not_stated': [False] * 10 + [True] * 2,
    })
    bench = benchmark_targets(moa)

    row = summarise(score(mechanisms, links, bench), links, statuses, bench, by=['phase_group']).row(0, named=True)

    assert row['pools'] == 12
    assert row['scored'] == 10
    assert row['correct'] == 9
    assert row['precision'] == pytest.approx(0.9)
    assert row['drug_precision'] == pytest.approx(0.9)
    assert row['recall'] == pytest.approx(9 / 12)
    assert row['not_stated_rate'] == pytest.approx(2 / 12)
    assert row['action_agreement'] == pytest.approx(1.0)


def test_precision_by_grounding_separates_source_and_suggestion():
    scored = pl.DataFrame({
        'grounding': ['source', 'source', 'suggestion', 'suggestion', 'unresolved'],
        'in_headline': [True, True, True, True, False],
        'correct': [True, True, True, False, None],
    })

    table = precision_by_grounding(scored).sort('grounding')

    assert table.select('grounding', 'scored', 'correct').rows() == [('source', 2, 2), ('suggestion', 2, 1)]
    assert table['precision'].to_list() == [1.0, 0.5]
