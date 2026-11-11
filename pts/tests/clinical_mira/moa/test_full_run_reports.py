import polars as pl

from pts.transformers.clinical_mira.moa.benchmark import precision_by_threshold
from pts.transformers.clinical_mira.moa.judge import handcheck, select_for_judging

EVIDENCE = pl.List(pl.Struct({'nctId': pl.String, 'quote': pl.String}))


def _scored(rows: list[dict]) -> pl.DataFrame:
    base = {
        'mechanism_idx': 0,
        'kept': True,
        'isHumanTarget': True,
        'grounding': 'source',
        'class_level': False,
        'eligible': True,
        'duplicate': False,
        'benchmarked': True,
        'in_headline': True,
        'correct': True,
        'confidence': 0.9,
        'targetFromSource': 'T',
        'evidence': [{'nctId': 'n', 'quote': 'q T'}],
    }
    return pl.DataFrame(
        [base | r for r in rows],
        schema_overrides={'mechanism_idx': pl.UInt32, 'evidence': EVIDENCE, 'correct': pl.Boolean},
    )


def _links(pool_ids: list[str], groups: list[str]) -> pl.DataFrame:
    return pl.DataFrame({'pool_id': pool_ids, 'phase_group': groups, 'maximumClinicalStage': groups})


def test_matches_for_calibration_are_spread_across_phases():
    pools = [f'e{i}' for i in range(50)] + [f'l{i}' for i in range(50)]
    scored = _scored([{'pool_id': p} for p in pools])
    links = _links(pools, ['early'] * 50 + ['late'] * 50)

    selected = select_for_judging(scored, n_matches=20, strata=links.select('pool_id', 'phase_group'))

    counts = selected.join(links, on='pool_id').group_by('phase_group').len()
    assert dict(counts.iter_rows()) == {'early': 10, 'late': 10}


def test_handcheck_samples_new_moa_predictions_per_phase():
    rows = [
        {'pool_id': f'p{i}', 'benchmarked': False, 'in_headline': False, 'correct': None, 'judge_correct': True}
        for i in range(30)
    ] + [{'pool_id': 'bench', 'judge_correct': True}]
    adjusted = _scored(rows).with_columns(
        targetSupported=pl.lit('yes'),
        actionSupported=pl.lit('yes'),
        primaryMechanism=pl.lit('yes'),
        rationale=pl.lit('r'),
    )
    pools = pl.DataFrame({'pool_id': [r['pool_id'] for r in rows], 'drug_name': [r['pool_id'] for r in rows]})
    links = _links([r['pool_id'] for r in rows], ['PHASE_1'] * 20 + ['PHASE_2'] * 10 + ['APPROVAL'])

    sample = handcheck(adjusted, pools, links, per_stratum=5, seed=0)

    assert 'bench' not in sample['pool_id'].to_list()
    assert dict(sample.group_by('maximumClinicalStage').len().iter_rows()) == {'PHASE_1': 5, 'PHASE_2': 5}


def test_precision_by_threshold_per_phase_group():
    scored = _scored([
        {'pool_id': 'a', 'confidence': 0.95},
        {'pool_id': 'b', 'confidence': 0.85, 'correct': False},
        {'pool_id': 'c', 'confidence': 0.75},
    ])
    links = _links(['a', 'b', 'c'], ['early'] * 3)

    table = precision_by_threshold(scored, links, thresholds=(0.0, 0.8, 0.9))

    rows = {r['threshold']: r for r in table.iter_rows(named=True)}
    assert (rows[0.0]['kept'], rows[0.0]['precision']) == (3, 2 / 3)
    assert (rows[0.8]['kept'], rows[0.8]['precision']) == (2, 0.5)
    assert (rows[0.9]['kept'], rows[0.9]['precision'], rows[0.9]['correct_kept_share']) == (1, 1.0, 0.5)
