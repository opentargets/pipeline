import polars as pl

from pts.transformers.clinical_mira.moa.pilot import pilot_sample
from pts.transformers.clinical_mira.moa.report import go_line


def _links(rows: list[tuple[str, str | None, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'pool_id': [r[0] for r in rows],
            'drugId': [r[1] for r in rows],
            'link': [r[2] for r in rows],
            'maximumClinicalStage': [r[3] for r in rows],
            'phase_group': [
                'early' if r[3] in {'EARLY_PHASE_1', 'PHASE_1', 'PHASE_1_2', 'PHASE_2'} else 'late' for r in rows
            ],
        },
        schema_overrides={'drugId': pl.String},
    )


def _scenario() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = (
        [(f'p1_{i}', f'C1_{i}', 'linked', 'PHASE_1') for i in range(5)]
        + [(f'p2_{i}', f'C2_{i}', 'linked', 'PHASE_2') for i in range(500)]
        + [(f'p23_{i}', f'C23_{i}', 'linked', 'PHASE_2_3') for i in range(3)]
        + [(f'pa_{i}', f'CA_{i}', 'linked', 'APPROVAL') for i in range(150)]
        + [(f'pu_{i}', None, 'unlinked', 'PHASE_1') for i in range(250)]
        + [('pnb', 'CNB', 'linked', 'PHASE_1')]
    )
    links = _links(rows)
    bench = pl.DataFrame({'drugId': [r[1] for r in rows if r[1] and r[1] != 'CNB']})
    return links, bench


def test_pilot_strata_sizes_and_caps():
    links, bench = _scenario()

    sample = pilot_sample(links, bench).join(links, on='pool_id')
    counts = dict(
        sample
        .group_by('maximumClinicalStage', 'link')
        .len()
        .select(pl.concat_str('maximumClinicalStage', 'link', separator='/'), 'len')
        .iter_rows()
    )

    assert sample.filter(pl.col('pool_id').str.starts_with('p1_')).height == 5
    assert counts['PHASE_2/linked'] == 400
    assert counts['PHASE_2_3/linked'] == 3
    assert counts['APPROVAL/linked'] == 100
    assert sample.filter(pl.col('pool_id').str.starts_with('pu_') | (pl.col('pool_id') == 'pnb')).height == 200


def test_unbenchmarked_linked_pools_count_as_unbenchmarked():
    links, bench = _scenario()
    links = links.filter(~pl.col('pool_id').str.starts_with('pu_'))

    assert 'pnb' in pilot_sample(links, bench)['pool_id'].to_list()


def test_pilot_is_seeded():
    links, bench = _scenario()

    assert (
        pilot_sample(links, bench, seed=3)['pool_id'].to_list()
        == pilot_sample(links, bench, seed=3)['pool_id'].to_list()
    )
    assert (
        pilot_sample(links, bench, seed=3)['pool_id'].to_list()
        != pilot_sample(links, bench, seed=4)['pool_id'].to_list()
    )


def test_go_line_marks_inconclusive():
    assert 'INCONCLUSIVE' in go_line({
        'precision': 0.92,
        'precision_lo': 0.80,
        'precision_hi': 0.97,
        'scored': 40,
        'correct': 37,
    })


def test_go_line_go_and_no_go():
    assert 'GO' in go_line({
        'precision': 0.95,
        'precision_lo': 0.91,
        'precision_hi': 0.98,
        'scored': 400,
        'correct': 380,
    })
    assert 'NO GO' in go_line({
        'precision': 0.70,
        'precision_lo': 0.62,
        'precision_hi': 0.77,
        'scored': 200,
        'correct': 140,
    })
    assert 'no early-phase' in go_line(None)
