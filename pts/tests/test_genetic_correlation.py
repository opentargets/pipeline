"""Tests for the genetic correlation tasks: representatives, munge and pairs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from pts.pyspark.genetic_correlation_munge import _reusable
from pts.transformers.genetic_correlation import pairs as pairs_module
from pts.transformers.genetic_correlation.munge import MUNGE_VERSION, MUNGED_SCHEMA, munge_one, panel_size, read_panel
from pts.transformers.genetic_correlation.pairs import Study, Tile, assemble, results, run_tile, shared, tiles
from pts.transformers.genetic_correlation.previous import previous_path
from pts.transformers.genetic_correlation.representatives import (
    OUTPUT_SCHEMA,
    ld_ancestry,
    representative_changes,
    representatives,
)
from pts.transformers.genetic_correlation.settings import MungeDestination, MungeSource, PairsSettings
from pts.transformers.study import QualityFlag
from pts.transformers.utils.dataset import scan_dataset


def _study(
    study_id: str,
    diseases: list[str],
    *,
    populations: list[tuple[str, float]] | None = None,
    cases: int | None = 20_000,
    controls: int | None = 20_000,
    samples: int = 40_000,
    quality: list[str] | None = None,
    flags: list[str] | None = None,
    study_type: str = 'gwas',
    sumstats: bool = True,
) -> dict:
    return {
        'studyId': study_id,
        'studyType': study_type,
        'diseaseIds': diseases,
        'traitFromSourceMappedIds': diseases,
        'nCases': cases,
        'nControls': controls,
        'nSamples': samples,
        'ldPopulationStructure': [
            {'ldPopulation': p, 'relativeSampleSize': w} for p, w in (populations or [('nfe', 1.0)])
        ],
        'qualityControls': quality,
        'analysisFlags': flags,
        'hasSumstats': sumstats,
    }


def _h2(
    study_id: str, h2: float = 0.2, se: float = 0.02, lambda_gc: float | None = 1.1, status: str = 'success'
) -> dict:
    return {'studyId': study_id, 'runStatus': status, 'h2': h2, 'h2_se': se, 'lambda_gc': lambda_gc}


STUDIES = [
    _study('BIG', ['D1']),
    _study('SMALL', ['D1'], cases=5_000, controls=5_000),
    _study('QC', ['D1'], cases=50_000, controls=50_000, quality=[QualityFlag.FAILED_PZ_CHECK.value]),
    _study('FLAG', ['D1'], cases=50_000, controls=50_000, flags=['pQTL']),
    _study('EAS', ['D1'], cases=50_000, controls=50_000, populations=[('eas', 0.9), ('nfe', 0.1)]),
    _study('NOH2', ['D1'], cases=50_000, controls=50_000),
    _study('WEAKH2', ['D1'], cases=50_000, controls=50_000),
    _study('QUANT', ['D2', 'D3'], cases=None, controls=None, samples=30_000),
    _study('TINY', ['D2'], cases=None, controls=None, samples=5_000),
    _study('QTL', ['D3'], study_type='eqtl'),
]
HERITABILITY = [
    _h2('BIG'),
    _h2('SMALL'),
    _h2('QC'),
    _h2('FLAG'),
    _h2('EAS'),
    _h2('NOH2', status='error'),
    _h2('WEAKH2', h2=0.02, se=0.02),
    _h2('QUANT', lambda_gc=None),
    _h2('TINY'),
    _h2('QTL'),
]


def _representatives_destination(root: Path) -> dict[str, str]:
    return {'representatives': str(root / 'representatives'), 'changes': str(root / 'changes')}


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, str]:
    study = tmp_path / 'study.parquet'
    pl.DataFrame(STUDIES).write_parquet(study)
    heritability = tmp_path / 'heritability.parquet'
    pl.DataFrame(HERITABILITY).write_parquet(heritability)
    return {'study': str(study), 'heritability': str(heritability)}


def test_ld_ancestry_takes_the_dominant_population_and_breaks_ties_towards_nfe() -> None:
    frame = pl.DataFrame({
        'studyId': ['a', 'b', 'c', 'd'],
        'ldPopulationStructure': [
            [{'ldPopulation': 'EAS', 'relativeSampleSize': 0.6}, {'ldPopulation': 'nfe', 'relativeSampleSize': 0.4}],
            [{'ldPopulation': 'afr', 'relativeSampleSize': 0.5}, {'ldPopulation': 'nfe', 'relativeSampleSize': 0.5}],
            [{'ldPopulation': 'afr', 'relativeSampleSize': 0.5}, {'ldPopulation': 'amr', 'relativeSampleSize': 0.5}],
            [{'ldPopulation': 'sas', 'relativeSampleSize': 1.0}],
        ],
    })
    result = ld_ancestry(frame.lazy()).collect().sort('studyId')
    assert result.to_dict(as_series=False) == {'studyId': ['a', 'b', 'c'], 'ldAncestry': ['eas', 'nfe', 'afr']}


def test_representatives_choose_the_best_powered_eligible_study(inputs: dict[str, str], tmp_path: Path) -> None:
    destination = _representatives_destination(tmp_path)
    representatives(inputs, destination, {}, Mock())
    result = scan_dataset(destination['representatives']).collect().sort('diseaseId')
    assert result.schema == OUTPUT_SCHEMA
    assert result.select('diseaseId', 'ldAncestry', 'studyId').to_dict(as_series=False) == {
        'diseaseId': ['D1', 'D2', 'D3'],
        'ldAncestry': ['nfe', 'nfe', 'nfe'],
        'studyId': ['BIG', 'QUANT', 'QUANT'],
    }
    big = result.row(0, named=True)
    assert big['effectiveSampleSize'] == 40_000
    assert big['h2'] == 0.2
    assert big['lambdaGc'] == 1.1


def test_representatives_can_include_another_ancestry(inputs: dict[str, str], tmp_path: Path) -> None:
    destination = _representatives_destination(tmp_path)
    representatives(inputs, destination, {'ancestries': ['nfe', 'eas']}, Mock())
    result = (
        scan_dataset(destination['representatives']).collect().filter(pl.col('diseaseId') == 'D1').sort('ldAncestry')
    )
    assert result['studyId'].to_list() == ['EAS', 'BIG']


def _sumstats(rng: np.random.Generator, positions: np.ndarray, *, sample_size: float | None) -> pl.DataFrame:
    return pl.DataFrame({
        'studyId': 'S',
        'variantId': [f'1_{p}_A_G' for p in positions],
        'chromosome': '1',
        'position': positions,
        'beta': rng.normal(size=len(positions)),
        'standardError': np.abs(rng.normal(size=len(positions))) + 0.1,
        'sampleSize': sample_size,
    })


@pytest.fixture
def munge_inputs(tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    rng = np.random.default_rng(0)
    positions = np.arange(100, 1100, 10)
    sumstats = tmp_path / 'sumstats'
    for study_id, size in (('S1', 1000.0), ('S2', None)):
        (sumstats / study_id).mkdir(parents=True)
        frame = _sumstats(rng, positions, sample_size=size).with_columns(pl.lit(study_id).alias('studyId'))
        # a variant with no effect, one with a non-positive error and a duplicate key are dropped;
        # the first two are panel positions, the panel covering every other position
        frame[2, 'beta'] = None
        frame[4, 'standardError'] = 0.0
        pl.concat([frame, frame.tail(1)]).write_parquet(sumstats / study_id / 'part-0.parquet')
    ldscore = tmp_path / 'ldscore'
    ldscore.mkdir()
    # the panel covers every other position, one with the alleles the other way round, and one rare SNP
    panel = pl.DataFrame({
        'CHR': '1',
        'BP': positions[::2],
        'ref': 'A',
        'alt': 'G',
        'AC': 10,
        'AF': 0.3,
        'AN': 100,
        'homozygote_count': 0,
        'L2': rng.gamma(4.0, 25.0, size=len(positions[::2])),
        'BP_hg38': positions[::2],
    })
    panel[0, 'ref'], panel[0, 'alt'] = 'G', 'A'
    panel[1, 'AF'] = 0.01
    panel.write_csv(ldscore / 'panel_nfe.csv', separator='\t')
    reps = tmp_path / 'representatives'
    reps.mkdir()
    pl.DataFrame({
        'studyId': ['S1', 'S2'],
        'ldAncestry': 'nfe',
        'effectiveSampleSize': [1000.0, 2500.0],
    }).write_parquet(reps / 'part-0.parquet')
    source = {'representatives': str(reps), 'ldscore': str(ldscore), 'sumstats': str(sumstats)}
    destination = {'munged': str(tmp_path / 'munged'), 'panels': str(tmp_path / 'panels')}
    return source, destination


def test_munge_one_joins_a_study_to_its_panel(munge_inputs: tuple[dict, dict], tmp_path: Path) -> None:
    source, destination = munge_inputs
    panel = read_panel(f'{source["ldscore"]}/panel_nfe.csv', tmp_path / 'panel')
    assert panel_size(panel, 0.05) == 49
    for study in pl.read_parquet(f'{source["representatives"]}/*.parquet').iter_rows(named=True):
        n_snps = munge_one(study, panel, source['sumstats'], destination['munged'], tmp_path / 'scratch')
        assert n_snps == 48
    s1 = pl.read_parquet(f'{destination["munged"]}/nfe/S1.parquet')
    s2 = pl.read_parquet(f'{destination["munged"]}/nfe/S2.parquet')
    assert s1.schema == MUNGED_SCHEMA
    # 50 panel SNPs, minus the one with no effect and the one with no error
    assert s1.height == 48
    assert s1['snp'].is_sorted()
    assert s1['sampleSize'].unique().to_list() == [1000.0]
    assert s2['sampleSize'].unique().to_list() == [2500.0]
    assert s1['snp'].to_list() == s2['snp'].to_list()
    # the panel lists position 100 as G>A and the study as A>G: kept, with the z-score negated
    raw = pl.read_parquet(f'{source["sumstats"]}/S1/part-0.parquet').row(0, named=True)
    swapped = panel.filter(pl.col('position') == 100)['snp'][0]
    assert s1.filter(pl.col('snp') == swapped)['z'][0] == pytest.approx(-raw['beta'] / raw['standardError'])


def _munged(
    rng: np.random.Generator, positions: np.ndarray, ld: np.ndarray, genetic: np.ndarray, n: float
) -> pl.DataFrame:
    return pl.DataFrame({
        'snp': np.arange(len(positions)),
        'z': np.sqrt(n) * genetic + rng.normal(size=len(positions)),
        'sampleSize': n,
        'ldScore': ld,
    }).cast(MUNGED_SCHEMA)


@pytest.fixture
def pairs_inputs(tmp_path: Path) -> tuple[dict[str, str], str]:
    rng = np.random.default_rng(1)
    n_snps, m = 3000, 1e5
    positions = np.arange(n_snps) * 100
    ld = rng.gamma(4.0, 25.0, size=n_snps)
    g1 = rng.normal(size=n_snps) * np.sqrt(0.3 * ld / m)
    g2 = 0.8 * g1 + rng.normal(size=n_snps) * np.sqrt(0.36 * 0.3 * ld / m)
    g3 = rng.normal(size=n_snps) * np.sqrt(0.3 * ld / m)
    munged = tmp_path / 'munged' / 'nfe'
    munged.mkdir(parents=True)
    for study_id, genetic in (('A', g1), ('B', g2), ('C', g3)):
        _munged(rng, positions, ld, genetic, 50_000.0).write_parquet(munged / f'{study_id}.parquet')
    studies = tmp_path / 'studies'
    studies.mkdir()
    pl.DataFrame({'studyId': ['A', 'B', 'C'], 'ldAncestry': 'nfe', 'nSnps': n_snps, 'isUsable': True}).write_parquet(
        studies / 'part-0.parquet'
    )
    panels = tmp_path / 'panels'
    panels.mkdir()
    pl.DataFrame({'ldAncestry': ['nfe'], 'mSnps': [int(m)]}).write_parquet(panels / 'part-0.parquet')
    source = {'studies': str(studies), 'munged': str(tmp_path / 'munged'), 'panels': str(panels)}
    return source, str(tmp_path / 'output')


def test_tiles_cover_every_pair_within_an_ancestry_once() -> None:
    ids = [f's{i:02d}' for i in range(7)]
    studies = pl.DataFrame({'ldAncestry': ['nfe'] * 7 + ['eas'] * 2, 'studyId': [*ids, 'x', 'y']})
    work = tiles(studies, 3)
    pairs = [(tile.ancestry, *pair) for tile in work for pair in tile.pairs()]
    assert sorted(pairs) == sorted(
        [('nfe', a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]] + [('eas', 'x', 'y')]
    )
    assert len(pairs) == len(set(pairs))
    assert max(len(set(tile.left + tile.right)) for tile in work) <= 6


def test_shared_snps_are_matched_in_genomic_order() -> None:
    left = Study(np.array([2, 5, 9, 11]), np.zeros(4), np.ones(4), np.ones(4))
    right = Study(np.array([1, 5, 9, 12]), np.zeros(4), np.ones(4), np.ones(4))
    i, j = shared(left, right)
    assert left.snp[i].tolist() == right.snp[j].tolist() == [5, 9]


def _run_pairs(source: dict[str, str], tmp_path: Path, **settings) -> pl.DataFrame:
    studies = pl.read_parquet(f'{source["studies"]}/*.parquet')
    options = PairsSettings.model_validate(settings)
    rows = [
        row
        for tile in tiles(studies, options.tile_studies)
        for row in run_tile(tile, source['munged'], {'nfe': 100_000}, options, tmp_path / 'scratch')
    ]
    return results(rows)


def test_a_tile_off_the_diagonal_pairs_its_two_blocks() -> None:
    assert Tile('nfe', ('a', 'b'), ('c',)).pairs() == [('a', 'c'), ('b', 'c')]
    assert Tile('nfe', ('a', 'b', 'c'), ('a', 'b', 'c')).pairs() == [('a', 'b'), ('a', 'c'), ('b', 'c')]


def test_pairs_correlate_every_pair(pairs_inputs: tuple[dict, str], tmp_path: Path) -> None:
    source, _ = pairs_inputs
    result = _run_pairs(source, tmp_path, n_blocks=20, tile_studies=2)
    assert result.schema == pairs_module.OUTPUT_SCHEMA
    assert result.select('leftStudyId', 'rightStudyId').rows() == [('A', 'B'), ('A', 'C'), ('B', 'C')]
    assert result['runStatus'].to_list() == ['success'] * 3
    assert result['nSnps'].to_list() == [3000] * 3
    assert result['mSnps'].to_list() == [100_000] * 3
    by_pair = {(r['leftStudyId'], r['rightStudyId']): r for r in result.iter_rows(named=True)}
    assert by_pair['A', 'B']['rg'] > 0.5
    assert abs(by_pair['A', 'C']['rg']) < 4 * by_pair['A', 'C']['rgStandardError']
    assert by_pair['A', 'B']['leftH2'] == pytest.approx(by_pair['A', 'C']['leftH2'])


def test_pairs_skip_pairs_with_too_few_shared_snps(pairs_inputs: tuple[dict, str], tmp_path: Path) -> None:
    source, _ = pairs_inputs
    result = _run_pairs(source, tmp_path, n_blocks=5000)
    assert result['runStatus'].to_list() == ['skipped'] * 3
    assert result['skipReason'][0] == '3000 shared SNPs, fewer than 5000 jackknife blocks'
    assert result['rg'].null_count() == 3


def test_representative_changes_compare_each_cell_with_the_previous_run(inputs: dict[str, str], tmp_path: Path) -> None:
    first = _representatives_destination(tmp_path / 'first')
    representatives(inputs, first, {}, Mock())
    assert scan_dataset(first['changes']).collect()['change'].unique().to_list() == ['added']
    second = _representatives_destination(tmp_path / 'second' / 'output')
    representatives({**inputs, 'previous_output': str(tmp_path / 'first')}, second, {}, Mock())
    assert scan_dataset(second['changes']).collect()['change'].unique().to_list() == ['unchanged']


def test_representative_changes_name_every_kind_of_change() -> None:
    def cells(rows: list[tuple[str, str]]) -> pl.DataFrame:
        return pl.DataFrame(rows, schema=['diseaseId', 'studyId'], orient='row').with_columns(
            pl.lit('nfe').alias('ldAncestry')
        )

    current = cells([('D1', 'A'), ('D2', 'B2'), ('D4', 'C')])
    previous = cells([('D1', 'A'), ('D2', 'B1'), ('D3', 'X')])
    changes = representative_changes(current, previous)
    assert changes.select('diseaseId', 'studyId', 'previousStudyId', 'change').rows() == [
        ('D1', 'A', 'A', 'unchanged'),
        ('D2', 'B2', 'B1', 'replaced'),
        ('D3', None, 'X', 'dropped'),
        ('D4', 'C', None, 'added'),
    ]


def test_previous_path_reads_the_same_name_in_the_previous_output(tmp_path: Path) -> None:
    (tmp_path / 'old' / 'munged').mkdir(parents=True)
    assert previous_path(None, 'gs://b/run/output/munged') is None
    assert previous_path(str(tmp_path / 'old'), 'gs://b/run/output/munged') == str(tmp_path / 'old' / 'munged')
    with pytest.raises(FileNotFoundError, match='no'):
        previous_path(str(tmp_path / 'old'), 'gs://b/run/output/panels')


def test_tiles_with_reused_studies_cover_only_the_pairs_with_a_new_study() -> None:
    ids = [f's{i}' for i in range(6)]
    studies = pl.DataFrame({'ldAncestry': 'nfe', 'studyId': ids})
    reused = studies.filter(pl.col('studyId').is_in(['s0', 's1', 's2', 's3']))
    pairs = sorted(pair for tile in tiles(studies, 2, reused) for pair in tile.pairs())
    expected = sorted((a, b) for i, a in enumerate(ids) for b in ids[i + 1 :] if not {a, b} <= {'s0', 's1', 's2', 's3'})
    assert pairs == expected


def _pair_rows(pairs: list[tuple[str, str]], rg: float) -> pl.DataFrame:
    return results([
        {'ldAncestry': 'nfe', 'leftStudyId': a, 'rightStudyId': b, 'runStatus': 'success', 'rg': rg} for a, b in pairs
    ])


def test_assemble_copies_reused_pairs_and_deprecates_the_rest() -> None:
    # previous run: A, B, C, D; now D has been dropped, E added, and C has too few SNPs
    previous = _pair_rows([('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')], 0.5)
    studies = pl.DataFrame({
        'ldAncestry': 'nfe',
        'studyId': ['A', 'B', 'C', 'E'],
        'isUsable': [True, True, False, True],
        'isReused': [True, True, True, False],
    })
    reused = pl.DataFrame({'ldAncestry': 'nfe', 'studyId': ['A', 'B']})
    computed = _pair_rows([('A', 'E'), ('B', 'E')], 0.1)
    output = assemble(computed, studies, previous, reused, min_snps=200_000)
    current = output.filter(~pl.col('isDeprecated'))
    assert current.select('leftStudyId', 'rightStudyId', 'rg').rows() == [
        ('A', 'B', 0.5),
        ('A', 'E', 0.1),
        ('B', 'E', 0.1),
    ]
    deprecated = dict(
        output
        .filter(pl.col('isDeprecated'))
        .select(pl.concat_str('leftStudyId', 'rightStudyId'), 'deprecationReason')
        .iter_rows()
    )
    assert deprecated == {
        'AC': 'C has fewer than 200000 panel SNPs',
        'AD': 'D is no longer a representative study',
        'BC': 'C has fewer than 200000 panel SNPs',
        'BD': 'D is no longer a representative study',
        'CD': 'C has fewer than 200000 panel SNPs; D is no longer a representative study',
    }
    # a later run keeps the deprecated pairs with their reasons, and a pair current again replaces its deprecated row
    later = assemble(
        _pair_rows([('A', 'D')], 0.9),
        studies.vstack(
            pl.DataFrame({'ldAncestry': ['nfe'], 'studyId': ['D'], 'isUsable': [True], 'isReused': [False]})
        ),
        output,
        None,
    )
    assert later.filter(pl.col('leftStudyId') == 'A', pl.col('rightStudyId') == 'D').select(
        'rg', 'isDeprecated'
    ).rows() == [(0.9, False)]
    assert later.filter(pl.col('isDeprecated'))['deprecationReason'].null_count() == 0


def test_munge_reuses_a_study_only_with_the_same_panel_and_version(tmp_path: Path) -> None:
    previous = tmp_path / 'previous'
    for name, frame in {
        'munged_studies': pl.DataFrame({
            'studyId': ['A', 'B'],
            'ldAncestry': ['nfe', 'eas'],
            'nSnps': [900_000, 800_000],
        }),
        'panels': pl.DataFrame({
            'ldAncestry': ['nfe', 'eas'],
            'mSnps': [1, 1],
            'panelChecksum': ['x', 'y'],
            'mungeVersion': MUNGE_VERSION,
        }),
    }.items():
        (previous / name).mkdir(parents=True)
        frame.write_parquet(previous / name / 'part-0.parquet')
    (previous / 'munged').mkdir()
    studies = pl.DataFrame({
        'studyId': ['A', 'B', 'C'],
        'ldAncestry': ['nfe', 'eas', 'nfe'],
        'effectiveSampleSize': 1.0,
    })
    # the eas panel changed since the previous run
    panels = pl.DataFrame({
        'ldAncestry': ['nfe', 'eas'],
        'mSnps': [1, 1],
        'panelChecksum': ['x', 'changed'],
        'mungeVersion': MUNGE_VERSION,
    })
    sources = MungeSource(representatives='r', ldscore='l', sumstats='s', previous_output=str(previous))
    run = tmp_path / 'run' / 'output'
    destinations = MungeDestination(
        munged=str(run / 'munged'), studies=str(run / 'munged_studies'), panels=str(run / 'panels')
    )
    reusable = _reusable(studies, panels, sources, destinations).sort('studyId')
    assert reusable.select('studyId', 'previousSnps').rows() == [('A', 900_000), ('B', None), ('C', None)]
    assert reusable['previousMunged'][0] == f'{previous}/munged/nfe/A.parquet'
