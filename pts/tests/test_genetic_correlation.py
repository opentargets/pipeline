"""Tests for the genetic correlation tasks: representatives, munge and pairs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from pts.transformers.genetic_correlation import munge as munge_module
from pts.transformers.genetic_correlation import pairs as pairs_module
from pts.transformers.genetic_correlation import representatives as representatives_module
from pts.transformers.genetic_correlation.munge import MUNGED_SCHEMA, munge
from pts.transformers.genetic_correlation.pairs import merge_pair, pairs, study_pairs
from pts.transformers.genetic_correlation.representatives import OUTPUT_SCHEMA, ld_ancestry, representatives
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
    destination = tmp_path / 'representatives'
    representatives(inputs, str(destination), {}, Mock())
    result = scan_dataset(str(destination)).collect().sort('diseaseId')
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
    destination = tmp_path / 'representatives'
    representatives(inputs, str(destination), {'ancestries': ['nfe', 'eas']}, Mock())
    result = scan_dataset(str(destination)).collect().filter(pl.col('diseaseId') == 'D1').sort('ldAncestry')
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


def test_munge_joins_each_study_to_its_panel(munge_inputs: tuple[dict, dict], tmp_path: Path) -> None:
    source, destination = munge_inputs
    munge(
        source,
        destination,
        {'ldscore_template': 'panel_{ancestry}.csv', 'parallel_studies': 2},
        Mock(work_path=tmp_path),
    )
    panels = scan_dataset(destination['panels']).collect()
    assert panels.to_dict(as_series=False) == {'ldAncestry': ['nfe'], 'mSnps': [49]}
    s1 = pl.read_parquet(f'{destination["munged"]}/nfe/S1.parquet')
    s2 = pl.read_parquet(f'{destination["munged"]}/nfe/S2.parquet')
    assert s1.schema == MUNGED_SCHEMA
    # 50 panel SNPs, minus the one with swapped alleles, the one with no effect and the one with no error
    assert s1.height == 47
    assert s1['position'].is_sorted()
    assert s1['sampleSize'].unique().to_list() == [1000.0]
    assert s2['sampleSize'].unique().to_list() == [2500.0]
    assert s1['position'].to_list() == s2['position'].to_list()


def _munged(
    rng: np.random.Generator, positions: np.ndarray, ld: np.ndarray, genetic: np.ndarray, n: float
) -> pl.DataFrame:
    se = 1 / np.sqrt(n)
    z = np.sqrt(n) * genetic + rng.normal(size=len(positions))
    return pl.DataFrame({
        'chromosome': '1',
        'position': positions,
        'ref': 'A',
        'alt': 'G',
        'beta': z * se,
        'standardError': se,
        'sampleSize': n,
        'ldScore': ld,
    }).cast(dict(MUNGED_SCHEMA))


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
    reps = tmp_path / 'representatives'
    reps.mkdir()
    pl.DataFrame({
        'diseaseId': ['D1', 'D2', 'D3'],
        'ldAncestry': 'nfe',
        'studyId': ['A', 'B', 'C'],
        'effectiveSampleSize': 50_000.0,
    }).write_parquet(reps / 'part-0.parquet')
    panels = tmp_path / 'panels'
    panels.mkdir()
    pl.DataFrame({'ldAncestry': ['nfe'], 'mSnps': [int(m)]}).write_parquet(panels / 'part-0.parquet')
    source = {'representatives': str(reps), 'munged': str(tmp_path / 'munged'), 'panels': str(panels)}
    return source, str(tmp_path / 'output')


def test_study_pairs_are_within_ancestry_and_ordered() -> None:
    reps = pl.DataFrame({'ldAncestry': ['nfe', 'nfe', 'eas', 'nfe', 'eas'], 'studyId': ['b', 'a', 'x', 'c', 'y']})
    assert study_pairs(reps).rows() == [('eas', 'x', 'y'), ('nfe', 'a', 'b'), ('nfe', 'a', 'c'), ('nfe', 'b', 'c')]


def test_merge_pair_matches_swapped_alleles_with_the_effect_negated() -> None:
    left = pl.DataFrame({
        'chromosome': '1',
        'position': [10, 20, 30],
        'ref': 'A',
        'alt': 'G',
        'beta': [0.1, 0.2, 0.3],
        'standardError': 0.1,
        'sampleSize': 100.0,
        'ldScore': [1.0, 2.0, 3.0],
    })
    right = pl.DataFrame({
        'chromosome': '1',
        'position': [30, 20, 40],
        'ref': ['G', 'A', 'A'],
        'alt': ['A', 'G', 'G'],
        'beta': [0.5, 0.6, 0.7],
        'standardError': 0.2,
        'sampleSize': 200.0,
        'ldScore': [3.0, 2.0, 4.0],
    })
    merged = merge_pair(left, right)
    assert merged['position'].to_list() == [20, 30]
    assert merged['rightBeta'].to_list() == [0.6, -0.5]
    assert merged['ldScore'].to_list() == [2.0, 3.0]


def test_pairs_correlate_every_pair(pairs_inputs: tuple[dict, str], tmp_path: Path) -> None:
    source, destination = pairs_inputs
    settings = {'n_blocks': 20, 'parallel_pairs': 2, 'pairs_per_chunk': 2, 'worker_threads': 1}
    pairs(source, destination, settings, Mock(work_path=tmp_path))
    result = scan_dataset(destination).collect()
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
    source, destination = pairs_inputs
    pairs(source, destination, {'n_blocks': 5000, 'parallel_pairs': 1}, Mock(work_path=tmp_path))
    result = scan_dataset(destination).collect()
    assert result['runStatus'].to_list() == ['skipped'] * 3
    assert result['skipReason'][0] == '3000 shared SNPs, fewer than 5000 jackknife blocks'
    assert result['rg'].null_count() == 3


def test_modules_expose_their_transformers() -> None:
    assert callable(representatives_module.representatives)
    assert callable(munge_module.munge)
    assert callable(pairs_module.pairs)
