"""End-to-end test for the l2g_train transformer, on tiny local fixtures."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES
from pts.transformers.l2g.model import load_model
from pts.transformers.l2g_train import l2g_train


@pytest.fixture
def workspace(tmp_path):
    """Write a feature matrix, credible set, gold standard and pinned test set to disk."""
    rng = np.random.default_rng(0)
    loci = [f'sl{i}' for i in range(40)]
    genes = [f'ENSG{i % 8:011d}' for i in range(40)]

    matrix = pl.DataFrame(
        {
            'studyLocusId': loci,
            'geneId': genes,
            'isProteinCoding': [1] * 40,
            **{name: rng.random(40) for name in FEATURES},
        }
    )
    (tmp_path / 'fm').mkdir()
    matrix.write_parquet(tmp_path / 'fm' / 'part-0.parquet')

    credible = pl.DataFrame({
        'studyLocusId': loci,
        'variantId': [f'1_{i}_A_G' for i in range(40)],
        'studyId': [f'GCST{i}' for i in range(40)],
        'studyType': ['gwas'] * 40,
    })
    (tmp_path / 'cs').mkdir()
    credible.write_parquet(tmp_path / 'cs' / 'part-0.parquet')

    gold = pl.DataFrame({
        'studyLocusId': loci,
        'geneId': genes,
        'diseaseIds': [['EFO_1']] * 40,
        'variantId': [f'1_{i}_A_G' for i in range(40)],
        'studyId': [f'GCST{i}' for i in range(40)],
        'goldStandardSet': ['positive' if i % 4 == 0 else 'negative' for i in range(40)],
    })
    (tmp_path / 'gs').mkdir()
    gold.write_ndjson(tmp_path / 'gs' / 'part-0.json')

    pinned = pl.DataFrame({
        'studyLocusId': loci[:8],
        'geneId': genes[:8],
        'goldStandardSet': ['positive' if i % 4 == 0 else 'negative' for i in range(8)],
    })
    (tmp_path / 'pt').mkdir()
    pinned.write_parquet(tmp_path / 'pt' / 'part-0.parquet')

    return tmp_path


def _source(workspace) -> dict[str, str]:
    return {
        'feature_matrix': str(workspace / 'fm'),
        'credible_set': str(workspace / 'cs'),
        'gold_standard': str(workspace / 'gs'),
        'predefined_test': str(workspace / 'pt'),
    }


def _destination(workspace) -> dict[str, str]:
    out = workspace / 'out'
    return {
        'model': str(out / 'model' / 'classifier.skops'),
        'background': str(out / 'model' / 'shap_background.parquet'),
        'metrics': str(out / 'model' / 'metrics.json'),
        'train_split': str(out / 'train'),
        'test_split': str(out / 'test'),
        'split_stats': str(out / 'split_stats.json'),
    }


SETTINGS = {
    'features_list': list(FEATURES),
    'hyperparameters': {'n_estimators': 5, 'max_depth': 2, 'random_state': 777},
    'train_on_full_dataset': True,
    'shap_background_size': 10,
    'shap_background_seed': 42,
}


def test_l2g_train_writes_every_declared_artifact(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    for key, path in destination.items():
        assert Path(path).exists(), f'{key} was not written to {path}'


def test_l2g_train_writes_a_loadable_model(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    model = load_model(destination['model'])
    assert model.get_booster().num_boosted_rounds() == 5


def test_l2g_train_metrics_carry_the_seven_scores_and_the_run_settings(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    metrics = json.loads(Path(destination['metrics']).read_text())
    assert set(metrics['heldOut']) == {
        'areaUnderROC',
        'accuracy',
        'weightedPrecision',
        'averagePrecision',
        'averagePrecisionFromScores',
        'weightedRecall',
        'f1',
    }
    assert metrics['shapBackgroundSeed'] == 42
    assert metrics['shapBackgroundSize'] == 10
    assert metrics['hyperparameters']['n_estimators'] == 5
    assert metrics['trainOnFullDataset'] is True
    assert set(metrics['featureMissingness']) == set(FEATURES)


def test_l2g_train_split_stats_reconcile(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    stats = json.loads(Path(destination['split_stats']).read_text())
    assert stats['n_train'] + stats['n_test_new'] + stats['n_lost_total'] == stats['n_original_total']


def test_l2g_train_background_has_the_configured_shape(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    background = pl.read_parquet(destination['background'])
    assert background.columns == list(FEATURES)
    assert background.height == 10
