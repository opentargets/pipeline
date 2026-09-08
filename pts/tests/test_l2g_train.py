"""End-to-end test for the l2g_train transformer, on tiny local fixtures."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES
from pts.transformers.l2g.model import load_model, to_matrix
from pts.transformers.l2g_train import l2g_train
from pts.transformers.utils.dataset import scan_dataset


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


def test_l2g_train_is_reproducible_across_runs(workspace) -> None:
    """Two runs from one fixture must produce the same model and the same SHAP background.

    `derive_splits` ends in polars joins, which leave their output order unspecified, so the rows
    reaching the fit arrive in a different order from run to run unless something imposes one.
    Both things that follow are POSITIONAL: XGBoost's `subsample` draws rows by position, and
    `build_background` draws the background by position. Without the sort in `l2g_train` the two
    saved models score the same rows differently and the two backgrounds differ, while
    `metrics.json` goes on recording `shapBackgroundSeed` as though the draw were reproducible.

    Asserted on the SAVED artifacts, not on in-memory frames, because those artifacts are what the
    release ships and what `l2g_predict` reads back.
    """
    # `subsample` is what makes the FIT positional, so it has to be below 1.0 here as it is in
    # production; `SETTINGS`'s three-key hyperparameter block leaves it at XGBoost's default of
    # 1.0, under which every row is used and the row order cannot reach the model.
    settings = dict(SETTINGS) | {
        'hyperparameters': {'n_estimators': 20, 'max_depth': 3, 'random_state': 777, 'subsample': 0.8}
    }
    first = _destination(workspace / 'first')
    second = _destination(workspace / 'second')
    l2g_train(_source(workspace), first, dict(settings), None)
    l2g_train(_source(workspace), second, dict(settings), None)

    background_first = pl.read_parquet(first['background'])
    background_second = pl.read_parquet(second['background'])
    assert background_first.equals(background_second), 'the SHAP background is not reproducible'

    # Scored on one fixed matrix, so any difference is the model's and not the ordering of the
    # rows it is asked about.
    scored = to_matrix(scan_dataset(first['test_split']).collect(), FEATURES)
    proba_first = load_model(first['model']).predict_proba(scored)[:, 1]
    proba_second = load_model(second['model']).predict_proba(scored)[:, 1]
    np.testing.assert_array_equal(proba_first, proba_second, err_msg='the fitted model is not reproducible')


@pytest.fixture
def duplicate_key_workspace(workspace):
    """`workspace`, plus one curated row that gives a training locus two labels for one gene.

    The only way a `(studyLocusId, geneId)` pair survives `annotate` twice is for the rows to
    differ in some other column, since it deduplicates on the whole row. Two curated rows agreeing
    on `(studyId, variantId, geneId)` and disagreeing on `goldStandardSet` do exactly that: the
    join fans the single feature-matrix row out to two, identical but for the label. `sl10` is
    used because it survives the contamination anti-join, so the duplicate lands in the TRAIN
    split rather than being filtered away before the check can see it.
    """
    gold = pl.read_ndjson(workspace / 'gs' / 'part-0.json')
    flipped = gold.filter(pl.col('studyLocusId') == 'sl10').with_columns(
        pl.lit('positive').alias('goldStandardSet')
    )
    assert flipped.height == 1, 'fixture no longer has exactly one sl10 row to flip'
    assert gold.filter(pl.col('studyLocusId') == 'sl10')['goldStandardSet'][0] == 'negative'
    pl.concat([gold, flipped]).write_ndjson(workspace / 'gs' / 'part-0.json')
    return workspace


def test_l2g_train_refuses_a_split_whose_key_repeats(duplicate_key_workspace) -> None:
    """A repeated key would silently undo the sort, so the run must fail instead.

    The sort is a total order only while `(studyLocusId, geneId)` is unique -- polars' `sort` is
    not tie-stable, and `maintain_order=True` would not help, because it stabilises against the
    input order and the input order is the unspecified thing. Tied rows would go back to the
    arbitrary order the joins produced, and the positional draws downstream would go back to
    varying between runs, with `metrics.json` still recording its seeds and
    `test_l2g_train_is_reproducible_across_runs` still passing on its key-unique fixture. Nothing
    would show, which is why this raises rather than warns.
    """
    with pytest.raises(ValueError, match=r'the train split repeats 1 .* pair'):
        l2g_train(_source(duplicate_key_workspace), _destination(duplicate_key_workspace), dict(SETTINGS), None)


@pytest.fixture
def leak_sensitive_workspace(tmp_path):
    """A fixture on which a refit-before-evaluate leak is impossible to miss.

    `workspace` above cannot detect that leak: every positive row in it lands on one of only
    two gene ids, and the pinned test set uses those same two genes as its positives, so
    `derive_splits`'s contamination rule (drop any locus that shares a gene with a test
    positive) strips EVERY positive row out of training. With zero positives to learn from,
    the model is already degenerate before a leak has a chance to matter, and held-out metrics
    come out identical whether or not the leak happens.

    Here, train and test are given disjoint gene pools -- the eight test loci each carry a gene
    id that appears nowhere else -- so contamination cannot touch training at all, and training
    keeps its own positives. A model that then also sees the test rows during fitting can
    memorise them outright (perfect metrics); a model that never saw them cannot, on features
    that carry no real signal. That gap is what makes the leak observable.
    """
    rng = np.random.default_rng(1)

    n_train = 40
    train_loci = [f'trainloc{i}' for i in range(n_train)]
    train_genes = [f'ENSGTRAIN{i % 8:07d}' for i in range(n_train)]
    train_labels = ['positive' if i % 4 == 0 else 'negative' for i in range(n_train)]

    n_test = 8
    test_loci = [f'testloc{i}' for i in range(n_test)]
    test_genes = [f'ENSGTEST{i:08d}' for i in range(n_test)]  # unique per test locus
    test_labels = ['positive' if i % 4 == 0 else 'negative' for i in range(n_test)]

    loci = train_loci + test_loci
    genes = train_genes + test_genes
    labels = train_labels + test_labels
    n = len(loci)

    matrix = pl.DataFrame(
        {
            'studyLocusId': loci,
            'geneId': genes,
            'isProteinCoding': [1] * n,
            **{name: rng.random(n) for name in FEATURES},
        }
    )
    (tmp_path / 'fm').mkdir()
    matrix.write_parquet(tmp_path / 'fm' / 'part-0.parquet')

    credible = pl.DataFrame({
        'studyLocusId': loci,
        'variantId': [f'1_{i}_A_G' for i in range(n)],
        'studyId': [f'GCST{i}' for i in range(n)],
        'studyType': ['gwas'] * n,
    })
    (tmp_path / 'cs').mkdir()
    credible.write_parquet(tmp_path / 'cs' / 'part-0.parquet')

    gold = pl.DataFrame({
        'studyLocusId': loci,
        'geneId': genes,
        'diseaseIds': [['EFO_1']] * n,
        'variantId': [f'1_{i}_A_G' for i in range(n)],
        'studyId': [f'GCST{i}' for i in range(n)],
        'goldStandardSet': labels,
    })
    (tmp_path / 'gs').mkdir()
    gold.write_ndjson(tmp_path / 'gs' / 'part-0.json')

    pinned = pl.DataFrame({
        'studyLocusId': test_loci,
        'geneId': test_genes,
        'goldStandardSet': test_labels,
    })
    (tmp_path / 'pt').mkdir()
    pinned.write_parquet(tmp_path / 'pt' / 'part-0.parquet')

    return tmp_path


def test_l2g_train_held_out_metrics_do_not_depend_on_train_on_full_dataset(leak_sensitive_workspace) -> None:
    """The refit must run strictly after `evaluate`, so it can never touch the reported metrics.

    Reordering them would still leave `heldOut` with the same seven keys -- every assertion above
    would keep passing -- but the numbers would be computed on data the model had just trained on.
    This pins the invariant directly instead of guessing a threshold: run once with
    `train_on_full_dataset: True` and once `False`, from the same fixture, and require the held-out
    blocks to match exactly. A reordering shifts the `True` run's numbers and this fails.
    """
    workspace = leak_sensitive_workspace
    settings = dict(SETTINGS) | {'hyperparameters': {'n_estimators': 50, 'max_depth': 6, 'random_state': 777}}
    destination_full = _destination(workspace / 'full')
    destination_partial = _destination(workspace / 'partial')

    l2g_train(_source(workspace), destination_full, dict(settings), None)
    l2g_train(_source(workspace), destination_partial, dict(settings) | {'train_on_full_dataset': False}, None)

    metrics_full = json.loads(Path(destination_full['metrics']).read_text())
    metrics_partial = json.loads(Path(destination_partial['metrics']).read_text())
    assert metrics_full['heldOut'] == metrics_partial['heldOut']

    # And the flag is not silently inert: it does change what gets SAVED, even though it must
    # never change what gets REPORTED. Same test rows, two differently-trained models.
    test_matrix = to_matrix(scan_dataset(destination_full['test_split']).collect(), FEATURES)
    proba_full = load_model(destination_full['model']).predict_proba(test_matrix)[:, 1]
    proba_partial = load_model(destination_partial['model']).predict_proba(test_matrix)[:, 1]
    assert not np.allclose(proba_full, proba_partial)


def test_l2g_train_leaves_no_local_gs_tree_for_a_cloud_destination(workspace, tmp_path, monkeypatch) -> None:
    """Every declared destination must survive being a `gs://…` URI, which in production they are.

    `release_uri` is set in production, so otter resolves each relative destination in
    `config.yaml` to a bucket URI before this transformer runs. POSIX collapses `gs://` to `gs:/`,
    so any writer built on `pathlib` -- `Path(...).write_text`, `Path(...).parent.mkdir`,
    `sio.dump` -- silently deposits the artifact under the working directory instead. The step
    still reports success, and `l2g_predict` reads the model back from the same collapsed path in
    the same directory, so nothing downstream notices that the release has no classifier, no
    `metrics.json` and no split statistics.

    The cloud boundary is stubbed rather than reached: what is asserted is that the writers route
    through `StorageHandle` (and, for the background, through polars' own cloud-aware writer with
    `mkdir=True`), and that not one byte lands in a local directory named `gs:`.
    """
    written: dict[str, object] = {}
    parquet_calls: list[tuple[str, dict]] = []

    class _Handle:
        def __init__(self, location: str) -> None:
            self.location = location

        def write(self, data: bytes) -> None:
            written[self.location] = data

        def write_text(self, data: str, encoding: str = 'utf-8') -> None:
            written[self.location] = data

    def _fake_write_parquet(self, path, **kwargs):  # a stand-in for the polars method
        parquet_calls.append((path, kwargs))

    monkeypatch.setattr('pts.transformers.l2g_train.StorageHandle', _Handle)
    monkeypatch.setattr('pts.transformers.l2g.model.StorageHandle', _Handle)
    monkeypatch.setattr('pts.transformers.l2g_train.write_dataset', lambda frame, path: None)
    monkeypatch.setattr(pl.DataFrame, 'write_parquet', _fake_write_parquet)
    monkeypatch.chdir(tmp_path)

    base = 'gs://a-release-bucket/a-run'
    destination = {
        'model': f'{base}/etc/model/locus_to_gene_model/classifier.skops',
        'background': f'{base}/etc/model/locus_to_gene_model/shap_background.parquet',
        'metrics': f'{base}/etc/model/locus_to_gene_model/metrics.json',
        'train_split': f'{base}/intermediate/l2g_train',
        'test_split': f'{base}/intermediate/l2g_test',
        'split_stats': f'{base}/intermediate/l2g_train_test_split_stats.json',
    }

    l2g_train(_source(workspace), destination, dict(SETTINGS), None)

    assert set(written) == {destination['split_stats'], destination['model'], destination['metrics']}
    assert parquet_calls == [(destination['background'], {'mkdir': True})]
    assert not (tmp_path / 'gs:').exists(), 'an artifact was written to a local gs:/ tree'
