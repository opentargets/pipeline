"""Tests for the l2g_predict transformer and its output assembly."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from pts.transformers.l2g.features import FEATURES
from pts.transformers.l2g.model import DEFAULT_HYPERPARAMETERS, fit, save_model
from pts.transformers.l2g_predict import build_output, l2g_predict


def test_build_output_pins_the_release_schema() -> None:
    keys = pl.DataFrame({'studyLocusId': ['sl1'], 'geneId': ['g1'], 'score': [0.5]})
    out = build_output(keys, np.array([[1.0, 2.0]], dtype=np.float32),
                       np.array([[0.1, 0.2]], dtype=np.float32), 0.25, ['a', 'b'])
    assert out.columns == ['studyLocusId', 'geneId', 'score', 'features', 'shapBaseValue']
    assert out.schema['studyLocusId'] == pl.String
    assert out.schema['geneId'] == pl.String
    assert out.schema['score'] == pl.Float64
    assert out.schema['shapBaseValue'] == pl.Float32
    assert out.schema['features'] == pl.List(
        pl.Struct({'name': pl.String, 'value': pl.Float32, 'shapValue': pl.Float32})
    )


def test_build_output_keeps_the_features_in_fitted_order() -> None:
    keys = pl.DataFrame({'studyLocusId': ['sl1'], 'geneId': ['g1'], 'score': [0.5]})
    out = build_output(keys, np.array([[1.0, 2.0]], dtype=np.float32),
                       np.array([[0.1, 0.2]], dtype=np.float32), 0.25, ['b', 'a'])
    assert [entry['name'] for entry in out['features'][0]] == ['b', 'a']
    assert [entry['value'] for entry in out['features'][0]] == [1.0, 2.0]
    assert [entry['shapValue'] for entry in out['features'][0]] == pytest.approx([0.1, 0.2])


def test_build_output_gives_every_row_the_same_base_value() -> None:
    keys = pl.DataFrame({'studyLocusId': ['sl1', 'sl2'], 'geneId': ['g1', 'g2'], 'score': [0.5, 0.6]})
    out = build_output(keys, np.zeros((2, 1), dtype=np.float32),
                       np.zeros((2, 1), dtype=np.float32), 0.25, ['a'])
    assert out['shapBaseValue'].unique().to_list() == [pytest.approx(0.25)]


@pytest.fixture
def workspace(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    matrix = pl.DataFrame(
        {
            'studyLocusId': [f'sl{i}' for i in range(n)],
            'geneId': [f'ENSG{i:011d}' for i in range(n)],
            'isProteinCoding': [1] * (n - 5) + [0] * 5,
            **{name: rng.random(n) for name in FEATURES},
        }
    )
    (tmp_path / 'fm').mkdir()
    matrix.write_parquet(tmp_path / 'fm' / 'part-0.parquet')

    credible = pl.DataFrame({
        'studyLocusId': [f'sl{i}' for i in range(n)],
        'studyType': ['gwas'] * (n - 10) + ['eqtl'] * 10,
    })
    (tmp_path / 'cs').mkdir()
    credible.write_parquet(tmp_path / 'cs' / 'part-0.parquet')

    x = rng.random((80, len(FEATURES))).astype(np.float32)
    y = (rng.random(80) > 0.5).astype(np.int32)
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 5, 'max_depth': 2})
    save_model(model, str(tmp_path / 'classifier.skops'))
    pl.DataFrame(x[:10], schema=list(FEATURES)).write_parquet(tmp_path / 'background.parquet')
    return tmp_path


@pytest.fixture
def multi_gene_workspace(tmp_path):
    """Twelve loci carrying four genes each, staged in descending key order.

    Written descending on BOTH key components so no assertion about ordering can pass by accident
    on the order the parts happen to be read in: every row has to move for the output to come out
    sorted. Everything else matches `workspace` -- all rows GWAS and protein-coding, so the row
    set is the whole matrix and only the order is under test.
    """
    rng = np.random.default_rng(0)
    pairs = [(f'sl{i:02d}', f'ENSG{g:011d}') for i in reversed(range(12)) for g in reversed(range(4))]
    n = len(pairs)

    matrix = pl.DataFrame(
        {
            'studyLocusId': [locus for locus, _ in pairs],
            'geneId': [gene for _, gene in pairs],
            'isProteinCoding': [1] * n,
            **{name: rng.random(n) for name in FEATURES},
        }
    )
    (tmp_path / 'fm').mkdir()
    matrix.write_parquet(tmp_path / 'fm' / 'part-0.parquet')

    credible = pl.DataFrame({
        'studyLocusId': [f'sl{i:02d}' for i in reversed(range(12))],
        'studyType': ['gwas'] * 12,
    })
    (tmp_path / 'cs').mkdir()
    credible.write_parquet(tmp_path / 'cs' / 'part-0.parquet')

    x = rng.random((80, len(FEATURES))).astype(np.float32)
    y = (rng.random(80) > 0.5).astype(np.int32)
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 5, 'max_depth': 2})
    save_model(model, str(tmp_path / 'classifier.skops'))
    pl.DataFrame(x[:10], schema=list(FEATURES)).write_parquet(tmp_path / 'background.parquet')
    return tmp_path


def _source(workspace) -> dict[str, str]:
    return {
        'feature_matrix': str(workspace / 'fm'),
        'credible_set': str(workspace / 'cs'),
        'model': str(workspace / 'classifier.skops'),
        'background': str(workspace / 'background.parquet'),
    }


SETTINGS = {
    'features_list': list(FEATURES),
    'l2g_threshold': 0.0,
    'explain_predictions': True,
    'shap_workers': 1,
}
"""No `shap_background_size`: the masker is sized from the background parquet, as in production."""


def test_l2g_predict_excludes_non_gwas_and_non_protein_coding_rows(workspace) -> None:
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, dict(SETTINGS), None)
    out = pl.read_parquet(f'{destination}/*.parquet')
    # 60 rows, 10 non-gwas at the tail, 5 non-protein-coding at the tail; they overlap.
    assert out.height == 50
    assert 'sl59' not in out['studyLocusId'].to_list()


def test_l2g_predict_applies_the_threshold(workspace) -> None:
    """An empty prediction set is a real production outcome, so pin exactly what it produces.

    `write_dataset` sinks through `pl.PartitionBy`, which writes one part even for a zero-row
    frame, so the dataset is a single readable `00000000.parquet` carrying the full schema and no
    rows -- not an empty directory, and not a missing one. Asserted directly rather than
    accommodated with a branch: a consumer globbing `*.parquet` sees a schema either way, and if
    that ever changes this should fail rather than quietly accept the new behaviour.
    """
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, {**SETTINGS, 'l2g_threshold': 1.1}, None)
    parts = sorted(Path(destination).glob('*.parquet'))
    assert [part.name for part in parts] == ['00000000.parquet']
    empty = pl.read_parquet(parts)
    assert empty.height == 0
    assert empty.columns == ['studyLocusId', 'geneId', 'score', 'features', 'shapBaseValue']


def test_l2g_predict_output_is_sorted_by_key(multi_gene_workspace) -> None:
    """The full `(studyLocusId, geneId)` key, on a fixture where the second component matters.

    With one gene per locus -- as in `workspace` -- an implementation that sorted on
    `studyLocusId` alone would satisfy this while violating the ordering the transformer
    declares, so the fixture gives every locus several genes and the assertion compares the
    key TUPLES.
    """
    destination = str(multi_gene_workspace / 'out')
    l2g_predict(_source(multi_gene_workspace), destination, dict(SETTINGS), None)
    out = pl.read_parquet(f'{destination}/*.parquet')

    genes_per_locus = out.group_by('studyLocusId').len()['len']
    assert int(genes_per_locus.min()) > 1, 'fixture no longer exercises the geneId component'

    keys = out.select('studyLocusId', 'geneId').rows()
    assert keys == sorted(keys)


def test_l2g_predict_writes_one_feature_entry_per_feature(workspace) -> None:
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, dict(SETTINGS), None)
    out = pl.read_parquet(f'{destination}/*.parquet')
    assert out['features'].list.len().unique().to_list() == [len(FEATURES)]


def test_l2g_predict_without_explanations_leaves_shap_null(workspace) -> None:
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, {**SETTINGS, 'explain_predictions': False}, None)
    out = pl.read_parquet(f'{destination}/*.parquet')
    assert out['shapBaseValue'].is_null().all()
    # Null, not NaN: the flag says no explanation was computed, and every downstream
    # consumer reads those as different values.
    first = out['features'][0]
    assert all(entry['shapValue'] is None for entry in first)


def test_the_two_config_tasks_share_one_feature_list() -> None:
    """The alias must survive yamlfmt: a drifted copy would fit and score on different orders."""
    config = yaml.safe_load(Path(__file__).parents[1].joinpath('config.yaml').read_text())
    tasks = {task['name']: task for task in config['steps']['l2g']}
    train = tasks['transform l2g_train']['settings']['features_list']
    predict = tasks['transform l2g_predict']['settings']['features_list']
    assert train == predict
    # And against `FEATURES`, not just against each other: two copies that drift together would
    # still agree while `features.py` went on claiming to define the order the model was fitted on.
    assert train == list(FEATURES)
    assert len(train) == 31


def test_l2g_predict_keeps_a_row_scoring_exactly_at_the_threshold(workspace) -> None:
    """The comparison is `>=`, and no gate over a released dataset can observe that.

    Flipping it to `>` leaves the parity gate passing, because no released row scores exactly at
    0.05 in float64 -- so the boundary has to be tested where the threshold can be derived from
    the data instead of imposed on it. Score everything, take a score the model actually produced,
    then re-run with that value as the threshold: under `>=` the row survives, under `>` it does
    not. No floating-point engineering is needed, because the two numbers are the same object.
    """
    settings = {**SETTINGS, 'explain_predictions': False}
    everything = str(workspace / 'everything')
    l2g_predict(_source(workspace), everything, dict(settings), None)
    scores = pl.read_parquet(f'{everything}/*.parquet')['score']
    boundary = float(scores.min())

    at_boundary = str(workspace / 'at_boundary')
    l2g_predict(_source(workspace), at_boundary, {**settings, 'l2g_threshold': boundary}, None)
    kept = pl.read_parquet(f'{at_boundary}/*.parquet')

    assert boundary in kept['score'].to_list(), 'the row scoring exactly at the threshold was dropped'
    assert kept.height == scores.len()
