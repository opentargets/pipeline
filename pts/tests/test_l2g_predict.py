"""Tests for the l2g_predict transformer and its output assembly."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from pts.transformers.l2g.features import FEATURES
from pts.transformers.l2g_predict import build_output


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








"""No `shap_background_size`: the masker is sized from the background parquet, as in production."""












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


