"""Every discovered dataset model, output or view, accepts its own example."""

import pytest

from ot_datamodels import DatasetModel
from ot_datamodels.discovery import discover_models

MODELS = sorted(discover_models().values(), key=lambda model: model.dataset_name)


@pytest.mark.parametrize('model', MODELS, ids=lambda model: model.dataset_name)
def test_example_is_valid(model: type[DatasetModel]) -> None:
    model.model_validate(model.example)
