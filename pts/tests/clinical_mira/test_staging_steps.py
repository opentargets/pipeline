"""The clinical staging steps in ``config.yaml`` load as transformers with valid settings."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from pts.tasks.transform import Transform
from pts.transformers.clinical_mira.aact_trial_extraction import AactTrialExtractionSettings, aact_trial_extraction
from pts.transformers.clinical_mira.aact_trial_models import AactTrialModelsSettings, aact_trial_models
from pts.transformers.clinical_mira.clinical_mining_models import (
    ClinicalMiningModelsSettings,
    clinical_mining_models,
)

CONFIG_PATH = Path(__file__).parents[2] / 'config.yaml'
"""The config file that ships in the image."""

STEPS: dict[str, tuple[Callable[..., None], type[BaseModel]]] = {
    'aact_trial_extraction': (aact_trial_extraction, AactTrialExtractionSettings),
    'aact_trial_models': (aact_trial_models, AactTrialModelsSettings),
    'clinical_mining_models': (clinical_mining_models, ClinicalMiningModelsSettings),
}


def _step(name: str) -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())['steps'][name][0]


@pytest.mark.parametrize('name', STEPS)
def test_staging_steps_load_from_config(name: str) -> None:
    step = _step(name)
    assert step['name'] == f'transform {name}'

    function, settings_model = STEPS[name]
    assert Transform.load_transformer(step['transformer']) is function

    settings = {key: '2026-09-23' if value == '{{snapshot}}' else value for key, value in step['settings'].items()}
    settings_model.model_validate(settings)

    with pytest.raises(ValidationError, match='concurency'):
        settings_model.model_validate({**settings, 'concurency': 3})
