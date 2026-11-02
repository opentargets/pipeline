"""The clinical staging steps in ``config.yaml`` load as transformers with valid settings."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from pts.tasks.transform import Transform
from pts.transformers.clinical_staging.aact_trial_extraction import AactTrialExtractionSettings
from pts.transformers.clinical_staging.aact_trial_models import AactTrialModelsSettings
from pts.transformers.clinical_staging.clinical_mining_models import ClinicalMiningModelsSettings

CONFIG_PATH = Path(__file__).parents[2] / 'config.yaml'
"""The config file that ships in the image."""

SETTINGS: dict[str, type[BaseModel]] = {
    'aact_trial_extraction': AactTrialExtractionSettings,
    'aact_trial_models': AactTrialModelsSettings,
    'clinical_mining_models': ClinicalMiningModelsSettings,
}


def _step(name: str) -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())['steps'][name][0]


@pytest.mark.parametrize('name', SETTINGS)
def test_staging_steps_load_from_config(name: str) -> None:
    step = _step(name)
    assert step['name'] == f'transform {name}'

    transformer = Transform.load_transformer(step['transformer'])
    assert getattr(transformer, '__name__', None) == name

    settings = {key: '2026-09-23' if value == '{{snapshot}}' else value for key, value in step['settings'].items()}
    SETTINGS[name].model_validate(settings)

    with pytest.raises(ValidationError, match='concurency'):
        SETTINGS[name].model_validate({**settings, 'concurency': 3})
