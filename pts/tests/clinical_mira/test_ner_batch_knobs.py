import pytest
from pydantic import ValidationError

from pts.transformers.clinical_mira.aact_trial_models import AactTrialModelsSettings
from pts.transformers.clinical_mira.clinical_mining_models import ClinicalMiningModelsSettings

Settings = AactTrialModelsSettings | ClinicalMiningModelsSettings

SETTINGS: list[type[Settings]] = [AactTrialModelsSettings, ClinicalMiningModelsSettings]


def _settings(cls: type[Settings], **knobs: int) -> Settings:
    return cls(store_uri='store', snapshot='s', **knobs)


@pytest.mark.parametrize('cls', SETTINGS)
def test_ner_model_batch_size_defaults_to_32(cls: type[Settings]) -> None:
    assert cls.model_fields['ner_model_batch_size'].default == 32


@pytest.mark.parametrize('cls', SETTINGS)
@pytest.mark.parametrize(('knob', 'value'), [('ner_batch_size', 0), ('ner_model_batch_size', -1)])
def test_ner_batch_knobs_must_be_positive(cls: type[Settings], knob: str, value: int) -> None:
    with pytest.raises(ValidationError):
        _settings(cls, **{knob: value})
