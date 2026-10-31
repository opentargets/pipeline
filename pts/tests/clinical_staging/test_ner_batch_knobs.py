import pytest
from otter.task.model import Spec
from pydantic import ValidationError

from pts.tasks.mining_models import MiningModelsSpec
from pts.tasks.trial_models import TrialModelsSpec

SPECS: list[type[Spec]] = [TrialModelsSpec, MiningModelsSpec]


def _spec(cls: type[Spec], **knobs: int) -> Spec:
    return cls(name='t', source={}, destination={}, store_uri='store', snapshot='s', **knobs)


@pytest.mark.parametrize('cls', SPECS)
def test_ner_model_batch_size_defaults_to_32(cls: type[Spec]) -> None:
    assert cls.model_fields['ner_model_batch_size'].default == 32


@pytest.mark.parametrize('cls', SPECS)
@pytest.mark.parametrize(('knob', 'value'), [('ner_batch_size', 0), ('ner_model_batch_size', -1)])
def test_ner_batch_knobs_must_be_positive(cls: type[Spec], knob: str, value: int) -> None:
    with pytest.raises(ValidationError):
        _spec(cls, **{knob: value})
