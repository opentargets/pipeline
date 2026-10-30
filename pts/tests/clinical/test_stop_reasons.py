import subprocess
import sys

import pytest

from pts.transformers.clinical import stop_reasons
from pts.transformers.clinical.stop_reasons import (
    STOP_REASONS_MODEL,
    predict_trial_stop_reasons,
    stop_reasons_with_status,
)


def _fake_predict(monkeypatch: pytest.MonkeyPatch, fail_on: set[str] | None = None) -> list[list[str]]:
    calls: list[list[str]] = []

    def predict(texts: list[str], **kwargs: object) -> list[list[str]]:
        calls.append(texts)
        if fail_on and fail_on & set(texts):
            raise RuntimeError('boom')
        return [[f'label-{t}'] for t in texts]

    monkeypatch.setattr(stop_reasons, 'predict_trial_stop_reasons', predict)
    monkeypatch.setattr(stop_reasons, '_load_model_assets', lambda model_name: None)
    return calls


def test_stop_reasons_with_status_matches_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class FakeTokenizer:
        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
            return {'x': torch.tensor([[len(t)] for t in texts], dtype=torch.float32)}

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> object:
            logits = torch.stack([x[:, 0] - 5, -x[:, 0]], dim=1)
            return type('Out', (), {'logits': logits})()

    monkeypatch.setattr(
        stop_reasons,
        '_load_model_assets',
        lambda model_name: (FakeTokenizer(), FakeModel(), {0: 'Long', 1: 'Short'}),
    )
    texts = ['abcdefgh', '', 'abcdefghijkl'] * 4

    expected = predict_trial_stop_reasons(texts, batch_size=5)
    actual = stop_reasons_with_status(texts, batch_size=5)

    assert actual == [(labels, None) for labels in expected]
    assert expected[0] == ['Long']
    assert expected[1] == ['Short']


def test_stop_reasons_failed_batch_marks_its_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_predict(monkeypatch, fail_on={'bad'})

    out = stop_reasons_with_status(['a', 'b', 'bad', 'c', 'd'], batch_size=2)

    assert calls == [['a', 'b'], ['bad', 'c'], ['d']]
    assert out == [
        (['label-a'], None),
        (['label-b'], None),
        (None, 'stop reasons batch failed: boom'),
        (None, 'stop reasons batch failed: boom'),
        (['label-d'], None),
    ]


def test_stop_reasons_model_load_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_model(model_name: str) -> None:
        raise OSError('model not staged')

    monkeypatch.setattr(stop_reasons, '_load_model_assets', no_model)

    with pytest.raises(OSError, match='model not staged'):
        stop_reasons_with_status(['a', 'b'], batch_size=1)


def test_stop_reasons_with_status_of_no_texts() -> None:
    assert stop_reasons_with_status([]) == []


def test_stop_reasons_model_name() -> None:
    assert STOP_REASONS_MODEL == 'opentargets/clinical_trial_stop_reasons'


def test_clinical_report_module_does_not_import_torch() -> None:
    code = (
        'import sys\n'
        'import pts.transformers.clinical_report\n'
        "assert 'torch' not in sys.modules, 'torch'\n"
        "assert 'transformers' not in sys.modules, 'transformers'\n"
    )

    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
