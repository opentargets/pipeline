import socket
from pathlib import Path

import huggingface_hub.constants as hf_constants
import pytest
from huggingface_hub import hf_hub_download

from pts.transformers.clinical.hf_models import offline_hf_models, staged_model_revisions


def test_offline_hf_models_serves_staged_files_without_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = 'a' * 40
    staged = tmp_path / 'models' / 'some-org' / 'some-model' / revision
    staged.mkdir(parents=True)
    (staged / 'config.json').write_text('{"staged": true}')

    def no_network(*args, **kwargs):
        raise AssertionError('network access attempted')

    monkeypatch.setattr(socket, 'create_connection', no_network)
    previous = hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE

    with offline_hf_models(str(tmp_path / 'models')) as cache:
        assert hf_constants.HF_HUB_OFFLINE
        path = Path(hf_hub_download('some-org/some-model', 'config.json'))
        assert path.read_text() == '{"staged": true}'
        assert path.resolve().is_relative_to(cache.resolve())
        assert (cache / 'models--some-org--some-model' / 'refs' / 'main').read_text() == revision

    assert previous == (hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE)


def test_offline_hf_models_fails_when_nothing_is_staged(tmp_path: Path) -> None:
    (tmp_path / 'models').mkdir()

    with pytest.raises(ValueError, match='no models staged'), offline_hf_models(str(tmp_path / 'models')):
        pass


def _stage(root: Path, model: str, revision: str, *filenames: str) -> None:
    directory = root / model / revision
    directory.mkdir(parents=True)
    for filename in filenames:
        (directory / filename).write_text('{}')


def test_staged_model_revisions_reads_layout(tmp_path: Path) -> None:
    _stage(tmp_path, 'org-a/model-one', 'a' * 40, 'config.json', 'model.safetensors')
    _stage(tmp_path, 'org-b/model-two', 'b' * 40, 'config.json')

    assert staged_model_revisions(str(tmp_path)) == {
        'org-a/model-one': 'a' * 40,
        'org-b/model-two': 'b' * 40,
    }


def test_staged_model_revisions_rejects_two_revisions(tmp_path: Path) -> None:
    _stage(tmp_path, 'org-a/model-one', 'a' * 40, 'config.json')
    _stage(tmp_path, 'org-a/model-one', 'c' * 40, 'config.json')

    with pytest.raises(ValueError, match='org-a/model-one'):
        staged_model_revisions(str(tmp_path))
