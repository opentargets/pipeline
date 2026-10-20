"""Hugging Face models staged as ``<org>/<name>/<revision>/<file>`` under a storage prefix."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import huggingface_hub.constants as hf_constants
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

if TYPE_CHECKING:
    from collections.abc import Generator


def _staged_files(models_uri: str) -> list[tuple[str, str, str, str, str]]:
    """``(path, org, name, revision, filename)`` of every file staged under ``models_uri``."""
    root = StorageHandle(models_uri)
    prefix = root.absolute.rstrip('/')
    files = []
    for path in root.glob('*/*/*/*'):
        org, name, revision, filename = path.removeprefix(prefix).strip('/').split('/')
        files.append((path, org, name, revision, filename))
    if not files:
        raise ValueError(f'no models staged under {models_uri}')
    return files


def staged_model_revisions(models_uri: str) -> dict[str, str]:
    """The revision of each model staged under ``models_uri``, keyed ``org/name``."""
    revisions: dict[str, set[str]] = {}
    for _, org, name, revision, _ in _staged_files(models_uri):
        revisions.setdefault(f'{org}/{name}', set()).add(revision)
    for model, found in revisions.items():
        if len(found) > 1:
            raise ValueError(f'{model} is staged at {len(found)} revisions under {models_uri}: {sorted(found)}')
    return {model: next(iter(found)) for model, found in revisions.items()}


@contextmanager
def offline_hf_models(models_uri: str) -> Generator[Path]:
    """Serve the Hugging Face models staged under ``models_uri`` from a local hub cache, with the hub offline.

    ``models_uri`` holds ``<org>/<name>/<revision>/<file>``.
    """
    with TemporaryDirectory(prefix='hf-hub-') as cache:
        for path, org, name, revision, filename in _staged_files(models_uri):
            model_dir = Path(cache) / f'models--{org}--{name}'
            target = model_dir / 'snapshots' / revision / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            data, _ = StorageHandle(path).read()
            target.write_bytes(data)
            (model_dir / 'refs').mkdir(exist_ok=True)
            (model_dir / 'refs' / 'main').write_text(revision)
        logger.info(f'staged hugging face models from {models_uri}: {sorted(p.name for p in Path(cache).iterdir())}')

        previous = hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE
        hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE = True, cache
        try:
            yield Path(cache)
        finally:
            hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE = previous
