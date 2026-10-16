"""The Ensembl core downloads of the `target` step agree with the scratchpad's db names."""

import asyncio
import re
import threading
from pathlib import Path
from typing import Any

import pytest
from otter.config.model import Config
from otter.config.yaml import parse_yaml
from otter.manifest.model import Result
from otter.scratchpad.model import Scratchpad
from otter.task.model import Spec, Task
from otter.task.task_registry import TaskRegistry
from otter.tasks.copy import CopySpec

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

KEY_PREFIX = 'ensembl_core_'
EXPLODE_PREFIX = 'explode ensembl core '
HUMAN_EXTRA_TABLES = [
    'seq_region',
    'coord_system',
    'object_xref',
    'external_db',
    'analysis',
    'protein_feature',
    'translation',
    'transcript',
    'transcript_attrib',
    'attrib_type',
    'exon',
    'exon_transcript',
]


def _config() -> dict[str, Any]:
    return parse_yaml(CONFIG_PATH)


def _core_dbs() -> dict[str, str]:
    """Species to its core db name, from the scratchpad."""
    return {k.removeprefix(KEY_PREFIX): v for k, v in _config()['scratchpad'].items() if k.startswith(KEY_PREFIX)}


def _target_task(name: str) -> dict[str, Any]:
    return next(task for task in _config()['steps']['target'] if task['name'] == name)


def _core_explodes() -> dict[str, dict[str, Any]]:
    """Species to its `explode ensembl core <species>` task."""
    return {
        task['name'].removeprefix(EXPLODE_PREFIX): task
        for task in _config()['steps']['target']
        if task['name'].startswith(EXPLODE_PREFIX)
    }


async def _run(task: Task) -> None:
    await task.run()


def _expand(registry: TaskRegistry, spec: Spec) -> list[CopySpec]:
    """Build `spec` the way the step coordinator does, running explodes to reach the copy tasks."""
    task = registry.build(spec)
    if spec.task_type != 'explode':
        assert isinstance(task.spec, CopySpec)
        return [task.spec]
    task.context.abort = threading.Event()
    asyncio.run(_run(task))
    assert task.manifest.result not in {Result.FAILURE, Result.ABORTED}, task.manifest.failure_reason
    return [leaf for child in task.context.specs for leaf in _expand(registry, child)]


def _core_copies() -> list[CopySpec]:
    """Every copy task the core explodes generate, fully resolved."""
    config_dict = _config()
    registry = TaskRegistry(
        Config(step='target', steps=list(config_dict['steps'])),
        Scratchpad(sentinel_dict=config_dict['scratchpad']),
    )
    registry.register('otter.tasks')
    registry.register('pis.tasks')
    return [leaf for task in _core_explodes().values() for leaf in _expand(registry, Spec(**task))]


@pytest.fixture(scope='module')
def core_copies() -> list[CopySpec]:
    return _core_copies()


def test_there_is_one_core_db_per_homology_species() -> None:
    """The core dbs are the Compara species, read from the `explode compara homologies` task by name."""
    assert sorted(_core_dbs()) == sorted(_target_task('explode compara homologies')['foreach'])
    assert len(_core_dbs()) == 13


def test_every_core_db_is_from_the_configured_release() -> None:
    release = _config()['scratchpad']['ensembl_version']
    for species, db in _core_dbs().items():
        assert f'_{release}_' in db, f'{species}: {db} is not from release {release}'
        assert re.fullmatch(rf'{species}_core_(\d+_)?{release}_\d+', db), f'{species}: {db} is not its core db'


def test_every_species_has_one_core_explode() -> None:
    assert sorted(_core_explodes()) == sorted(_core_dbs())


def test_expanded_copies_are_concrete(core_copies: list[CopySpec]) -> None:
    for spec in core_copies:
        assert '$' not in spec.name + spec.source + spec.destination, spec.name


def test_expanded_destinations(core_copies: list[CopySpec]) -> None:
    expected = {
        f'input/ensembl/{species}/{name}'
        for species, db in _core_dbs().items()
        for name in (f'{db}.sql.gz', 'gene.txt.gz', 'xref.txt.gz')
    } | {f'input/ensembl/homo_sapiens/{table}.txt.gz' for table in HUMAN_EXTRA_TABLES}
    destinations = [spec.destination for spec in core_copies]
    assert len(destinations) == 13 * 3 + 12
    assert sorted(destinations) == sorted(expected)


def test_each_copy_reads_its_own_species_db(core_copies: list[CopySpec]) -> None:
    release = _config()['scratchpad']['ensembl_version']
    dbs = _core_dbs()
    for spec in core_copies:
        species, filename = Path(spec.destination).parts[-2:]
        assert spec.source == f'https://ftp.ensembl.org/pub/release-{release}/mysql/{dbs[species]}/{filename}'
