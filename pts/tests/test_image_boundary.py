"""The lean pts image carries no torch, transformers, openai or pyspark."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

CONFIG = Path(__file__).parents[1] / 'config.yaml'
PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'
LOCK = Path(__file__).parents[1] / 'uv.lock'

FORBIDDEN = ('torch', 'transformers', 'openai', 'pyspark')
STAGING_STEPS = frozenset({
    'aact_moa',
    'moa_literature',
    'aact_trial_extraction_v2',
    'aact_trial_models',
    'clinical_mining_models',
})

BLOCK_AND_IMPORT = """
import importlib
import importlib.machinery
import sys

forbidden = set(sys.argv[1].split(','))


class HidingPathFinder(importlib.machinery.PathFinder):
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name.split('.')[0] in forbidden:
            return None
        return super().find_spec(name, path, target)


sys.meta_path[:] = [HidingPathFinder if f is importlib.machinery.PathFinder else f for f in sys.meta_path]
for module in sys.argv[2:]:
    importlib.import_module(module)
"""


def _transformers(node: object) -> set[str]:
    if isinstance(node, dict):
        found = {node['transformer']} if isinstance(node.get('transformer'), str) else set()
        return found.union(*(_transformers(value) for value in node.values()))
    if isinstance(node, list):
        return set().union(*(_transformers(item) for item in node))
    return set()


def _is_spark_step(tasks: list[dict]) -> bool:
    return any(str(task.get('name', '')).startswith('pyspark') for task in tasks)


def unified_vm_transformer_modules() -> list[str]:
    steps = yaml.safe_load(CONFIG.read_text())['steps']
    names = set().union(
        *(
            _transformers(tasks)
            for step, tasks in steps.items()
            if step not in STAGING_STEPS and not _is_spark_step(tasks)
        )
    )
    return sorted(f'pts.transformers.{name}' for name in names)


def test_unified_vm_steps_import_without_heavy_dependencies():
    modules = unified_vm_transformer_modules()
    assert modules

    result = subprocess.run(
        [sys.executable, '-c', BLOCK_AND_IMPORT, ','.join(FORBIDDEN), *modules],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not [name for name in FORBIDDEN if f"No module named '{name}" in result.stderr]


def _requirement_name(requirement: str) -> str:
    return re.split(r'[\s;\[<>=!~]', requirement, maxsplit=1)[0].lower()


def test_pts_does_not_depend_on_openai():
    project = tomllib.loads(PYPROJECT.read_text())['project']
    extras = project['optional-dependencies'].values()
    requirements = [*project['dependencies'], *(r for extra in extras for r in extra)]
    assert 'openai' not in {_requirement_name(r) for r in requirements}

    packages = tomllib.loads(LOCK.read_text())['package']
    assert 'openai' not in {package['name'] for package in packages}
