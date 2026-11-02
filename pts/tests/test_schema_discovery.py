"""Tests for pts.schemas.dataset.discovery, against throwaway packages."""

import textwrap
from pathlib import Path

import pytest

from pts.schemas.dataset.discovery import discover_datasets

_HEADER = """\
from typing import ClassVar
from pts.schemas.dataset import DatasetModel, StructModel
"""


def _package(root: Path, name: str, modules: dict[str, str]) -> None:
    """Write package `name` under `root`, with `modules` given by dotted name relative to it."""
    for module, source in modules.items():
        parts = [name, *module.split('.')]
        for depth in range(1, len(parts)):
            package_dir = root.joinpath(*parts[:depth])
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / '__init__.py').touch()
        root.joinpath(*parts[:-1], f'{parts[-1]}.py').write_text(_HEADER + textwrap.dedent(source))


def test_finds_concrete_models_in_every_module_and_ignores_the_rest(tmp_path: Path, monkeypatch) -> None:
    _package(
        tmp_path,
        '_outputs_found',
        {
            'shared': """
                class Shared(DatasetModel):
                    x: str

                class Part(StructModel):
                    y: str

                class First(Shared):
                    dataset_name: ClassVar[str] = 'first'
            """,
            'deeper.more': """
                from _outputs_found.shared import First

                class Second(First):
                    dataset_name: ClassVar[str] = 'second'
            """,
        },
    )
    monkeypatch.syspath_prepend(tmp_path)

    found = discover_datasets('_outputs_found')

    assert {name: model.__name__ for name, model in found.items()} == {'first': 'First', 'second': 'Second'}


def test_rejects_a_dataset_name_declared_twice(tmp_path: Path, monkeypatch) -> None:
    _package(
        tmp_path,
        '_outputs_twice',
        {
            'a': """
                class A(DatasetModel):
                    dataset_name: ClassVar[str] = 'same'
            """,
            'b': """
                class B(DatasetModel):
                    dataset_name: ClassVar[str] = 'same'
            """,
        },
    )
    monkeypatch.syspath_prepend(tmp_path)

    with pytest.raises(ValueError, match="'same' is declared twice"):
        discover_datasets('_outputs_twice')
