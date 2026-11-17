"""The step validating colocalisation, which gentropy writes: it reads it, writes nothing, and fails on a violation."""

from pathlib import Path

import polars as pl
import pytest

from pts.schemas.dataset import to_polars_schema
from pts.schemas.outputs.colocalisation import Colocalisation
from pts.transformers.colocalisation_validation import colocalisation_validation, colocalisation_validation_validate


def _colocalisation(tmp_path: Path, **overrides) -> dict[str, str]:
    path = tmp_path / 'output/colocalisation'
    path.mkdir(parents=True)
    row = Colocalisation.example | overrides
    pl.DataFrame([row], schema=to_polars_schema(Colocalisation)).write_parquet(path / 'part.parquet')
    return {'colocalisation': str(path)}


def test_a_valid_colocalisation_passes_and_nothing_is_written(tmp_path: Path) -> None:
    source = _colocalisation(tmp_path)

    colocalisation_validation(source, {}, {}, None)
    colocalisation_validation_validate(source, {}, {}, None)

    assert [p.name for p in tmp_path.iterdir()] == ['output']


def test_a_colocalisation_breaking_its_model_fails(tmp_path: Path) -> None:
    source = _colocalisation(tmp_path, h4=1.5)

    with pytest.raises(ValueError, match='breaks its model Colocalisation'):
        colocalisation_validation_validate(source, {}, {}, None)


def test_the_step_refuses_a_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='destination must be empty'):
        colocalisation_validation(_colocalisation(tmp_path), {'report': str(tmp_path / 'r')}, {}, None)
