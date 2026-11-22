"""Tests for casting a dataset to its model's dtypes before it is written."""

from pathlib import Path
from typing import Annotated, ClassVar, Literal

import polars as pl
import pytest
from ot_datamodels import DatasetModel, PolarsDtype, PrimaryKey, StructModel, to_polars_schema
from pydantic import Field

from pts.schemas.dataset.conform import conform
from pts.schemas.dataset.validate import OUTSIDE_ENUM, validation_report

Colour = Literal['red', 'green', 'blue']
COLOUR = pl.Enum(['red', 'green', 'blue'])


class _Paint(StructModel):
    colour: Annotated[Colour, PolarsDtype(pl.Enum)]
    litres: Annotated[int, PolarsDtype(pl.Int32)]


class _Palette(DatasetModel):
    dataset_name: ClassVar[str] = '_palette'
    dataset_title: ClassVar[str] = 'palette'
    dataset_description: ClassVar[str] = 'palette'
    example: ClassVar[dict] = {
        'id': 'a',
        'colour': 'red',
        'shades': ['green'],
        'paint': {'colour': 'blue', 'litres': 1},
        'paints': [{'colour': 'red', 'litres': 2}],
    }

    id: Annotated[str, PrimaryKey()]
    colour: Annotated[Colour, PolarsDtype(pl.Enum)]
    shades: list[Annotated[Colour, PolarsDtype(pl.Enum)]] | None
    paint: _Paint | None
    paints: list[_Paint] | None = Field(default=None)


def _strings(**overrides: object) -> pl.DataFrame:
    """The palette as Spark writes it: every Enum a string, every integer an Int64."""
    row = {**_Palette.example, **overrides}
    return pl.DataFrame([{k: row[k] for k in _Palette.example}])


def test_strings_become_the_models_enums_at_every_depth() -> None:
    conformed = conform(_strings(), _Palette)

    assert conformed.schema == to_polars_schema(_Palette)
    assert conformed.row(0, named=True) == _Palette.example


def test_a_lazy_frame_is_cast_when_collected() -> None:
    conformed = conform(_strings().lazy(), _Palette)

    assert isinstance(conformed, pl.LazyFrame)
    assert conformed.collect().schema == to_polars_schema(_Palette)


def test_a_frame_already_in_the_model_is_returned_as_it_is() -> None:
    frame = pl.DataFrame([_Palette.example], schema=to_polars_schema(_Palette))

    assert conform(frame, _Palette) is frame


def test_an_undeclared_column_or_struct_field_is_kept() -> None:
    frame = _strings(paint={'colour': 'blue', 'litres': 1, 'finish': 'matt'}).with_columns(extra=pl.lit('x'))

    conformed = conform(frame, _Palette)

    assert conformed['extra'].dtype == pl.String()
    assert conformed.schema['paint'] == pl.Struct({'colour': COLOUR, 'litres': pl.Int32, 'finish': pl.String})


@pytest.mark.parametrize(
    ('overrides', 'path'),
    [
        ({'colour': 'mauve'}, 'colour'),
        ({'shades': ['green', 'mauve']}, 'shades[]'),
        ({'paint': {'colour': 'mauve', 'litres': 1}}, 'paint/colour'),
        ({'paints': [{'colour': 'mauve', 'litres': 1}]}, 'paints[]/colour'),
    ],
)
def test_a_value_outside_the_enum_raises_naming_its_path_and_value(overrides: dict, path: str) -> None:
    with pytest.raises(ValueError, match=rf"_palette .*{path.replace('[', r'\[').replace(']', r'\]')}: \['mauve'\]"):
        conform(_strings(**overrides), _Palette)


def test_a_lazy_value_outside_the_enum_raises_naming_the_column_and_value() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match=r'(?s)\["mauve"\].*col\("shades"\)'):
        conform(_strings(shades=['mauve']).lazy(), _Palette).collect()


def _write(path: Path, frame: pl.DataFrame) -> str:
    path.mkdir(parents=True)
    frame.write_parquet(path / '00000000.parquet')
    return str(path)


def test_validation_holds_a_dataset_of_strings_to_the_enums(tmp_path: Path) -> None:
    """Spark writes plain strings: a dataset whose values are all in the Enums is valid."""
    stored = _strings().with_columns(
        pl.col('paint').cast(pl.Struct({'colour': pl.String, 'litres': pl.Int32})),
        pl.col('paints').cast(pl.List(pl.Struct({'colour': pl.String, 'litres': pl.Int32}))),
    )

    assert validation_report(_write(tmp_path / 'ds', stored), _Palette).is_empty()


def test_validation_reports_a_string_outside_the_enum(tmp_path: Path) -> None:
    stored = _strings(shades=['mauve', 'teal', 'red']).with_columns(
        pl.col('paint').cast(pl.Struct({'colour': pl.String, 'litres': pl.Int32})),
        pl.col('paints').cast(pl.List(pl.Struct({'colour': pl.String, 'litres': pl.Int32}))),
    )

    report = validation_report(_write(tmp_path / 'ds', stored), _Palette)

    outside = report.filter(pl.col('check') == OUTSIDE_ENUM)
    assert outside.select('column', 'failures', 'examples').rows() == [('shades[]', 2, ['mauve', 'teal'])]


def test_validation_does_not_cast_a_wrong_integer_type(tmp_path: Path) -> None:
    """Only strings are cast to Enums; any other dtype difference is still reported."""
    report = validation_report(_write(tmp_path / 'ds', _strings()), _Palette)

    assert set(report['column']) == {'paint', 'paints'}
