"""Reading a source's credible sets and variants: required columns without nulls, and quality controls as lists."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.schemas.credible_set import (
    credible_set_required,
    credible_set_schema,
    credible_set_variant_required,
    credible_set_variant_schema,
)
from pts.transformers.credible_set.source_dataset import read_source_dataset

CREDIBLE_SET = {
    'id': '0' * 32,
    'chromosome': '1',
    'locusStart': 1,
    'locusEnd': 10,
    'studyId': 's',
    'studyType': 'gwas',
    'leadVariantId': '1_5_A_T',
    'fineMappingMethod': 'SuSiE',
    'qualityControls': [],
}
CREDIBLE_SET_VARIANT = {
    'credibleSetId': '0' * 32,
    'variantId': '1_5_A_T',
    'posteriorProbability': 1.0,
    'qualityControls': [],
}


def _write(tmp_path: Path, schema: pl.Schema, *rows: dict[str, Any]) -> str:
    path = tmp_path / 'dataset'
    path.mkdir()
    pl.DataFrame([{name: row.get(name) for name in schema} for row in rows], schema=schema).write_parquet(
        path / 'part-0.parquet'
    )
    return str(path)


def test_the_required_columns_are_the_id_and_its_components_and_the_variant_keys_and_pip() -> None:
    assert credible_set_required == (
        'id',
        'studyId',
        'chromosome',
        'locusStart',
        'locusEnd',
        'leadVariantId',
        'fineMappingMethod',
    )
    assert credible_set_variant_required == ('credibleSetId', 'variantId', 'posteriorProbability')


@pytest.mark.parametrize(
    ('schema', 'required', 'row'),
    [
        *(
            (credible_set_schema, credible_set_required, {**CREDIBLE_SET, column: None})
            for column in credible_set_required
        ),
        *(
            (credible_set_variant_schema, credible_set_variant_required, {**CREDIBLE_SET_VARIANT, column: None})
            for column in credible_set_variant_required
        ),
    ],
)
def test_a_null_in_a_required_column_raises_with_the_column_and_its_count(
    tmp_path: Path, schema: pl.Schema, required: tuple[str, ...], row: dict[str, Any]
) -> None:
    clean = CREDIBLE_SET if schema == credible_set_schema else CREDIBLE_SET_VARIANT
    [column] = [name for name in required if row[name] is None]
    path = _write(tmp_path, schema, row, row, clean)
    with pytest.raises(ValueError, match=rf"{path}.*'{column}': 2"):
        read_source_dataset(path, schema, required)


def test_null_quality_controls_become_an_empty_list(tmp_path: Path) -> None:
    path = _write(tmp_path, credible_set_schema, {**CREDIBLE_SET, 'qualityControls': None}, CREDIBLE_SET)
    frame = read_source_dataset(path, credible_set_schema, credible_set_required).collect()
    assert frame['qualityControls'].to_list() == [[], []]
    assert frame.schema == credible_set_schema


def test_a_dataset_without_a_schema_column_raises(tmp_path: Path) -> None:
    path = tmp_path / 'dataset'
    path.mkdir()
    pl.DataFrame({'credibleSetId': ['a'], 'variantId': ['1_5_A_T']}).write_parquet(path / 'part-0.parquet')
    with pytest.raises(pl.exceptions.SchemaError, match='missing columns'):
        read_source_dataset(str(path), credible_set_variant_schema, credible_set_variant_required)


def test_nullable_columns_may_be_null(tmp_path: Path) -> None:
    path = _write(tmp_path, credible_set_schema, {**CREDIBLE_SET, 'studyType': None})
    assert read_source_dataset(path, credible_set_schema, credible_set_required).collect().height == 1
