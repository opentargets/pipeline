"""Tests over every discovered dataset model -- a new model is covered without being listed here.

`test_real_output_is_valid` validates actual step outputs, which no fixture here can stand in for
(e.g. struct field order, which only real data fixes). Point `PTS_OUTPUT_DIR` at a local output
directory holding one folder per dataset to run it::

    PTS_OUTPUT_DIR=../work/output uv run pytest tests/test_schema_outputs.py
"""

import os
from pathlib import Path

import pandera.errors as pa_errors
import polars as pl
import pytest
from ot_datamodels import DatasetModel, to_polars_schema
from ot_datamodels.discovery import discover_models

from pts.schemas.dataset.converter import to_pandera
from pts.schemas.dataset.validate import format_report, validation_report

DATASETS = discover_models()
MODELS = sorted(DATASETS.values(), key=lambda model: model.dataset_name)


def _name(model: type[DatasetModel]) -> str:
    return model.dataset_name


@pytest.mark.parametrize('model', MODELS, ids=_name)
def test_example_is_valid(model: type[DatasetModel]) -> None:
    to_pandera(model).validate(pl.DataFrame([model.example], schema=to_polars_schema(model)))


@pytest.mark.parametrize('model', MODELS, ids=_name)
def test_real_output_is_valid(model: type[DatasetModel]) -> None:
    output_dir = os.environ.get('PTS_OUTPUT_DIR')
    if not output_dir:
        pytest.skip('PTS_OUTPUT_DIR is not set')
    location = Path(output_dir) / model.dataset_name
    if not location.exists():
        pytest.skip(f'no {location}')

    report = validation_report(str(location), model)
    assert report.is_empty(), f'{model.dataset_name} breaks its model:\n{format_report(report)}'


#: (dataset, a change to its example that only its row check rejects, the check)
ROW_CHECK_VIOLATIONS = [
    ('colocalisation', {'h3': 0.7, 'h4': 0.7}, 'h3_and_h4_are_exclusive'),
    ('biosample_enrichment', {'studyId': 'GCST90000001'}, 'one_study_or_one_pooled_disease'),
]


@pytest.mark.parametrize(
    ('dataset', 'change', 'check'), ROW_CHECK_VIOLATIONS, ids=[d for d, _, _ in ROW_CHECK_VIOLATIONS]
)
def test_row_checks_reject_what_they_describe(dataset: str, change: dict, check: str) -> None:
    model = DATASETS[dataset]
    frame = pl.DataFrame([model.example | change], schema=to_polars_schema(model))

    with pytest.raises(pa_errors.SchemaErrors) as caught:
        to_pandera(model).validate(frame, lazy=True)
    assert {failure.split(':')[0] for failure in caught.value.failure_cases['check'].to_list()} == {check}
