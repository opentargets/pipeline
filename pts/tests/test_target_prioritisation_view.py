"""Tests for the target prioritisation view transformer."""

import math
from pathlib import Path

import polars as pl
import pytest
from ot_datamodels.outputs.target_essentiality import TargetEssentiality
from ot_datamodels.outputs.target_prioritisation import TargetPrioritisation
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.target_prioritisation_view import TargetPrioritisationView

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers import target_prioritisation_view as module
from pts.transformers.target_prioritisation_view import COLUMNS, ch_float, target_prioritisation_view

PRIORITISATION = to_polars_schema(TargetPrioritisation)
ESSENTIALITY = to_polars_schema(TargetEssentiality)

#: Float64 values and what ClickHouse 26.8.2.7 `toString` gives for them (read off a live server)
CH_FLOATS = [
    (0.42952615721720067, '0.42952615721720067'),
    (0.1 + 0.2, '0.30000000000000004'),
    (4.0, '4'),
    (100.0, '100'),
    (-0.5, '-0.5'),
    (0.00001, '0.00001'),
    (0.000001, '0.000001'),
    (9.9e-7, '9.9e-7'),
    (1e-7, '1e-7'),
    (2.5e-7, '2.5e-7'),
    (-2.5e-8, '-2.5e-8'),
    (1.23e-18, '1.23e-18'),
    (1.5e20, '150000000000000000000'),
    (1e20, '100000000000000000000'),
    (123456789012345680000.0, '123456789012345680000'),
    (1e21, '1e21'),
    (1.5e300, '1.5e300'),
    (1.7976931348623157e308, '1.7976931348623157e308'),
    (5e-324, '5e-324'),
    (0.0, '0'),
    (-0.0, '-0'),
    (math.nan, 'nan'),
    (math.inf, 'inf'),
    (-math.inf, '-inf'),
]


@pytest.mark.parametrize(('value', 'expected'), CH_FLOATS, ids=[e for _, e in CH_FLOATS])
def test_ch_float_is_clickhouse_to_string(value: float, expected: str):
    assert ch_float(value) == expected


def _prioritisation(target_id: str, **values) -> dict:
    return {'targetId': target_id, **dict.fromkeys(COLUMNS), **values}


ROWS = [
    _prioritisation('T1', isInMembrane=1, hasSafetyEvent=-1, geneticConstraint=0.25, maxClinicalStage=4.0),
    _prioritisation('T2', paralogMaxIdentityPercentage=-0.0, hasPocket=0),
    _prioritisation('T3', tissueSpecificity=1e-7),
    _prioritisation('T4'),
    _prioritisation('T5', celltypeDistribution=0.5),
]
ESSENTIAL = [
    {'targetId': 'T1', 'isEssential': True, 'depMapEssentiality': []},
    {'targetId': 'T2', 'isEssential': False, 'depMapEssentiality': []},
    {'targetId': 'T4', 'isEssential': True, 'depMapEssentiality': []},
    {'targetId': 'T9', 'isEssential': True, 'depMapEssentiality': []},
]


def _run(tmp_path: Path, rows: list[dict] = ROWS, essential: list[dict] = ESSENTIAL) -> pl.DataFrame:
    source = {}
    for name, data, schema in (
        ('target_prioritisation', rows, PRIORITISATION),
        ('target_essentiality', essential, ESSENTIALITY),
    ):
        (tmp_path / name).mkdir()
        pl.DataFrame(data, schema=schema).write_parquet(tmp_path / name / 'part-0.parquet')
        source[name] = str(tmp_path / name)
    destination = {'view': str(tmp_path / 'out')}
    target_prioritisation_view(source, destination, {}, None)
    return pl.read_parquet(destination['view'])


def _items(frame: pl.DataFrame) -> dict[str, list[tuple[str, str]]]:
    return {r['targetId']: [(i['key'], i['value']) for i in r['items']] for r in frame.to_dicts()}


def test_items(tmp_path: Path):
    """Non-null values in served order as toString writes them, then geneEssentiality for a target DepMap
    screened: -1 essential, 0 not; none for a target absent from target_essentiality (David, 2026-11-24) or with
    every value null."""
    result = _run(tmp_path)

    assert result.schema == to_polars_schema(TargetPrioritisationView)
    assert result['targetId'].to_list() == ['T1', 'T2', 'T3', 'T4', 'T5']
    assert _items(result) == {
        'T1': [
            ('geneticConstraint', '0.25'),
            ('hasSafetyEvent', '-1'),
            ('isInMembrane', '1'),
            ('maxClinicalStage', '4'),
            ('geneEssentiality', '-1'),
        ],
        'T2': [('hasPocket', '0'), ('paralogMaxIdentityPercentage', '-0'), ('geneEssentiality', '0')],
        'T3': [('tissueSpecificity', '1e-7')],
        'T4': [],
        'T5': [('celltypeDistribution', '0.5')],
    }


def test_every_column_in_sql_order(tmp_path: Path):
    row = _prioritisation('T1', **{c: 1 if PRIORITISATION[c].is_integer() else 0.5 for c in COLUMNS})
    (items,) = _items(_run(tmp_path, [row], [])).values()
    assert [k for k, _ in items] == COLUMNS
    assert COLUMNS[:3] == ['geneticConstraint', 'hasHighQualityChemicalProbes', 'hasLigand']
    assert COLUMNS[-2:] == ['celltypeSpecificity', 'celltypeDistribution']


def test_writes_its_model(tmp_path: Path):
    essentiality = TargetEssentiality.example | {'targetId': TargetPrioritisation.example['targetId']}
    result = _run(tmp_path, [TargetPrioritisation.example], [essentiality])

    assert OutputValidation.of(target_prioritisation_view, module).outputs == {'view': TargetPrioritisationView}
    validate_dataset(TargetPrioritisationView, str(tmp_path / 'out'))
    (items,) = _items(result).values()
    assert items[-1] == ('geneEssentiality', '-1' if essentiality['isEssential'] else '0')
    assert len(items) == len(COLUMNS) + 1
