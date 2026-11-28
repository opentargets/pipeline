"""`copy_view` copies an output into its view model's columns, in model order, and the config places every view."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from ot_datamodels.discovery import discover_models
from ot_datamodels.outputs.evidence_eva import EvidenceEva
from ot_datamodels.outputs.go import Go
from ot_datamodels.outputs.literature_entity_lut import LiteratureEntityLut
from ot_datamodels.views.evidence_views import EVIDENCE_VIEWS
from ot_datamodels.views.go_view import GoView
from ot_datamodels.views.literature_entity_lut_view import LiteratureEntityLutView

from pts.schemas.dataset.conform import conform
from pts.tasks.transform import Transform
from pts.transformers.copy_view import copy_view, validate
from pts.transformers.utils.dataset import scan_dataset, write_dataset

CONFIG = Path(__file__).parents[1] / 'config.yaml'


def _output(tmp_path: Path, model: Any, name: str) -> str:
    path = str(tmp_path / name)
    frame = pl.DataFrame([model.example]).with_columns(pl.col(pl.Int64).cast(pl.Int32))
    write_dataset(conform(frame, model), path)
    return path


def test_a_whole_copy_changes_no_row_and_no_type(tmp_path: Path) -> None:
    source = _output(tmp_path, Go, 'go')
    destination = str(tmp_path / 'go_view')

    copy_view(source, destination, None, None)
    validate(source, destination, None, None)

    copied = scan_dataset(destination).collect()
    assert copied.columns == list(GoView.model_fields)
    assert copied.equals(scan_dataset(source).collect())
    assert copied.schema == scan_dataset(source).collect().schema


def test_a_projection_keeps_the_view_columns_in_model_order(tmp_path: Path) -> None:
    source = _output(tmp_path, LiteratureEntityLut, 'literature_entity_lut')
    destination = str(tmp_path / 'literature_entity_lut_view')

    copy_view(source, destination, None, None)
    validate(source, destination, None, None)

    original = scan_dataset(source).collect()
    copied = scan_dataset(destination).collect()
    assert copied.columns == ['pmid', 'pmcid', 'keywordId', 'relevance', 'date', 'year', 'month']
    assert copied.columns == list(LiteratureEntityLutView.model_fields)
    assert copied.equals(original.select(copied.columns))
    assert copied.schema == original.select(copied.columns).schema


def test_an_unknown_basename_raises(tmp_path: Path) -> None:
    source = _output(tmp_path, Go, 'go')
    with pytest.raises(ValueError, match='names no view model'):
        copy_view(source, str(tmp_path / 'not_a_view'), None, None)
    with pytest.raises(ValueError, match='names no view model'):
        copy_view(source, str(tmp_path / 'go'), None, None)


def test_a_source_missing_a_column_raises(tmp_path: Path) -> None:
    source = str(tmp_path / 'go')
    write_dataset(conform(pl.DataFrame([Go.example]), Go).drop('label'), source)
    destination = tmp_path / 'go_view'
    with pytest.raises(ValueError, match="lacks the columns \\['label'\\]"):
        copy_view(source, str(destination), None, None)
    assert not destination.exists()


def test_validation_fails_on_a_bad_copy(tmp_path: Path) -> None:
    destination = str(tmp_path / 'go_view')
    write_dataset(pl.DataFrame({'id': ['GO:1']}), destination)
    with pytest.raises(ValueError, match='GoView'):
        validate('', destination, None, None)


def _tasks() -> list[dict[str, Any]]:
    steps: dict[str, list[dict[str, Any]]] = yaml.safe_load(CONFIG.read_text())['steps']
    return [{'step': step, **task} for step, tasks in steps.items() for task in tasks or []]


def test_every_served_view_is_written_by_exactly_one_task() -> None:
    tasks = _tasks()
    served = [name for name, model in discover_models().items() if model.clickhouse is not None]
    assert len(served) >= 39
    for name in served:
        writers = [
            t['name']
            for t in tasks
            if name in [p.rstrip('/') for p in _destinations(t)] or f'view/{name}' in _destinations(t)
        ]
        assert len(writers) == 1, f'{name} is written by {writers}'


def _destinations(task: dict[str, Any]) -> list[str]:
    destination = task.get('destination')
    paths = destination.values() if isinstance(destination, dict) else [destination]
    return [p.rstrip('/') for p in paths if isinstance(p, str)]


def test_every_copy_task_requires_its_producer() -> None:
    tasks = _tasks()
    copies = [t for t in tasks if t.get('transformer') == 'copy_view']
    assert len(copies) == 13 + 49 + 3
    for copy in copies:
        view = copy['destination'].removeprefix('view/')
        assert discover_models()[view].clickhouse is not None
        producers = [
            t for t in tasks if t['step'] == copy['step'] and t is not copy and copy['source'] in _destinations(t)
        ]
        assert [p['name'] for p in producers] == copy['requires'], copy['name']
        assert Transform.load_output_validation('copy_view').hook is validate


def test_a_rerun_replaces_the_copy(tmp_path: Path) -> None:
    source = _output(tmp_path, Go, 'go')
    destination = str(tmp_path / 'go_view')

    copy_view(source, destination, None, None)
    copy_view(source, destination, None, None)

    assert scan_dataset(destination).collect().equals(scan_dataset(source).collect())


def _eva(tmp_path: Path) -> str:
    """An `evidence_eva` of two rows, the second with no variant."""
    rows = [EvidenceEva.example, {**EvidenceEva.example, 'id': 'f' * 40, 'variantId': None}]
    path = str(tmp_path / 'evidence_eva')
    write_dataset(
        conform(pl.DataFrame(rows, schema_overrides={'qualityControls': pl.List(pl.String)}), EvidenceEva), path
    )
    return path


def test_drop_nulls_drops_the_rows_null_in_a_non_nullable_view_column(tmp_path: Path) -> None:
    source = _eva(tmp_path)
    destination = str(tmp_path / 'evidence_eva_by_variant_view')

    copy_view(source, destination, {'drop_nulls': True}, None)
    validate(source, destination, None, None)

    copied = scan_dataset(destination).collect()
    expected = scan_dataset(source).collect().filter(pl.col('variantId').is_not_null())
    assert copied.columns == ['datasourceId', 'variantId', 'score', 'id']
    assert copied.equals(expected.select(copied.columns))


def test_without_drop_nulls_a_null_key_row_is_copied_and_fails_validation(tmp_path: Path) -> None:
    source = _eva(tmp_path)
    destination = str(tmp_path / 'evidence_eva_by_variant_view')

    copy_view(source, destination, None, None)

    assert scan_dataset(destination).collect().height == 2
    with pytest.raises(ValueError, match='EvidenceEvaByVariantView'):
        validate(source, destination, None, None)


def test_drop_nulls_leaves_a_whole_copy_unchanged(tmp_path: Path) -> None:
    source = _eva(tmp_path)
    destination = str(tmp_path / 'evidence_eva_view')

    copy_view(source, destination, {'drop_nulls': True}, None)
    validate(source, destination, None, None)

    assert scan_dataset(destination).collect().equals(scan_dataset(source).collect())


def test_drop_nulls_must_be_a_bool(tmp_path: Path) -> None:
    source = _eva(tmp_path)
    with pytest.raises(TypeError, match='drop_nulls must be true or false'):
        copy_view(source, str(tmp_path / 'evidence_eva_view'), {'drop_nulls': 'false'}, None)


def test_only_the_index_views_and_the_clinical_splits_drop_nulls() -> None:
    copies = {t['destination'].removeprefix('view/'): t for t in _tasks() if t.get('transformer') == 'copy_view'}
    dropping = {view for view, t in copies.items() if (t.get('settings') or {}).get('drop_nulls')}
    indexes = {m.dataset_name for m in EVIDENCE_VIEWS} - {f'{n}_view' for n in _evidence_outputs()}
    assert dropping == indexes | {
        'clinical_indication_by_drug_view',
        'clinical_indication_by_disease_view',
        'clinical_target_by_target_view',
    }
    assert len(indexes) == 26
    assert all('settings' not in copies[f'{n}_view'] for n in _evidence_outputs())


def _evidence_outputs() -> list[str]:
    return [n for n in discover_models() if n.startswith('evidence_') and not n.endswith('_view')]
