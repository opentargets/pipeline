"""Tests for the description of each kind of staged model output."""

import dataclasses
import inspect
import json
import re
from pathlib import Path

import polars as pl
import pytest

from pts.clinical_staging.staged_outputs import DISEASE_ENTITIES, DRUG_ENTITIES, STOP_REASONS, StagedOutput
from pts.record_store import publish_snapshot, read_records
from pts.transformers.clinical.ner import (
    BIOBERT_LABELS,
    CHEMICAL_MODEL,
    DISEASE_DROP_DIGITS,
    DISEASE_LABELS,
    DISEASE_MIN_LENGTH,
    DISEASE_MODEL,
    DRUG_NAME_PATTERNS,
    DRUG_NER_MODEL,
    DRUGTEMIST_LABELS,
    DRUGTEMIST_MODEL,
)
from pts.transformers.clinical.stop_reasons import (
    STOP_REASONS_MODEL,
    STOP_REASONS_THRESHOLD,
    predict_trial_stop_reasons,
)

REVISIONS = {STOP_REASONS_MODEL: 'a0', CHEMICAL_MODEL: 'b1', DRUGTEMIST_MODEL: 'c2', DISEASE_MODEL: 'd3'}


def upper(texts: list[str]) -> list[tuple[list[str], None]]:
    return [([t.upper()], None) for t in texts]


def stage(kind: StagedOutput, tmp_path: Path, texts: list[str], run_id: str):
    return kind.stage(
        pl.Series('text', texts, dtype=pl.String),
        upper,
        REVISIONS,
        str(tmp_path / 'store'),
        str(tmp_path / 'snap' / kind.name),
        run_id,
        None,
        max_attempts=3,
    )


def snapshot(tmp_path: Path, name: str, df: pl.DataFrame, metadata: dict) -> str:
    uri = str(tmp_path / name)
    publish_snapshot(df, uri, metadata, None)
    return uri


def drug_snapshot(tmp_path: Path, name: str, labels: list[str], drugs: list[list[str]], tag: str | None = None) -> str:
    tag = tag or DRUG_ENTITIES.tag(REVISIONS)
    metadata = {
        'kind': 'drug_entities',
        'model': DRUG_NER_MODEL,
        'tag': tag,
        'model_revisions': {CHEMICAL_MODEL: 'b1', DRUGTEMIST_MODEL: 'c2'},
    }
    df = pl.DataFrame(
        {'label': labels, 'extracted_drugs': drugs},
        schema={'label': pl.String, 'extracted_drugs': pl.List(pl.String)},
    )
    return snapshot(tmp_path, name, df, metadata)


def test_tag_joins_hub_revisions_and_a_param_digest() -> None:
    assert re.fullmatch(r'b1\+c2-p[0-9a-f]{8}', DRUG_ENTITIES.tag({CHEMICAL_MODEL: 'b1', DRUGTEMIST_MODEL: 'c2'}))


def test_tag_changes_with_params_and_logic_version() -> None:
    tag = STOP_REASONS.tag(REVISIONS)

    assert dataclasses.replace(STOP_REASONS, params={'threshold': 0.5}).tag(REVISIONS) != tag
    assert dataclasses.replace(STOP_REASONS, logic_version=2).tag(REVISIONS) != tag


def test_params_hold_the_threshold_and_patterns_the_code_runs() -> None:
    assert STOP_REASONS.params == {'threshold': STOP_REASONS_THRESHOLD}
    assert DRUG_ENTITIES.params['patterns'] == DRUG_NAME_PATTERNS


def test_params_hold_the_values_each_compute_reads() -> None:
    default = inspect.signature(predict_trial_stop_reasons).parameters['threshold'].default
    assert STOP_REASONS.params == {'threshold': default}
    assert DRUG_ENTITIES.params == {
        'patterns': DRUG_NAME_PATTERNS,
        'biobert_labels': BIOBERT_LABELS,
        'drugtemist_labels': DRUGTEMIST_LABELS,
    }
    assert DISEASE_ENTITIES.params == {
        'labels': DISEASE_LABELS,
        'min_length': DISEASE_MIN_LENGTH,
        'drop_digits': DISEASE_DROP_DIGITS,
    }


def test_require_names_the_missing_models() -> None:
    with pytest.raises(ValueError, match=re.escape(DRUGTEMIST_MODEL)):
        DRUG_ENTITIES.require({CHEMICAL_MODEL: 'b1'})


def test_stage_keys_the_store_on_the_tag_and_publishes_under_it(tmp_path: Path) -> None:
    result = stage(STOP_REASONS, tmp_path, ['b', 'a', 'b'], 'r1')

    records = read_records(str(tmp_path / 'store' / 'stop_reasons'), None)
    assert set(records['model_revision']) == {result.tag}
    assert result.computed
    assert result.published
    snap = tmp_path / 'snap' / 'stop_reasons' / result.tag
    assert json.loads((snap / 'metadata.json').read_text()) == {
        'kind': 'stop_reasons',
        'model': STOP_REASONS_MODEL,
        'tag': result.tag,
        'model_revisions': {STOP_REASONS_MODEL: 'a0'},
    }
    data = pl.read_parquet(snap / 'data.parquet')
    assert data.columns == ['text', 'text_sha256', 'categories']
    assert data.equals(result.frame)
    assert data.select('text', 'categories').to_dicts() == [
        {'text': 'a', 'categories': ['A']},
        {'text': 'b', 'categories': ['B']},
    ]


def test_stage_rerun_computes_and_publishes_nothing(tmp_path: Path) -> None:
    first = stage(DRUG_ENTITIES, tmp_path, ['x', 'y'], 'r1')
    second = stage(DRUG_ENTITIES, tmp_path, ['x', 'y'], 'r2')

    assert second.computed is False
    assert second.published is False
    assert second.frame.equals(first.frame)
    assert second.frame.columns == ['label', 'extracted_drugs']


def test_stats_count_the_texts_and_report_the_staging(tmp_path: Path) -> None:
    first = stage(DRUG_ENTITIES, tmp_path, ['x', 'y', 'x'], 'r1')
    second = stage(DRUG_ENTITIES, tmp_path, ['x', 'y'], 'r2')

    assert first.stats() == {'texts': 2, 'computed': True, 'published': True, 'tag': first.tag}
    assert second.stats() == {'texts': 2, 'computed': False, 'published': False, 'tag': first.tag}


def test_read_merges_snapshots_of_one_kind(tmp_path: Path) -> None:
    aact = drug_snapshot(tmp_path, 'aact', ['a'], [['y', 'x']])
    mining = drug_snapshot(tmp_path, 'mining', ['b'], [['z']])

    merged = DRUG_ENTITIES.read({'aact_drug_entities': aact, 'mining_drug_entities': mining})

    assert merged.sort('label').to_dicts() == [
        {'label': 'a', 'extracted_drugs': ['x', 'y']},
        {'label': 'b', 'extracted_drugs': ['z']},
    ]


def test_read_returns_a_single_snapshot_as_staged(tmp_path: Path) -> None:
    result = stage(STOP_REASONS, tmp_path, ['a'], 'r1')
    uri = str(tmp_path / 'snap' / 'stop_reasons' / result.tag)
    df = pl.read_parquet(f'{uri}/data.parquet').with_columns(categories=pl.lit(['Z', 'A']))
    df.write_parquet(f'{uri}/data.parquet')

    read = STOP_REASONS.read({'aact_stop_reasons': uri})

    assert read.equals(df)


def test_read_fails_on_conflicting_results(tmp_path: Path) -> None:
    aact = drug_snapshot(tmp_path, 'aact', ['a', 'b'], [['x'], ['z']])
    mining = drug_snapshot(tmp_path, 'mining', ['a', 'b'], [['y'], ['z']])

    with pytest.raises(ValueError, match=r"1 drug_entities keys staged with different results, e.g. \['a'\]"):
        DRUG_ENTITIES.read({'aact_drug_entities': aact, 'mining_drug_entities': mining})


def test_read_fails_on_snapshots_with_different_tags(tmp_path: Path) -> None:
    other_revisions = {**REVISIONS, CHEMICAL_MODEL: 'b9'}
    aact = drug_snapshot(tmp_path, 'aact', ['a'], [['x']])
    mining = drug_snapshot(tmp_path, 'mining', ['b'], [['z']], tag=DRUG_ENTITIES.tag(other_revisions))

    with pytest.raises(ValueError, match=r'different tags') as exc:
        DRUG_ENTITIES.read({'aact_drug_entities': aact, 'mining_drug_entities': mining})
    assert DRUG_ENTITIES.tag(REVISIONS) in str(exc.value)
    assert DRUG_ENTITIES.tag(other_revisions) in str(exc.value)


def test_read_fails_when_the_snapshot_was_staged_by_other_code(tmp_path: Path) -> None:
    aact = drug_snapshot(tmp_path, 'aact', ['a'], [['x']], tag='b1+c2-p00000000')

    with pytest.raises(ValueError, match='aact_drug_entities') as exc:
        DRUG_ENTITIES.read({'aact_drug_entities': aact})
    assert '00000000' in str(exc.value)
    assert DRUG_ENTITIES.tag(REVISIONS).rsplit('-p', 1)[1] in str(exc.value)


def test_read_fails_on_metadata_of_another_kind(tmp_path: Path) -> None:
    uri = snapshot(
        tmp_path,
        'disease',
        pl.DataFrame({'text': ['a'], 'extracted_diseases': [['x']]}),
        {
            'kind': 'drug_entities',
            'model': DRUG_NER_MODEL,
            'tag': DISEASE_ENTITIES.tag(REVISIONS),
            'model_revisions': {},
        },
    )

    with pytest.raises(ValueError, match="kind 'drug_entities'"):
        DISEASE_ENTITIES.read({'mining_disease_entities': uri})


def test_read_fails_on_metadata_without_a_tag(tmp_path: Path) -> None:
    uri = snapshot(tmp_path, 'disease', pl.DataFrame({'text': ['a']}), {'model': 'm', 'model_revisions': {}})

    with pytest.raises(ValueError, match=r"mining_disease_entities metadata lacks \['kind', 'tag'\]"):
        DISEASE_ENTITIES.read({'mining_disease_entities': uri})
