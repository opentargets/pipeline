"""Tests for the description of each kind of staged model output."""

import dataclasses
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from pts.clinical_staging.staged_outputs import (
    DISEASE_ENTITIES,
    DRUG_ENTITIES,
    EXTRACTION,
    STOP_REASONS,
    StagedOutput,
    schema_digest,
)
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
from pts.transformers.clinical.providers.aact_llm import TRIAL_FIELDS
from pts.transformers.clinical.schemas import ClinicalReportExtractionSchema
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


def test_params_hold_the_tuned_constants_each_compute_reads() -> None:
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


def extraction_snapshot(tmp_path: Path, name: str, **overrides: object) -> str:
    digest = schema_digest(ClinicalReportExtractionSchema)
    metadata = {**EXTRACTION.metadata(digest, None), **overrides}
    df = pl.DataFrame({'id': ['NCT1']})
    return snapshot(tmp_path, name, df, {k: v for k, v in metadata.items() if v is not None})


def test_schema_digest_is_the_hash_of_the_sorted_alias_schema() -> None:
    document = json.dumps(ClinicalReportExtractionSchema.model_json_schema(by_alias=True), sort_keys=True)

    assert schema_digest(ClinicalReportExtractionSchema) == hashlib.sha256(document.encode('utf-8')).hexdigest()


def test_extraction_describes_the_llm_extraction() -> None:
    assert EXTRACTION.name == 'extraction'
    assert EXTRACTION.model == 'aact_llm_extraction'
    assert EXTRACTION.logic_version == 1
    assert EXTRACTION.params == {'trial_fields': TRIAL_FIELDS}


def test_extraction_tag_is_the_schema_prefix_and_a_param_digest() -> None:
    digest = schema_digest(ClinicalReportExtractionSchema)
    document = json.dumps({'logic_version': 1, 'trial_fields': TRIAL_FIELDS, 'sample_size': None}, sort_keys=True)
    expected = hashlib.sha256(document.encode('utf-8')).hexdigest()[:8]

    assert EXTRACTION.tag(digest, None) == f'{digest[:12]}-p{expected}'
    assert re.fullmatch(r'[0-9a-f]{12}-p[0-9a-f]{8}', EXTRACTION.tag(digest, None))


def test_extraction_tag_differs_between_a_full_run_and_a_sample() -> None:
    digest = schema_digest(ClinicalReportExtractionSchema)

    assert EXTRACTION.tag(digest, None) != EXTRACTION.tag(digest, 5)
    assert EXTRACTION.tag(digest, 5) != EXTRACTION.tag(digest, 6)


def test_extraction_tag_changes_with_trial_fields_logic_version_and_schema() -> None:
    digest = schema_digest(ClinicalReportExtractionSchema)
    tag = EXTRACTION.tag(digest, None)

    assert dataclasses.replace(EXTRACTION, params={'trial_fields': {'a': 'A'}}).tag(digest, None) != tag
    assert dataclasses.replace(EXTRACTION, logic_version=2).tag(digest, None) != tag
    assert EXTRACTION.tag('f' * 64, None) != tag


def test_extraction_release_tag_is_the_full_run_of_the_current_schema() -> None:
    digest = schema_digest(ClinicalReportExtractionSchema)

    assert EXTRACTION.release_tag() == EXTRACTION.tag(digest, None)


def test_extraction_metadata_holds_the_full_schema_digest() -> None:
    digest = schema_digest(ClinicalReportExtractionSchema)

    assert EXTRACTION.metadata(digest, 5) == {
        'kind': 'extraction',
        'model': 'aact_llm_extraction',
        'tag': EXTRACTION.tag(digest, 5),
        'model_revisions': {'aact_llm_extraction': digest},
    }


def test_extraction_read_accepts_its_own_metadata(tmp_path: Path) -> None:
    uri = extraction_snapshot(tmp_path, 'extraction')

    assert EXTRACTION.read(uri).to_dict(as_series=False) == {'id': ['NCT1']}


def test_extraction_read_rejects_a_missing_key(tmp_path: Path) -> None:
    uri = extraction_snapshot(tmp_path, 'extraction', model_revisions=None)

    with pytest.raises(ValueError, match='model_revisions'):
        EXTRACTION.read(uri)


def test_extraction_read_rejects_another_kind(tmp_path: Path) -> None:
    uri = extraction_snapshot(tmp_path, 'extraction', kind='stop_reasons')

    with pytest.raises(ValueError, match="'stop_reasons'"):
        EXTRACTION.read(uri)


def test_extraction_read_rejects_another_tag_naming_both(tmp_path: Path) -> None:
    uri = extraction_snapshot(tmp_path, 'extraction', tag='abcdef012345-p00000000')

    with pytest.raises(ValueError, match='abcdef012345-p00000000') as error:
        EXTRACTION.read(uri)
    assert EXTRACTION.release_tag() in str(error.value)


def test_extraction_read_rejects_a_sampled_extraction(tmp_path: Path) -> None:
    digest = schema_digest(ClinicalReportExtractionSchema)
    uri = extraction_snapshot(tmp_path, 'extraction', tag=EXTRACTION.tag(digest, 5))

    with pytest.raises(ValueError, match='tag'):
        EXTRACTION.read(uri)


def test_stage_writes_the_extra_metadata(tmp_path: Path) -> None:
    result = STOP_REASONS.stage(
        pl.Series('text', ['a'], dtype=pl.String),
        upper,
        REVISIONS,
        str(tmp_path / 'store'),
        str(tmp_path / 'snap'),
        'run-1',
        None,
        max_attempts=3,
        extra_metadata={'extraction_tag': 'x-p1'},
    )

    metadata = json.loads((tmp_path / 'snap' / result.tag / 'metadata.json').read_text())
    assert metadata['extraction_tag'] == 'x-p1'
    assert metadata['kind'] == 'stop_reasons'


def test_read_checks_the_expected_metadata_of_the_named_sources(tmp_path: Path) -> None:
    tag = DRUG_ENTITIES.tag(REVISIONS)
    a = snapshot(
        tmp_path,
        'a',
        pl.DataFrame({'label': ['x'], 'extracted_drugs': [['X']]}),
        {
            'kind': 'drug_entities',
            'model': DRUG_NER_MODEL,
            'tag': tag,
            'model_revisions': {},
            'extraction_tag': 'right',
        },
    )
    b = drug_snapshot(tmp_path, 'b', ['y'], [['Y']])

    assert DRUG_ENTITIES.read({'a': a, 'b': b}, expect={'a': {'extraction_tag': 'right'}}).height == 2
    with pytest.raises(ValueError, match=r"a.*extraction_tag.*'wrong'.*'right'"):
        DRUG_ENTITIES.read({'a': a, 'b': b}, expect={'a': {'extraction_tag': 'wrong'}})
    with pytest.raises(ValueError, match=r"b.*extraction_tag.*'wrong'.*None"):
        DRUG_ENTITIES.read({'a': a, 'b': b}, expect={'b': {'extraction_tag': 'wrong'}})


def test_staged_outputs_loads_no_heavy_library() -> None:
    code = (
        'import sys; import pts.clinical_staging.staged_outputs; '
        "bad = [m for m in ('openai', 'torch', 'transformers') if m in sys.modules]; "
        'sys.exit(repr(bad) if bad else 0)'
    )

    assert subprocess.run([sys.executable, '-c', code], check=False, capture_output=True).returncode == 0
