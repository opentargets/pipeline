"""Tests for the trial_models staging task."""

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.clinical_staging import trial_models
from pts.clinical_staging.staged_outputs import DRUG_ENTITIES, EXTRACTION, STOP_REASONS
from pts.clinical_staging.trial_models import (
    read_extraction,
    seed_drug_entities,
    stage_trial_models,
    why_stopped_texts,
)
from pts.record_store import map_texts, read_records, text_sha256
from pts.transformers.clinical import extract_aact_clinical_report
from pts.transformers.clinical.hf_models import staged_model_revisions
from pts.transformers.clinical.ner import CHEMICAL_MODEL, DRUG_NER_MODEL, DRUGTEMIST_MODEL
from pts.transformers.clinical.providers.aact import llm_drug_labels
from pts.transformers.clinical.stop_reasons import STOP_REASONS_MODEL
from pts.transformers.clinical_report import attach_stop_reasons

REVISIONS = {STOP_REASONS_MODEL: 'rs', CHEMICAL_MODEL: 'rc', DRUGTEMIST_MODEL: 'rd'}
STOP_TAG = STOP_REASONS.tag(REVISIONS)
DRUG_TAG = DRUG_ENTITIES.tag(REVISIONS)
DRUG_STRUCT = pl.Struct({
    'drug': pl.String,
    'route': pl.String,
    'formulation': pl.String,
    'synonyms': pl.List(pl.String),
    'dosages': pl.List(pl.String),
    'evidence_quote': pl.String,
})
DISEASE_STRUCT = pl.Struct({
    'name': pl.String,
    'severity': pl.String,
    'stage': pl.String,
    'onset': pl.String,
    'etiology': pl.String,
    'evidence_quote': pl.String,
})


def _drug(name: str | None) -> dict[str, Any]:
    return {
        'drug': name,
        'route': None,
        'formulation': None,
        'synonyms': None,
        'dosages': None,
        'evidence_quote': None,
    }


def _extractions(drugs: list[list[str | None]]) -> pl.DataFrame:
    disease = {'name': 'pain', 'severity': None, 'stage': None, 'onset': None, 'etiology': None, 'evidence_quote': None}
    return pl.DataFrame(
        {
            'id': [f'NCT{i}' for i in range(len(drugs))],
            'primary_indications': [[disease] for _ in drugs],
            'investigated_drugs': [[_drug(d) for d in row] for row in drugs],
        },
        schema={
            'id': pl.String,
            'primary_indications': pl.List(DISEASE_STRUCT),
            'investigated_drugs': pl.List(DRUG_STRUCT),
        },
    )


def _studies(why_stopped: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {'nct_id': [f'NCT{i}' for i in range(len(why_stopped))], 'why_stopped': why_stopped},
        schema={'nct_id': pl.String, 'why_stopped': pl.String},
    )


@pytest.fixture
def models_uri(tmp_path: Path) -> str:
    for model, revision in REVISIONS.items():
        directory = tmp_path / 'models' / model / revision
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text('{}')
    return str(tmp_path / 'models')


class Models:
    """Fake model computes that remember what they were asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.stop_calls: list[list[str]] = []
        self.ner_calls: list[list[str]] = []
        monkeypatch.setattr(trial_models, 'stop_reasons_with_status', self.stop_reasons)
        monkeypatch.setattr(trial_models, 'drug_entities_with_status', self.drug_entities)

    def stop_reasons(self, texts: list[str], batch_size: int = 32) -> list[tuple[list[str] | None, str | None]]:
        self.stop_calls.append(list(texts))
        return [([f'reason:{t}'], None) for t in texts]

    def drug_entities(self, texts: list[str], batch_size: int) -> list[tuple[list[str] | None, str | None]]:
        self.ner_calls.append(list(texts))
        return [([f'drug:{t}'], None) for t in texts]


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Models:
    return Models(monkeypatch)


def _stage(
    tmp_path: Path, models_uri: str, studies: pl.DataFrame, extractions: pl.DataFrame, run_id: str
) -> dict[str, Any]:
    return stage_trial_models(
        studies,
        extractions,
        models_uri,
        {'stop_reasons': str(tmp_path / 'snap/stop_reasons'), 'drug_entities': str(tmp_path / 'snap/drug_entities')},
        str(tmp_path / 'store'),
        run_id,
        None,
        ner_batch_size=4,
        stop_reasons_batch_size=2,
        max_attempts=3,
        extraction_tag=EXTRACTION.release_tag(),
    )


def test_release_drug_labels_are_staged() -> None:
    studies = pl.DataFrame({
        'nct_id': ['NCT0', 'NCT1'],
        'study_type': ['INTERVENTIONAL'] * 2,
        'phase': ['PHASE2'] * 2,
        'official_title': ['t'] * 2,
    })
    interventions = pl.DataFrame({'nct_id': ['NCT0'], 'intervention_type': ['DRUG'], 'name': ['Aspirin']})
    conditions = pl.DataFrame({'nct_id': ['NCT0'], 'downcase_name': ['pain']})
    raw = ['Aspirin', ' Ibuprofen ', r'drug \[X\]', 'a\x0041b', 'nul\x00x']
    extractions = _extractions([[*raw, None], [''], []])

    report = extract_aact_clinical_report(studies, interventions, conditions, llm_extractions=extractions).df

    released = report['drugs'].explode().struct.field('drugFromSource').drop_nulls().unique().sort()
    assert released.to_list() == llm_drug_labels(extractions).to_list() == sorted(['', *raw])


def test_trial_models_publishes_both_snapshots(tmp_path: Path, models_uri: str, models: Models) -> None:
    studies = _studies(['no funds', None, 'no funds', 'slow'])
    stats = _stage(tmp_path, models_uri, studies, _extractions([['a', 'b']]), 'r1')

    stop = pl.read_parquet(tmp_path / f'snap/stop_reasons/{STOP_TAG}/data.parquet')
    ner = pl.read_parquet(tmp_path / f'snap/drug_entities/{DRUG_TAG}/data.parquet')
    assert stop.columns == ['text', 'text_sha256', 'categories']
    assert stop['text'].to_list() == ['no funds', 'slow']
    assert stop['categories'].to_list() == [['reason:no funds'], ['reason:slow']]
    assert stop['text_sha256'].to_list() == [text_sha256('no funds'), text_sha256('slow')]
    assert ner.columns == ['label', 'extracted_drugs']
    assert ner.to_dicts() == [
        {'label': 'a', 'extracted_drugs': ['drug:a']},
        {'label': 'b', 'extracted_drugs': ['drug:b']},
    ]
    kinds = (
        (STOP_REASONS, STOP_TAG, STOP_REASONS_MODEL, {STOP_REASONS_MODEL: 'rs'}),
        (DRUG_ENTITIES, DRUG_TAG, DRUG_NER_MODEL, {CHEMICAL_MODEL: 'rc', DRUGTEMIST_MODEL: 'rd'}),
    )
    for kind, tag, model, hub in kinds:
        metadata = json.loads((tmp_path / f'snap/{kind.name}/{tag}/metadata.json').read_text())
        expected = {'kind': kind.name, 'model': model, 'tag': tag, 'model_revisions': hub}
        if kind is DRUG_ENTITIES:
            expected['extraction_tag'] = EXTRACTION.release_tag()
        assert metadata == expected
    assert stats['stop_reasons'] == {'texts': 2, 'computed': True, 'published': True, 'tag': STOP_TAG}
    assert stats['drug_entities'] == {'texts': 2, 'computed': True, 'published': True, 'tag': DRUG_TAG}


def test_trial_models_publish_under_their_tags(tmp_path: Path, models_uri: str, models: Models) -> None:
    _stage(tmp_path, models_uri, _studies(['slow']), _extractions([['a']]), 'r1')

    assert sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / 'snap').rglob('data.parquet')) == [
        f'snap/drug_entities/{DRUG_TAG}/data.parquet',
        f'snap/stop_reasons/{STOP_TAG}/data.parquet',
    ]


def test_rerun_computes_nothing_and_rewrites_nothing(tmp_path: Path, models_uri: str, models: Models) -> None:
    studies, extractions = _studies(['no funds', 'slow']), _extractions([['a']])
    _stage(tmp_path, models_uri, studies, extractions, 'r1')
    snapshot = (tmp_path / f'snap/stop_reasons/{STOP_TAG}/data.parquet').stat().st_mtime_ns
    models.stop_calls.clear()
    models.ner_calls.clear()

    stats = _stage(tmp_path, models_uri, studies, extractions, 'r2')

    assert models.stop_calls == models.ner_calls == []
    assert stats['stop_reasons'] == {'texts': 2, 'computed': False, 'published': False, 'tag': STOP_TAG}
    assert stats['drug_entities'] == {'texts': 1, 'computed': False, 'published': False, 'tag': DRUG_TAG}
    assert (tmp_path / f'snap/stop_reasons/{STOP_TAG}/data.parquet').stat().st_mtime_ns == snapshot
    assert not (tmp_path / 'store/stop_reasons/runs/r2').exists()


def test_seed_imports_the_ner_cache_once(tmp_path: Path) -> None:
    cache = tmp_path / 'ner_cache.parquet'
    pl.DataFrame({
        'query_label': ['aspirin', None, 'x'],
        'extracted_drugs': [['aspirin'], ['dropped'], None],
    }).write_parquet(cache)
    store = str(tmp_path / 'store')

    assert seed_drug_entities(str(cache), store, DRUG_NER_MODEL, DRUG_TAG, None) == 2
    assert seed_drug_entities(str(cache), store, DRUG_NER_MODEL, DRUG_TAG, None) == 0

    records = read_records(f'{store}/drug_entities', None).sort('text')
    assert records['text'].to_list() == ['aspirin', 'x']
    assert records['output'].to_list() == [['aspirin'], []]
    assert records['status'].to_list() == ['ok', 'ok']
    assert records['run_id'].to_list() == ['seed-ner-cache'] * 2
    assert records['model'].to_list() == [DRUG_NER_MODEL] * 2
    assert records['model_revision'].to_list() == [DRUG_TAG] * 2
    assert records['text_sha256'].to_list() == [text_sha256('aspirin'), text_sha256('x')]


def test_seeded_labels_are_not_recomputed(tmp_path: Path, models_uri: str, models: Models) -> None:
    cache = tmp_path / 'ner_cache.parquet'
    pl.DataFrame({'query_label': ['a'], 'extracted_drugs': [['seeded']]}).write_parquet(cache)
    seed_drug_entities(str(cache), str(tmp_path / 'store'), DRUG_NER_MODEL, DRUG_TAG, None)

    _stage(tmp_path, models_uri, _studies([]), _extractions([['a', 'b']]), 'r1')

    assert models.ner_calls == [['b']]
    ner = pl.read_parquet(tmp_path / f'snap/drug_entities/{DRUG_TAG}/data.parquet')
    assert ner.to_dicts() == [
        {'label': 'a', 'extracted_drugs': ['seeded']},
        {'label': 'b', 'extracted_drugs': ['drug:b']},
    ]


def test_empty_and_blank_texts_are_staged_verbatim(tmp_path: Path, models_uri: str, models: Models) -> None:
    studies = _studies(['', '  ', 'x', None])
    extractions = _extractions([['', ' ', 'a']])

    assert why_stopped_texts(studies).to_list() == ['', '  ', 'x']
    _stage(tmp_path, models_uri, studies, extractions, 'r1')

    assert models.stop_calls == [['', '  ', 'x']]
    assert models.ner_calls == [['', ' ', 'a']]
    assert pl.read_parquet(tmp_path / f'snap/stop_reasons/{STOP_TAG}/data.parquet')['text'].to_list() == ['', '  ', 'x']
    assert pl.read_parquet(tmp_path / f'snap/drug_entities/{DRUG_TAG}/data.parquet')['label'].to_list() == [
        '',
        ' ',
        'a',
    ]


ESCAPED = r'Principal investigator \[PI\] has left \_the\_ institution.'
SANITISED = 'Principal investigator [PI] has left _the_ institution.'


def test_why_stopped_is_staged_as_the_release_sanitises_it() -> None:
    studies = _studies([ESCAPED, SANITISED, None, 'a\x0041b'])

    assert why_stopped_texts(studies).to_list() == sorted([SANITISED, 'aAb'])


def test_staged_stop_reasons_join_the_release_report(tmp_path: Path, models_uri: str, models: Models) -> None:
    studies = pl.DataFrame({
        'nct_id': ['NCT0'],
        'study_type': ['INTERVENTIONAL'],
        'phase': ['PHASE2'],
        'official_title': ['t'],
        'why_stopped': [ESCAPED],
    })
    interventions = pl.DataFrame({'nct_id': ['NCT0'], 'intervention_type': ['DRUG'], 'name': ['Aspirin']})
    conditions = pl.DataFrame({'nct_id': ['NCT0'], 'downcase_name': ['pain']})
    extractions = _extractions([['Aspirin']])
    _stage(tmp_path, models_uri, studies.select('nct_id', 'why_stopped'), extractions, 'r1')

    aact = extract_aact_clinical_report(studies, interventions, conditions, llm_extractions=extractions).df
    staged = pl.read_parquet(tmp_path / f'snap/stop_reasons/{STOP_TAG}/data.parquet')
    attached = attach_stop_reasons(aact, staged)

    assert attached['trialWhyStopped'].to_list() == [SANITISED]
    assert attached['trialStopReasonCategories'].to_list() == [[f'reason:{SANITISED}']]


def test_stage_trial_models_names_missing_models(tmp_path: Path, models_uri: str, models: Models) -> None:
    shutil.rmtree(Path(models_uri) / DRUGTEMIST_MODEL)

    with pytest.raises(ValueError, match=DRUGTEMIST_MODEL):
        _stage(tmp_path, models_uri, _studies(['slow']), _extractions([['a']]), 'r1')

    assert not (tmp_path / 'snap').exists()


def test_seed_records_use_the_drug_entities_tag(tmp_path: Path, models_uri: str, models: Models) -> None:
    cache = tmp_path / 'ner_cache.parquet'
    pl.DataFrame({'query_label': ['a'], 'extracted_drugs': [['seeded']]}).write_parquet(cache)
    store = str(tmp_path / 'store')
    seed_drug_entities(str(cache), store, DRUG_NER_MODEL, DRUG_ENTITIES.tag(staged_model_revisions(models_uri)), None)

    _stage(tmp_path, models_uri, _studies([]), _extractions([['a', 'b']]), 'r1')

    records = read_records(f'{store}/drug_entities', None)
    assert sorted(records['run_id'].unique().to_list()) == ['r1', 'seed-ner-cache']
    assert records['model_revision'].unique().to_list() == [DRUG_TAG]


def test_seed_never_outranks_a_computed_record(tmp_path: Path) -> None:
    store = str(tmp_path / 'store')
    revision = DRUG_TAG
    drug_store = f'{store}/drug_entities'
    map_texts(
        pl.Series(['a']),
        lambda texts: [(['computed'], None) for _ in texts],
        drug_store,
        DRUG_NER_MODEL,
        revision,
        'r1',
        None,
    )
    cache = tmp_path / 'ner_cache.parquet'
    pl.DataFrame({'query_label': ['a'], 'extracted_drugs': [['seeded']]}).write_parquet(cache)

    seed_drug_entities(str(cache), store, DRUG_NER_MODEL, revision, None)

    result = map_texts(pl.Series(['a']), lambda texts: [], drug_store, DRUG_NER_MODEL, revision, 'r2', None)
    assert result['output'].to_list() == [['computed']]


def test_the_drug_snapshot_records_the_extraction_it_was_staged_from(
    tmp_path: Path, models_uri: str, models: Models
) -> None:
    _stage(tmp_path, models_uri, _studies([]), _extractions([['a']]), 'r1')

    metadata = json.loads((tmp_path / f'snap/drug_entities/{DRUG_TAG}/metadata.json').read_text())
    stop = json.loads((tmp_path / f'snap/stop_reasons/{STOP_TAG}/metadata.json').read_text())
    assert metadata['extraction_tag'] == EXTRACTION.release_tag()
    assert 'extraction_tag' not in stop


def test_read_extraction_reads_the_release_tag(tmp_path: Path) -> None:
    directory = tmp_path / 'extraction' / EXTRACTION.release_tag()
    directory.mkdir(parents=True)
    _extractions([['a']]).write_parquet(directory / 'data.parquet')
    other = tmp_path / 'extraction' / 'other-tag'
    other.mkdir()
    _extractions([['b'], ['c']]).write_parquet(other / 'data.parquet')

    assert read_extraction(str(tmp_path / 'extraction'), None).height == 1


def test_read_extraction_names_the_path_it_expected(tmp_path: Path) -> None:
    other = tmp_path / 'extraction' / 'other-tag'
    other.mkdir(parents=True)
    _extractions([['b']]).write_parquet(other / 'data.parquet')

    with pytest.raises(FileNotFoundError, match='extraction/' + EXTRACTION.release_tag() + '/data.parquet'):
        read_extraction(str(tmp_path / 'extraction'), None)
