"""Tests for the trial_models staging task."""

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts import aact_trial_models
from pts.aact_trial_models import seed_drug_entities, stage_trial_models, why_stopped_texts
from pts.record_store import read_records, text_sha256
from pts.transformers.clinical import extract_aact_clinical_report
from pts.transformers.clinical.ner import CHEMICAL_MODEL, DRUG_NER_MODEL, DRUGTEMIST_MODEL, drug_ner_revision
from pts.transformers.clinical.providers.aact import llm_drug_labels
from pts.transformers.clinical.stop_reasons import STOP_REASONS_MODEL, stop_reasons_revision

REVISIONS = {STOP_REASONS_MODEL: 'rs', CHEMICAL_MODEL: 'rc', DRUGTEMIST_MODEL: 'rd'}
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
        monkeypatch.setattr(aact_trial_models, 'stop_reasons_with_status', self.stop_reasons)
        monkeypatch.setattr(aact_trial_models, 'drug_entities_with_status', self.drug_entities)

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
    extractions = _extractions([['Aspirin', ' Ibuprofen ', None], [''], []])

    report = extract_aact_clinical_report(studies, interventions, conditions, llm_extractions=extractions).df

    released = set(report['drugs'].explode().struct.field('drugFromSource').drop_nulls())
    assert released
    assert released <= set(llm_drug_labels(extractions))
    assert llm_drug_labels(extractions).to_list() == ['', ' Ibuprofen ', 'Aspirin']


def test_trial_models_publishes_both_snapshots(tmp_path: Path, models_uri: str, models: Models) -> None:
    studies = _studies(['no funds', None, 'no funds', 'slow'])
    stats = _stage(tmp_path, models_uri, studies, _extractions([['a', 'b']]), 'r1')

    stop = pl.read_parquet(tmp_path / 'snap/stop_reasons/data.parquet')
    ner = pl.read_parquet(tmp_path / 'snap/drug_entities/data.parquet')
    assert stop.columns == ['text', 'text_sha256', 'categories']
    assert stop['text'].to_list() == ['no funds', 'slow']
    assert stop['categories'].to_list() == [['reason:no funds'], ['reason:slow']]
    assert stop['text_sha256'].to_list() == [text_sha256('no funds'), text_sha256('slow')]
    assert ner.columns == ['label', 'extracted_drugs']
    assert ner.to_dicts() == [
        {'label': 'a', 'extracted_drugs': ['drug:a']},
        {'label': 'b', 'extracted_drugs': ['drug:b']},
    ]
    revisions = {'stop_reasons': stop_reasons_revision('rs'), 'drug_entities': drug_ner_revision(REVISIONS)}
    for name, model in (('stop_reasons', STOP_REASONS_MODEL), ('drug_entities', DRUG_NER_MODEL)):
        metadata = json.loads((tmp_path / f'snap/{name}/metadata.json').read_text())
        assert metadata == {'model': model, 'model_revision': revisions[name], 'model_revisions': REVISIONS}
    assert stats['stop_reasons'] == {'texts': 2, 'computed': True, 'published': True}
    assert stats['drug_entities'] == {'texts': 2, 'computed': True, 'published': True}


def test_rerun_computes_nothing_and_rewrites_nothing(tmp_path: Path, models_uri: str, models: Models) -> None:
    studies, extractions = _studies(['no funds', 'slow']), _extractions([['a']])
    _stage(tmp_path, models_uri, studies, extractions, 'r1')
    snapshot = (tmp_path / 'snap/stop_reasons/data.parquet').stat().st_mtime_ns
    models.stop_calls.clear()
    models.ner_calls.clear()

    stats = _stage(tmp_path, models_uri, studies, extractions, 'r2')

    assert models.stop_calls == models.ner_calls == []
    assert stats['stop_reasons'] == {'texts': 2, 'computed': False, 'published': False}
    assert stats['drug_entities'] == {'texts': 1, 'computed': False, 'published': False}
    assert (tmp_path / 'snap/stop_reasons/data.parquet').stat().st_mtime_ns == snapshot
    assert not (tmp_path / 'store/stop_reasons/runs/r2').exists()


def test_seed_imports_the_ner_cache_once(tmp_path: Path) -> None:
    cache = tmp_path / 'ner_cache.parquet'
    pl.DataFrame({
        'query_label': ['aspirin', None, 'x'],
        'extracted_drugs': [['aspirin'], ['dropped'], None],
    }).write_parquet(cache)
    store = str(tmp_path / 'store')

    assert seed_drug_entities(str(cache), store, DRUG_NER_MODEL, 'rev', None) == 2
    assert seed_drug_entities(str(cache), store, DRUG_NER_MODEL, 'rev', None) == 0

    records = read_records(f'{store}/drug_entities', None).sort('text')
    assert records['text'].to_list() == ['aspirin', 'x']
    assert records['output'].to_list() == [['aspirin'], []]
    assert records['status'].to_list() == ['ok', 'ok']
    assert records['run_id'].to_list() == ['seed-ner-cache'] * 2
    assert records['model'].to_list() == [DRUG_NER_MODEL] * 2
    assert records['model_revision'].to_list() == ['rev'] * 2
    assert records['text_sha256'].to_list() == [text_sha256('aspirin'), text_sha256('x')]


def test_seeded_labels_are_not_recomputed(tmp_path: Path, models_uri: str, models: Models) -> None:
    cache = tmp_path / 'ner_cache.parquet'
    pl.DataFrame({'query_label': ['a'], 'extracted_drugs': [['seeded']]}).write_parquet(cache)
    seed_drug_entities(str(cache), str(tmp_path / 'store'), DRUG_NER_MODEL, drug_ner_revision(REVISIONS), None)

    _stage(tmp_path, models_uri, _studies([]), _extractions([['a', 'b']]), 'r1')

    assert models.ner_calls == [['b']]
    ner = pl.read_parquet(tmp_path / 'snap/drug_entities/data.parquet')
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
    assert pl.read_parquet(tmp_path / 'snap/stop_reasons/data.parquet')['text'].to_list() == ['', '  ', 'x']
    assert pl.read_parquet(tmp_path / 'snap/drug_entities/data.parquet')['label'].to_list() == ['', ' ', 'a']
