"""Tests for the mining_models staging task."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers.clinical_mira import clinical_mining_models as mining_models
from pts.transformers.clinical_mira.clinical_mining_models import clinical_mining_models, stage_mining_models
from pts.transformers.clinical_mira.mining import (
    RecordingNer,
    mining_reports,
    read_mining_inputs,
    report_drug_labels,
    staged_lookup,
)
from pts.transformers.clinical_mira.ner import (
    CHEMICAL_MODEL,
    DISEASE_MODEL,
    DISEASE_NER_MODEL,
    DRUG_NER_MODEL,
    DRUGTEMIST_MODEL,
)
from pts.transformers.clinical_mira.record_store import map_texts
from pts.transformers.clinical_mira.staged_outputs import DISEASE_ENTITIES, DRUG_ENTITIES
from tests.clinical_mira.test_mining import inputs  # noqa: F401

REVISIONS = {DISEASE_MODEL: 'rx', CHEMICAL_MODEL: 'rc', DRUGTEMIST_MODEL: 'rd'}
DISEASE_TAG = DISEASE_ENTITIES.tag(REVISIONS)
DRUG_TAG = DRUG_ENTITIES.tag(REVISIONS)


def _report(labels: list[str]) -> Any:
    drugs = [[{'drugFromSource': label, 'drugId': None}] for label in labels]
    return SimpleNamespace(df=pl.DataFrame({'drugs': drugs}))


@pytest.fixture
def models_uri(tmp_path: Path) -> str:
    for model, revision in REVISIONS.items():
        directory = tmp_path / 'models' / model / revision
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text('{}')
    return str(tmp_path / 'models')


class Models:
    """Fake reports and model computes that remember what they were asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.disease_calls: list[list[str]] = []
        self.ner_calls: list[list[str]] = []
        self.batch_sizes: dict[str, list[tuple[int, int]]] = {'disease': [], 'drug': []}
        self.labels = {'ttd': ['aspirin'], 'ema': ['ibuprofen'], 'pmda': ['aspirin', 'x']}
        self.texts = ['pain', 'fever']
        self.reads: list[tuple[str, str, str]] = []
        self.built_from: list[object] = []
        self.sentinel = object()
        monkeypatch.setattr(mining_models, 'read_mining_inputs', self.read_inputs)
        monkeypatch.setattr(mining_models, 'mining_reports', self.reports)
        monkeypatch.setattr(mining_models, 'disease_entities_with_status', self.diseases)
        monkeypatch.setattr(mining_models, 'drug_entities_with_status', self.drug_entities)

    def read_inputs(self, ttd: str, ema: str, pmda: str) -> object:
        self.reads.append((ttd, ema, pmda))
        return self.sentinel

    def reports(self, parsed: object, extract_diseases: Any) -> dict[str, Any]:
        self.built_from.append(parsed)
        extract_diseases(pl.Series('text', self.texts))
        return {key: _report(labels) for key, labels in self.labels.items()}

    def diseases(
        self, texts: list[str], batch_size: int, model_batch_size: int = 1
    ) -> list[tuple[list[str] | None, str | None]]:
        self.disease_calls.append(list(texts))
        self.batch_sizes['disease'].append((batch_size, model_batch_size))
        return [([f'disease:{t}'], None) for t in texts]

    def drug_entities(
        self, texts: list[str], batch_size: int, model_batch_size: int
    ) -> list[tuple[list[str] | None, str | None]]:
        self.ner_calls.append(list(texts))
        self.batch_sizes['drug'].append((batch_size, model_batch_size))
        return [([f'drug:{t}'], None) for t in texts]


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Models:
    return Models(monkeypatch)


def _stage(tmp_path: Path, models_uri: str, run_id: str) -> dict[str, Any]:
    return stage_mining_models(
        'ttd',
        'ema',
        'pmda',
        models_uri,
        {
            'drug_entities': str(tmp_path / 'snap/drug_entities'),
            'disease_entities': str(tmp_path / 'snap/disease_entities'),
        },
        str(tmp_path / 'store'),
        run_id,
        None,
        ner_batch_size=4,
        ner_model_batch_size=3,
        max_attempts=3,
    )


def test_mining_models_pass_the_model_batch_size_to_drug_ner_only(
    tmp_path: Path, models_uri: str, models: Models
) -> None:
    _stage(tmp_path, models_uri, 'r1')

    assert models.batch_sizes == {'disease': [(4, 1)], 'drug': [(4, 3)]}


def test_staging_reads_the_inputs_once(tmp_path: Path, models_uri: str, models: Models) -> None:
    _stage(tmp_path, models_uri, 'r1')

    assert models.reads == [('ttd', 'ema', 'pmda')]
    assert models.built_from == [models.sentinel, models.sentinel]


def test_stage_mining_models_names_missing_models(tmp_path: Path, models_uri: str, models: Models) -> None:
    shutil.rmtree(Path(models_uri) / DRUGTEMIST_MODEL)

    with pytest.raises(ValueError, match=DRUGTEMIST_MODEL):
        _stage(tmp_path, models_uri, 'r1')

    assert not (tmp_path / 'snap').exists()


def test_mining_models_publishes_both_snapshots(tmp_path: Path, models_uri: str, models: Models) -> None:
    stats = _stage(tmp_path, models_uri, 'r1')

    disease = pl.read_parquet(tmp_path / f'snap/disease_entities/{DISEASE_TAG}/data.parquet')
    drugs = pl.read_parquet(tmp_path / f'snap/drug_entities/{DRUG_TAG}/data.parquet')
    assert disease.columns == ['text', 'extracted_diseases']
    assert disease.to_dicts() == [
        {'text': 'fever', 'extracted_diseases': ['disease:fever']},
        {'text': 'pain', 'extracted_diseases': ['disease:pain']},
    ]
    assert drugs.columns == ['label', 'extracted_drugs']
    assert drugs['label'].to_list() == ['aspirin', 'ibuprofen', 'x']
    assert models.ner_calls == [['aspirin', 'ibuprofen', 'x']]
    kinds = (
        (DISEASE_ENTITIES, DISEASE_TAG, DISEASE_NER_MODEL, {DISEASE_MODEL: 'rx'}),
        (DRUG_ENTITIES, DRUG_TAG, DRUG_NER_MODEL, {CHEMICAL_MODEL: 'rc', DRUGTEMIST_MODEL: 'rd'}),
    )
    for kind, tag, model, hub in kinds:
        metadata = json.loads((tmp_path / f'snap/{kind.name}/{tag}/metadata.json').read_text())
        assert metadata == {'kind': kind.name, 'model': model, 'tag': tag, 'model_revisions': hub}
    assert stats['disease_entities'] == {
        'texts': 2,
        'computed': True,
        'computed_texts': 2,
        'published': True,
        'tag': DISEASE_TAG,
    }
    assert stats['drug_entities'] == {
        'texts': 3,
        'computed': True,
        'computed_texts': 3,
        'published': True,
        'tag': DRUG_TAG,
    }


def test_mining_rerun_computes_nothing(tmp_path: Path, models_uri: str, models: Models) -> None:
    _stage(tmp_path, models_uri, 'r1')
    models.disease_calls.clear()
    models.ner_calls.clear()

    stats = _stage(tmp_path, models_uri, 'r2')

    assert models.disease_calls == models.ner_calls == []
    assert stats['disease_entities'] == {
        'texts': 2,
        'computed': False,
        'computed_texts': 0,
        'published': False,
        'tag': DISEASE_TAG,
    }
    assert stats['drug_entities'] == {
        'texts': 3,
        'computed': False,
        'computed_texts': 0,
        'published': False,
        'tag': DRUG_TAG,
    }


def test_a_label_shared_with_aact_is_not_recomputed(tmp_path: Path, models_uri: str, models: Models) -> None:
    map_texts(
        pl.Series(['aspirin']),
        lambda texts: [(['aact'], None) for _ in texts],
        str(tmp_path / 'store/drug_entities'),
        DRUG_NER_MODEL,
        DRUG_TAG,
        'aact-run',
        None,
    )

    _stage(tmp_path, models_uri, 'r1')

    assert models.ner_calls == [['ibuprofen', 'x']]
    drugs = pl.read_parquet(tmp_path / f'snap/drug_entities/{DRUG_TAG}/data.parquet')
    assert drugs.filter(label='aspirin')['extracted_drugs'].to_list() == [['aact']]


def test_mining_drug_labels_come_from_reports_built_with_staged_diseases(
    tmp_path: Path,
    models_uri: str,
    inputs: dict[str, str],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mining_models, 'disease_entities_with_status', lambda texts, batch_size: [([f'd:{t}'], None) for t in texts]
    )
    monkeypatch.setattr(
        mining_models,
        'drug_entities_with_status',
        lambda texts, batch_size, model_batch_size: [([f'drug:{t}'], None) for t in texts],
    )
    ttd, ema, pmda = inputs['ttd'], inputs['ema'], inputs['pmda']

    stage_mining_models(
        ttd,
        ema,
        pmda,
        models_uri,
        {
            'drug_entities': str(tmp_path / 'snap/drug_entities'),
            'disease_entities': str(tmp_path / 'snap/disease_entities'),
        },
        str(tmp_path / 'store'),
        'r1',
        None,
        ner_batch_size=4,
        ner_model_batch_size=1,
        max_attempts=3,
    )

    staged = pl.read_parquet(tmp_path / f'snap/disease_entities/{DISEASE_TAG}/data.parquet')
    lookup = staged_lookup(staged, 'text', 'extracted_diseases', 'disease texts')
    parsed = read_mining_inputs(ttd, ema, pmda)
    expected = report_drug_labels(r.df for r in mining_reports(parsed, lookup).values())
    blind = report_drug_labels(r.df for r in mining_reports(parsed, RecordingNer()).values())
    labels = pl.read_parquet(tmp_path / f'snap/drug_entities/{DRUG_TAG}/data.parquet')['label']
    assert labels.to_list() == expected.to_list()
    assert set(blind.to_list()) < set(expected.to_list())
    assert {'alphamab', 'gammastat'} <= set(labels.to_list())


def test_the_transformer_stages_the_providers_from_its_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({'args': args, **kwargs})
        return {}

    monkeypatch.setattr(mining_models, 'stage_mining_models', fake_stage)

    clinical_mining_models(
        {'ttd': 'ttd.txt', 'ema': 'ema.xlsx', 'pmda': 'pmda.pdf', 'models': 'models'},
        {'drug_entities': 'drugs', 'disease_entities': 'diseases'},
        {'store_uri': 'store/', 'snapshot': '2026-10', 'ner_model_batch_size': 4},
        Config(step='test', steps=['test'], work_path=tmp_path),
    )

    (call,) = calls
    ttd, ema, pmda, models_uri, destination, store_uri, run_id, _ = call['args']
    assert (ttd, ema, pmda, models_uri) == ('ttd.txt', 'ema.xlsx', 'pmda.pdf', 'models')
    assert destination == {'drug_entities': 'drugs', 'disease_entities': 'diseases'}
    assert store_uri == 'store'
    assert run_id.startswith('2026-10-')
    assert call['ner_batch_size'] == 256
    assert call['ner_model_batch_size'] == 4
    assert call['max_attempts'] == 3
