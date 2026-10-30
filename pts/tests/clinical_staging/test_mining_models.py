"""Tests for the mining_models staging task."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from pts.clinical_staging import mining_models
from pts.clinical_staging.mining_models import stage_mining_models
from pts.clinical_staging.staged_outputs import DISEASE_ENTITIES, DRUG_ENTITIES
from pts.record_store import map_texts
from pts.transformers.clinical.mining import RecordingNer, build_mining_reports, report_drug_labels, staged_lookup
from pts.transformers.clinical.ner import (
    CHEMICAL_MODEL,
    DISEASE_MODEL,
    DISEASE_NER_MODEL,
    DRUG_NER_MODEL,
    DRUGTEMIST_MODEL,
)
from tests.clinical.test_mining import inputs  # noqa: F401

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
        self.labels = {'ttd': ['aspirin'], 'ema': ['ibuprofen'], 'pmda': ['aspirin', 'x']}
        self.texts = ['pain', 'fever']
        monkeypatch.setattr(mining_models, 'build_mining_reports', self.reports)
        monkeypatch.setattr(mining_models, 'disease_entities_with_status', self.diseases)
        monkeypatch.setattr(mining_models, 'drug_entities_with_status', self.drug_entities)

    def reports(self, ttd: str, ema: str, pmda: str, extract_diseases: Any) -> dict[str, Any]:
        extract_diseases(pl.Series('text', self.texts))
        return {key: _report(labels) for key, labels in self.labels.items()}

    def diseases(self, texts: list[str]) -> list[tuple[list[str] | None, str | None]]:
        self.disease_calls.append(list(texts))
        return [([f'disease:{t}'], None) for t in texts]

    def drug_entities(self, texts: list[str], batch_size: int) -> list[tuple[list[str] | None, str | None]]:
        self.ner_calls.append(list(texts))
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
        max_attempts=3,
    )


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
    assert stats['disease_entities'] == {'texts': 2, 'computed': True, 'published': True, 'tag': DISEASE_TAG}
    assert stats['drug_entities'] == {'texts': 3, 'computed': True, 'published': True, 'tag': DRUG_TAG}


def test_mining_rerun_computes_nothing(tmp_path: Path, models_uri: str, models: Models) -> None:
    _stage(tmp_path, models_uri, 'r1')
    models.disease_calls.clear()
    models.ner_calls.clear()

    stats = _stage(tmp_path, models_uri, 'r2')

    assert models.disease_calls == models.ner_calls == []
    assert stats['disease_entities'] == {'texts': 2, 'computed': False, 'published': False, 'tag': DISEASE_TAG}
    assert stats['drug_entities'] == {'texts': 3, 'computed': False, 'published': False, 'tag': DRUG_TAG}


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
        mining_models, 'disease_entities_with_status', lambda texts: [([f'd:{t}'], None) for t in texts]
    )
    monkeypatch.setattr(
        mining_models, 'drug_entities_with_status', lambda texts, batch_size: [([f'drug:{t}'], None) for t in texts]
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
        max_attempts=3,
    )

    staged = pl.read_parquet(tmp_path / f'snap/disease_entities/{DISEASE_TAG}/data.parquet')
    lookup = staged_lookup(staged, 'text', 'extracted_diseases', 'disease texts')
    expected = report_drug_labels(r.df for r in build_mining_reports(ttd, ema, pmda, lookup).values())
    blind = report_drug_labels(r.df for r in build_mining_reports(ttd, ema, pmda, RecordingNer()).values())
    labels = pl.read_parquet(tmp_path / f'snap/drug_entities/{DRUG_TAG}/data.parquet')['label']
    assert labels.to_list() == expected.to_list()
    assert set(blind.to_list()) < set(expected.to_list())
    assert {'alphamab', 'gammastat'} <= set(labels.to_list())
