"""Tests for the mining_models staging task."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from pts import mining_models
from pts.mining_models import stage_mining_models
from pts.record_store import map_texts
from pts.transformers.clinical.hf_models import revision_tags
from pts.transformers.clinical.ner import (
    CHEMICAL_MODEL,
    DISEASE_MODEL,
    DISEASE_NER_MODEL,
    DRUG_NER_MODEL,
    DRUGTEMIST_MODEL,
    drug_ner_revision,
)
from pts.transformers.clinical.stop_reasons import STOP_REASONS_MODEL

REVISIONS = {DISEASE_MODEL: 'rx', CHEMICAL_MODEL: 'rc', DRUGTEMIST_MODEL: 'rd'}


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


def test_revision_tags_from_staged_revisions() -> None:
    aact = {STOP_REASONS_MODEL: 'rs', CHEMICAL_MODEL: 'rc', DRUGTEMIST_MODEL: 'rd'}

    assert revision_tags({**aact, DISEASE_MODEL: 'rx'}) == {
        'stop_reasons': 'rs',
        'drug_entities': 'rc+rd',
        'disease_entities': 'rx',
    }
    assert revision_tags(aact) == {'stop_reasons': 'rs', 'drug_entities': 'rc+rd'}
    assert revision_tags(REVISIONS) == {'drug_entities': 'rc+rd', 'disease_entities': 'rx'}


def test_mining_models_publishes_both_snapshots(tmp_path: Path, models_uri: str, models: Models) -> None:
    stats = _stage(tmp_path, models_uri, 'r1')

    disease = pl.read_parquet(tmp_path / 'snap/disease_entities/rx/data.parquet')
    drugs = pl.read_parquet(tmp_path / 'snap/drug_entities/rc+rd/data.parquet')
    assert disease.columns == ['text', 'extracted_diseases']
    assert disease.to_dicts() == [
        {'text': 'fever', 'extracted_diseases': ['disease:fever']},
        {'text': 'pain', 'extracted_diseases': ['disease:pain']},
    ]
    assert drugs.columns == ['label', 'extracted_drugs']
    assert drugs['label'].to_list() == ['aspirin', 'ibuprofen', 'x']
    assert models.ner_calls == [['aspirin', 'ibuprofen', 'x']]
    for path, model, revision in (
        ('disease_entities/rx', DISEASE_NER_MODEL, 'rx'),
        ('drug_entities/rc+rd', DRUG_NER_MODEL, drug_ner_revision(REVISIONS)),
    ):
        metadata = json.loads((tmp_path / f'snap/{path}/metadata.json').read_text())
        assert metadata == {'model': model, 'model_revision': revision, 'model_revisions': REVISIONS}
    assert stats['disease_entities'] == {'texts': 2, 'computed': True, 'published': True}
    assert stats['drug_entities'] == {'texts': 3, 'computed': True, 'published': True}


def test_mining_rerun_computes_nothing(tmp_path: Path, models_uri: str, models: Models) -> None:
    _stage(tmp_path, models_uri, 'r1')
    models.disease_calls.clear()
    models.ner_calls.clear()

    stats = _stage(tmp_path, models_uri, 'r2')

    assert models.disease_calls == models.ner_calls == []
    assert stats['disease_entities'] == {'texts': 2, 'computed': False, 'published': False}
    assert stats['drug_entities'] == {'texts': 3, 'computed': False, 'published': False}


def test_a_label_shared_with_aact_is_not_recomputed(tmp_path: Path, models_uri: str, models: Models) -> None:
    map_texts(
        pl.Series(['aspirin']),
        lambda texts: [(['aact'], None) for _ in texts],
        str(tmp_path / 'store/drug_entities'),
        DRUG_NER_MODEL,
        drug_ner_revision(REVISIONS),
        'aact-run',
        None,
    )

    _stage(tmp_path, models_uri, 'r1')

    assert models.ner_calls == [['ibuprofen', 'x']]
    drugs = pl.read_parquet(tmp_path / 'snap/drug_entities/rc+rd/data.parquet')
    assert drugs.filter(label='aspirin')['extracted_drugs'].to_list() == [['aact']]
