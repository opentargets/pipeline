from collections.abc import Iterator
from typing import Any

import polars as pl
import pytest
from loguru import logger

from pts.transformers.clinical import extract_disease_entities, extract_drug_entities, ner

Entity = dict[str, Any]

DISEASE_MODEL = 'alvaroalon2/biobert_diseases_ner'
CHEMICAL_MODEL = 'alvaroalon2/biobert_chemical_ner'
DRUGTEMIST_MODEL = 'BSC-NLP4BIA/bsc-bio-ehr-es-carmen-drugtemist'


def _ent(group: str, word: str) -> Entity:
    return {'entity_group': group, 'word': word}


class FakePipeline:
    """Stands in for a transformers NER pipeline: texts in, entity dicts out."""

    def __init__(self, by_text: dict[str, list[Entity]], fail_on: set[str] | None = None) -> None:
        self.by_text = by_text
        self.fail_on = fail_on or set()
        self.calls: list[Any] = []

    def __call__(self, texts: str | list[str]) -> list[Entity] | list[list[Entity]]:
        self.calls.append(texts)
        batch = [texts] if isinstance(texts, str) else texts
        if self.fail_on & set(batch):
            raise RuntimeError('boom')
        if isinstance(texts, str):
            return self.by_text.get(texts, [])
        return [self.by_text.get(t, []) for t in texts]


def _patch_factory(monkeypatch: pytest.MonkeyPatch, pipelines: dict[str, FakePipeline]) -> list[dict[str, Any]]:
    created: list[dict[str, Any]] = []

    def factory(model_name: str, tokenizer_name: str | None = None, aggregation_strategy: str = 'max') -> FakePipeline:
        created.append({'model': model_name, 'tokenizer': tokenizer_name, 'aggregation_strategy': aggregation_strategy})
        return pipelines[model_name]

    monkeypatch.setattr(ner, '_create_ner_pipeline', factory)
    return created


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m).strip()), level='WARNING', format='{message}')
    yield messages
    logger.remove(sink)


def test_disease_filters_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline({
        'text': [
            _ent('DISEASE', ' Breast Cancer '),
            _ent('disease', 'asthma'),
            _ent('DISEASE', 'ASTHMA'),
            _ent('DISEASE', 'x'),
            _ent('DISEASE', '2024'),
            _ent('CHEMICAL', 'aspirin'),
            _ent('B-DISEASE', 'Anaemia'),
        ],
    })
    created = _patch_factory(monkeypatch, {DISEASE_MODEL: fake})

    out = extract_disease_entities(pl.Series(['text']))

    assert out.dtype == pl.List(pl.String)
    assert out.to_list() == [['anaemia', 'asthma', 'breast cancer']]
    assert created == [{'model': DISEASE_MODEL, 'tokenizer': DISEASE_MODEL, 'aggregation_strategy': 'max'}]


def test_disease_empty_and_blank_text_yield_empty_without_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline({'a': [_ent('DISEASE', 'gout')]})
    _patch_factory(monkeypatch, {DISEASE_MODEL: fake})

    out = extract_disease_entities(pl.Series(['', '  ', None, 'a']))

    assert out.to_list() == [[], [], [], ['gout']]
    assert fake.calls == ['a']


def test_disease_runs_once_per_distinct_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline({'a': [_ent('DISEASE', 'gout')], 'b': [_ent('DISEASE', 'flu')]})
    _patch_factory(monkeypatch, {DISEASE_MODEL: fake})

    out = extract_disease_entities(pl.Series(['b', 'a', 'b', 'a', 'b']))

    assert out.to_list() == [['flu'], ['gout'], ['flu'], ['gout'], ['flu']]
    assert sorted(fake.calls) == ['a', 'b']


def test_disease_failure_yields_empty_and_counts(monkeypatch: pytest.MonkeyPatch, warnings: list[str]) -> None:
    fake = FakePipeline({'a': [_ent('DISEASE', 'gout')]}, fail_on={'bad', 'worse'})
    _patch_factory(monkeypatch, {DISEASE_MODEL: fake})

    out = extract_disease_entities(pl.Series(['a', 'bad', 'worse', 'bad']))

    assert out.to_list() == [['gout'], [], [], []]
    assert any('disease ner failed for 2 of 3 distinct texts' in m for m in warnings)


def _drug_pipelines(
    biobert: dict[str, list[Entity]], drugtemist: dict[str, list[Entity]], fail_on: set[str] | None = None
) -> dict[str, FakePipeline]:
    return {
        CHEMICAL_MODEL: FakePipeline(biobert, fail_on=fail_on),
        DRUGTEMIST_MODEL: FakePipeline(drugtemist),
    }


def test_drug_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    pipelines = _drug_pipelines(
        biobert={
            'pembrolizumab 100mg': [_ent('CHEMICAL', 'Pembrolizumab'), _ent('CHEM', 'x')],
            'aspirin tablet': [_ent('CHEMICAL', 'Aspirin'), _ent('DRUG', '##sodium'), _ent('DISEASE', 'pain')],
        },
        drugtemist={
            'brandname': [_ent('FARMACO', 'BrandName'), _ent('OTHER', 'zzz')],
            'pembrolizumab 100mg': [_ent('FARMACO', 'should-not-appear')],
        },
    )
    created = _patch_factory(monkeypatch, pipelines)

    labels = pl.Series(['pembrolizumab 100mg', 'aspirin tablet', 'brandname', 'nothing', 'pembrolizumab 100mg'])
    out = extract_drug_entities(labels, batch_size=2)

    assert out.dtype == pl.List(pl.String)
    assert out.to_list() == [['pembrolizumab'], ['aspirin', 'sodium'], ['brandname'], [], ['pembrolizumab']]
    assert [c['model'] for c in created] == [CHEMICAL_MODEL, DRUGTEMIST_MODEL]
    assert all(c['aggregation_strategy'] == 'max' for c in created)
    fallback_texts = [t for call in pipelines[DRUGTEMIST_MODEL].calls for t in call]
    assert sorted(fallback_texts) == ['brandname', 'nothing']


def test_drug_regex_unions_with_biobert_and_skips_drugtemist(monkeypatch: pytest.MonkeyPatch) -> None:
    pipelines = _drug_pipelines(
        biobert={'osimertinib with Foo': [_ent('CHEMICAL', 'Foo Bar')]},
        drugtemist={'osimertinib with Foo': [_ent('FARMACO', 'nope')]},
    )
    _patch_factory(monkeypatch, pipelines)

    out = extract_drug_entities(pl.Series(['osimertinib with Foo']), batch_size=8)

    assert out.to_list() == [['foo bar', 'osimertinib']]
    assert pipelines[DRUGTEMIST_MODEL].calls == []


def test_drug_batch_failure_yields_empty_and_counts(monkeypatch: pytest.MonkeyPatch, warnings: list[str]) -> None:
    pipelines = _drug_pipelines(
        biobert={'a': [_ent('CHEMICAL', 'aspirin')], 'c': [_ent('CHEMICAL', 'ibuprofen')]},
        drugtemist={},
        fail_on={'b'},
    )
    _patch_factory(monkeypatch, pipelines)

    out = extract_drug_entities(pl.Series(['a', 'b', 'c']), batch_size=1)

    assert out.to_list() == [['aspirin'], [], ['ibuprofen']]
    assert warnings == ['BioBERT failed for 1 of 3 texts']


def test_drug_after_biobert_batch_failure_keeps_regex_hits_and_runs_drugtemist(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    pipelines = _drug_pipelines(
        biobert={},
        drugtemist={'b': [_ent('FARMACO', 'Rescued')]},
        fail_on={'xmab', 'b'},
    )
    _patch_factory(monkeypatch, pipelines)

    out = extract_drug_entities(pl.Series(['xmab', 'b']), batch_size=8)

    assert out.to_list() == [['xmab'], ['rescued']]
    assert pipelines[DRUGTEMIST_MODEL].calls == [['b']]
    assert warnings == ['BioBERT failed for 2 of 2 texts']
