from typing import Any

import pytest

from pts.transformers.clinical import ner
from pts.transformers.clinical.ner import disease_entities_with_status, drug_entities_with_status

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

    out = disease_entities_with_status(['text'])

    assert out == [(['anaemia', 'asthma', 'breast cancer'], None)]
    assert created == [{'model': DISEASE_MODEL, 'tokenizer': DISEASE_MODEL, 'aggregation_strategy': 'max'}]


def test_disease_filter_reads_its_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline({'text': [_ent('DISEASE', 'ab'), _ent('DISEASE', 'abc'), _ent('DISEASE', '2024')]})
    _patch_factory(monkeypatch, {DISEASE_MODEL: fake})
    monkeypatch.setattr(ner, 'DISEASE_MIN_LENGTH', 3)
    monkeypatch.setattr(ner, 'DISEASE_DROP_DIGITS', False)

    out = disease_entities_with_status(['text'])

    assert out == [(['2024', 'abc'], None)]


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

    texts = ['pembrolizumab 100mg', 'aspirin tablet', 'brandname', 'nothing', '', 'osimertinib']
    out = drug_entities_with_status(texts, batch_size=2)

    assert out == [
        (['pembrolizumab'], None),
        (['aspirin', 'sodium'], None),
        (['brandname'], None),
        ([], None),
        ([], None),
        (['osimertinib'], None),
    ]
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

    out = drug_entities_with_status(['osimertinib with Foo'], batch_size=8)

    assert out == [(['foo bar', 'osimertinib'], None)]
    assert pipelines[DRUGTEMIST_MODEL].calls == []


def test_drug_after_biobert_batch_failure_still_runs_drugtemist_on_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    pipelines = _drug_pipelines(
        biobert={},
        drugtemist={'b': [_ent('FARMACO', 'Rescued')]},
        fail_on={'xmab', 'b'},
    )
    _patch_factory(monkeypatch, pipelines)

    out = drug_entities_with_status(['xmab', 'b'], batch_size=8)

    assert out == [(None, 'ner batch failed: boom'), (None, 'ner batch failed: boom')]
    assert pipelines[DRUGTEMIST_MODEL].calls == [['b']]


def test_staging_ner_marks_texts_of_a_failed_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    pipelines = _drug_pipelines(
        biobert={'a': [_ent('CHEMICAL', 'aspirin')], 'c': [_ent('CHEMICAL', 'ibuprofen')]},
        drugtemist={},
        fail_on={'b'},
    )
    _patch_factory(monkeypatch, pipelines)

    out = drug_entities_with_status(['a', 'b', 'c'], batch_size=1)

    assert out == [(['aspirin'], None), (None, 'ner batch failed: boom'), (['ibuprofen'], None)]


def test_staging_ner_marks_regex_hits_of_a_failed_biobert_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    pipelines = _drug_pipelines(biobert={}, drugtemist={}, fail_on={'xmab'})
    _patch_factory(monkeypatch, pipelines)

    out = drug_entities_with_status(['xmab', 'ok'], batch_size=1)

    assert out[0] == (None, 'ner batch failed: boom')
    assert out[1] == ([], None)


def test_drugtemist_failure_marks_only_fallback_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    pipelines = _drug_pipelines(biobert={'hit': [_ent('CHEMICAL', 'aspirin')]}, drugtemist={})
    pipelines[DRUGTEMIST_MODEL] = FakePipeline({}, fail_on={'miss'})
    _patch_factory(monkeypatch, pipelines)

    out = drug_entities_with_status(['hit', 'miss', 'other'], batch_size=1)

    assert out == [(['aspirin'], None), (None, 'ner batch failed: boom'), ([], None)]
    assert pipelines[DRUGTEMIST_MODEL].calls == [['miss'], ['other']]


def test_drug_ner_model_name() -> None:
    assert f'{CHEMICAL_MODEL}+{DRUGTEMIST_MODEL}+regex' == ner.DRUG_NER_MODEL


def test_disease_status_marks_only_failing_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline({'a': [_ent('DISEASE', 'Gout')], 'b': [_ent('DISEASE', 'Flu')]}, fail_on={'bad'})
    _patch_factory(monkeypatch, {DISEASE_MODEL: fake})

    out = disease_entities_with_status(['a', 'bad', '', '  ', 'b'])

    assert out == [(['gout'], None), (None, 'disease ner failed: boom'), ([], None), ([], None), (['flu'], None)]
    assert fake.calls == ['a', 'bad', 'b']


def test_disease_model_load_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(*_: Any, **__: Any) -> FakePipeline:
        raise OSError('no such model')

    monkeypatch.setattr(ner, '_create_ner_pipeline', factory)

    with pytest.raises(OSError, match='no such model'):
        disease_entities_with_status(['a'])
    assert disease_entities_with_status(['', ' ']) == [([], None), ([], None)]
    assert ner.DISEASE_NER_MODEL == DISEASE_MODEL
