import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
import yaml
from otter.config.model import Config

from pts.transformers.clinical_mira import aact_moa as aact_moa_module
from pts.transformers.clinical_mira import aact_trial_extraction as extraction
from pts.transformers.clinical_mira.aact_moa import AactMoaSettings, aact_moa, stage_moa
from pts.transformers.clinical_mira.moa import schemas
from pts.transformers.clinical_mira.moa.staging import import_run_id
from pts.transformers.clinical_mira.record_store import ComputeAborted, read_records, run_exists
from pts.transformers.clinical_mira.schemas import ClinicalReportExtractionSchema
from pts.transformers.clinical_mira.staged_outputs import EXTRACTION, MOA, schema_digest
from pts.transformers.clinical_mira.staging_errors import ComputeTooLarge, SeedTooSmall

EXTRACTION_TAG = EXTRACTION.release_tag()
KRAS_QUOTE = 'Sotorasib is a selective KRAS G12C inhibitor'
MEK_QUOTE = 'Sotorasib is a MEK pathway modulator'
COX_QUOTE = 'Aspirinx is a COX inhibitor'

WORLD = {
    'nct1': ('sotorasib', f'{KRAS_QUOTE} in NSCLC.', 'Adults with NSCLC.'),
    'nct2': ('sotorasib', 'Sotorasib in colorectal cancer.', f'{MEK_QUOTE} in this arm.'),
    'nct3': ('aspirinx', f'{COX_QUOTE} in pain.', 'Adults with pain.'),
}
COLUMNS = [
    'drugName',
    'nctIds',
    'targetFromSource',
    'targetSymbolSuggestion',
    'actsOnReceptor',
    'targetType',
    'actionType',
    'mechanismOfAction',
    'relation',
    'statementType',
    'confidence',
    'effect_wording',
    'evidence',
    'isHumanTarget',
]


def _mechanism(target: str, nct_id: str, quote: str) -> dict:
    return {
        'targetFromSource': target,
        'targetSymbolSuggestion': [target],
        'isHumanTarget': True,
        'actsOnReceptor': False,
        'relation': 'direct',
        'statementType': 'drug_description',
        'targetType': 'single protein',
        'actionType': 'INHIBITOR',
        'mechanismOfAction': f'{target} inhibitor',
        'evidence': [{'nctId': nct_id, 'quote': quote}],
        'confidence': 0.9,
    }


ANSWERS = {
    'sotorasib': [_mechanism('KRAS', 'nct1', KRAS_QUOTE), _mechanism('MEK', 'nct2', MEK_QUOTE)],
    'aspirinx': [_mechanism('COX', 'nct3', COX_QUOTE)],
}


class FakeBatches:
    """Message Batches stand-in answering each pool from ``ANSWERS``, and failing the drugs in ``failing``."""

    def __init__(self, failing: frozenset[str] = frozenset()) -> None:
        self.failing = failing
        self.created: list[list[dict]] = []

    def create(self, *, requests: list[dict]) -> SimpleNamespace:
        self.created.append(requests)
        return SimpleNamespace(id=f'batch_{len(self.created)}', processing_status='ended')

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=batch_id, processing_status='ended', request_counts=SimpleNamespace(processing=0))

    def results(self, batch_id: str):
        for request in self.created[int(batch_id.split('_')[1]) - 1]:
            text = request['params']['messages'][0]['content']
            drug = text.split('\n', 1)[0].removeprefix('Drug: ')
            if drug in self.failing:
                error = SimpleNamespace(type='api_error')
                yield SimpleNamespace(
                    custom_id=request['custom_id'], result=SimpleNamespace(type='errored', error=error)
                )
                continue
            body = {'drugName': drug, 'notStated': False, 'mechanisms': ANSWERS[drug]}
            message = SimpleNamespace(
                stop_reason='end_turn',
                content=[SimpleNamespace(type='text', text=json.dumps(body))],
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
                ),
            )
            yield SimpleNamespace(
                custom_id=request['custom_id'], result=SimpleNamespace(type='succeeded', message=message)
            )

    @property
    def drugs(self) -> list[str]:
        """The drug of every request submitted, in order."""
        return [
            r['params']['messages'][0]['content'].split('\n', 1)[0].removeprefix('Drug: ')
            for requests in self.created
            for r in requests
        ]


def _client(batches: FakeBatches) -> Any:
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _config(tmp_path: Path, name: str = 'work') -> Config:
    return Config(step='test', steps=['test'], work_path=tmp_path / name)


def _stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batches: FakeBatches | None,
    world: dict[str, tuple[str, str, str]] = WORLD,
    snapshot: str = 'v1',
    work: str = 'work',
    **settings: Any,
) -> dict[str, Any]:
    """Stage the mechanisms of ``world``, a trial id to its drug, description and detailed description."""
    ids = list(world)
    report = pl.DataFrame({
        'id': ids,
        'trialOfficialTitle': [f'title {i}' for i in ids],
        'trialDescription': [world[i][1] for i in ids],
        'trialDetailedDescription': [world[i][2] for i in ids],
    })
    monkeypatch.setattr(extraction, 'trial_report', lambda *args: report)
    prompts = extraction.prompts_for(report)
    drug = pl.Struct({'drug': pl.String})
    table = prompts.select('id', 'prompt_sha256').with_columns(
        investigated_drugs=pl.Series([[{'drug': world[i][0]}] for i in ids], dtype=pl.List(drug))
    )
    directory = tmp_path / snapshot / 'extraction' / EXTRACTION_TAG
    directory.mkdir(parents=True, exist_ok=True)
    table.write_parquet(directory / 'data.parquet')
    digest = schema_digest(ClinicalReportExtractionSchema)
    (directory / 'metadata.json').write_text(json.dumps(EXTRACTION.metadata(digest, None)))
    return stage_moa(
        str(tmp_path / 'aact.zip'),
        str(tmp_path / snapshot / 'extraction'),
        str(tmp_path / snapshot / 'extraction'),
        AactMoaSettings(store_uri=str(tmp_path / 'store'), snapshot=snapshot, **settings),
        _config(tmp_path, work),
        client=None if batches is None else _client(batches),
    )


def _snapshot(tmp_path: Path, snapshot: str = 'v1') -> Path:
    return tmp_path / snapshot / 'extraction' / EXTRACTION_TAG / 'moa' / MOA.tag()


def _rows(tmp_path: Path, snapshot: str = 'v1') -> list[tuple[str, str]]:
    frame = pl.read_parquet(_snapshot(tmp_path, snapshot) / 'data.parquet')
    return sorted(zip(frame['drugName'], frame['targetFromSource'], strict=True))


def test_first_run_computes_every_pool_and_publishes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches)

    assert stats == {
        'pools': 2,
        'seeded': 0,
        'carried': 0,
        'computed': 2,
        'failed': 0,
        'mechanisms': 3,
        'dropped_by_quote_check': 0,
        'published': True,
    }
    assert sorted(batches.drugs) == ['aspirinx', 'sotorasib']
    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    assert published.columns == COLUMNS
    assert _rows(tmp_path) == [('aspirinx', 'COX'), ('sotorasib', 'KRAS'), ('sotorasib', 'MEK')]
    kras = published.filter(pl.col('targetFromSource') == 'KRAS').row(0, named=True)
    assert kras['nctIds'] == ['nct1', 'nct2']
    assert kras['evidence'] == [{'nctId': 'nct1', 'quote': KRAS_QUOTE}]
    pools = pl.read_parquet(_snapshot(tmp_path) / 'pools.parquet').sort('drug_name')
    assert pools.columns == ['drug_name', 'n_trials', 'n_trials_used', 'extracted', 'status', 'carried']
    assert pools['extracted'].to_list() == [True, True]
    assert pools['carried'].to_list() == [False, False]


def test_second_run_on_the_same_inputs_computes_nothing_and_republishes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches)

    assert batches.created == []
    assert stats['computed'] == 0
    assert stats['carried'] == 0
    assert stats['mechanisms'] == 3
    assert stats['published'] is False


def test_carried_output_loses_a_mechanism_whose_quote_is_gone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    edited = {**WORLD, 'nct1': ('sotorasib', 'Sotorasib in NSCLC patients.', 'Adults with NSCLC.')}
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, world=edited, snapshot='v2')

    assert batches.created == []
    assert stats['carried'] == 1
    assert stats['computed'] == 0
    assert stats['mechanisms'] == 2
    assert stats['dropped_by_quote_check'] == 1
    assert _rows(tmp_path, 'v2') == [('aspirinx', 'COX'), ('sotorasib', 'MEK')]
    pools = pl.read_parquet(_snapshot(tmp_path, 'v2') / 'pools.parquet').sort('drug_name')
    assert pools['carried'].to_list() == [False, True]


def test_pool_whose_trials_are_all_gone_has_no_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    remaining = {k: v for k, v in WORLD.items() if v[0] == 'sotorasib'}
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, world=remaining, snapshot='v2')

    assert batches.created == []
    assert stats['pools'] == 1
    assert _rows(tmp_path, 'v2') == [('sotorasib', 'KRAS'), ('sotorasib', 'MEK')]


def test_a_new_pool_is_computed_and_never_carried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches(), world={k: v for k, v in WORLD.items() if k != 'nct3'})
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, snapshot='v2')

    assert batches.drugs == ['aspirinx']
    assert stats['carried'] == 0
    assert stats['computed'] == 1


def test_a_pool_is_computed_when_a_cited_trial_is_gone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, world={k: v for k, v in WORLD.items() if k != 'nct2'}, snapshot='v2')

    assert batches.drugs == ['sotorasib']
    assert (stats['carried'], stats['computed']) == (0, 1)


def test_records_name_the_prompt_they_were_computed_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())

    records = read_records(str(tmp_path / 'store' / MOA.name), _config(tmp_path), pl.String())

    expected = hashlib.sha256(schemas.load_prompt('extract').encode('utf-8')).hexdigest()
    assert {json.loads(o)['prompt_sha256'] for o in records['output']} == {expected}


def test_metadata_names_the_extraction_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())

    metadata = json.loads((_snapshot(tmp_path) / 'metadata.json').read_text())

    assert metadata == MOA.metadata(EXTRACTION_TAG)
    assert metadata['extraction_tag'] == EXTRACTION_TAG


def test_failed_pool_is_retried_next_run_until_max_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = _stage(monkeypatch, tmp_path, FakeBatches(failing=frozenset({'aspirinx'})), max_attempts=1)
    second_batches = FakeBatches(failing=frozenset({'aspirinx'}))
    second = _stage(monkeypatch, tmp_path, second_batches, snapshot='v2', max_attempts=2)
    third_batches = FakeBatches(failing=frozenset({'aspirinx'}))
    third = _stage(monkeypatch, tmp_path, third_batches, snapshot='v3', max_attempts=2)

    assert (first['computed'], first['failed']) == (2, 1)
    assert _rows(tmp_path) == [('sotorasib', 'KRAS'), ('sotorasib', 'MEK')]
    assert second_batches.drugs == ['aspirinx']
    assert (second['computed'], second['failed']) == (1, 1)
    assert third_batches.created == []
    assert (third['computed'], third['failed']) == (0, 1)
    pools = pl.read_parquet(_snapshot(tmp_path, 'v3') / 'pools.parquet').sort('drug_name')
    assert pools['extracted'].to_list() == [False, True]
    assert pools['status'].to_list() == ['failed', 'stated']


def test_cache_of_an_earlier_run_seeds_the_store_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    cache = tmp_path / 'work' / 'moa_cache.jsonl'
    batches = FakeBatches()

    stats = _stage(
        monkeypatch,
        tmp_path / 'other',
        batches,
        work='other_work',
        import_experiment_cache=str(cache),
        import_for_tag=MOA.tag(),
        import_for_snapshot='v1',
    )

    assert batches.created == []
    assert stats['seeded'] == 2
    assert stats['computed'] == 0
    assert stats['mechanisms'] == 3


def test_missing_workspace_id_fails_before_any_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID', raising=False)

    with pytest.raises(ComputeAborted, match='ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID'):
        _stage(monkeypatch, tmp_path, None)


def test_entry_point_reads_source_and_writes_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called: dict[str, Any] = {}

    def fake_stage(archive, extraction_dir, destination, settings, config, client=None):
        called.update(archive=archive, extraction_dir=extraction_dir, destination=destination, settings=settings)
        return {}

    monkeypatch.setattr('pts.transformers.clinical_mira.aact_moa.stage_moa', fake_stage)

    aact_moa(
        {'aact': 'a.zip', 'extraction': 'e'},
        {'moa': 'm'},
        {'store_uri': 's', 'snapshot': 'v1'},
        _config(tmp_path),
    )

    assert called['archive'] == 'a.zip'
    assert called['extraction_dir'] == 'e'
    assert called['destination'] == 'm'
    assert called['settings'] == AactMoaSettings(store_uri='s', snapshot='v1')


def test_config_imports_no_seed_and_caps_compute() -> None:
    config = yaml.safe_load((Path(__file__).parents[2] / 'config.yaml').read_text())
    settings = AactMoaSettings.model_validate(config['steps']['aact_moa'][0]['settings'])

    assert settings.import_experiment_cache is None
    assert settings.import_for_tag is None
    assert settings.import_for_snapshot is None
    assert settings.max_compute == 5000


def _seed_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: dict[str, tuple[str, str, str]]) -> Path:
    """Compute ``world`` in a separate store and return the result cache it leaves."""
    _stage(monkeypatch, tmp_path / 'seed', FakeBatches(), world=world, work='seed_work')
    return tmp_path / 'seed' / 'seed_work' / 'moa_cache.jsonl'


def _seed_settings(cache: Path, snapshot: str = 'v1', **settings: Any) -> dict[str, Any]:
    return {
        'import_experiment_cache': str(cache),
        'import_for_tag': MOA.tag(),
        'import_for_snapshot': snapshot,
        **settings,
    }


def test_a_seed_covering_too_few_pools_aborts_before_any_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, {k: v for k, v in WORLD.items() if k != 'nct3'})
    batches = FakeBatches()

    with pytest.raises(SeedTooSmall, match='1 of 2'):
        _stage(monkeypatch, tmp_path, batches, **_seed_settings(cache, min_seeded_share=0.6))

    assert batches.created == []


def test_a_seed_below_share_writes_no_import_and_an_adequate_seed_imports_on_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = str(tmp_path / 'store' / MOA.name)
    small = _seed_cache(monkeypatch, tmp_path / 'small', {k: v for k, v in WORLD.items() if k != 'nct3'})

    with pytest.raises(SeedTooSmall, match='not written'):
        _stage(monkeypatch, tmp_path, FakeBatches(), **_seed_settings(small, min_seeded_share=0.6))

    assert not run_exists(store, import_run_id(), _config(tmp_path))
    full = _seed_cache(monkeypatch, tmp_path / 'full', WORLD)
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, **_seed_settings(full, min_seeded_share=0.6))

    assert (stats['seeded'], stats['computed']) == (2, 0)
    assert batches.created == []


def test_an_existing_import_covering_too_few_pools_aborts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, {k: v for k, v in WORLD.items() if k != 'nct3'})
    _stage(monkeypatch, tmp_path, FakeBatches(), **_seed_settings(cache, min_seeded_share=0.0))
    batches = FakeBatches()

    with pytest.raises(SeedTooSmall, match='1 of 2'):
        _stage(monkeypatch, tmp_path, batches, snapshot='v2', **_seed_settings(cache, 'v2', min_seeded_share=0.6))

    assert batches.created == []


def test_a_rerun_after_a_full_import_does_not_abort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, WORLD)
    _stage(monkeypatch, tmp_path, FakeBatches(), **_seed_settings(cache))
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, snapshot='v2', **_seed_settings(cache, 'v2'))

    assert batches.created == []
    assert (stats['seeded'], stats['computed']) == (0, 0)


def test_a_later_snapshot_with_the_seed_settings_left_in_place_neither_imports_nor_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = str(tmp_path / 'store' / MOA.name)
    cache = _seed_cache(monkeypatch, tmp_path, {k: v for k, v in WORLD.items() if k != 'nct3'})
    _stage(monkeypatch, tmp_path, FakeBatches(), **_seed_settings(cache, min_seeded_share=0.5))
    imported = read_records(store, _config(tmp_path), pl.String()).filter(pl.col('run_id') == import_run_id()).height
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, snapshot='v2', **_seed_settings(cache, min_seeded_share=0.9))

    assert stats['seeded'] == 0
    assert batches.created == []
    after = read_records(store, _config(tmp_path), pl.String()).filter(pl.col('run_id') == import_run_id()).height
    assert after == imported


def test_a_later_snapshot_with_the_seed_settings_left_in_place_does_not_import_a_fresh_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, WORLD)
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, snapshot='v2', **_seed_settings(cache, min_seeded_share=0.9))

    assert stats['seeded'] == 0
    assert stats['computed'] == 2
    assert not run_exists(str(tmp_path / 'store' / MOA.name), import_run_id(), _config(tmp_path))


def test_import_for_another_tag_raises_before_reading_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def unread(*args: Any) -> None:
        raise AssertionError('inputs read')

    monkeypatch.setattr(aact_moa_module, 'EXTRACTION', SimpleNamespace(release_tag=unread, read=unread))
    monkeypatch.setattr(aact_moa_module.pooling, 'trial_texts', unread)
    settings = AactMoaSettings(
        store_uri=str(tmp_path / 'store'),
        snapshot='v1',
        import_experiment_cache=str(tmp_path / 'cache.jsonl'),
        import_for_tag='claude-opus-5-5-pold',
        import_for_snapshot='v1',
    )

    with pytest.raises(ValueError, match='import_for_tag claude-opus-5-5-pold is not the current tag'):
        stage_moa(
            'a.zip', str(tmp_path / 'e'), str(tmp_path / 'e'), settings, _config(tmp_path), _client(FakeBatches())
        )

    assert not (tmp_path / 'store').exists()


def test_more_pools_to_compute_than_max_compute_raises_before_any_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batches = FakeBatches()

    with pytest.raises(ComputeTooLarge, match='2 pools to compute, more than max_compute 1'):
        _stage(monkeypatch, tmp_path, batches, max_compute=1)

    assert batches.created == []
    assert not _snapshot(tmp_path).exists()


def test_max_compute_counts_only_pools_without_an_ok_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, {k: v for k, v in WORLD.items() if k != 'nct3'})
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, max_compute=1, **_seed_settings(cache, min_seeded_share=0.5))

    assert stats['computed'] == 1
    assert batches.drugs == ['aspirinx']


def test_max_compute_counts_carried_pools_as_done(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    edited = {**WORLD, 'nct1': ('sotorasib', 'Sotorasib in NSCLC patients.', 'Adults with NSCLC.')}
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, world=edited, snapshot='v2', max_compute=0)

    assert (stats['carried'], stats['computed']) == (1, 0)


def test_batch_state_is_named_by_snapshot_and_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())

    assert (tmp_path / 'store' / MOA.name / 'batches' / f'v1-{MOA.tag()}.json').exists()


def test_a_missing_pool_table_is_written_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    (_snapshot(tmp_path) / 'pools.parquet').unlink()

    stats = _stage(monkeypatch, tmp_path, FakeBatches())

    assert stats['published'] is False
    assert pl.read_parquet(_snapshot(tmp_path) / 'pools.parquet').height == 2
