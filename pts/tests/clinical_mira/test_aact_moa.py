import json
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import polars as pl
import pytest
import yaml
from otter.config.model import Config

from pts.transformers.clinical_mira import aact_trial_extraction as extraction
from pts.transformers.clinical_mira.aact_moa import AactMoaSettings, SeedTooSmall, aact_moa, stage_moa
from pts.transformers.clinical_mira.moa import papers, schemas
from pts.transformers.clinical_mira.record_store import ComputeAborted, read_records, write_run
from pts.transformers.clinical_mira.schemas import ClinicalReportExtractionSchema
from pts.transformers.clinical_mira.staged_outputs import EXTRACTION, MOA, schema_digest

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
    'papers',
]
PAPER = f'{KRAS_QUOTE} in the publication.'
QUIET_PAPER = 'Adults were enrolled and followed.'


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

    def __init__(self, failing: frozenset[str] = frozenset(), answers: dict[str, list[dict]] | None = None) -> None:
        self.failing = failing
        self.answers = answers or ANSWERS
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
            body = {'drugName': drug, 'notStated': False, 'mechanisms': self.answers[drug]}
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
    def prompts(self) -> dict[str, str]:
        """The system prompt of the request submitted for each drug."""
        return {
            r['params']['messages'][0]['content'].split('\n', 1)[0].removeprefix('Drug: '): r['params']['system'][0][
                'text'
            ]
            for requests in self.created
            for r in requests
        }

    @property
    def drugs(self) -> list[str]:
        """The drug of every request submitted, in order."""
        return [
            r['params']['messages'][0]['content'].split('\n', 1)[0].removeprefix('Drug: ')
            for requests in self.created
            for r in requests
        ]


def _europepmc(abstracts: dict[str, tuple[str, str]]) -> Any:
    """A Europe PMC stand-in returning the abstracts it knows among the PMIDs asked for."""

    def get(url: str) -> dict:
        query = parse_qs(urlparse(url).query)['query'][0]
        found = [i for i in re.findall(r'EXT_ID:(\d+)', query) if i in abstracts]
        return {
            'resultList': {
                'result': [{'pmid': i, 'title': abstracts[i][0], 'abstractText': abstracts[i][1]} for i in found]
            }
        }

    return get


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
    links: Sequence[tuple[str, str]] = (),
    abstracts: dict[str, tuple[str, str]] | None = None,
    get: Any = None,
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
    linked = pl.DataFrame(
        {'nct_id': [n for n, _ in links], 'pmid': [p for _, p in links]},
        schema={'nct_id': pl.String, 'pmid': pl.String},
    )
    monkeypatch.setattr(papers, 'trial_papers', lambda *args: linked)
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
        get=get or _europepmc(abstracts or {}),
    )


def _cache(root: Path, work: str) -> Path:
    """The result cache of the extraction prompt in a work directory."""
    (cache,) = (root / work).glob('moa_cache-*.jsonl')
    return cache


def _snapshot(tmp_path: Path, snapshot: str = 'v1') -> Path:
    return tmp_path / snapshot / 'extraction' / EXTRACTION_TAG / 'moa' / MOA.tag()


def _rows(tmp_path: Path, snapshot: str = 'v1') -> list[tuple[str, str]]:
    frame = pl.read_parquet(_snapshot(tmp_path, snapshot) / 'data.parquet')
    return sorted(zip(frame['drugName'], frame['targetFromSource'], strict=True))


def test_first_run_computes_every_pool_and_publishes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches)

    assert stats == {
        'papers': 0,
        'abstracts_fetched': 0,
        'adopted': 0,
        'pools_with_papers': 0,
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
    assert kras['evidence'] == [{'nctId': 'nct1', 'pmid': None, 'quote': KRAS_QUOTE}]
    assert kras['papers'] == []
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
    cache = _cache(tmp_path, 'work')
    batches = FakeBatches()

    stats = _stage(
        monkeypatch,
        tmp_path / 'other',
        batches,
        work='other_work',
        import_experiment_cache=str(cache),
        import_for_tag=MOA.tag(),
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


def test_config_imports_the_seed_for_the_current_tag_only() -> None:
    config = yaml.safe_load((Path(__file__).parents[2] / 'config.yaml').read_text())
    settings = AactMoaSettings.model_validate(config['steps']['aact_moa'][0]['settings'])

    assert settings.import_for_tag == MOA.tag()
    assert settings.min_seeded_share == 0.5
    assert settings.adopt_from_tag == 'claude-opus-5-5-p4e6708cf'


def _seed_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: dict[str, tuple[str, str, str]]) -> Path:
    """Compute ``world`` in a separate store and return the result cache it leaves."""
    _stage(monkeypatch, tmp_path / 'seed', FakeBatches(), world=world, work='seed_work')
    return _cache(tmp_path / 'seed', 'seed_work')


def test_a_seed_covering_too_few_pools_aborts_before_any_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, {k: v for k, v in WORLD.items() if k != 'nct3'})
    batches = FakeBatches()

    with pytest.raises(SeedTooSmall, match='1 of 2'):
        _stage(
            monkeypatch,
            tmp_path,
            batches,
            import_experiment_cache=str(cache),
            import_for_tag=MOA.tag(),
            min_seeded_share=0.6,
        )

    assert batches.created == []


def test_an_existing_import_covering_too_few_pools_aborts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, {k: v for k, v in WORLD.items() if k != 'nct3'})
    seed = {'import_experiment_cache': str(cache), 'import_for_tag': MOA.tag()}
    _stage(monkeypatch, tmp_path, FakeBatches(), min_seeded_share=0.0, **seed)
    batches = FakeBatches()

    with pytest.raises(SeedTooSmall, match='1 of 2'):
        _stage(monkeypatch, tmp_path, batches, snapshot='v2', min_seeded_share=0.6, **seed)

    assert batches.created == []


def test_a_rerun_after_a_full_import_does_not_abort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, WORLD)
    seed = {'import_experiment_cache': str(cache), 'import_for_tag': MOA.tag()}
    _stage(monkeypatch, tmp_path, FakeBatches(), **seed)
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, snapshot='v2', **seed)

    assert batches.created == []
    assert (stats['seeded'], stats['computed']) == (0, 0)


def test_a_seed_for_another_tag_is_not_imported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _seed_cache(monkeypatch, tmp_path, WORLD)
    batches = FakeBatches()

    stats = _stage(
        monkeypatch, tmp_path, batches, import_experiment_cache=str(cache), import_for_tag='claude-opus-5-5-pold'
    )

    assert stats['seeded'] == 0
    assert stats['computed'] == 2
    assert sorted(batches.drugs) == ['aspirinx', 'sotorasib']
    assert not list((tmp_path / 'store' / MOA.name / 'runs').glob('import-*'))


def test_batch_state_is_named_by_snapshot_and_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())

    assert list((tmp_path / 'store' / MOA.name / 'batches').glob(f'v1-{MOA.tag()}-*.json'))


def test_a_missing_pool_table_is_written_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    (_snapshot(tmp_path) / 'pools.parquet').unlink()

    stats = _stage(monkeypatch, tmp_path, FakeBatches())

    assert stats['published'] is False
    assert pl.read_parquet(_snapshot(tmp_path) / 'pools.parquet').height == 2


def _with_paper_answers(nct_or_pmid: str) -> dict[str, list[dict]]:
    return {**ANSWERS, 'sotorasib': [_mechanism('KRAS', nct_or_pmid, PAPER), _mechanism('MEK', 'nct2', MEK_QUOTE)]}


PAPER_WORLD = {'links': [('nct1', '111')], 'abstracts': {'111': ('A report', PAPER)}}


def test_snapshot_evidence_splits_trial_and_paper_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches(answers=_with_paper_answers('pmid111'))

    _stage(monkeypatch, tmp_path, batches, **PAPER_WORLD)

    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    kras = published.filter(pl.col('targetFromSource') == 'KRAS').row(0, named=True)
    mek = published.filter(pl.col('targetFromSource') == 'MEK').row(0, named=True)
    assert kras['evidence'] == [{'nctId': None, 'pmid': '111', 'quote': PAPER.removesuffix('.')}]
    assert mek['evidence'] == [{'nctId': 'nct2', 'pmid': None, 'quote': MEK_QUOTE}]
    assert kras['nctIds'] == ['nct1', 'nct2']
    assert batches.prompts['sotorasib'] == schemas.papers_prompt()
    assert batches.prompts['aspirinx'] == schemas.load_prompt('extract')


def test_snapshot_papers_column_lists_pool_papers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    world = {
        'links': [('nct1', '111'), ('nct2', '222'), ('nct2', '333')],
        'abstracts': {'111': ('A', PAPER), '222': ('B', QUIET_PAPER)},
    }

    stats = _stage(monkeypatch, tmp_path, FakeBatches(), **world)

    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    for row in published.filter(pl.col('drugName') == 'sotorasib').iter_rows(named=True):
        assert row['papers'] == [{'pmid': '222', 'nctId': 'nct2'}, {'pmid': '111', 'nctId': 'nct1'}]
        assert row['nctIds'] == ['nct1', 'nct2']
    assert published.filter(pl.col('drugName') == 'aspirinx')['papers'].to_list() == [[]]
    assert (stats['papers'], stats['abstracts_fetched'], stats['pools_with_papers']) == (2, 2, 1)


def test_quote_citing_foreign_pmid_is_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches(answers=_with_paper_answers('pmid999'))

    stats = _stage(monkeypatch, tmp_path, batches, **PAPER_WORLD)

    assert stats['dropped_by_quote_check'] == 1
    assert _rows(tmp_path) == [('aspirinx', 'COX'), ('sotorasib', 'MEK')]


def test_stage_moa_publishes_when_europepmc_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr('pts.transformers.clinical_mira.moa.papers.time.sleep', lambda seconds: None)

    def down(url: str) -> dict:
        raise OSError('Europe PMC is down')

    stats = _stage(monkeypatch, tmp_path, FakeBatches(), get=down, links=[('nct1', '111')])

    assert stats['published'] is True
    assert (stats['papers'], stats['abstracts_fetched'], stats['pools_with_papers']) == (0, 0, 0)
    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    assert published['papers'].to_list() == [[]] * 3
    assert _rows(tmp_path) == [('aspirinx', 'COX'), ('sotorasib', 'KRAS'), ('sotorasib', 'MEK')]


def test_abstracts_are_cached_between_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches(), **PAPER_WORLD)

    def down(url: str) -> dict:
        raise OSError('Europe PMC is down')

    batches = FakeBatches()
    stats = _stage(monkeypatch, tmp_path, batches, snapshot='v2', get=down, links=[('nct1', '111')])

    assert batches.created == []
    assert (stats['papers'], stats['abstracts_fetched'], stats['pools_with_papers']) == (1, 0, 1)
    assert len(list((tmp_path / 'store' / 'moa' / 'papers').glob('*.parquet'))) == 1


def test_stage_moa_adopts_pools_without_papers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path / 'seed', FakeBatches(), work='seed_work')
    seed_store = str(tmp_path / 'seed' / 'store' / MOA.name)
    old = read_records(seed_store, _config(tmp_path), pl.String()).with_columns(model_revision=pl.lit('old-tag'))
    write_run(old, str(tmp_path / 'other' / 'store' / MOA.name), 'old-run', _config(tmp_path), pl.String())
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path / 'other', batches, work='other_work', adopt_from_tag='old-tag', **PAPER_WORLD)

    assert batches.drugs == ['sotorasib']
    assert (stats['adopted'], stats['computed'], stats['pools_with_papers']) == (1, 1, 1)
    assert _rows(tmp_path / 'other') == [('aspirinx', 'COX'), ('sotorasib', 'KRAS'), ('sotorasib', 'MEK')]


def test_carry_forward_covers_a_new_paper_without_vocab(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    batches = FakeBatches()

    stats = _stage(
        monkeypatch,
        tmp_path,
        batches,
        snapshot='v2',
        links=[('nct1', '222')],
        abstracts={'222': ('A report', QUIET_PAPER)},
    )

    assert batches.created == []
    assert (stats['carried'], stats['computed'], stats['pools_with_papers']) == (1, 0, 1)
    assert stats['mechanisms'] == 3
    published = pl.read_parquet(_snapshot(tmp_path, 'v2') / 'data.parquet')
    assert published.filter(pl.col('drugName') == 'sotorasib')['papers'].to_list()[0] == [
        {'pmid': '222', 'nctId': 'nct1'}
    ]


def test_a_pool_with_papers_is_never_adopted_even_under_its_current_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage(monkeypatch, tmp_path / 'seed', FakeBatches(), work='seed_work', **PAPER_WORLD)
    seed_store = str(tmp_path / 'seed' / 'store' / MOA.name)
    old = read_records(seed_store, _config(tmp_path), pl.String()).with_columns(model_revision=pl.lit('old-tag'))
    write_run(old, str(tmp_path / 'other' / 'store' / MOA.name), 'old-run', _config(tmp_path), pl.String())
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path / 'other', batches, work='other_work', adopt_from_tag='old-tag', **PAPER_WORLD)

    assert batches.drugs == ['sotorasib']
    assert batches.prompts['sotorasib'] == schemas.papers_prompt()
    assert (stats['adopted'], stats['computed']) == (1, 1)
