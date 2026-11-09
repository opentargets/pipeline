import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers.clinical_mira.aact_moa import SeedTooSmall
from pts.transformers.clinical_mira.moa import schemas
from pts.transformers.clinical_mira.moa.literature_staging import parse_output
from pts.transformers.clinical_mira.moa_literature import (
    MoaLiteratureSettings,
    moa_literature,
    stage_moa_literature,
)
from pts.transformers.clinical_mira.record_store import ComputeAborted, SnapshotConflictError, read_records
from pts.transformers.clinical_mira.staged_outputs import METADATA_KEYS, MOA_LITERATURE

TAG = MOA_LITERATURE.tag()
KRAS_QUOTE = 'Sotorasib is a selective KRAS G12C inhibitor'
MEK_QUOTE = 'Sotorasib is a MEK pathway modulator'
COX_QUOTE = 'Aspirinx is a COX inhibitor'

SENTENCES = [
    ('CHEMBL1', 'sotorasib', '1', f'{KRAS_QUOTE} in NSCLC.'),
    ('CHEMBL1', 'sotorasib', '2', f'{MEK_QUOTE} in this arm.'),
    ('CHEMBL2', 'aspirinx', '3', f'{COX_QUOTE} used for pain relief.'),
]
"""Each match sentence as ``(molecule, label, pmid, text)``."""

MOLECULES = pl.DataFrame({
    'id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL9'],
    'parentId': [None, None, None],
    'maximumClinicalStage': ['APPROVAL', 'PHASE_2', 'PRECLINICAL'],
    'name': ['SOTORASIB', 'ASPIRINX', 'NOTCLINICAL'],
})

COLUMNS = [
    'drugId',
    'docIds',
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


def _mechanism(target: str, doc: str, quote: str) -> dict:
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
        'evidence': [{'nctId': doc, 'quote': quote}],
        'confidence': 0.9,
    }


ANSWERS = {
    'sotorasib': [_mechanism('KRAS', 'pmid1', KRAS_QUOTE), _mechanism('MEK', 'pmid2', MEK_QUOTE)],
    'aspirinx': [_mechanism('COX', 'pmid3', COX_QUOTE)],
}


class FakeBatches:
    """Message Batches stand-in answering each drug from ``answers``."""

    def __init__(self, answers: dict[str, list[dict]] = ANSWERS) -> None:
        self.answers = answers
        self.created: list[list[dict]] = []

    def create(self, *, requests: list[dict]) -> SimpleNamespace:
        self.created.append(requests)
        return SimpleNamespace(id=f'batch_{len(self.created)}', processing_status='ended')

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=batch_id, processing_status='ended', request_counts=SimpleNamespace(processing=0))

    def results(self, batch_id: str):
        for request in self.created[int(batch_id.split('_')[1]) - 1]:
            drug = _drug(request)
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
    def drugs(self) -> list[str]:
        """The drug of every request submitted, in order."""
        return [_drug(r) for requests in self.created for r in requests]


def _drug(request: dict) -> str:
    return request['params']['messages'][0]['content'].split('\n', 1)[0].removeprefix('Drug: ')


def _client(batches: FakeBatches) -> Any:
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _config(tmp_path: Path, name: str = 'work') -> Config:
    return Config(step='test', steps=['test'], work_path=tmp_path / name)


def _write_inputs(root: Path, sentences: list[tuple[str, str, str, str]]) -> tuple[Path, Path]:
    """Write the drug ``sentences`` and a KRAS gene mention in the first one as a match file."""
    match = root / 'match'
    match.mkdir(parents=True, exist_ok=True)
    rows = [('CD', *s) for s in sentences] + [('GP', 'ENSG00000133703', 'KRAS', SENTENCES[0][2], SENTENCES[0][3])]
    pl.DataFrame(
        {
            'pmid': [r[3] for r in rows],
            'pmcid': [None] * len(rows),
            'section': ['abstract'] * len(rows),
            'year': [2020] * len(rows),
            'type': [r[0] for r in rows],
            'mappedId': [r[1] for r in rows],
            'label': [r[2] for r in rows],
            'text': [r[4] for r in rows],
            'isValid': [True] * len(rows),
        },
        schema={
            'pmid': pl.String,
            'pmcid': pl.String,
            'section': pl.String,
            'year': pl.Int64,
            'type': pl.String,
            'mappedId': pl.String,
            'label': pl.String,
            'text': pl.String,
            'isValid': pl.Boolean,
        },
    ).write_parquet(match / 'part-0000.parquet')
    molecule = root / 'drug_molecule'
    molecule.mkdir(parents=True, exist_ok=True)
    MOLECULES.write_parquet(molecule / 'part-0000.parquet')
    return match, molecule


def _stage(
    tmp_path: Path,
    batches: FakeBatches | None,
    sentences: list[tuple[str, str, str, str]] = SENTENCES,
    release: str = '26.09',
    work: str = 'work',
    **settings: Any,
) -> dict[str, Any]:
    match, molecule = _write_inputs(tmp_path / 'inputs' / work, sentences)
    return stage_moa_literature(
        str(match),
        str(molecule),
        str(tmp_path / 'literature'),
        MoaLiteratureSettings(store_uri=str(tmp_path / 'store'), input_release=release, workers=1, **settings),
        _config(tmp_path, work),
        client=None if batches is None else _client(batches),
    )


def _snapshot(tmp_path: Path, release: str = '26.09') -> Path:
    return tmp_path / 'literature' / release / 'moa_literature' / TAG


def _rows(tmp_path: Path, release: str = '26.09') -> list[tuple[str, str]]:
    frame = pl.read_parquet(_snapshot(tmp_path, release) / 'data.parquet')
    return sorted(zip(frame['drugId'], frame['targetFromSource'], strict=True))


def _seed(tmp_path: Path, parents: set[str]) -> Path:
    """Compute every drug in a separate store and write the outputs of ``parents`` as an experiment seed."""
    _stage(tmp_path / 'seed', FakeBatches())
    records = read_records(str(tmp_path / 'seed' / 'store' / MOA_LITERATURE.name), _config(tmp_path), pl.String())
    prompt_sha = hashlib.sha256(schemas.literature_system_prompt().encode('utf-8')).hexdigest()
    lines = []
    for key, output in records.select('text', 'output').iter_rows():
        parsed = parse_output(output)
        if parsed['parent'] in parents:
            lines.append({
                'content_key': key,
                'parent': parsed['parent'],
                'response': parsed['extraction'],
                'usage': {'input_tokens': 3},
                'prompt_sha256': prompt_sha,
                'run': 'pilot',
            })
    path = tmp_path / 'seed.jsonl'
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))
    return path


def test_first_run_imports_seed_and_computes_the_rest(tmp_path: Path) -> None:
    seed = _seed(tmp_path, {'CHEMBL1'})
    batches = FakeBatches()

    stats = _stage(tmp_path, batches, import_experiment_cache=str(seed), import_for_tag=TAG, min_seeded_share=0.5)

    assert stats == {
        'drugs': 2,
        'sentences': 3,
        'pools': 2,
        'seeded': 1,
        'carried': 0,
        'computed': 1,
        'failed': 0,
        'mechanisms': 3,
        'dropped_by_quote_check': 0,
        'published': True,
    }
    assert batches.drugs == ['aspirinx']
    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    assert published.columns == COLUMNS
    assert _rows(tmp_path) == [('CHEMBL1', 'KRAS'), ('CHEMBL1', 'MEK'), ('CHEMBL2', 'COX')]
    kras = published.filter(pl.col('targetFromSource') == 'KRAS').row(0, named=True)
    assert kras['docIds'] == ['pmid1']
    assert kras['evidence'] == [{'nctId': 'pmid1', 'quote': KRAS_QUOTE}]
    metadata = json.loads((_snapshot(tmp_path) / 'metadata.json').read_text())
    assert set(METADATA_KEYS) <= set(metadata)
    assert metadata['kind'] == MOA_LITERATURE.name
    assert metadata['tag'] == TAG
    assert metadata['input_release'] == '26.09'
    assert 'extraction_tag' not in metadata
    drugs = pl.read_parquet(_snapshot(tmp_path) / 'drugs.parquet').sort('parent')
    assert drugs.to_dicts() == [
        {
            'parent': 'CHEMBL1',
            'name': 'sotorasib',
            'n_sentences': 2,
            'extracted': True,
            'status': 'stated',
            'carried': False,
        },
        {
            'parent': 'CHEMBL2',
            'name': 'aspirinx',
            'n_sentences': 1,
            'extracted': True,
            'status': 'stated',
            'carried': False,
        },
    ]
    assert (tmp_path / 'store' / MOA_LITERATURE.name / 'batches' / f'26.09-{TAG}.json').exists()


def test_second_run_on_same_input_computes_nothing(tmp_path: Path) -> None:
    _stage(tmp_path, FakeBatches())
    batches = FakeBatches()

    stats = _stage(tmp_path, batches)

    assert batches.created == []
    assert (stats['computed'], stats['carried'], stats['seeded']) == (0, 0, 0)
    assert stats['mechanisms'] == 3
    assert stats['published'] is False


def test_quote_not_in_drug_sentences_is_dropped(tmp_path: Path) -> None:
    shared = [*SENTENCES, ('CHEMBL2', 'aspirinx', '1', 'Aspirinx was given together with sotorasib.')]
    answers = {**ANSWERS, 'aspirinx': [*ANSWERS['aspirinx'], _mechanism('KRAS', 'pmid1', KRAS_QUOTE)]}

    stats = _stage(tmp_path, FakeBatches(answers), sentences=shared)

    assert stats['dropped_by_quote_check'] == 1
    assert _rows(tmp_path) == [('CHEMBL1', 'KRAS'), ('CHEMBL1', 'MEK'), ('CHEMBL2', 'COX')]


def test_unset_workspace_aborts_before_compute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID', raising=False)

    with pytest.raises(ComputeAborted, match='ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID'):
        _stage(tmp_path, None)

    assert not _snapshot(tmp_path).exists()


def test_seed_below_share_raises(tmp_path: Path) -> None:
    seed = _seed(tmp_path, {'CHEMBL1'})
    batches = FakeBatches()

    with pytest.raises(SeedTooSmall, match='1 of 2'):
        _stage(tmp_path, batches, import_experiment_cache=str(seed), import_for_tag=TAG, min_seeded_share=0.6)

    assert batches.created == []


def test_snapshot_is_write_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs = iter(range(100))
    monkeypatch.setattr(
        'pts.transformers.clinical_mira.moa_literature.new_run_id', lambda release: f'{release}-run{next(runs)}'
    )
    _stage(tmp_path, FakeBatches())
    before = (_snapshot(tmp_path) / 'data.parquet').read_bytes()
    edited = [*SENTENCES[:2], ('CHEMBL2', 'aspirinx', '3', 'Aspirinx is a PTGS2 inhibitor given for chronic pain.')]
    answers = {**ANSWERS, 'aspirinx': [_mechanism('PTGS2', 'pmid3', 'Aspirinx is a PTGS2 inhibitor')]}

    with pytest.raises(SnapshotConflictError):
        _stage(tmp_path, FakeBatches(answers), sentences=edited, work='work2')

    assert (_snapshot(tmp_path) / 'data.parquet').read_bytes() == before
    stats = _stage(tmp_path, FakeBatches(), sentences=edited, release='26.12', work='work3')
    assert stats['published'] is True


def test_entry_point_reads_source_and_writes_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called: dict[str, Any] = {}

    def fake_stage(match, drug_molecule, destination, settings, config, client=None):
        called.update(match=match, drug_molecule=drug_molecule, destination=destination, settings=settings)
        return {}

    monkeypatch.setattr('pts.transformers.clinical_mira.moa_literature.stage_moa_literature', fake_stage)

    moa_literature(
        {'match': 'm', 'drug_molecule': 'd'},
        {'moa': 'out'},
        {'store_uri': 's', 'input_release': '26.09'},
        _config(tmp_path),
    )

    assert called == {
        'match': 'm',
        'drug_molecule': 'd',
        'destination': 'out',
        'settings': MoaLiteratureSettings(store_uri='s', input_release='26.09'),
    }
