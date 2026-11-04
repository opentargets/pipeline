import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from pts.transformers.clinical_mira.moa import __main__ as cli
from pts.transformers.clinical_mira.moa import llm

DRUG = pl.Struct({'drug': pl.String, 'synonyms': pl.List(pl.String)})
LABELS = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))
REPORT_DRUGS = pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String}))

TRIALS = {
    'n1': (
        'sotorasib',
        'Trial ID: n1\nDescription: sotorasib, a KRAS inhibitor, in lung cancer.',
        'CHEMBL1',
        'PHASE_1',
    ),
    'n2': ('drugz', 'Trial ID: n2\nDescription: drugz, a KRAS inhibitor, in colon cancer.', 'CHEMBL2', 'PHASE_2'),
    'n3': ('novelmab', 'Trial ID: n3\nDescription: novelmab, a KRAS inhibitor, first in human.', None, 'PHASE_1'),
}


JUDGE_INPUTS: list[str] = []


class FakeClient:
    """Extracts KRAS from every pool; the judge agrees with every mechanism."""

    def __init__(self, *_: object, **__: object) -> None:
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, *, instructions: str, input: str, **_: object) -> SimpleNamespace:
        if 'pharmacologist' in instructions:
            JUDGE_INPUTS.append(input)
            body = {'quoteSupports': 'yes', 'primaryMechanism': 'yes', 'rationale': 'stated'}
        else:
            nct = input.split('Trial ID: ')[1].split('\n', maxsplit=1)[0]
            body = {
                'drugName': 'x',
                'notStated': False,
                'mechanisms': [
                    {
                        'targetFromSource': 'KRAS',
                        'targetSymbolSuggestion': ['KRAS'],
                        'isHumanTarget': True,
                        'targetType': 'single protein',
                        'actionType': 'INHIBITOR',
                        'mechanismOfAction': 'KRAS inhibitor',
                        'evidence': [{'nctId': nct, 'quote': 'a KRAS inhibitor'}],
                        'confidence': 0.9,
                    }
                ],
            }
        return SimpleNamespace(output_text=json.dumps(body))

    async def close(self) -> None:
        return None


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    d = tmp_path / 'inputs'
    for sub in ('clinical_report', 'molecule', 'moa', 'target_view'):
        (d / sub).mkdir(parents=True)
    extraction = pl.DataFrame(
        {
            'id': list(TRIALS),
            'investigated_drugs': [[{'drug': t[0], 'synonyms': []}] for t in TRIALS.values()],
            'prompt_sha256': ['h'] * 3,
        },
        schema={'id': pl.String, 'investigated_drugs': pl.List(DRUG), 'prompt_sha256': pl.String},
    )
    extraction.write_parquet(d / 'extraction.parquet')
    pl.DataFrame(
        {
            'id': list(TRIALS),
            'source': ['ClinicalTrials.gov'] * 3,
            'drugs': [[{'drugFromSource': t[0], 'drugId': t[2]}] for t in TRIALS.values()],
            'clinicalStage': [t[3] for t in TRIALS.values()],
        },
        schema={'id': pl.String, 'source': pl.String, 'drugs': REPORT_DRUGS, 'clinicalStage': pl.String},
    ).write_parquet(d / 'clinical_report' / 'part.parquet')
    pl.DataFrame({
        'id': ['CHEMBL1', 'CHEMBL2'],
        'maximumClinicalStage': ['PHASE_1', 'PHASE_2'],
        'drugType': ['Small molecule', 'Small molecule'],
    }).write_parquet(d / 'molecule' / 'part.parquet')
    pl.DataFrame({
        'mechanismOfAction': ['KRAS inhibitor', 'EGFR inhibitor'],
        'actionType': ['INHIBITOR', 'INHIBITOR'],
        'chemblIds': [['CHEMBL1'], ['CHEMBL2']],
        'targets': [['ENSG_KRAS'], ['ENSG_EGFR']],
    }).write_parquet(d / 'moa' / 'part.parquet')
    pl.DataFrame(
        {
            'id': ['ENSG_KRAS', 'ENSG_EGFR'],
            'approvedSymbol': ['KRAS', 'EGFR'],
            'approvedName': ['KRAS proto-oncogene', 'epidermal growth factor receptor'],
            'symbolSynonyms': [[], []],
            'obsoleteSymbols': [[], []],
            'nameSynonyms': [[], []],
        },
        schema_overrides={'symbolSynonyms': LABELS, 'obsoleteSymbols': LABELS, 'nameSynonyms': LABELS},
    ).write_parquet(d / 'target_view' / 'part.parquet')
    return {'dir': d}


def test_cli_end_to_end_with_fake_client(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(llm, 'AsyncOpenAI', FakeClient)
    run = tmp_path / 'run'
    run.mkdir()
    pl.DataFrame({
        'id': list(TRIALS),
        'prompt': [t[1] for t in TRIALS.values()],
        'prompt_sha256': ['h'] * 3,
    }).write_parquet(run / 'texts.parquet')
    d = inputs['dir']

    def main(*args: str) -> None:
        cli.main([args[0], '--run-dir', str(run), *args[1:]])

    main('pools', '--extraction', str(d / 'extraction.parquet'))
    main(
        'link',
        '--clinical-report',
        str(d / 'clinical_report'),
        '--molecule',
        str(d / 'molecule'),
        '--moa',
        str(d / 'moa'),
    )
    main('pilot')
    main('extract')
    main('ground', '--target-view', str(d / 'target_view'))
    main('judge')
    main('evaluate')

    report = (run / 'report.md').read_text()
    assert report.startswith('# AACT MoA extraction')
    assert '## Headline precision by how the target was grounded' in report
    assert JUDGE_INPUTS
    assert not any('CHEMBL' in text for text in JUDGE_INPUTS)
    scored = pl.read_parquet(run / 'scored.parquet').sort('pool_id')
    assert scored['correct'].to_list().count(True) == 1
    assert scored['correct'].to_list().count(False) == 1
    assert (run / 'handcheck.csv').exists()
    early = (
        pl.read_parquet(run / 'summary_phase_group.parquet').filter(pl.col('phase_group') == 'early').row(0, named=True)
    )
    assert (early['scored'], early['correct']) == (2, 1)


def test_extract_limit_runs_only_that_many_pools(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(llm, 'AsyncOpenAI', FakeClient)
    run = tmp_path / 'run'
    run.mkdir()
    pl.DataFrame({
        'id': list(TRIALS),
        'prompt': [t[1] for t in TRIALS.values()],
        'prompt_sha256': ['h'] * 3,
    }).write_parquet(run / 'texts.parquet')
    d = inputs['dir']
    cli.main(['pools', '--run-dir', str(run), '--extraction', str(d / 'extraction.parquet')])

    cli.main(['extract', '--run-dir', str(run), '--all', '--limit', '2'])

    assert pl.read_parquet(run / 'statuses.parquet').height == 2
