"""Tests for the drug_mechanism_of_action_aact transformer."""

from pathlib import Path

import polars as pl
import yaml
from loguru import logger

from pts.transformers.clinical_mira.record_store import publish_snapshot
from pts.transformers.clinical_mira.staged_outputs import MOA
from pts.transformers.drug_mechanism_of_action_aact import build_candidates, drug_mechanism_of_action_aact
from pts.transformers.utils.dataset import scan_dataset, write_dataset

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
SETTINGS = {'link_share': 0.8, 'class_level_genes': 20, 'report_min_confidence': 0.70}

LABELS = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))
DRUGS = pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String}))
EVIDENCE = pl.List(pl.Struct({'nctId': pl.String, 'quote': pl.String}))


def _target() -> pl.DataFrame:
    genes = [('ENSG_EGFR', 'EGFR', 'epidermal growth factor receptor')]
    genes += [(f'ENSG_K{i}', f'K{i}', f'kinase {i}') for i in range(21)]
    return pl.DataFrame(
        {
            'id': [g[0] for g in genes],
            'approvedSymbol': [g[1] for g in genes],
            'approvedName': [g[2] for g in genes],
            'symbolSynonyms': [[{'label': 'KINASEFAM', 'source': 's'}] if g[1].startswith('K') else [] for g in genes],
            'obsoleteSymbols': [[] for _ in genes],
            'nameSynonyms': [[] for _ in genes],
        },
        schema={
            'id': pl.String,
            'approvedSymbol': pl.String,
            'approvedName': pl.String,
            'symbolSynonyms': LABELS,
            'obsoleteSymbols': LABELS,
            'nameSynonyms': LABELS,
        },
    )


def _molecule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'id': ['CHEMBL1', 'CHEMBL1SALT', 'CHEMBL2'],
            'parentId': ['CHEMBL1', 'CHEMBL1', None],
            'childChemblIds': [['CHEMBL1SALT'], [], []],
            'drugType': ['Small molecule', 'Small molecule', 'Antibody'],
        },
        schema={
            'id': pl.String,
            'parentId': pl.String,
            'childChemblIds': pl.List(pl.String),
            'drugType': pl.String,
        },
    )


def _report(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'id': [r[0] for r in rows],
            'source': ['ClinicalTrials.gov'] * len(rows),
            'drugs': [[{'drugFromSource': 'x', 'drugId': r[1]}] for r in rows],
            'clinicalStage': [r[2] for r in rows],
        },
        schema={'id': pl.String, 'source': pl.String, 'drugs': DRUGS, 'clinicalStage': pl.String},
    )


def _mechanism(**overrides: object) -> dict:
    return {
        'drugName': 'drug a',
        'nctIds': ['NCT1', 'NCT2'],
        'targetFromSource': 'EGFR',
        'targetSymbolSuggestion': ['EGFR'],
        'actsOnReceptor': False,
        'targetType': 'PROTEIN',
        'actionType': 'INHIBITOR',
        'mechanismOfAction': 'EGFR inhibitor',
        'relation': 'direct',
        'statementType': 'drug_description',
        'confidence': 0.9,
        'effect_wording': 'inhibits',
        'evidence': [{'nctId': 'nct2', 'quote': 'q2'}, {'nctId': 'nct1', 'quote': 'q1'}],
        'isHumanTarget': True,
    } | overrides


def _moa(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(
        list(rows) or [_mechanism()],
        schema={
            'drugName': pl.String,
            'nctIds': pl.List(pl.String),
            'targetFromSource': pl.String,
            'targetSymbolSuggestion': pl.List(pl.String),
            'actsOnReceptor': pl.Boolean,
            'targetType': pl.String,
            'actionType': pl.String,
            'mechanismOfAction': pl.String,
            'relation': pl.String,
            'statementType': pl.String,
            'confidence': pl.Float64,
            'effect_wording': pl.String,
            'evidence': EVIDENCE,
            'isHumanTarget': pl.Boolean,
        },
    )


REPORT = _report([('NCT1', 'CHEMBL1', 'PHASE_2'), ('NCT2', 'CHEMBL1', 'PHASE_3')])


def _build(moa: pl.DataFrame, report: pl.DataFrame = REPORT, **settings: object) -> pl.DataFrame:
    merged = SETTINGS | settings
    return build_candidates(
        moa,
        report,
        _molecule(),
        _target(),
        link_share=merged['link_share'],
        class_level_genes=merged['class_level_genes'],
        report_min_confidence=merged['report_min_confidence'],
    )


def test_a_linked_grounded_mechanism_becomes_a_candidate() -> None:
    row = _build(_moa()).row(0, named=True)

    assert row['parentChemblId'] == 'CHEMBL1'
    assert row['chemblIds'] == ['CHEMBL1', 'CHEMBL1SALT']
    assert row['maximumClinicalStage'] == 'PHASE_3'
    assert row['drugType'] == 'Small molecule'
    assert row['targetFromSource'] == 'EGFR'
    assert row['targets'] == ['ENSG_EGFR']
    assert row['targetSymbols'] == ['EGFR']
    assert row['targetType'] == 'protein'
    assert row['actionType'] == 'INHIBITOR'
    assert row['nctIds'] == ['NCT1', 'NCT2']
    assert (row['quote'], row['quoteNctId']) == ('q2', 'NCT2')


def test_output_nct_ids_are_the_cited_trials_not_the_pool() -> None:
    moa = _moa(_mechanism(nctIds=['NCT1', 'NCT2', 'NCT3'], evidence=[{'nctId': 'nct2', 'quote': 'q'}]))

    assert _build(moa)['nctIds'].to_list() == [['NCT2']]


def test_salt_link_publishes_under_the_parent() -> None:
    report = _report([('NCT1', 'CHEMBL1SALT', 'PHASE_1'), ('NCT2', 'CHEMBL1SALT', 'PHASE_1')])

    row = _build(_moa(), report).row(0, named=True)

    assert row['parentChemblId'] == 'CHEMBL1'
    assert row['chemblIds'] == ['CHEMBL1', 'CHEMBL1SALT']
    assert row['drugType'] == 'Small molecule'
    assert row['maximumClinicalStage'] == 'PHASE_1'


def test_a_parent_absent_from_the_molecules_still_lists_itself_in_chembl_ids() -> None:
    molecule = pl.concat([
        _molecule(),
        pl.DataFrame(
            {'id': ['CHEMBL5SALT'], 'parentId': ['CHEMBL5'], 'childChemblIds': [[]], 'drugType': ['Small molecule']},
            schema=_molecule().schema,
        ),
    ])
    report = _report([('NCT1', 'CHEMBL5SALT', 'PHASE_2'), ('NCT2', 'CHEMBL5SALT', 'PHASE_2')])

    candidates = build_candidates(_moa(), report, molecule, _target(), **SETTINGS)

    assert candidates.select('parentChemblId', 'chemblIds').rows() == [('CHEMBL5', ['CHEMBL5'])]


def test_a_pool_linked_to_a_drug_absent_from_the_molecules_is_dropped_and_counted() -> None:
    report = _report([('NCT1', 'CHEMBL404', 'PHASE_2'), ('NCT2', 'CHEMBL404', 'PHASE_2')])
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(message.record['message']))
    try:
        candidates = _build(_moa(), report)
    finally:
        logger.remove(handler)

    assert candidates.is_empty()
    assert any('1 linked pools' in m and 'chembl_molecule' in m for m in messages), messages


def test_pool_with_split_trial_drugs_is_not_linked() -> None:
    nct_ids = [f'NCT{i}' for i in range(1, 6)]
    report = _report([(n, 'CHEMBL1' if i < 3 else 'CHEMBL2', 'PHASE_2') for i, n in enumerate(nct_ids)])

    assert _build(_moa(_mechanism(nctIds=nct_ids)), report).is_empty()
    assert _build(_moa(_mechanism(nctIds=nct_ids)), report, link_share=0.6).height == 1


def test_indirect_hypothesis_other_and_class_level_are_dropped() -> None:
    moa = _moa(
        _mechanism(relation='indirect'),
        _mechanism(statementType='hypothesis'),
        _mechanism(actionType='OTHER'),
        _mechanism(isHumanTarget=False),
        _mechanism(targetFromSource='KINASEFAM', targetSymbolSuggestion=[f'K{i}' for i in range(21)]),
        _mechanism(targetFromSource='unknown thing', targetSymbolSuggestion=[]),
        _mechanism(mechanismOfAction='kept'),
    )

    assert _build(moa)['mechanismOfAction'].to_list() == ['kept']
    assert _build(moa, class_level_genes=30).height == 2


def test_confidence_below_the_report_minimum_is_dropped() -> None:
    moa = _moa(_mechanism(confidence=0.69, mechanismOfAction='low'), _mechanism(confidence=0.70))

    assert _build(moa)['mechanismOfAction'].to_list() == ['EGFR inhibitor']
    assert _build(moa, report_min_confidence=0.5).height == 2


def test_pool_without_a_linking_trial_is_dropped() -> None:
    assert _build(_moa(), _report([('NCT9', 'CHEMBL1', 'PHASE_2')])).is_empty()


def test_stage_comes_from_clinical_report_not_drug_molecule() -> None:
    task = yaml.safe_load(CONFIG_PATH.read_text())['steps']['drug_mechanism_of_action_aact'][0]

    assert task['transformer'] == 'drug_mechanism_of_action_aact'
    assert set(task['source']) == {'moa', 'clinical_report', 'chembl_molecule', 'target'}
    assert task['destination'] == {'candidates': 'intermediate/drug_mechanism_of_action_aact'}
    assert task['settings'] == SETTINGS


def test_the_step_reads_and_writes_datasets(tmp_path: Path) -> None:
    publish_snapshot(_moa(), str(tmp_path / 'moa'), MOA.metadata('abc'), None)
    write_dataset(REPORT, str(tmp_path / 'report'))
    write_dataset(_molecule(), str(tmp_path / 'molecule'))
    write_dataset(_target(), str(tmp_path / 'target'))
    source = {
        'moa': str(tmp_path / 'moa'),
        'clinical_report': str(tmp_path / 'report'),
        'chembl_molecule': str(tmp_path / 'molecule'),
        'target': str(tmp_path / 'target'),
    }

    drug_mechanism_of_action_aact(source, {'candidates': str(tmp_path / 'out')}, SETTINGS, None)  # type: ignore[arg-type]

    assert scan_dataset(str(tmp_path / 'out')).collect().height == 1
