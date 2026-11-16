"""Tests for the trial and literature candidate mechanisms of drug_mechanism_of_action."""

from pathlib import Path

import polars as pl
import yaml
from loguru import logger

from pts.transformers.drug_mechanism_of_action_candidates import (
    CANDIDATE_COLUMNS,
    build_literature_candidates,
    build_trial_candidates,
)

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
SETTINGS = {
    'link_share': 0.8,
    'class_level_genes': 20,
    'report_min_confidence': 0.70,
    'suggestion_min_confidence': 0.80,
}

LABELS = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))
DRUGS = pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String}))
EVIDENCE = pl.List(pl.Struct({'nctId': pl.String, 'quote': pl.String}))


def _synonyms(symbol: str) -> list[dict]:
    if symbol.startswith('K'):
        return [{'label': 'KINASEFAM', 'source': 's'}]
    if symbol == 'IL3RA':
        return [{'label': 'CD123', 'source': 's'}]
    return []


def _target() -> pl.DataFrame:
    genes = [
        ('ENSG_EGFR', 'EGFR', 'epidermal growth factor receptor', '7'),
        ('ENSG_DRD2', 'DRD2', 'dopamine receptor D2', '11'),
        ('ENSG_IL3RA_X', 'IL3RA', 'interleukin 3 receptor subunit alpha', 'X'),
        ('ENSG_IL3RA_Y', 'IL3RA', 'interleukin 3 receptor subunit alpha', 'Y'),
    ]
    genes += [(f'ENSG_K{i}', f'K{i}', f'kinase {i}', '1') for i in range(21)]
    return pl.DataFrame(
        {
            'id': [g[0] for g in genes],
            'approvedSymbol': [g[1] for g in genes],
            'biotype': ['protein_coding' for _ in genes],
            'genomicLocation': [{'chromosome': g[3], 'start': 1, 'end': 2, 'strand': 1} for g in genes],
            'approvedName': [g[2] for g in genes],
            'symbolSynonyms': [_synonyms(g[1]) for g in genes],
            'obsoleteSymbols': [[] for _ in genes],
            'nameSynonyms': [[] for _ in genes],
        },
        schema={
            'id': pl.String,
            'approvedSymbol': pl.String,
            'biotype': pl.String,
            'genomicLocation': pl.Struct({
                'chromosome': pl.String,
                'start': pl.Int64,
                'end': pl.Int64,
                'strand': pl.Int64,
            }),
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
    return build_trial_candidates(
        moa,
        report,
        _molecule(),
        _target(),
        link_share=merged['link_share'],
        class_level_genes=merged['class_level_genes'],
        report_min_confidence=merged['report_min_confidence'],
        suggestion_min_confidence=merged['suggestion_min_confidence'],
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
    assert row['source'] == 'aact'
    assert row['docIds'] == ['NCT1', 'NCT2']
    assert (row['quote'], row['quoteDocId']) == ('q2', 'NCT2')


def test_output_nct_ids_are_the_cited_trials_not_the_pool() -> None:
    moa = _moa(_mechanism(nctIds=['NCT1', 'NCT2', 'NCT3'], evidence=[{'nctId': 'nct2', 'quote': 'q'}]))

    assert _build(moa)['docIds'].to_list() == [['NCT2']]


def test_salt_link_publishes_under_the_parent() -> None:
    report = _report([('NCT1', 'CHEMBL1SALT', 'PHASE_1'), ('NCT2', 'CHEMBL1SALT', 'PHASE_1')])

    row = _build(_moa(), report).row(0, named=True)

    assert row['parentChemblId'] == 'CHEMBL1'
    assert row['chemblIds'] == ['CHEMBL1', 'CHEMBL1SALT']
    assert row['drugType'] == 'Small molecule'
    assert row['maximumClinicalStage'] == 'PHASE_1'


def test_candidate_stage_is_the_best_of_the_parents_family() -> None:
    report = _report([
        ('NCT1', 'CHEMBL1', 'PHASE_2'),
        ('NCT2', 'CHEMBL1', 'PHASE_2'),
        ('NCT3', 'CHEMBL1SALT', 'APPROVAL'),
    ])

    row = _build(_moa(), report).row(0, named=True)

    assert row['maximumClinicalStage'] == 'APPROVAL'


def test_a_parent_absent_from_the_molecules_still_lists_itself_in_chembl_ids() -> None:
    molecule = pl.concat([
        _molecule(),
        pl.DataFrame(
            {'id': ['CHEMBL5SALT'], 'parentId': ['CHEMBL5'], 'childChemblIds': [[]], 'drugType': ['Small molecule']},
            schema=_molecule().schema,
        ),
    ])
    report = _report([('NCT1', 'CHEMBL5SALT', 'PHASE_2'), ('NCT2', 'CHEMBL5SALT', 'PHASE_2')])

    candidates = build_trial_candidates(_moa(), report, molecule, _target(), **SETTINGS)

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


def test_confident_unsupported_suggestion_grounds_by_fallback() -> None:
    moa = _moa(
        _mechanism(targetFromSource='DA-2', targetSymbolSuggestion=['DRD2'], confidence=0.9, mechanismOfAction='hi'),
        _mechanism(targetFromSource='DA-2', targetSymbolSuggestion=['DRD2'], confidence=0.75, mechanismOfAction='lo'),
    )

    candidates = _build(moa)

    assert candidates.select('mechanismOfAction', 'targets').to_dicts() == [
        {'mechanismOfAction': 'hi', 'targets': ['ENSG_DRD2']}
    ]
    assert _build(moa, suggestion_min_confidence=0.7).height == 2


def test_pseudoautosomal_target_grounds_to_its_x_copy() -> None:
    candidates = _build(_moa(_mechanism(targetFromSource='CD123', targetSymbolSuggestion=['IL3RA'])))

    assert candidates['targets'].to_list() == [['ENSG_IL3RA_X']]


def test_pool_without_a_linking_trial_is_dropped() -> None:
    assert _build(_moa(), _report([('NCT9', 'CHEMBL1', 'PHASE_2')])).is_empty()


def test_the_release_step_carries_the_candidate_settings() -> None:
    task = yaml.safe_load(CONFIG_PATH.read_text())['steps']['drug_mechanism_of_action'][0]

    assert task['transformer'] == 'drug_mechanism_of_action'
    assert {'aact', 'literature', 'clinical_report', 'chembl_molecule', 'target'} <= set(task['source'])
    assert task['destination']['candidates'] == 'intermediate/drug_mechanism_of_action_candidates'
    assert SETTINGS.items() <= task['settings'].items()
    assert 'drug_mechanism_of_action_aact' not in yaml.safe_load(CONFIG_PATH.read_text())['steps']


def _paper(**overrides: object) -> dict:
    row = _mechanism(evidence=[{'nctId': 'pmid22', 'quote': 'p22'}, {'nctId': 'pmc7', 'quote': 'p7'}]) | overrides
    row.pop('nctIds', None)
    row.pop('drugName', None)
    return {'drugId': 'CHEMBL1', 'docIds': ['pmc7', 'pmid22']} | row


def _literature(*rows: dict) -> pl.DataFrame:
    schema = dict(_moa().schema)
    schema.pop('drugName')
    schema.pop('nctIds')
    return pl.DataFrame(
        list(rows) or [_paper()],
        schema={'drugId': pl.String, 'docIds': pl.List(pl.String), **schema},
    )


def _build_literature(moa: pl.DataFrame, report: pl.DataFrame = REPORT, **settings: object) -> pl.DataFrame:
    merged = SETTINGS | settings
    return build_literature_candidates(
        moa,
        report,
        _molecule(),
        _target(),
        class_level_genes=merged['class_level_genes'],
        report_min_confidence=merged['report_min_confidence'],
        suggestion_min_confidence=merged['suggestion_min_confidence'],
    )


def test_a_grounded_literature_mechanism_becomes_a_candidate() -> None:
    candidates = _build_literature(_literature())

    assert candidates.columns == CANDIDATE_COLUMNS
    row = candidates.row(0, named=True)
    assert row['source'] == 'literature'
    assert row['parentChemblId'] == 'CHEMBL1'
    assert row['chemblIds'] == ['CHEMBL1', 'CHEMBL1SALT']
    assert row['maximumClinicalStage'] == 'PHASE_3'
    assert row['drugType'] == 'Small molecule'
    assert row['targets'] == ['ENSG_EGFR']
    assert row['targetType'] == 'protein'
    assert row['docIds'] == ['pmc7', 'pmid22']
    assert (row['quote'], row['quoteDocId']) == ('p22', 'pmid22')


def test_literature_candidates_reparent_salts_and_drop_missing_ids() -> None:
    moa = _literature(
        _paper(drugId='CHEMBL1SALT', mechanismOfAction='salt'),
        _paper(drugId='CHEMBL404', mechanismOfAction='missing'),
        _paper(drugId='CHEMBL2', mechanismOfAction='antibody'),
    )
    report = _report([('NCT1', 'CHEMBL1', 'PHASE_2'), ('NCT3', 'CHEMBL2', 'APPROVAL')])
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(message.record['message']))
    try:
        candidates = _build_literature(moa, report)
    finally:
        logger.remove(handler)

    assert candidates.select('mechanismOfAction', 'parentChemblId', 'chemblIds', 'maximumClinicalStage').rows() == [
        ('salt', 'CHEMBL1', ['CHEMBL1', 'CHEMBL1SALT'], 'PHASE_2'),
        ('antibody', 'CHEMBL2', ['CHEMBL2'], 'APPROVAL'),
    ]
    assert candidates['drugType'].to_list() == ['Small molecule', 'Antibody']
    assert any('1 literature mechanisms' in m and 'chembl_molecule' in m for m in messages), messages


def test_literature_candidates_are_filtered_and_grounded_like_trial_text() -> None:
    moa = _literature(
        _paper(relation='indirect'),
        _paper(statementType='hypothesis'),
        _paper(isHumanTarget=False),
        _paper(confidence=0.69),
        _paper(targetFromSource='unknown thing', targetSymbolSuggestion=[]),
        _paper(mechanismOfAction='kept'),
    )

    assert _build_literature(moa)['mechanismOfAction'].to_list() == ['kept']


def test_literature_doc_ids_are_lower_cased() -> None:
    moa = _literature(_paper(evidence=[{'nctId': 'PMID22', 'quote': 'p22'}, {'nctId': 'PMC7', 'quote': 'p7'}]))

    row = _build_literature(moa).row(0, named=True)

    assert row['docIds'] == ['pmc7', 'pmid22']
    assert row['quoteDocId'] == 'pmid22'
