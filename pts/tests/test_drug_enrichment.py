"""Tests for the drug_enrichment transformer."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.stats import fisher_exact

from pts.schemas.drug_enrichment import drug_enrichment_schema
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.drug_enrichment import (
    Settings,
    clinical_pairs,
    disease_lineage,
    drug_enrichment,
    excluded_diseases,
    flagged_set_evidence,
    indirect_scores,
    phase_test,
    promiscuous_drugs,
)
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from tests.credible_set_fixtures import published_credible_set

CANCER = 'MONDO_0045024'


def _disease() -> pl.DataFrame:
    return pl.DataFrame({
        'id': ['ROOT', 'A', 'A1', 'B', CANCER, 'C1'],
        'ancestors': [None, ['ROOT'], ['A', 'ROOT'], ['ROOT'], ['ROOT'], [CANCER, 'ROOT']],
    })


def _moa() -> pl.DataFrame:
    return pl.DataFrame({
        'chemblIds': [['DRUG1'], ['DRUG2', 'DRUG2B'], ['DRUG5'], ['DRUG6'], ['DRUG6']],
        'targets': [['T1'], ['T2'], [f'T{i}' for i in range(1, 6)], ['T1', 'T2', 'T3'], ['T3', 'T4', 'T5', 'T6']],
    })


def _clinical() -> pl.DataFrame:
    rows = [
        ('T1', 'A', 'DRUG1', 'PHASE_3'),
        ('T1', 'A', 'DRUG1', 'WITHDRAWAL'),
        ('T2', 'A1', 'DRUG2', 'PHASE_2_3'),
        ('T2', 'B', 'DRUG2B', 'APPROVAL'),
        ('T3', 'B', 'DRUG5', 'PHASE_1'),
        ('T4', 'B', 'DRUG5', 'UNKNOWN'),
        ('T5', 'C1', 'DRUG5', 'APPROVAL'),
        ('T6', 'B', 'DRUG6', 'APPROVAL'),
    ]
    return pl.DataFrame(rows, schema=['targetId', 'diseaseId', 'drugId', 'clinicalStage'], orient='row')


def _evidence() -> pl.DataFrame:
    return pl.DataFrame({
        'datasourceId': ['gwas_credible_sets'] * 4,
        'targetId': ['T1', 'T1', 'T2', 'T5'],
        'diseaseId': ['A1', 'A1', 'B', 'C1'],
        'score': [0.6, 0.9, 0.2, 0.9],
    })


def test_lineage_includes_the_disease_itself() -> None:
    lineage = disease_lineage(_disease().lazy()).collect()
    assert set(lineage.filter(pl.col('diseaseId') == 'A1')['ancestorId']) == {'A1', 'A', 'ROOT'}
    assert set(lineage.filter(pl.col('diseaseId') == 'ROOT')['ancestorId']) == {'ROOT'}


def test_excluded_diseases_are_the_ancestor_and_its_descendants() -> None:
    lineage = disease_lineage(_disease().lazy())
    assert set(excluded_diseases(lineage, (CANCER,)).collect()['diseaseId']) == {CANCER, 'C1'}


def test_promiscuous_drugs_count_distinct_targets_over_all_mechanisms() -> None:
    # DRUG5 has 5 targets, DRUG6 has 6 over two mechanisms
    assert promiscuous_drugs(_moa().lazy(), 5).collect()['drugId'].to_list() == ['DRUG6']


def test_clinical_pairs_keep_the_highest_known_phase() -> None:
    lineage = disease_lineage(_disease().lazy())
    pairs = clinical_pairs(
        _clinical().lazy(),
        excluded_diseases(lineage, (CANCER,)),
        promiscuous_drugs(_moa().lazy(), 5),
    ).collect()
    phases = dict(zip(pairs['targetId'] + ':' + pairs['diseaseId'], pairs['maxPhase'], strict=True))
    # withdrawal, unknown, cancer and the six-target drug are left out
    assert phases == {'T1:A': 3, 'T2:A1': 2, 'T2:B': 4, 'T3:B': 1}


def test_indirect_scores_take_the_highest_score_over_descendants() -> None:
    lineage = disease_lineage(_disease().lazy())
    scores = indirect_scores(_evidence().lazy(), lineage, excluded_diseases(lineage, (CANCER,))).collect()
    t1 = dict(zip(*scores.filter(pl.col('targetId') == 'T1').select('diseaseId', 'score'), strict=True))
    assert t1 == {'A1': 0.9, 'A': 0.9, 'ROOT': 0.9}
    assert scores.filter(pl.col('targetId') == 'T5').is_empty()


def test_phase_test_matches_fisher_and_the_risk_ratio() -> None:
    stats = phase_test(supported=40, supported_reached=12, unsupported=300, unsupported_reached=45)
    table = [[255, 45], [28, 12]]
    odds_ratio, p_value = fisher_exact(table)
    assert stats['oddsRatio'] == pytest.approx(odds_ratio)
    assert stats['oddsRatioPValue'] == pytest.approx(p_value)
    se = np.sqrt(1 / 255 + 1 / 45 + 1 / 28 + 1 / 12)
    assert stats['oddsRatioCiLow'] == pytest.approx(odds_ratio * np.exp(-1.96 * se))
    assert stats['relativeSuccess'] == pytest.approx((12 / 40) / (45 / 300))
    assert stats['relativeSuccessCiLow'] < stats['relativeSuccess'] < stats['relativeSuccessCiHigh']
    assert 0 < stats['relativeSuccessPValue'] < 0.05


def test_phase_test_with_an_empty_cell_is_null_effect() -> None:
    stats = phase_test(supported=10, supported_reached=0, unsupported=100, unsupported_reached=30)
    assert (stats['oddsRatio'], stats['oddsRatioCiLow'], stats['oddsRatioCiHigh']) == (1.0, 1.0, 1.0)
    assert (stats['relativeSuccess'], stats['relativeSuccessPValue']) == (1.0, 1.0)


def test_unknown_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match='unknown settings'):
        Settings.from_settings({'score_thr': 0.5})


def test_drug_enrichment(tmp_path: Path) -> None:
    clinical = _clinical().with_columns(
        pl.col('clinicalStage').cast(pl.Enum(sorted(set(_clinical()['clinicalStage']))))
    )
    inputs = {
        'disease': _disease(),
        'drug_mechanism_of_action': _moa(),
        'clinical_precedence': clinical,
        'evidence_gwas_credible_sets': _evidence(),
    }
    source = {}
    for key, frame in inputs.items():
        source[key] = str(tmp_path / key)
        write_dataset(frame, source[key])
    destination = str(tmp_path / 'drug_enrichment')

    drug_enrichment(source, destination, {}, None)

    output = scan_dataset(destination).collect()
    assert output.schema == drug_enrichment_schema
    assert output['clinicalPhase'].to_list() == ['2+', '3+', '4+']
    # T1:A is supported (0.9 propagated from A1) and reached phase 3; T2:B at 0.2 is not supported
    phase_3 = output.row(1, named=True)
    assert (phase_3['supportedReachedPhase'], phase_3['supportedBelowPhase']) == (1, 0)
    assert (phase_3['unsupportedReachedPhase'], phase_3['unsupportedBelowPhase']) == (1, 2)
    assert output['supportedAssociations'].unique().to_list() == [3]


def _credible_set_evidence() -> pl.DataFrame:
    """T1:A1 from a qualified set and T2:A1 from an unqualified one."""
    return pl.DataFrame({
        'datasourceId': ['gwas_credible_sets'] * 2,
        'targetId': ['T1', 'T2'],
        'diseaseId': ['A1', 'A1'],
        'score': [0.9, 0.9],
        'studyLocusId': ['qualified', 'unqualified'],
    })


def _credible_set() -> pl.DataFrame:
    return published_credible_set(
        {'id': 'qualified', 'qualityControls': [CredibleSetFlag.QUALIFIED_DISEASE_SET.value, 'other']},
        {'id': 'unqualified', 'qualityControls': ['other']},
    )


def test_flagged_set_evidence_keeps_the_evidence_of_flagged_sets() -> None:
    kept = flagged_set_evidence(
        _credible_set_evidence().lazy(), _credible_set().lazy(), (CredibleSetFlag.QUALIFIED_DISEASE_SET,)
    ).collect()
    assert kept['studyLocusId'].to_list() == ['qualified']


def test_unknown_credible_set_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match='unknown credible set flags'):
        Settings.from_settings({'credible_set_flags': ['QUALIFIED']})


def test_credible_set_flags_need_a_credible_set_source(tmp_path: Path) -> None:
    source = {'disease': '', 'drug_mechanism_of_action': '', 'clinical_precedence': '', 'evidence_x': ''}
    with pytest.raises(ValueError, match='needs a credible_set source'):
        drug_enrichment(source, str(tmp_path), {'credible_set_flags': ['QUALIFIED_DISEASE_SET']}, None)


# each evidence row supports its disease A1 and the ancestors A and ROOT
@pytest.mark.parametrize(('flags', 'supported'), [([], 6), (['QUALIFIED_DISEASE_SET'], 3)])
def test_drug_enrichment_counts_only_the_evidence_of_flagged_sets(
    tmp_path: Path, flags: list[str], supported: int
) -> None:
    inputs = {
        'disease': _disease(),
        'drug_mechanism_of_action': _moa(),
        'clinical_precedence': _clinical(),
        'credible_set': _credible_set(),
        'evidence_gwas_credible_sets': _credible_set_evidence(),
    }
    source = {}
    for key, frame in inputs.items():
        source[key] = str(tmp_path / key)
        write_dataset(frame, source[key])
    destination = str(tmp_path / 'drug_enrichment')

    drug_enrichment(source, destination, {'credible_set_flags': flags}, None)

    assert scan_dataset(destination).collect()['supportedAssociations'].unique().to_list() == [supported]
