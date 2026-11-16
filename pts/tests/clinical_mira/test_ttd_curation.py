from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pytest
from loguru import logger

from pts.transformers.clinical_mira import ClinicalReport
from pts.transformers.clinical_mira.providers.ttd import extract_clinical_report, extract_indication, ttd_report_id
from pts.transformers.clinical_mira.ttd_curation import (
    CURATED_UNKNOWN_PHASE,
    apply_ttd_curation,
    known_disease_ids,
    read_ttd_curation,
)

TTD_LINES = [
    'TTDDRUID\tD0A001',
    'DRUGNAME\tAlphamab',
    'INDICATI\tplaceholder row the parser drops\tICD-11: XX00\tApproved',
    'INDICATI\tAge-related Macular Degeneration\tICD-11: 9B75\tApproved',
    'TTDDRUID\tD0B002',
    'DRUGNAME\tBetanib',
    'INDICATI\tChronic obstructive pulmonary disease\tICD-11: CA22\tPhase 2',
    'INDICATI\tAsthma\tICD-11: CA23\tApproved',
]

DOWNGRADE_HEADER = 'ttdDrugId\tdrugFromSource\tdiseaseFromSource\treason'
CORRECTION_HEADER = 'ttdDrugId\tdrugFromSource\tdiseaseFromSource\tdrugId\tdiseaseId\treason'
DRUG_IDS = pl.Series(['CHEMBL1', 'CHEMBL2', 'CHEMBL3'])
DISEASE_IDS = pl.Series(['EFO_0001365', 'MONDO_0004979', 'EFO_0000341'])


def _id(ttd_id: str, indication: str) -> str:
    return pl.select(ttd_report_id(pl.lit(ttd_id), pl.lit(indication))).item()


AMD = _id('D0A001', 'Age-related Macular Degeneration')
ASTHMA = _id('D0B002', 'Asthma')
COPD = _id('D0B002', 'Chronic obstructive pulmonary disease')


def test_ttd_report_id_matches_extracted_report_ids() -> None:
    indications = extract_indication(TTD_LINES)

    reports = extract_clinical_report(indications)
    expected = indications.select(ttd_report_id(pl.col('ttd_id'), pl.col('diseaseFromSource')).alias('id'))

    assert set(reports.df['id']) == set(expected['id'])
    assert len(set(expected['id'])) == 3


def _mapped_reports() -> ClinicalReport:
    """The fixture TTD reports as they leave `map_report_entities`, plus one AACT report with AMD's labels."""
    ttd = extract_clinical_report(extract_indication(TTD_LINES)).df
    drugs = {
        AMD: [{'drugFromSource': 'alphamab', 'drugId': 'CHEMBL1'}, {'drugFromSource': 'alphamab', 'drugId': 'CHEMBL2'}],
        ASTHMA: [{'drugFromSource': 'betanib', 'drugId': 'CHEMBL2'}],
        COPD: [{'drugFromSource': 'betanib', 'drugId': 'CHEMBL2'}],
    }
    diseases = {
        AMD: [{'diseaseFromSource': 'age-related macular degeneration', 'diseaseId': 'EFO_0001365'}],
        ASTHMA: [{'diseaseFromSource': 'asthma', 'diseaseId': 'MONDO_0004979'}],
        COPD: [{'diseaseFromSource': 'chronic obstructive pulmonary disease', 'diseaseId': 'EFO_0000341'}],
    }
    ttd = ttd.with_columns(
        drugs=pl.col('id').replace_strict(
            drugs, return_dtype=pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String}))
        ),
        diseases=pl.col('id').replace_strict(
            diseases, return_dtype=pl.List(pl.Struct({'diseaseFromSource': pl.String, 'diseaseId': pl.String}))
        ),
    ).drop('clinicalStage')
    aact = pl.DataFrame({
        'id': ['NCT00000001'],
        'origin': ['CLINICAL_TRIAL'],
        'url': ['https://clinicaltrials.gov/study/NCT00000001'],
        'phaseFromSource': ['phase3'],
        'type': ['INDICATION'],
        'source': ['ClinicalTrials.gov'],
        'provider': ['AACT'],
        'diseases': [diseases[AMD]],
        'drugs': [drugs[AMD]],
    })
    return ClinicalReport(df=pl.concat([ttd, aact.select(ttd.columns).cast(ttd.schema)]))


def _write(tmp_path: Path, downgrades: list[str], corrections: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    (tmp_path / 'd.tsv').write_text('\n'.join([DOWNGRADE_HEADER, *downgrades]) + '\n')
    (tmp_path / 'c.tsv').write_text('\n'.join([CORRECTION_HEADER, *corrections]) + '\n')
    return read_ttd_curation(str(tmp_path / 'd.tsv'), str(tmp_path / 'c.tsv'))


def _apply(tmp_path: Path, downgrades: list[str], corrections: list[str]) -> pl.DataFrame:
    curation = _write(tmp_path, downgrades, corrections)
    return apply_ttd_curation(_mapped_reports(), *curation, drug_ids=DRUG_IDS, disease_ids=DISEASE_IDS).df


def _row(df: pl.DataFrame, report_id: str) -> dict:
    return df.filter(pl.col('id') == report_id).to_dicts()[0]


@contextmanager
def _logged() -> Generator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m).strip()), level='INFO', format='{message}')
    yield messages
    logger.remove(sink)


def test_downgrade_sets_unknown(tmp_path: Path) -> None:
    assert _row(_mapped_reports().df, AMD)['clinicalStage'] == 'APPROVAL'

    out = _apply(tmp_path, ['D0A001\talphamab\tAge-related Macular Degeneration\ttrial only'], [])

    assert _row(out, AMD)['clinicalStage'] == 'UNKNOWN'
    assert _row(out, AMD)['phaseFromSource'] == CURATED_UNKNOWN_PHASE
    assert _row(out, ASTHMA)['clinicalStage'] == 'APPROVAL'


def test_downgrade_matches_case_insensitively(tmp_path: Path) -> None:
    out = _apply(tmp_path, ['D0B002\tbetanib\tASTHMA\toff-label'], [])

    assert _row(out, ASTHMA)['clinicalStage'] == 'UNKNOWN'


def test_drug_correction_replaces_drug_list(tmp_path: Path) -> None:
    out = _apply(tmp_path, [], ['D0A001\talphamab\tAge-related Macular Degeneration\tCHEMBL3\t\twrong molecule'])

    assert _row(out, AMD)['drugs'] == [{'drugFromSource': 'alphamab', 'drugId': 'CHEMBL3'}]
    assert _row(out, AMD)['diseases'] == [
        {'diseaseFromSource': 'age-related macular degeneration', 'diseaseId': 'EFO_0001365'}
    ]


def test_unmapped_correction_nulls_disease_id(tmp_path: Path) -> None:
    out = _apply(tmp_path, [], ['D0B002\tbetanib\tAsthma\t\tunmapped\tno matching disease'])

    assert _row(out, ASTHMA)['diseases'] == [{'diseaseFromSource': 'asthma', 'diseaseId': None}]
    assert _row(out, ASTHMA)['drugs'] == [{'drugFromSource': 'betanib', 'drugId': 'CHEMBL2'}]


def test_downgrade_and_correction_both_apply(tmp_path: Path) -> None:
    out = _apply(
        tmp_path,
        ['D0A001\talphamab\tAge-related Macular Degeneration\ttrial only'],
        ['D0A001\talphamab\tAge-related Macular Degeneration\tCHEMBL3\t\twrong molecule'],
    )

    assert _row(out, AMD)['clinicalStage'] == 'UNKNOWN'
    assert _row(out, AMD)['drugs'] == [{'drugFromSource': 'alphamab', 'drugId': 'CHEMBL3'}]


def test_other_providers_untouched(tmp_path: Path) -> None:
    before = _row(_mapped_reports().df, 'nct00000001')

    out = _apply(
        tmp_path,
        ['D0A001\talphamab\tAge-related Macular Degeneration\ttrial only'],
        ['D0A001\talphamab\tAge-related Macular Degeneration\tCHEMBL3\tunmapped\twrong'],
    )

    assert _row(out, 'nct00000001') == before
    assert out.height == _mapped_reports().df.height


def test_unmatched_rows_are_counted(tmp_path: Path) -> None:
    with _logged() as messages:
        _apply(tmp_path, ['D9Z999\tnobody\tAsthma\tgone', 'D0B002\tbetanib\tAsthma\toff-label'], [])

    assert any('stage downgrades' in m and 'matched 1' in m and 'unmatched 1' in m for m in messages)


def test_correction_to_unknown_id_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='CHEMBL999'):
        _apply(tmp_path, [], ['D0B002\tbetanib\tAsthma\tCHEMBL999\t\twrong'])


def test_read_rejects_duplicate_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='duplicate'):
        _write(tmp_path, ['D0B002\tbetanib\tAsthma\ta', 'D0B002\tbetanib\tasthma\tb'], [])


def test_read_rejects_empty_correction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='neither'):
        _write(tmp_path, [], ['D0B002\tbetanib\tAsthma\t\t\tnothing'])


def test_read_rejects_missing_column(tmp_path: Path) -> None:
    (tmp_path / 'd.tsv').write_text('ttdDrugId\tdiseaseFromSource\nD0B002\tAsthma\n')
    (tmp_path / 'c.tsv').write_text(CORRECTION_HEADER + '\n')

    with pytest.raises(ValueError, match='reason'):
        read_ttd_curation(str(tmp_path / 'd.tsv'), str(tmp_path / 'c.tsv'))


def test_known_disease_ids_include_obsolete_terms() -> None:
    index = pl.DataFrame({'id': ['MONDO_0005301', 'EFO_0803536'], 'obsoleteTerms': [['EFO_0003885'], None]})

    assert set(known_disease_ids(index)) == {'MONDO_0005301', 'EFO_0803536', 'EFO_0003885'}
