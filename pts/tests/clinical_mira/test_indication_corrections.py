"""Tests for the curated fixes to the diseases of indication reports."""

from pathlib import Path

import polars as pl
import pytest

from pts.transformers.clinical_mira import ClinicalProvider, ClinicalReport
from pts.transformers.clinical_mira.indication_corrections import (
    apply_indication_corrections,
    read_indication_corrections,
)
from pts.transformers.clinical_mira.providers.chembl_indications import extract_clinical_report

HEADER = 'reportId\tdrugFromSource\tdiseaseFromSource\tdiseaseId\tcorrectedDiseaseId\treason'
LABEL = '188815dd-9d1c-4e0f-934e-972dce0d4599'
OTHER = 'aaaaaaaa-0000-0000-0000-000000000000'
DISEASE_INDEX = pl.DataFrame(
    {
        'id': ['MONDO_0009807', 'EFO_0005842', 'MONDO_0002280', 'MONDO_0001046'],
        'obsoleteTerms': [['EFO_0000637'], [], [], []],
    },
    schema={'id': pl.String, 'obsoleteTerms': pl.List(pl.String)},
)


def _reports() -> ClinicalReport:
    """Two DailyMed labels: levoleucovorin with three indications, and another drug with anemia alone."""
    indications = [
        (1, 1, 'EFO:0000637', 'osteosarcoma'),
        (2, 1, 'EFO:0005842', 'colorectal cancer'),
        (3, 1, 'MONDO:0002280', 'anemia'),
        (4, 2, 'MONDO:0002280', 'anemia'),
    ]
    drug_indication = pl.DataFrame(
        indications, schema=['drugind_id', 'molregno', 'efo_id', 'efo_term'], orient='row'
    ).with_columns(max_phase_for_ind=pl.lit(4.0))
    molecule_dictionary = pl.DataFrame({
        'molregno': [1, 2],
        'pref_name': ['LEVOLEUCOVORIN', 'OTHER DRUG'],
        'chembl_id': ['CHEMBL1908841', 'CHEMBL2'],
    })
    indication_refs = pl.DataFrame({
        'drugind_id': [1, 2, 3, 4],
        'ref_type': ['DailyMed'] * 4,
        'ref_id': [LABEL, LABEL, LABEL, OTHER],
        'ref_url': [f'https://dailymed.nlm.nih.gov/dailymed/search.cfm?query=setid:{i}' for i in [LABEL] * 3 + [OTHER]],
    })
    return extract_clinical_report(drug_indication, molecule_dictionary, indication_refs)


def _apply(tmp_path: Path, rows: list[str]) -> pl.DataFrame:
    (tmp_path / 'c.tsv').write_text('\n'.join([HEADER, *rows]) + '\n')
    corrections = read_indication_corrections(str(tmp_path / 'c.tsv'))
    return apply_indication_corrections(
        _reports(), corrections, disease_index=DISEASE_INDEX, provider=ClinicalProvider.CHEMBL.value
    ).df


def _diseases(df: pl.DataFrame, report_id: str) -> list[str]:
    row = df.filter(pl.col('id') == report_id)
    return sorted(d['diseaseId'] for d in row['diseases'].item()) if row.height else []


def test_removal_drops_only_that_disease(tmp_path: Path) -> None:
    out = _apply(tmp_path, [f'{LABEL.upper()}\tlevoleucovorin\tanemia\tMONDO_0002280\tunmapped\tlimitation of use'])
    assert _diseases(out, LABEL) == ['EFO_0000637', 'EFO_0005842']
    assert _diseases(out, OTHER) == ['MONDO_0002280']


def test_report_left_without_diseases_is_dropped(tmp_path: Path) -> None:
    out = _apply(tmp_path, [f'{OTHER}\tother drug\tanemia\tMONDO_0002280\tunmapped\tlimitation of use'])
    assert OTHER not in out['id'].to_list()
    assert out.height == 1


def test_replacement_keeps_the_source_label(tmp_path: Path) -> None:
    out = _apply(tmp_path, [f'{OTHER}\tother drug\tanemia\tMONDO_0002280\tMONDO_0001046\tnarrower disease'])
    (disease,) = out.filter(pl.col('id') == OTHER)['diseases'].item()
    assert disease == {'diseaseId': 'MONDO_0001046', 'diseaseFromSource': 'anemia'}


def test_unmatched_rows_leave_reports_untouched(tmp_path: Path) -> None:
    out = _apply(tmp_path, ['nope\tx\tanemia\tMONDO_0002280\tunmapped\tr', f'{LABEL}\tx\tasthma\tMONDO_9\tunmapped\tr'])
    assert out.height == 2
    assert _diseases(out, LABEL) == ['EFO_0000637', 'EFO_0005842', 'MONDO_0002280']
    assert _diseases(out, OTHER) == ['MONDO_0002280']


def test_unknown_replacement_id_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='MONDO_9999999'):
        _apply(tmp_path, [f'{OTHER}\tx\tanemia\tMONDO_0002280\tMONDO_9999999\tr'])


@pytest.mark.parametrize(
    ('rows', 'message'),
    [
        ([f'{LABEL}\tx\tanemia\tMONDO_0002280\t\tr'], 'need reportId'),
        ([f'{LABEL}\tx\tanemia\t\tunmapped\tr'], 'remove without a diseaseId'),
        ([f'{LABEL}\tx\tanemia\tMONDO_0002280\tunmapped\tr', f'{LABEL.upper()}\ty\tAnemia\tMONDO_0002280\tunmapped\ts'],
         'duplicate keys'),
    ],
)
def test_invalid_files_are_refused(tmp_path: Path, rows: list[str], message: str) -> None:
    (tmp_path / 'c.tsv').write_text('\n'.join([HEADER, *rows]) + '\n')
    with pytest.raises(ValueError, match=message):
        read_indication_corrections(str(tmp_path / 'c.tsv'))


def test_addition_appends_a_disease_once(tmp_path: Path) -> None:
    out = _apply(
        tmp_path,
        [f'{OTHER}\tx\tAnemia Of Something\t\tMONDO_0001046\tr', f'{LABEL}\tx\tosteosarcoma\t\tEFO_0000637\tr'],
    )
    added = [d for d in out.filter(pl.col('id') == OTHER)['diseases'].item() if d['diseaseId'] == 'MONDO_0001046']
    assert added == [{'diseaseFromSource': 'anemia of something', 'diseaseId': 'MONDO_0001046'}]
    assert _diseases(out, LABEL) == ['EFO_0000637', 'EFO_0005842', 'MONDO_0002280']


def test_other_providers_are_untouched(tmp_path: Path) -> None:
    (tmp_path / 'c.tsv').write_text('\n'.join([HEADER, f'{OTHER}\tx\tanemia\tMONDO_0002280\tunmapped\tr']) + '\n')
    corrections = read_indication_corrections(str(tmp_path / 'c.tsv'))
    out = apply_indication_corrections(
        _reports(), corrections, disease_index=DISEASE_INDEX, provider=ClinicalProvider.EMA.value
    ).df
    assert _diseases(out, OTHER) == ['MONDO_0002280']


def test_obsolete_report_id_matches_its_current_id(tmp_path: Path) -> None:
    out = _apply(tmp_path, [f'{LABEL}\tx\tosteosarcoma\tMONDO_0009807\tunmapped\tr'])
    assert _diseases(out, LABEL) == ['EFO_0005842', 'MONDO_0002280']


def test_corrected_reports_keep_the_input_disease_type(tmp_path: Path) -> None:
    reports = _reports().df
    rows = [f'{OTHER}\tx\tanemia\tMONDO_0002280\tMONDO_0001046\tr', f'{LABEL}\tx\tx\t\tMONDO_0001046\tr']
    out = _apply(tmp_path, rows)
    assert out.schema['diseases'] == reports.schema['diseases']
    assert pl.concat([reports, out]).height == reports.height + out.height
