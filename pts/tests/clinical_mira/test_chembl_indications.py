import polars as pl
import pytest

from pts.transformers.clinical_mira import extract_chembl_clinical_report as extract_clinical_report


def _extract(ref_type: str, max_phase_for_ind: float, ref_id: str = 'C09XA01') -> pl.DataFrame:
    drug_indication = pl.DataFrame(
        {
            'drugind_id': [1],
            'molregno': [10],
            'max_phase_for_ind': [max_phase_for_ind],
            'efo_id': ['EFO:0000319'],
            'efo_term': ['cardiovascular disease'],
        },
        schema_overrides={'max_phase_for_ind': pl.Decimal(38, 10)},
    )
    molecule_dictionary = pl.DataFrame({'molregno': [10], 'chembl_id': ['CHEMBL31601'], 'pref_name': ['REMIKIREN']})
    indication_refs = pl.DataFrame({
        'drugind_id': [1],
        'ref_type': [ref_type],
        'ref_id': [ref_id],
        'ref_url': ['https://www.whocc.no/atc_ddd_index/?code=C09XA01'],
    })
    return extract_clinical_report(drug_indication, molecule_dictionary, indication_refs).df


@pytest.mark.parametrize(
    ('max_phase_for_ind', 'stage'),
    [(4.0, 'APPROVAL'), (3.0, 'PHASE_3'), (2.0, 'PHASE_2'), (1.0, 'PHASE_1'), (-1.0, 'UNKNOWN')],
)
def test_atc_reference_takes_the_chembl_phase(max_phase_for_ind: float, stage: str) -> None:
    """An ATC code is a classification, not an approval, so the stage is ChEMBL's phase for the indication."""
    assert _extract('ATC', max_phase_for_ind)['clinicalStage'].to_list() == [stage]


@pytest.mark.parametrize('ref_type', ['EMA', 'FDA', 'DailyMed'])
def test_regulator_reference_is_an_approval(ref_type: str) -> None:
    assert _extract(ref_type, 2.0, ref_id='ref1')['clinicalStage'].to_list() == ['APPROVAL']
