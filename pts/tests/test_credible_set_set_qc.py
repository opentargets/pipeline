"""Set QC: the flags that depend only on the credible set row, on the worked examples of the PRD."""

import polars as pl
import pytest

from pts.schemas.credible_set import credible_set_schema
from pts.transformers.credible_set.enums import FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.set_qc import (
    SetQcSettings,
    flag_invalid_chromosome,
    flag_invalid_cs_id,
    flag_invalid_finemapping_method,
    flag_lead_chromosome_mismatch,
    flag_sets,
)

CS_ID = '0' * 32


def _set(**values: str) -> pl.LazyFrame:
    """One credible set row; unset columns are null, flags empty."""
    row = {
        'id': CS_ID,
        'chromosome': '1',
        'leadVariantId': '1_100_A_T',
        'fineMappingMethod': 'SuSiE',
        'qualityControls': [],
        **values,
    }
    return pl.LazyFrame([{name: row.get(name) for name in credible_set_schema}]).cast(credible_set_schema)


def _carries(frame: pl.LazyFrame, flag: str) -> bool:
    return frame.select(pl.col('qualityControls').list.contains(flag)).collect().item()


@pytest.mark.parametrize(
    ('credible_set_id', 'invalid'),
    [(CS_ID, False), ('0123456789abcdef' * 2, False), ('0123456789ABCDEF' * 2, True), ('0' * 31, True), ('', True)],
)
def test_invalid_cs_id(credible_set_id: str, invalid: bool) -> None:
    assert _carries(flag_invalid_cs_id(_set(id=credible_set_id)), CredibleSetFlag.INVALID_CS_ID) is invalid


@pytest.mark.parametrize(
    ('chromosome', 'invalid'),
    [
        ('1', False),
        ('22', False),
        ('X', False),
        ('XY', False),
        ('MT', False),
        ('23', True),
        ('24', True),
        ('chr1', True),
    ],
)
def test_invalid_chromosome(chromosome: str, invalid: bool) -> None:
    lead = f'{chromosome}_100_A_T'
    frame = flag_invalid_chromosome(_set(chromosome=chromosome, leadVariantId=lead))
    assert _carries(frame, CredibleSetFlag.INVALID_CHROMOSOME) is invalid


@pytest.mark.parametrize(
    ('chromosome', 'lead', 'mismatch'),
    [('1', '2_100_A_T', True), ('1', '1_100_A_T', False), ('23', '23_100_A_T', False), ('X', '23_100_A_T', True)],
)
def test_lead_chromosome_mismatch(chromosome: str, lead: str, mismatch: bool) -> None:
    frame = flag_lead_chromosome_mismatch(_set(chromosome=chromosome, leadVariantId=lead))
    assert _carries(frame, CredibleSetFlag.LEAD_CHROMOSOME_MISMATCH) is mismatch


@pytest.mark.parametrize(
    ('method', 'invalid'),
    [('PICS', False), ('SuSiE', False), ('SuSiE-inf', False), ('SuSie', True), ('susie', True), ('FINEMAP', True)],
)
def test_invalid_finemapping_method_with_the_default_methods(method: str, invalid: bool) -> None:
    frame = flag_invalid_finemapping_method(_set(fineMappingMethod=method), SetQcSettings().allowed_fm_methods)
    assert _carries(frame, CredibleSetFlag.INVALID_FINEMAPPING_METHOD) is invalid


def test_a_method_left_out_of_the_allowed_methods_is_invalid() -> None:
    frame = flag_invalid_finemapping_method(_set(fineMappingMethod='PICS'), [FinemappingMethod.SUSIE])
    assert _carries(frame, CredibleSetFlag.INVALID_FINEMAPPING_METHOD)


def test_the_allowed_methods_default_to_every_method_and_reject_unknown_ones() -> None:
    assert SetQcSettings().allowed_fm_methods == [
        FinemappingMethod.PICS,
        FinemappingMethod.SUSIE,
        FinemappingMethod.SUSIE_INF,
    ]
    with pytest.raises(ValueError, match='allowed_fm_methods'):
        SetQcSettings.model_validate({'allowed_fm_methods': ['SuSie']})


def test_set_qc_adds_every_row_flag() -> None:
    frame = flag_sets(
        _set(id='X', chromosome='6', leadVariantId='7_30000000_A_T', fineMappingMethod='SuSie'),
        SetQcSettings().allowed_fm_methods,
    ).collect()
    assert set(frame['qualityControls'].item()) == {
        CredibleSetFlag.INVALID_CS_ID,
        CredibleSetFlag.IN_MHC,
        CredibleSetFlag.LEAD_CHROMOSOME_MISMATCH,
        CredibleSetFlag.INVALID_FINEMAPPING_METHOD,
    }
