"""Set QC: the flags that depend only on the credible set row, on the worked examples of the PRD."""

from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config

from pts.schemas.credible_set import credible_set_schema
from pts.transformers.credible_set.enums import FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.set_qc import (
    SetQcSettings,
    flag_invalid_chromosome,
    flag_invalid_cs_id,
    flag_invalid_finemapping_method,
    flag_lead_chromosome_mismatch,
    flag_study,
    set_qc,
)
from pts.transformers.study import QualityFlag as StudyQualityFlag

CS_ID = '0' * 32


def _set(**values: str | list[str] | None) -> pl.LazyFrame:
    """One credible set row; unset columns are null, flags empty."""
    row = {
        'id': CS_ID,
        'chromosome': '1',
        'locusStart': 1,
        'locusEnd': 1_000_000,
        'studyId': 's',
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


def test_set_qc_writes_every_set_flag(tmp_path: Path) -> None:
    sets = _set(
        id='X', chromosome='6', leadVariantId='7_30000000_A_T', fineMappingMethod='SuSie', studyId='s', studyType='eqtl'
    )
    studies = _studies(('s', 'gwas', [STUDY_FLAGGED]))
    sets.collect().write_parquet(tmp_path / 'sets.parquet')
    studies.collect().write_parquet(tmp_path / 'study.parquet')
    set_qc(
        source={
            'credible_set': str(tmp_path / 'sets.parquet'),
            'study': str(tmp_path / 'study.parquet'),
        },
        destination=str(tmp_path / 'qc'),
        settings={},
        config=Mock(spec=Config),
    )
    result = pl.read_parquet(tmp_path / 'qc')
    assert set(result['qualityControls'].item()) == {
        CredibleSetFlag.INVALID_CS_ID,
        CredibleSetFlag.IN_MHC,
        CredibleSetFlag.LEAD_CHROMOSOME_MISMATCH,
        CredibleSetFlag.INVALID_FINEMAPPING_METHOD,
        CredibleSetFlag.STUDY_TYPE_MISMATCH,
    }
    assert result['studyType'].item() == 'eqtl'


STUDY_FLAGGED = 'Some study quality flag'


def _studies(*rows: tuple[str, str, list[str]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        [
            {'studyId': study_id, 'studyType': study_type, 'qualityControls': flags}
            for study_id, study_type, flags in rows
        ],
        schema={'studyId': pl.String, 'studyType': pl.String, 'qualityControls': pl.List(pl.String)},
    )


def _with_study(frame: pl.LazyFrame, studies: pl.LazyFrame) -> pl.DataFrame:
    return flag_study(frame, studies).collect()


def test_a_set_of_an_unknown_study_is_missing_and_keeps_its_study_type() -> None:
    result = _with_study(_set(studyId='unknown', studyType='gwas'), _studies(('known', 'gwas', [])))
    assert CredibleSetFlag.MISSING_STUDY in result['qualityControls'].item()
    assert CredibleSetFlag.STUDY_TYPE_MISMATCH not in result['qualityControls'].item()
    assert result['studyType'].item() == 'gwas'


@pytest.mark.parametrize(
    ('source_type', 'index_type', 'mismatch'),
    [
        ('gwas', 'gwas', False),
        ('eqtl', 'eqtl', False),
        ('eqtl', 'gwas', True),
        ('eQTL', 'eqtl', True),
    ],
)
def test_a_study_type_different_from_the_study_index_is_a_mismatch_and_is_kept(
    source_type: str | None, index_type: str, mismatch: bool
) -> None:
    result = _with_study(_set(studyId='s', studyType=source_type), _studies(('s', index_type, [])))
    assert (CredibleSetFlag.STUDY_TYPE_MISMATCH in result['qualityControls'].item()) is mismatch
    assert result['studyType'].item() == source_type


def test_a_missing_study_type_is_taken_from_the_study_index_without_a_flag() -> None:
    result = _with_study(_set(studyId='s', studyType=None), _studies(('s', 'pqtl', [])))
    assert result['studyType'].item() == 'pqtl'
    assert result['qualityControls'].item().to_list() == []


def test_a_missing_study_type_of_an_unknown_study_stays_missing() -> None:
    result = _with_study(_set(studyId='unknown', studyType=None), _studies(('s', 'pqtl', [])))
    assert result['studyType'].item() is None
    assert set(result['qualityControls'].item()) == {CredibleSetFlag.MISSING_STUDY}


@pytest.mark.parametrize(
    ('source_type', 'invalid'), [('gwas', False), ('sctuqtl', False), ('eQTL', True), ('pQTL', True)]
)
def test_invalid_study_type(source_type: str, invalid: bool) -> None:
    result = _with_study(_set(studyId='s', studyType=source_type), _studies(('s', 'gwas', [])))
    assert (CredibleSetFlag.INVALID_STUDY_TYPE in result['qualityControls'].item()) is invalid


def test_the_flags_of_a_study_leave_its_sets_unflagged() -> None:
    """The study's own flags are shown with the study; its sets do not repeat them."""
    result = _with_study(_set(studyId='s', studyType='gwas'), _studies(('s', 'gwas', [STUDY_FLAGGED])))
    assert result['qualityControls'].to_list() == [[]]


@pytest.mark.parametrize(
    ('set_flags', 'study_flags', 'flagged'),
    [
        ([CredibleSetFlag.TOP_HIT.value], [], True),
        ([CredibleSetFlag.TOP_HIT.value], [StudyQualityFlag.SUMSTATS_NOT_AVAILABLE.value], False),
        ([], [], False),
    ],
    ids=['top hit, sumstats available', 'top hit, no sumstats', 'not a top hit'],
)
def test_top_hit_and_summary_stats(set_flags: list[str], study_flags: list[str], flagged: bool) -> None:
    result = _with_study(
        _set(studyId='s', studyType='gwas', qualityControls=set_flags), _studies(('s', 'gwas', study_flags))
    )
    assert (CredibleSetFlag.TOP_HIT_AND_SUMMARY_STATS in result['qualityControls'].item()) is flagged


def test_study_flags_keep_rows_and_columns() -> None:
    sets = pl.concat([
        _set(id='a' * 32, studyId='s', studyType='gwas'),
        _set(id='b' * 32, studyId='t', studyType='gwas'),
    ])
    result = _with_study(sets, _studies(('s', 'gwas', []), ('s2', 'eqtl', [])))
    assert result.height == 2
    assert result.columns == list(credible_set_schema)
