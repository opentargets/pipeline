"""Duplicated credible set ids: within a source every copy leaves; across sources the smallest valid copy is kept."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import polars as pl
from otter.config.model import Config

from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.validation import validation

DUPLICATED = CredibleSetFlag.DUPLICATED_CREDIBLE_SET_ID_ACROSS_SOURCES.value
WITHIN = CredibleSetFlag.DUPLICATED_CREDIBLE_SET_ID.value
EXCLUDED = CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED.value
DUP_ID = 'd' * 32


def _set(set_id: str, study_id: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {
        'id': set_id,
        'chromosome': '1',
        'locusStart': 1,
        'locusEnd': 1_000_000,
        'studyId': study_id,
        'studyType': 'gwas',
        'leadVariantId': '1_100_A_T',
        'fineMappingMethod': 'SuSiE',
        'qualityControls': flags or [],
    }


def _variants(set_id: str, size: int) -> list[dict[str, Any]]:
    return [
        {
            'credibleSetId': set_id,
            'variantId': f'1_{100 + i}_A_T',
            'posteriorProbability': 1 / size,
            'qualityControls': [],
        }
        for i in range(size)
    ]


def _write(path: Path, schema: pl.Schema, rows: list[dict[str, Any]]) -> str:
    path.mkdir(parents=True)
    pl.DataFrame([{name: row.get(name) for name in schema} for row in rows], schema=schema).write_parquet(
        path / 'part-0.parquet'
    )
    return str(path)


def _validate(
    tmp_path: Path, sources: dict[str, tuple[list[dict], list[dict]]], **settings: Any
) -> dict[str, pl.DataFrame]:
    """Validate QC'd sources of (sets, variants), in the given order; return the four written datasets."""
    set_paths, variant_paths = [], []
    for name, (sets, variants) in sources.items():
        set_paths.append(_write(tmp_path / f'qc/gwas/{name}/credible_set', credible_set_schema, sets))
        variant_paths.append(
            _write(tmp_path / f'qc/gwas/{name}/credible_set_variant', credible_set_variant_schema, variants)
        )
    out = tmp_path / 'out'
    names = ('valid_set', 'valid_variant', 'excluded_set', 'excluded_variant')
    validation(
        source={'credible_set': set_paths, 'credible_set_variant': variant_paths},
        destination={name: str(out / name) for name in names},
        settings={'cs_invalid_qc_reasons': ['IN_MHC', 'DUPLICATED_CREDIBLE_SET_ID_ACROSS_SOURCES'], **settings},
        config=Mock(spec=Config, work_path=str(tmp_path), release_uri=None),
    )
    return {name: pl.read_parquet(out / name / '*.parquet') for name in names}


def _studies(frame: pl.DataFrame) -> list[str]:
    return sorted(frame['studyId'])


def test_the_copy_with_fewest_variants_is_kept_and_the_others_are_excluded_with_their_variants(tmp_path: Path) -> None:
    written = _validate(
        tmp_path,
        {'a': ([_set(DUP_ID, 'large')], _variants(DUP_ID, 3)), 'b': ([_set(DUP_ID, 'small')], _variants(DUP_ID, 2))},
    )
    assert _studies(written['valid_set']) == ['small']
    assert written['valid_variant'].height == 2
    [excluded] = written['excluded_set'].to_dicts()
    assert excluded['studyId'] == 'large'
    assert excluded['qualityControls'] == [DUPLICATED]
    assert written['excluded_variant'].height == 3
    assert written['excluded_variant']['qualityControls'].list.contains(EXCLUDED).all()


def test_a_copy_excluded_by_another_flag_does_not_make_the_other_a_duplicate(tmp_path: Path) -> None:
    written = _validate(
        tmp_path,
        {
            'a': ([_set(DUP_ID, 'mhc', [CredibleSetFlag.IN_MHC.value])], _variants(DUP_ID, 1)),
            'b': ([_set(DUP_ID, 'kept')], _variants(DUP_ID, 2)),
        },
    )
    assert _studies(written['valid_set']) == ['kept']
    assert written['valid_set']['qualityControls'].to_list() == [[]]
    [excluded] = written['excluded_set'].to_dicts()
    assert excluded['qualityControls'] == [CredibleSetFlag.IN_MHC.value]


def test_copies_of_equal_size_keep_the_one_of_the_first_source(tmp_path: Path) -> None:
    written = _validate(
        tmp_path,
        {
            'first': ([_set(DUP_ID, 'first')], _variants(DUP_ID, 2)),
            'second': ([_set(DUP_ID, 'second')], _variants(DUP_ID, 2)),
        },
    )
    assert _studies(written['valid_set']) == ['first']
    assert _studies(written['excluded_set']) == ['second']


def test_without_the_flag_in_the_exclusion_list_every_copy_stays_and_the_others_are_flagged(tmp_path: Path) -> None:
    written = _validate(
        tmp_path,
        {'a': ([_set(DUP_ID, 'large')], _variants(DUP_ID, 3)), 'b': ([_set(DUP_ID, 'small')], _variants(DUP_ID, 2))},
        cs_invalid_qc_reasons=['IN_MHC'],
    )
    flags = {row['studyId']: row['qualityControls'] for row in written['valid_set'].to_dicts()}
    assert flags == {'large': [DUPLICATED], 'small': []}
    assert written['excluded_set'].height == 0


def test_copies_within_one_source_all_leave_with_every_variant_of_the_id(tmp_path: Path) -> None:
    """The variants of two sets under one id cannot be told apart, so no copy can be kept."""
    written = _validate(tmp_path, {'a': ([_set(DUP_ID, 'first'), _set(DUP_ID, 'second')], _variants(DUP_ID, 2))})
    assert written['valid_set'].height == 0
    assert written['excluded_set']['qualityControls'].to_list() == [[WITHIN], [WITHIN]]
    assert written['valid_variant'].height == 0
    assert written['excluded_variant']['qualityControls'].to_list() == [[EXCLUDED], [EXCLUDED]]


def test_copies_within_one_source_leave_even_without_an_exclusion_list(tmp_path: Path) -> None:
    """Excluding them is structural: no setting can keep a copy whose variants are not its own."""
    written = _validate(
        tmp_path,
        {'a': ([_set(DUP_ID, 'first'), _set(DUP_ID, 'second')], _variants(DUP_ID, 2))},
        cs_invalid_qc_reasons=[],
    )
    assert written['valid_set'].height == 0
    assert written['valid_variant'].height == 0


def test_copies_within_one_source_with_different_leads_leave_once_with_their_variants(tmp_path: Path) -> None:
    """The copies of an id may differ in what the id should have hashed; their variants are not doubled."""
    other_lead = {**_set(DUP_ID, 'second'), 'leadVariantId': '1_101_A_T'}
    written = _validate(tmp_path, {'a': ([_set(DUP_ID, 'first'), other_lead], _variants(DUP_ID, 2))})
    assert written['valid_set'].height == 0
    assert written['excluded_set']['qualityControls'].to_list() == [[WITHIN], [WITHIN]]
    assert written['excluded_variant'].height == 2
