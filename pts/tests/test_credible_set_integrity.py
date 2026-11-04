"""Validation: integrity checks before writing, the exclusion log, and the output sort order."""

from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from loguru import logger
from otter.config.model import Config

from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.validation import (
    ValidationSettings,
    exclusion_counts,
    integrity_checks,
    integrity_errors,
    sort_sets,
    sort_variants,
    validation,
)

MHC = CredibleSetFlag.IN_MHC.value
EXCLUDED = CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED.value


def _sets(*rows: tuple) -> pl.LazyFrame:
    """Sets of (id, leadVariantId, studyType, qualityControls)."""
    return pl.LazyFrame(
        [
            {
                name: {'id': i, 'chromosome': '1', 'leadVariantId': lead, 'studyType': t, 'qualityControls': qc}.get(
                    name
                )
                for name in credible_set_schema
            }
            for i, lead, t, qc in rows
        ],
        schema=credible_set_schema,
    )


def _variants(*rows: tuple) -> pl.LazyFrame:
    """Variants of (credibleSetId, variantId, qualityControls)."""
    return pl.LazyFrame(
        [
            {
                name: {'credibleSetId': cs, 'variantId': v, 'qualityControls': qc}.get(name)
                for name in credible_set_variant_schema
            }
            for cs, v, qc in rows
        ],
        schema=credible_set_variant_schema,
    )


VALID_SET = ('a', '1_1_A_T', 'gwas', [])
EXCLUDED_SET = ('b', '1_5_A_T', 'gwas', [MHC])
VALID_VARIANTS = [('a', '1_1_A_T', []), ('a', '1_2_A_T', [])]
EXCLUDED_VARIANTS = [('b', '1_5_A_T', [EXCLUDED])]


def _errors(
    valid_set: pl.LazyFrame | None = None,
    valid_variant: pl.LazyFrame | None = None,
    excluded_set: pl.LazyFrame | None = None,
    excluded_variant: pl.LazyFrame | None = None,
) -> list[str]:
    """The integrity errors of one source, whose inputs are VALID_SET, EXCLUDED_SET and their variants."""
    checks = integrity_checks(
        'qc/gwas/s',
        _sets(VALID_SET, EXCLUDED_SET),
        _variants(*VALID_VARIANTS, *EXCLUDED_VARIANTS),
        valid_set if valid_set is not None else _sets(VALID_SET),
        valid_variant if valid_variant is not None else _variants(*VALID_VARIANTS),
        excluded_set if excluded_set is not None else _sets(EXCLUDED_SET),
        excluded_variant if excluded_variant is not None else _variants(*EXCLUDED_VARIANTS),
    )
    return integrity_errors(checks.collect())


def test_consistent_outputs_have_no_integrity_error() -> None:
    assert _errors() == []


def test_a_lost_set_is_an_error() -> None:
    [error] = _errors(excluded_set=_sets())
    assert 'qc/gwas/s' in error and 'sets' in error


def test_a_lost_variant_is_an_error() -> None:
    [error] = _errors(valid_variant=_variants(VALID_VARIANTS[0]), excluded_variant=_variants(*EXCLUDED_VARIANTS))
    assert 'qc/gwas/s' in error and 'variants' in error


def test_a_valid_variant_without_a_valid_set_is_an_error() -> None:
    errors = _errors(valid_variant=_variants(*VALID_VARIANTS, ('b', '1_5_A_T', [])), excluded_variant=_variants())
    assert any('without a valid set' in error for error in errors)


@pytest.mark.parametrize(
    'variants',
    [[('a', '1_2_A_T', []), ('a', '1_3_A_T', [])], [('a', '1_1_A_T', []), ('a', '1_1_A_T', [])]],
    ids=['no lead row', 'two lead rows'],
)
def test_a_valid_set_without_exactly_one_lead_row_is_an_error(variants: list[tuple]) -> None:
    errors = _errors(valid_variant=_variants(*variants))
    assert any('exactly one lead row' in error for error in errors)


def test_a_valid_set_without_a_study_type_is_an_error() -> None:
    errors = _errors(valid_set=_sets(('a', '1_1_A_T', None, [])))
    assert any('studyType' in error for error in errors)


def test_validation_raises_before_writing_when_the_policy_keeps_a_set_without_its_lead(tmp_path: Path) -> None:
    lead_missing = CredibleSetFlag.LEAD_VARIANT_MISSING.value
    sets = tmp_path / 'qc/gwas/s/credible_set'
    variants = tmp_path / 'qc/gwas/s/credible_set_variant'
    sets.mkdir(parents=True)
    variants.mkdir(parents=True)
    _sets(('a', '1_9_A_T', 'gwas', [lead_missing])).collect().write_parquet(sets / 'part-0.parquet')
    _variants(('a', '1_1_A_T', [])).collect().write_parquet(variants / 'part-0.parquet')
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='exactly one lead row'):
        validation(
            source={'credible_set': [str(sets)], 'credible_set_variant': [str(variants)]},
            destination={
                name: str(out / name) for name in ('valid_set', 'valid_variant', 'excluded_set', 'excluded_variant')
            },
            settings={},
            config=Mock(spec=Config, work_path=str(tmp_path), release_uri=None),
        )
    assert not out.exists()


def test_exclusion_counts_are_per_source_dataset_and_flag() -> None:
    settings = ValidationSettings(cs_invalid_qc_reasons=['IN_MHC'], csv_row_invalid_qc_reasons=['EXCEEDS_COVERAGE_95'])
    exceeds = CredibleSetVariantFlag.EXCEEDS_COVERAGE_95.value
    counts = exclusion_counts(
        'qc/gwas/s',
        _sets(EXCLUDED_SET, ('c', '1_7_A_T', 'gwas', [MHC, CredibleSetFlag.TOP_HIT.value])),
        _variants(('b', '1_5_A_T', [EXCLUDED]), ('a', '1_3_A_T', [exceeds])),
        settings,
    ).collect()
    assert sorted(counts.iter_rows()) == [
        ('qc/gwas/s', 'credible_set', CredibleSetFlag.IN_MHC.name, 2),
        ('qc/gwas/s', 'credible_set_variant', CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED.name, 1),
        ('qc/gwas/s', 'credible_set_variant', CredibleSetVariantFlag.EXCEEDS_COVERAGE_95.name, 1),
    ]


def test_validation_logs_the_exclusion_counts_per_source(tmp_path: Path) -> None:
    sets = tmp_path / 'qc/gwas/s/credible_set'
    variants = tmp_path / 'qc/gwas/s/credible_set_variant'
    sets.mkdir(parents=True)
    variants.mkdir(parents=True)
    _sets(VALID_SET, EXCLUDED_SET).collect().write_parquet(sets / 'part-0.parquet')
    _variants(('a', '1_1_A_T', []), ('b', '1_5_A_T', [])).collect().write_parquet(variants / 'part-0.parquet')
    messages: list[str] = []
    sink = logger.add(messages.append, level='INFO')
    try:
        validation(
            source={'credible_set': [str(sets)], 'credible_set_variant': [str(variants)]},
            destination={
                name: str(tmp_path / 'out' / name)
                for name in ('valid_set', 'valid_variant', 'excluded_set', 'excluded_variant')
            },
            settings={'cs_invalid_qc_reasons': ['IN_MHC']},
            config=Mock(spec=Config, work_path=str(tmp_path), release_uri=None),
        )
    finally:
        logger.remove(sink)
    log = ''.join(messages)
    assert 'qc/gwas/s' in log and 'IN_MHC' in log and 'CREDIBLE_SET_EXCLUDED' in log


def test_sets_are_sorted_by_chromosome_rank_then_locus_then_lead() -> None:
    rows = [
        ('10', 5, 5, 'c'),
        ('2', 5, 5, 'b'),
        ('X', 1, 1, 'x'),
        ('1', 9, 9, 'z'),
        ('1', 1, 9, 'b'),
        ('1', 1, 9, 'a'),
        ('MT', 1, 1, 'm'),
        ('1', 1, 2, 'y'),
    ]
    sets = pl.LazyFrame(
        [{'chromosome': c, 'locusStart': s, 'locusEnd': e, 'leadVariantId': lead} for c, s, e, lead in rows],
        schema={'chromosome': pl.String, 'locusStart': pl.Int32, 'locusEnd': pl.Int32, 'leadVariantId': pl.String},
    )
    ordered = sort_sets(sets).collect()
    assert list(zip(ordered['chromosome'], ordered['leadVariantId'], strict=True)) == [
        ('1', 'y'),
        ('1', 'a'),
        ('1', 'b'),
        ('1', 'z'),
        ('2', 'b'),
        ('10', 'c'),
        ('X', 'x'),
        ('MT', 'm'),
    ]


def test_variants_are_sorted_by_credible_set_then_variant() -> None:
    variants = pl.LazyFrame({'credibleSetId': ['b', 'a', 'a'], 'variantId': ['1_1_A_T', '1_9_A_T', '1_2_A_T']})
    assert sort_variants(variants).collect().rows() == [('a', '1_2_A_T'), ('a', '1_9_A_T'), ('b', '1_1_A_T')]
