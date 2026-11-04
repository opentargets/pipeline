"""Validation: row exclusion, the lead and orphan flags, and the explanation of every excluded row."""

import polars as pl
import pytest
from pydantic import ValidationError

from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.set_qc import flag_lead_variant_missing
from pts.transformers.credible_set.validation import ValidationSettings, cross_set_flags_schema, validate
from pts.transformers.credible_set.variant_qc import flag_orphan_variants, flag_variant_chromosome_mismatch

EXCEEDS = CredibleSetVariantFlag.EXCEEDS_COVERAGE_95.value
SETTINGS = ValidationSettings(
    cs_invalid_qc_reasons=['IN_MHC', 'LEAD_VARIANT_MISSING'],
    csv_set_invalid_qc_reasons=['INVALID_VARIANT_ID'],
    csv_row_invalid_qc_reasons=['EXCEEDS_COVERAGE_95', 'ORPHAN_VARIANT'],
)


def _sets(*rows: tuple[str, str, str, list[str]]) -> pl.LazyFrame:
    """Sets of (id, chromosome, leadVariantId, qualityControls)."""
    return pl.LazyFrame(
        [
            {
                name: {'id': i, 'chromosome': c, 'leadVariantId': lead, 'qualityControls': qc}.get(name)
                for name in credible_set_schema
            }
            for i, c, lead, qc in rows
        ],
        schema=credible_set_schema,
    )


def _variants(*rows: tuple[str, str, list[str]]) -> pl.LazyFrame:
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


def _flags(frame: pl.LazyFrame, key: str) -> dict[str, set[str]]:
    return {row[key]: set(row['qualityControls']) for row in frame.collect().iter_rows(named=True)}


def _validate(sets: pl.LazyFrame, variants: pl.LazyFrame) -> tuple[dict, dict, dict, dict]:
    valid_set, valid_variant, excluded_set, excluded_variant = validate(
        sets, variants, pl.LazyFrame(schema=cross_set_flags_schema), SETTINGS
    )
    return (
        _flags(valid_set, 'id'),
        _flags(valid_variant, 'variantId'),
        _flags(excluded_set, 'id'),
        _flags(excluded_variant, 'variantId'),
    )


# Quality control of one source: orphans, chromosome mismatch, missing lead.


def test_a_variant_of_a_set_absent_from_its_source_is_an_orphan() -> None:
    sets = _sets(('a', '1', '1_1_A_T', []))
    variants = flag_orphan_variants(_variants(('a', '1_1_A_T', []), ('missing', '1_2_A_T', [])), sets)
    assert _flags(variants, 'variantId') == {'1_1_A_T': set(), '1_2_A_T': {CredibleSetVariantFlag.ORPHAN_VARIANT}}


def test_a_variant_on_another_chromosome_than_its_set_is_flagged() -> None:
    sets = _sets(('a', '1', '1_1_A_T', []))
    variants = flag_variant_chromosome_mismatch(
        _variants(('a', '1_1_A_T', []), ('a', '2_5_A_T', []), ('orphan', '3_5_A_T', [])), sets
    )
    assert _flags(variants, 'variantId') == {
        '1_1_A_T': set(),
        '2_5_A_T': {CredibleSetVariantFlag.VARIANT_CHROMOSOME_MISMATCH},
        '3_5_A_T': set(),
    }


def test_a_set_without_its_lead_row_among_its_variants_is_flagged() -> None:
    sets = flag_lead_variant_missing(
        _sets(('with_lead', '1', '1_1_A_T', []), ('no_lead', '1', '1_9_A_T', [])),
        _variants(('with_lead', '1_1_A_T', []), ('no_lead', '1_2_A_T', [])),
    )
    assert _flags(sets, 'id') == {'with_lead': set(), 'no_lead': {CredibleSetFlag.LEAD_VARIANT_MISSING}}


# Validation.


def test_a_member_beyond_the_coverage_is_excluded_alone_and_its_set_stays_valid() -> None:
    valid_sets, valid_variants, excluded_sets, excluded_variants = _validate(
        _sets(('a', '1', '1_1_A_T', [])), _variants(('a', '1_1_A_T', []), ('a', '1_2_A_T', [EXCEEDS]))
    )
    assert valid_sets.keys() == {'a'}
    assert valid_variants.keys() == {'1_1_A_T'}
    assert excluded_sets == {}
    assert excluded_variants == {'1_2_A_T': {EXCEEDS}}


def test_a_set_whose_lead_row_is_excluded_loses_its_lead_and_is_excluded() -> None:
    valid_sets, valid_variants, excluded_sets, excluded_variants = _validate(
        _sets(('a', '1', '1_1_A_T', [])), _variants(('a', '1_1_A_T', [EXCEEDS]), ('a', '1_2_A_T', []))
    )
    assert valid_sets == {} and valid_variants == {}
    assert excluded_sets == {'a': {CredibleSetFlag.LEAD_VARIANT_MISSING}}
    assert excluded_variants == {
        '1_1_A_T': {EXCEEDS},
        '1_2_A_T': {CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED},
    }


def test_an_orphan_variant_is_excluded_without_excluding_any_set() -> None:
    valid_sets, valid_variants, _, excluded_variants = _validate(
        _sets(('a', '1', '1_1_A_T', [])),
        _variants(('a', '1_1_A_T', []), ('gone', '1_9_A_T', [CredibleSetVariantFlag.ORPHAN_VARIANT.value])),
    )
    assert valid_sets.keys() == {'a'}
    assert valid_variants.keys() == {'1_1_A_T'}
    assert excluded_variants == {'1_9_A_T': {CredibleSetVariantFlag.ORPHAN_VARIANT}}


def test_a_set_missing_its_lead_at_input_is_excluded_with_its_variants() -> None:
    sets = _sets(('a', '1', '1_9_A_T', [CredibleSetFlag.LEAD_VARIANT_MISSING.value]))
    valid_sets, _, excluded_sets, excluded_variants = _validate(sets, _variants(('a', '1_1_A_T', [])))
    assert valid_sets == {}
    assert excluded_sets == {'a': {CredibleSetFlag.LEAD_VARIANT_MISSING}}
    assert excluded_variants == {'1_1_A_T': {CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED}}


def test_a_variant_on_another_chromosome_stays_valid() -> None:
    mismatch = CredibleSetVariantFlag.VARIANT_CHROMOSOME_MISMATCH.value
    valid_sets, valid_variants, _, _ = _validate(
        _sets(('a', '1', '1_1_A_T', [])), _variants(('a', '1_1_A_T', []), ('a', '2_5_A_T', [mismatch]))
    )
    assert valid_sets.keys() == {'a'}
    assert valid_variants == {'1_1_A_T': set(), '2_5_A_T': {mismatch}}


def test_every_excluded_row_carries_the_flag_that_explains_it() -> None:
    sets = _sets(
        ('mhc', '6', '6_30000000_A_T', [CredibleSetFlag.IN_MHC.value]),
        ('bad_variant', '1', '1_1_A_T', []),
        ('lead_cut', '1', '1_5_A_T', []),
        ('fine', '1', '1_7_A_T', []),
    )
    variants = _variants(
        ('mhc', '6_30000000_A_T', []),
        ('bad_variant', '1_1_A_T', []),
        ('bad_variant', '1_2_A_A', [CredibleSetVariantFlag.INVALID_VARIANT_ID.value]),
        ('lead_cut', '1_5_A_T', [EXCEEDS]),
        ('fine', '1_7_A_T', []),
        ('fine', '1_8_A_T', [EXCEEDS]),
        ('orphan', '1_9_A_T', [CredibleSetVariantFlag.ORPHAN_VARIANT.value]),
    )
    valid_set, _, excluded_set, excluded_variant = validate(
        sets, variants, pl.LazyFrame(schema=cross_set_flags_schema), SETTINGS
    )
    set_reasons = [*SETTINGS.cs_invalid_qc_reasons, CredibleSetFlag.INVALID_VARIANT_IN_SET]
    variant_reasons = [*SETTINGS.csv_row_invalid_qc_reasons, CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED]
    excluded_sets = excluded_set.collect()
    excluded_variants = excluded_variant.collect()
    assert sorted(excluded_sets['id']) == ['bad_variant', 'lead_cut', 'mhc']
    assert valid_set.collect()['id'].to_list() == ['fine']
    set_explained = excluded_sets['qualityControls'].list.set_intersection([f.value for f in set_reasons])
    variant_explained = excluded_variants['qualityControls'].list.set_intersection([f.value for f in variant_reasons])
    assert all(count > 0 for count in set_explained.list.len())
    assert all(count > 0 for count in variant_explained.list.len())


# Configuration.


@pytest.mark.parametrize(
    ('setting', 'flag'),
    [
        ('cs_invalid_qc_reasons', 'INVALID_VARIANT_IN_SET'),
        ('csv_set_invalid_qc_reasons', 'CREDIBLE_SET_EXCLUDED'),
        ('csv_row_invalid_qc_reasons', 'CREDIBLE_SET_EXCLUDED'),
    ],
)
def test_a_structural_flag_in_an_exclusion_list_fails_validation(setting: str, flag: str) -> None:
    with pytest.raises(ValidationError, match='structural'):
        ValidationSettings.model_validate({setting: [flag]})


def test_the_lead_and_orphan_flags_are_ordinary_flags() -> None:
    settings = ValidationSettings.model_validate({
        'cs_invalid_qc_reasons': ['LEAD_VARIANT_MISSING'],
        'csv_row_invalid_qc_reasons': ['ORPHAN_VARIANT'],
    })
    assert settings.cs_invalid_qc_reasons == [CredibleSetFlag.LEAD_VARIANT_MISSING]
    assert settings.csv_row_invalid_qc_reasons == [CredibleSetVariantFlag.ORPHAN_VARIANT]
