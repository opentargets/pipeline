"""Cross-set QC in validation: the flags that compare credible sets across sources."""

import polars as pl
import pytest

from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.validation import (
    ValidationSettings,
    cross_set_flags_schema,
    flag_across_sets,
    validate,
)

TOP_HIT = CredibleSetFlag.TOP_HIT.value
EXCEEDS = CredibleSetVariantFlag.EXCEEDS_COVERAGE_95.value


def _sets(*rows: dict) -> pl.LazyFrame:
    defaults = {'chromosome': '1', 'studyId': 's', 'studyType': 'gwas', 'qualityControls': []}
    return pl.LazyFrame(
        [{name: {**defaults, **row}.get(name) for name in credible_set_schema} for row in rows],
        schema=credible_set_schema,
    )


def _variants(*rows: tuple[str, str, list[str]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        [
            {
                name: {'credibleSetId': cs, 'variantId': v, 'posteriorProbability': 1.0, 'qualityControls': qc}.get(
                    name
                )
                for name in credible_set_variant_schema
            }
            for cs, v, qc in rows
        ],
        schema=credible_set_variant_schema,
    )


def _flags(sets: pl.LazyFrame, variants: pl.LazyFrame | None = None) -> dict[str, set[str]]:
    variants = variants if variants is not None else _variants()
    table = flag_across_sets(sets, variants).collect()
    assert table.schema == cross_set_flags_schema
    return {row['id']: set(row['qualityControls']) for row in table.iter_rows(named=True)}


def _pics(set_id: str, **row) -> dict:
    return {'id': set_id, 'fineMappingMethod': 'PICS', 'leadVariantId': '1_100_A_T', **row}


def _susie(set_id: str, start: int, end: int, method: str = 'SuSiE', **row) -> dict:
    return {
        'id': set_id,
        'fineMappingMethod': method,
        'locusStart': start,
        'locusEnd': end,
        'leadVariantId': f'1_{start}_A_T',
        **row,
    }


def test_a_pics_top_hit_of_a_study_with_a_summary_statistics_pics_set_is_redundant() -> None:
    flags = _flags(_sets(_pics('top', qualityControls=[TOP_HIT]), _pics('sumstats')))
    assert flags == {'top': {CredibleSetFlag.REDUNDANT_PICS_TOP_HIT}}


@pytest.mark.parametrize(
    'other',
    [_pics('other_top', qualityControls=[TOP_HIT]), _pics('other_study', studyId='t')],
    ids=['only top hits in the study', 'summary statistics set of another study'],
)
def test_a_top_hit_is_not_redundant_without_a_summary_statistics_pics_set_of_its_study(other: dict) -> None:
    assert _flags(_sets(_pics('top', qualityControls=[TOP_HIT]), other)) == {}


def test_every_copy_of_a_duplicated_id_is_flagged() -> None:
    sets = _sets(_susie('dup', 1, 10), _susie('dup', 1, 10, studyId='t'), _susie('unique', 1, 10))
    assert _flags(sets) == {'dup': {CredibleSetFlag.DUPLICATED_STUDYLOCUS_ID}}
    variants = _variants(('dup', '1_1_A_T', []), ('dup', '1_1_A_T', []), ('unique', '1_1_A_T', []))
    settings = ValidationSettings(cs_invalid_qc_reasons=['DUPLICATED_STUDYLOCUS_ID'])
    valid, _, excluded, _ = validate(sets, variants, flag_across_sets(sets, variants), settings)
    assert sorted(excluded.collect()['id']) == ['dup', 'dup']
    assert valid.collect()['id'].to_list() == ['unique']


def _explained(sets: pl.LazyFrame, members: list[tuple[str, list[str]]]) -> bool:
    variants = _variants(*(('pics', variant_id, qc) for variant_id, qc in members))
    return CredibleSetFlag.EXPLAINED_BY_SUSIE in _flags(sets, variants).get('pics', set())


@pytest.mark.parametrize('method', ['SuSiE', 'SuSiE-inf'])
def test_a_pics_set_with_a_member_in_a_susie_region_of_its_study_is_explained(method: str) -> None:
    sets = _sets(_pics('pics'), _susie('susie', 1000, 2000, method))
    assert _explained(sets, [('1_500_A_T', []), ('1_1500_A_T', [])])


@pytest.mark.parametrize('position', [1000, 2000])
def test_the_susie_region_bounds_are_inclusive(position: int) -> None:
    sets = _sets(_pics('pics'), _susie('susie', 1000, 2000))
    assert _explained(sets, [(f'1_{position}_A_T', [])])


def test_a_member_beyond_the_coverage_does_not_explain_the_set() -> None:
    sets = _sets(_pics('pics'), _susie('susie', 1000, 2000))
    assert not _explained(sets, [('1_500_A_T', []), ('1_1500_A_T', [EXCEEDS])])


@pytest.mark.parametrize(
    'susie',
    [_susie('susie', 1000, 2000, studyId='other'), _susie('susie', 1000, 2000, chromosome='2')],
    ids=['another study', 'another chromosome'],
)
def test_a_susie_set_of_another_study_or_chromosome_does_not_explain(susie: dict) -> None:
    assert not _explained(_sets(_pics('pics'), susie), [('1_1500_A_T', [])])


def test_a_member_on_an_alternate_contig_has_no_position_and_does_not_explain() -> None:
    sets = _sets(_pics('pics'), _susie('susie', 1, 10_000_000))
    assert not _explained(sets, [('1_KI270706v1_random_1500_A_T', [])])


def test_susie_sets_are_never_explained() -> None:
    sets = _sets(_susie('inner', 1400, 1600), _susie('outer', 1000, 2000))
    variants = _variants(('inner', '1_1500_A_T', []), ('outer', '1_1500_A_T', []))
    assert _flags(sets, variants) == {}


def test_a_set_gets_every_cross_set_flag_that_applies() -> None:
    sets = _sets(
        _pics('pics', qualityControls=[TOP_HIT]),
        _pics('pics', qualityControls=[TOP_HIT]),
        _pics('sumstats'),
        _susie('susie', 1000, 2000),
    )
    variants = _variants(('pics', '1_1500_A_T', []), ('sumstats', '1_100_A_T', []), ('susie', '1_1500_A_T', []))
    assert _flags(sets, variants)['pics'] == {
        CredibleSetFlag.REDUNDANT_PICS_TOP_HIT,
        CredibleSetFlag.DUPLICATED_STUDYLOCUS_ID,
        CredibleSetFlag.EXPLAINED_BY_SUSIE,
    }
