"""Annotation: the effect allele frequency of every validated variant, from the source or from gnomAD."""

import math

import polars as pl
import pytest

from pts.schemas.credible_set import credible_set_variant_schema
from pts.schemas.study import study_schema
from pts.schemas.variant import variant_schema
from pts.transformers.credible_set.annotation import effect_allele_frequency
from pts.transformers.credible_set.flags import CredibleSetVariantFlag

EAF_FLAGS = {
    CredibleSetVariantFlag.EAF_FROM_SOURCE,
    CredibleSetVariantFlag.EAF_FROM_GNOMAD,
    CredibleSetVariantFlag.MISSING_EAF,
}
NFE_EAS = [{'ldPopulation': 'nfe', 'relativeSampleSize': 0.8}, {'ldPopulation': 'eas', 'relativeSampleSize': 0.2}]
"""A study of 80% non-Finnish Europeans and 20% East Asians."""


def _variants(*rows: tuple[str, float | None]) -> pl.LazyFrame:
    """Variants of (variantId, effectAlleleFrequency) in credible set `cs`, each with an earlier flag."""
    return pl.LazyFrame(
        [
            {
                name: {
                    'credibleSetId': 'cs',
                    'variantId': variant_id,
                    'effectAlleleFrequency': eaf,
                    'qualityControls': [CredibleSetVariantFlag.ABNORMAL_PIP.value],
                }.get(name)
                for name in credible_set_variant_schema
            }
            for variant_id, eaf in rows
        ],
        schema=credible_set_variant_schema,
    )


def _studies(populations: list[dict] | None) -> pl.LazyFrame:
    return pl.LazyFrame(
        [{name: {'studyId': 's', 'ldPopulationStructure': populations}.get(name) for name in study_schema}],
        schema=study_schema,
    )


def _index(**frequencies: dict[str, float]) -> pl.LazyFrame:
    """The variant index rows of `variant=population=frequency` keyword maps."""
    return pl.LazyFrame(
        [
            {
                name: {
                    'variantId': variant_id,
                    'alleleFrequencies': [
                        {'populationName': population, 'alleleFrequency': af} for population, af in afs.items()
                    ],
                }.get(name)
                for name in variant_schema
            }
            for variant_id, afs in frequencies.items()
        ],
        schema=variant_schema,
    )


SETS = pl.LazyFrame({'id': ['cs'], 'studyId': ['s']})
GNOMAD = {'v': {'nfe_adj': 0.30, 'eas_adj': 0.05, 'afr_adj': 0.9}}


def _annotate(
    variants: pl.LazyFrame, populations: list[dict] | None = NFE_EAS, index: pl.LazyFrame | None = None
) -> list[tuple[float | None, set[str]]]:
    """(effectAlleleFrequency, flags other than the earlier one) of every row, in input order."""
    index = index if index is not None else _index(**GNOMAD)
    result = effect_allele_frequency(variants, SETS, _studies(populations), index).collect()
    assert result.schema == credible_set_variant_schema
    flags = [set(qc) - {CredibleSetVariantFlag.ABNORMAL_PIP} for qc in result['qualityControls']]
    assert all(CredibleSetVariantFlag.ABNORMAL_PIP in qc for qc in result['qualityControls'])
    return list(zip(result['effectAlleleFrequency'], flags, strict=True))


def test_a_valid_source_frequency_is_kept() -> None:
    [(eaf, flags)] = _annotate(_variants(('v', 0.3)))
    assert eaf == pytest.approx(0.3)
    assert flags == {CredibleSetVariantFlag.EAF_FROM_SOURCE}


@pytest.mark.parametrize('bound', [0.0, 1.0])
def test_the_bounds_of_a_source_frequency_are_valid(bound: float) -> None:
    assert _annotate(_variants(('v', bound))) == [(bound, {CredibleSetVariantFlag.EAF_FROM_SOURCE})]


def test_the_sample_weighted_gnomad_frequency_replaces_a_missing_source_frequency() -> None:
    [(eaf, flags)] = _annotate(_variants(('v', None)))
    assert eaf == pytest.approx(0.8 * 0.30 + 0.2 * 0.05)
    assert eaf == pytest.approx(0.25)
    assert flags == {CredibleSetVariantFlag.EAF_FROM_GNOMAD}


@pytest.mark.parametrize('invalid', [1.4, -0.1, math.nan])
def test_an_invalid_source_frequency_is_flagged_and_replaced_by_gnomad(invalid: float) -> None:
    [(eaf, flags)] = _annotate(_variants(('v', invalid)))
    assert eaf == pytest.approx(0.25)
    assert flags == {CredibleSetVariantFlag.INVALID_EAF, CredibleSetVariantFlag.EAF_FROM_GNOMAD}


def test_a_study_population_without_a_gnomad_frequency_leaves_the_frequency_missing() -> None:
    index = _index(v={'nfe_adj': 0.30})
    assert _annotate(_variants(('v', None)), index=index) == [(None, {CredibleSetVariantFlag.MISSING_EAF})]


@pytest.mark.parametrize('populations', [None, []], ids=['null', 'empty'])
def test_a_study_without_population_structure_leaves_the_frequency_missing(populations: list | None) -> None:
    assert _annotate(_variants(('v', None)), populations) == [(None, {CredibleSetVariantFlag.MISSING_EAF})]


def test_a_variant_absent_from_the_index_leaves_the_frequency_missing() -> None:
    assert _annotate(_variants(('absent', None))) == [(None, {CredibleSetVariantFlag.MISSING_EAF})]


def test_an_invalid_source_frequency_without_gnomad_is_invalid_and_missing() -> None:
    assert _annotate(_variants(('absent', 1.4))) == [
        (None, {CredibleSetVariantFlag.INVALID_EAF, CredibleSetVariantFlag.MISSING_EAF})
    ]


@pytest.mark.parametrize('weight', [0.5, 0.0, None])
def test_a_single_population_uses_its_frequency_whatever_its_weight(weight: float | None) -> None:
    [(eaf, _)] = _annotate(_variants(('v', None)), [{'ldPopulation': 'afr', 'relativeSampleSize': weight}])
    assert eaf == pytest.approx(0.9)


def test_study_populations_map_to_the_adjusted_gnomad_populations() -> None:
    index = _index(v={'nfe': 0.30, 'eas': 0.05})
    assert _annotate(_variants(('v', None)), index=index) == [(None, {CredibleSetVariantFlag.MISSING_EAF})]


def test_every_row_has_exactly_one_origin_flag_and_the_rows_keep_their_order() -> None:
    variants = _variants(('v', None), ('absent', 0.4), ('v', 2.0), ('absent', None), ('v', 0.1))
    rows = _annotate(variants)
    assert len(rows) == 5
    assert all(len(flags & EAF_FLAGS) == 1 for _, flags in rows)
    assert [eaf for eaf, _ in rows] == pytest.approx([0.25, 0.4, 0.25, None, 0.1])
