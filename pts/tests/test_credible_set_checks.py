"""The single check of each credible set stage, on the worked examples of the PRDs."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers.credible_set.confidence import assign_confidence
from pts.transformers.credible_set.flags import CredibleSetConfidenceClasses, CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.set_qc import flag_in_mhc
from pts.transformers.credible_set.validation import (
    ValidationSettings,
    ValidationSource,
    pair_sources,
    type_valid_sets,
)
from pts.transformers.credible_set.variant_qc import flag_invalid_variant_id


def _flagged(frame: pl.LazyFrame, flag: str) -> bool:
    return frame.select(pl.col('qualityControls').list.contains(flag)).collect().item()


@pytest.mark.parametrize(
    ('variant_id', 'invalid'),
    [
        ('1_100_A_A', True),
        ('chr1_100_A_T', True),
        ('1_0_A_T', True),
        ('1_100_a_T', True),
        ('23_100_A_T', True),
        ('14_KI270846v1_alt_497149_C_T', True),
        ('XY_5_ACGT_A', True),
        ('1_100_A_T', False),
        ('22_100_A_T', False),
    ],
)
def test_invalid_variant_id(variant_id: str, invalid: bool) -> None:
    frame = pl.LazyFrame({'variantId': [variant_id], 'qualityControls': [[]]})
    assert _flagged(flag_invalid_variant_id(frame), CredibleSetVariantFlag.INVALID_VARIANT_ID) is invalid


def test_invalid_variant_id_keeps_existing_flags() -> None:
    frame = pl.LazyFrame({'variantId': ['1_100_A_A'], 'qualityControls': [[CredibleSetVariantFlag.NOT_IN_LOCUS.value]]})
    flags = flag_invalid_variant_id(frame).collect()['qualityControls'].item().to_list()
    assert set(flags) == {CredibleSetVariantFlag.NOT_IN_LOCUS, CredibleSetVariantFlag.INVALID_VARIANT_ID}


@pytest.mark.parametrize(
    ('chromosome', 'lead', 'in_mhc'),
    [
        ('6', '6_30000000_A_T', True),
        ('6', '6_25726063_A_T', True),
        ('6', '6_33400556_A_T', True),
        ('6', '6_50000000_A_T', False),
        ('6', '6_25726062_A_T', False),
        ('7', '7_30000000_A_T', False),
    ],
)
def test_in_mhc(chromosome: str, lead: str, in_mhc: bool) -> None:
    frame = pl.LazyFrame({'chromosome': [chromosome], 'leadVariantId': [lead], 'qualityControls': [[]]})
    assert _flagged(flag_in_mhc(frame), CredibleSetFlag.IN_MHC) is in_mhc


@pytest.mark.parametrize(
    ('method', 'flags', 'confidence'),
    [
        ('SuSiE-inf', [], CredibleSetConfidenceClasses.FINEMAPPED_IN_SAMPLE_LD),
        ('SuSiE-inf', [CredibleSetFlag.OUT_OF_SAMPLE_LD], CredibleSetConfidenceClasses.FINEMAPPED_OUT_OF_SAMPLE_LD),
        ('SuSiE', [], CredibleSetConfidenceClasses.FINEMAPPED_IN_SAMPLE_LD),
        ('PICS', [], CredibleSetConfidenceClasses.PICSED_SUMMARY_STATS),
        ('PICS', [CredibleSetFlag.TOP_HIT], CredibleSetConfidenceClasses.PICSED_TOP_HIT),
        ('SuSie', [], CredibleSetConfidenceClasses.UNKNOWN),
    ],
)
def test_confidence(method: str, flags: list[str], confidence: int) -> None:
    frame = pl.LazyFrame(
        {'fineMappingMethod': [method], 'qualityControls': [[str(flag) for flag in flags]]},
        schema={'fineMappingMethod': pl.String, 'qualityControls': pl.List(pl.String)},
    )
    result = assign_confidence(frame).collect()
    assert result['confidence'].dtype == pl.Int8
    assert result['confidence'].item() == confidence


def test_typing_valid_sets_rejects_a_chromosome_outside_the_enum() -> None:
    sets = pl.LazyFrame({'chromosome': ['23'], 'studyType': ['gwas'], 'fineMappingMethod': ['PICS']})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        type_valid_sets(sets).collect()


def test_typing_valid_sets_rejects_a_study_type_outside_the_enum() -> None:
    sets = pl.LazyFrame({'chromosome': ['1'], 'studyType': ['customqtl'], 'fineMappingMethod': ['PICS']})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        type_valid_sets(sets).collect()


def test_typing_valid_sets_rejects_a_finemapping_method_outside_the_enum() -> None:
    sets = pl.LazyFrame({'chromosome': ['1'], 'studyType': ['gwas'], 'fineMappingMethod': ['SuSie']})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        type_valid_sets(sets).collect()


def test_a_cross_set_flag_reaches_its_set_and_excludes_it_when_listed(validate_source) -> None:
    """A curated top hit of a study that also has a PICS set from summary statistics is redundant."""
    sets = pl.LazyFrame(
        {
            'id': ['top_hit', 'clean'],
            'studyId': ['s', 's'],
            'chromosome': ['1', '1'],
            'locusStart': [1, 2],
            'locusEnd': [1, 2],
            'leadVariantId': ['1_1_A_T', '1_2_A_T'],
            'fineMappingMethod': ['PICS', 'PICS'],
            'qualityControls': [[CredibleSetFlag.TOP_HIT.value], []],
        },
        schema={
            'id': pl.String,
            'studyId': pl.String,
            'chromosome': pl.String,
            'locusStart': pl.Int32,
            'locusEnd': pl.Int32,
            'leadVariantId': pl.String,
            'fineMappingMethod': pl.String,
            'qualityControls': pl.List(pl.String),
        },
    )
    variants = pl.LazyFrame(
        {'credibleSetId': ['top_hit', 'clean'], 'variantId': ['1_1_A_T', '1_2_A_T'], 'qualityControls': [[], []]},
        schema={'credibleSetId': pl.String, 'variantId': pl.String, 'qualityControls': pl.List(pl.String)},
    )
    settings = ValidationSettings(cs_invalid_qc_reasons=['REDUNDANT_PICS_TOP_HIT'])
    valid_set, _, excluded_set, excluded_variant = validate_source(sets, variants, settings)

    assert valid_set.collect()['id'].to_list() == ['clean']
    excluded = excluded_set.collect()
    assert excluded['id'].to_list() == ['top_hit']
    assert set(excluded['qualityControls'].item()) == {
        CredibleSetFlag.TOP_HIT,
        CredibleSetFlag.REDUNDANT_PICS_TOP_HIT,
    }
    assert excluded_variant.collect()['credibleSetId'].to_list() == ['top_hit']


def _config(release_uri: str | None = None) -> Config:
    return Mock(spec=Config, work_path='/work', release_uri=release_uri)


def test_validation_pairs_the_sets_and_variants_of_each_source() -> None:
    sources = ValidationSource(
        credible_set=['/work/qc/gwas/a/credible_set', '/work/qc/gwas/b/credible_set'],
        credible_set_variant=['/work/qc/gwas/a/credible_set_variant', '/work/qc/gwas/b/credible_set_variant'],
    )
    assert pair_sources(sources, _config()) == [
        ('qc/gwas/a', '/work/qc/gwas/a/credible_set', '/work/qc/gwas/a/credible_set_variant'),
        ('qc/gwas/b', '/work/qc/gwas/b/credible_set', '/work/qc/gwas/b/credible_set_variant'),
    ]


def test_validation_pairs_gcs_paths_by_their_folder_in_the_release() -> None:
    root = 'gs://bucket/26.12'
    sources = ValidationSource(
        credible_set=[f'{root}/qc/gwas/gcca/credible_set', f'{root}/qc/molqtl/eQTLGen/coeqtl/credible_set/'],
        credible_set_variant=[
            f'{root}/qc/gwas/gcca/credible_set_variant',
            f'{root}/qc/molqtl/eQTLGen/coeqtl/credible_set_variant',
        ],
    )
    assert [source for source, _, _ in pair_sources(sources, _config(root))] == [
        'qc/gwas/gcca',
        'qc/molqtl/eQTLGen/coeqtl',
    ]


@pytest.mark.parametrize(
    'variants',
    [
        ['/work/qc/gwas/a/credible_set_variant'],
        ['/work/qc/gwas/b/credible_set_variant', '/work/qc/gwas/a/credible_set_variant'],
    ],
    ids=['missing source', 'different order'],
)
def test_validation_rejects_set_and_variant_lists_that_do_not_pair_up(variants: list[str]) -> None:
    sources = ValidationSource(
        credible_set=['/work/qc/gwas/a/credible_set', '/work/qc/gwas/b/credible_set'], credible_set_variant=variants
    )
    with pytest.raises(ValueError, match='qc/gwas/a'):
        pair_sources(sources, _config())
