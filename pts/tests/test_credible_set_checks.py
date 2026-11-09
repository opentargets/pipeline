"""The single check of each credible set stage, on the worked examples of the PRDs."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config

from pts.schemas.credible_set import credible_set_schema
from pts.transformers.credible_set.confidence import CredibleSetConfidence, assign_confidence
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.validation import ValidationSettings, validation
from tests.credible_set_fixtures import published_credible_set_variant

CS_ID = '0' * 32


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
def test_invalid_variant_id(qc_variants, variant_id: str, invalid: bool) -> None:
    frame = _variant(variant_id)
    assert _flagged(qc_variants(frame), CredibleSetVariantFlag.INVALID_VARIANT_ID) is invalid


def test_invalid_variant_id_keeps_existing_flags(qc_variants) -> None:
    frame = _variant('1_100_A_A', [CredibleSetVariantFlag.NOT_IN_LOCUS.value])
    flags = qc_variants(frame).collect()['qualityControls'].item().to_list()
    assert {CredibleSetVariantFlag.NOT_IN_LOCUS, CredibleSetVariantFlag.INVALID_VARIANT_ID} <= set(flags)


def _variant(variant_id: str, flags: list[str] | None = None) -> pl.LazyFrame:
    """One credible set variant, the only member of its set."""
    return published_credible_set_variant({
        'credibleSetId': CS_ID,
        'variantId': variant_id,
        'posteriorProbability': 1.0,
        'qualityControls': flags or [],
    }).lazy()


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
def test_in_mhc(qc_sets, chromosome: str, lead: str, in_mhc: bool) -> None:
    assert _flagged(qc_sets(_set(chromosome=chromosome, leadVariantId=lead)), CredibleSetFlag.IN_MHC) is in_mhc


def _set(**values: str | int | list[str]) -> pl.LazyFrame:
    """One credible set row; unset columns are null, flags empty."""
    row = {
        'id': CS_ID,
        'chromosome': '1',
        'locusStart': 1,
        'locusEnd': 1_000_000,
        'studyId': 's',
        'studyType': 'gwas',
        'leadVariantId': '1_100_A_T',
        'fineMappingMethod': 'SuSiE',
        'qualityControls': [],
        **values,
    }
    return pl.LazyFrame([{name: row.get(name) for name in credible_set_schema}], schema=credible_set_schema)


@pytest.mark.parametrize(
    ('method', 'flags', 'confidence'),
    [
        ('SuSiE-inf', [], CredibleSetConfidence.FINEMAPPED_IN_SAMPLE_LD),
        ('SuSiE-inf', [CredibleSetFlag.OUT_OF_SAMPLE_LD], CredibleSetConfidence.FINEMAPPED_OUT_OF_SAMPLE_LD),
        ('SuSiE', [], CredibleSetConfidence.FINEMAPPED_IN_SAMPLE_LD),
        ('PICS', [], CredibleSetConfidence.PICSED_SUMMARY_STATS),
        ('PICS', [CredibleSetFlag.TOP_HIT], CredibleSetConfidence.PICSED_TOP_HIT),
        ('SuSie', [], CredibleSetConfidence.UNKNOWN),
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


@pytest.mark.parametrize(
    'outside',
    [{'chromosome': '23', 'leadVariantId': '23_100_A_T'}, {'studyType': 'customqtl'}, {'fineMappingMethod': 'SuSie'}],
    ids=['chromosome', 'study type', 'fine-mapping method'],
)
def test_validation_refuses_to_write_a_valid_set_outside_the_vocabularies(validate_source, outside: dict) -> None:
    """Set QC excludes such a set first; validation fails rather than write one it was given."""
    lead = outside.get('leadVariantId', '1_100_A_T')
    variant = published_credible_set_variant({'credibleSetId': CS_ID, 'variantId': lead, 'posteriorProbability': 1.0})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        validate_source(_set(**outside), variant.lazy(), ValidationSettings())


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


def _validate_paths(credible_set: list[str], credible_set_variant: list[str], release_uri: str | None = None) -> None:
    """Run validation on paths that are never read: the pairing of the sources fails first."""
    validation(
        source={'credible_set': credible_set, 'credible_set_variant': credible_set_variant},
        destination={
            name: '/out/' + name for name in ('valid_set', 'valid_variant', 'excluded_set', 'excluded_variant')
        },
        settings={},
        config=Mock(spec=Config, work_path='/work', release_uri=release_uri),
    )


@pytest.mark.parametrize(
    'variants',
    [
        ['/work/qc/gwas/a/credible_set_variant'],
        ['/work/qc/gwas/b/credible_set_variant', '/work/qc/gwas/a/credible_set_variant'],
    ],
    ids=['missing source', 'different order'],
)
def test_validation_rejects_set_and_variant_lists_that_do_not_pair_up(variants: list[str]) -> None:
    with pytest.raises(ValueError, match='qc/gwas/a'):
        _validate_paths(['/work/qc/gwas/a/credible_set', '/work/qc/gwas/b/credible_set'], variants)


def test_validation_pairs_gcs_paths_by_their_folder_in_the_release() -> None:
    root = 'gs://bucket/26.12'
    with pytest.raises(ValueError, match='qc/molqtl/eQTLGen/coeqtl'):
        _validate_paths(
            [f'{root}/qc/gwas/gcca/credible_set', f'{root}/qc/molqtl/eQTLGen/coeqtl/credible_set/'],
            [f'{root}/qc/molqtl/eQTLGen/coeqtl/credible_set_variant', f'{root}/qc/gwas/gcca/credible_set_variant'],
            root,
        )
