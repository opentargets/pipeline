"""Annotate the validated credible sets and variants, and publish them."""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.credible_set.enums import FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetConfidenceClasses, CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.replication import replicated_sets
from pts.transformers.credible_set.rescaling import rescaled_effect_size
from pts.transformers.credible_set.study_lookup import study_of_set
from pts.transformers.credible_set.trans_qtl import trans_qtls
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_matching_ids


class AnnotationSource(BaseModel):
    """The validated sets and variants of every source, the study index and the variant index."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str
    study: str
    """The study index, for the populations of each study."""
    variant: str
    """The variant index, for the gnomAD allele frequencies."""
    target: str
    """The target index, for the chromosome and TSS of the gene of a QTL study."""
    disease: str
    """The disease index, for the therapeutic areas that mark a term as a measurement."""


class AnnotationDestination(BaseModel):
    """Paths of the published credible sets and variants."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str


class AnnotationSettings(BaseModel):
    """Settings of the annotation."""

    model_config = ConfigDict(extra='forbid')

    trans_qtl_threshold: int = Field(default=5_000_000, ge=0)
    """The distance from the gene TSS beyond which a QTL variant is trans."""


def assign_confidence(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Set `confidence` from the fine-mapping method and the LD and top hit flags.

    The conditions are tested in order:

    - SuSiE or SuSiE-inf without `OUT_OF_SAMPLE_LD`: 4, `FINEMAPPED_IN_SAMPLE_LD`;
    - SuSiE or SuSiE-inf with `OUT_OF_SAMPLE_LD`: 3, `FINEMAPPED_OUT_OF_SAMPLE_LD`;
    - PICS without `TOP_HIT`: 2, `PICSED_SUMMARY_STATS`;
    - PICS with `TOP_HIT`: 1, `PICSED_TOP_HIT`;
    - any other set: 0, `UNKNOWN`.

    Examples:
        >>> out_of_sample, top_hit = CredibleSetFlag.OUT_OF_SAMPLE_LD.value, CredibleSetFlag.TOP_HIT.value
        >>> credible_set = pl.LazyFrame({
        ...     'fineMappingMethod': ['SuSiE', 'SuSiE-inf', 'PICS', 'PICS', 'SuSie'],
        ...     'qualityControls': [[], [out_of_sample], [], [top_hit], []],
        ... })
        >>> assign_confidence(credible_set).collect()['confidence'].to_list()
        [4, 3, 2, 1, 0]
    """
    quality_control = pl.col('qualityControls')
    susie = pl.col('fineMappingMethod').is_in([method.value for method in FinemappingMethod.susie()])
    pics = pl.col('fineMappingMethod') == FinemappingMethod.PICS.value
    confidence = (
        pl
        .when(susie & ~quality_control.list.contains(CredibleSetFlag.OUT_OF_SAMPLE_LD.value))
        .then(CredibleSetConfidenceClasses.FINEMAPPED_IN_SAMPLE_LD.value)
        .when(susie)
        .then(CredibleSetConfidenceClasses.FINEMAPPED_OUT_OF_SAMPLE_LD.value)
        .when(pics & ~quality_control.list.contains(CredibleSetFlag.TOP_HIT.value))
        .then(CredibleSetConfidenceClasses.PICSED_SUMMARY_STATS.value)
        .when(pics)
        .then(CredibleSetConfidenceClasses.PICSED_TOP_HIT.value)
        .otherwise(CredibleSetConfidenceClasses.UNKNOWN.value)
    )
    return frame.with_columns(confidence.cast(pl.Int8).alias('confidence'))


def gnomad_frequencies(
    credible_set_variant: pl.LazyFrame, credible_set: pl.LazyFrame, study: pl.LazyFrame, variant_index: pl.LazyFrame
) -> pl.LazyFrame:
    """The sample-weighted gnomAD frequency of each (studyId, variantId) of `credible_set_variant`.

    The frequency over the study populations p of `ldPopulationStructure` is
    `sum(w_p * AF_p) / sum(w_p)`, where `w_p` is `relativeSampleSize` and `AF_p` the `{p}_adj`
    entry of the variant `alleleFrequencies`. A single population gives its own frequency,
    whatever its weight. The frequency is null when the study has no population, or when a
    population has no gnomAD frequency or, among several, no weight.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({'credibleSetId': ['cs', 'cs'], 'variantId': ['v', 'w']})
        >>> credible_set = pl.LazyFrame({'id': ['cs'], 'studyId': ['s']})
        >>> study = pl.LazyFrame({
        ...     'studyId': ['s'],
        ...     'ldPopulationStructure': [[
        ...         {'ldPopulation': 'nfe', 'relativeSampleSize': 0.8},
        ...         {'ldPopulation': 'eas', 'relativeSampleSize': 0.2},
        ...     ]],
        ... })
        >>> index = pl.LazyFrame({
        ...     'variantId': ['v', 'w'],
        ...     'alleleFrequencies': [
        ...         [
        ...             {'populationName': 'nfe_adj', 'alleleFrequency': 0.3},
        ...             {'populationName': 'eas_adj', 'alleleFrequency': 0.05},
        ...         ],
        ...         [{'populationName': 'nfe_adj', 'alleleFrequency': 0.3}],
        ...     ],
        ... })
        >>> frequency = gnomad_frequencies(credible_set_variant, credible_set, study, index)
        >>> frequency.with_columns(pl.col('_gnomad').round(4)).sort('variantId').collect().rows()
        [('s', 'v', 0.25), ('s', 'w', None)]
    """
    study_variant = (
        credible_set_variant
        .select('credibleSetId', 'variantId')
        .join(study_of_set(credible_set), on='credibleSetId')
        .select('studyId', 'variantId')
        .unique()
    )
    population = (
        study
        .select('studyId', pl.col('ldPopulationStructure').alias('_population'))
        .explode('_population', empty_as_null=True)
        .select(
            'studyId',
            (pl.col('_population').struct.field('ldPopulation') + '_adj').alias('populationName'),
            pl.col('_population').struct.field('relativeSampleSize').alias('_weight'),
        )
    )
    frequency = (
        variant_index
        .select('variantId', 'alleleFrequencies')
        .join(study_variant.select('variantId').unique(), on='variantId', how='semi')
        .explode('alleleFrequencies')
        .select('variantId', pl.col('alleleFrequencies').struct.unnest())
    )
    af, weight = pl.col('alleleFrequency'), pl.col('_weight')
    single = pl.len() == 1
    missing = af.is_null().any() | (~single & weight.is_null().any())
    weighted = ((weight * af).sum() / weight.sum()).fill_nan(None)
    return (
        study_variant
        .join(population, on='studyId', how='left')
        .join(frequency, on=['variantId', 'populationName'], how='left')
        .group_by('studyId', 'variantId')
        .agg(pl.when(missing).then(None).when(single).then(af.first()).otherwise(weighted).alias('_gnomad'))
    )


def effect_allele_frequency(
    credible_set_variant: pl.LazyFrame, credible_set: pl.LazyFrame, study: pl.LazyFrame, variant_index: pl.LazyFrame
) -> pl.LazyFrame:
    """Set `effectAlleleFrequency` from the source, otherwise from gnomAD, and flag its origin.

    A source frequency is valid when it is in [0, 1] and is not NaN. Every row gets exactly one of:

    - `EAF_FROM_SOURCE`: the valid source frequency is kept;
    - `EAF_FROM_GNOMAD`: the sample-weighted gnomAD frequency of `gnomad_frequencies`;
    - `MISSING_EAF`: neither, and the frequency is null.

    An invalid source frequency also gets `INVALID_EAF`. The rows and their order are unchanged.

    Examples:
        >>> credible_set_variant = pl.LazyFrame(
        ...     {'credibleSetId': ['cs'] * 3, 'variantId': ['v'] * 3, 'effectAlleleFrequency': [0.3, 1.4, None]},
        ...     schema_overrides={'effectAlleleFrequency': pl.Float32},
        ... ).with_columns(pl.lit([], dtype=pl.List(pl.String)).alias('qualityControls'))
        >>> credible_set = pl.LazyFrame({'id': ['cs'], 'studyId': ['s']})
        >>> nfe = {'ldPopulation': 'nfe', 'relativeSampleSize': 1.0}
        >>> study = pl.LazyFrame({'studyId': ['s'], 'ldPopulationStructure': [[nfe]]})
        >>> nfe_adj = {'populationName': 'nfe_adj', 'alleleFrequency': 0.2}
        >>> index = pl.LazyFrame({'variantId': ['v'], 'alleleFrequencies': [[nfe_adj]]})
        >>> result = effect_allele_frequency(credible_set_variant, credible_set, study, index).collect()
        >>> eafs = pl.col('effectAlleleFrequency').cast(pl.Float64).round(2)
        >>> for eaf, quality_control in result.select(eafs, 'qualityControls').rows():
        ...     print(eaf, [CredibleSetVariantFlag(flag).name for flag in quality_control])
        0.3 ['EAF_FROM_SOURCE']
        0.2 ['INVALID_EAF', 'EAF_FROM_GNOMAD']
        0.2 ['EAF_FROM_GNOMAD']
    """
    eaf = pl.col('effectAlleleFrequency')
    valid = (eaf.is_between(0, 1) & eaf.is_not_nan()).fill_null(False)
    gnomad = gnomad_frequencies(credible_set_variant.filter(~valid), credible_set, study, variant_index)
    origin = (
        pl
        .when(valid)
        .then(pl.lit(CredibleSetVariantFlag.EAF_FROM_SOURCE.value))
        .when(pl.col('_gnomad').is_not_null())
        .then(pl.lit(CredibleSetVariantFlag.EAF_FROM_GNOMAD.value))
        .otherwise(pl.lit(CredibleSetVariantFlag.MISSING_EAF.value))
    )
    invalid = pl.when(eaf.is_not_null() & ~valid).then(pl.lit(CredibleSetVariantFlag.INVALID_EAF.value))
    return (
        credible_set_variant
        .join(
            study_of_set(credible_set),
            on='credibleSetId',
            how='left',
            maintain_order='left',
        )
        .join(gnomad, on=['studyId', 'variantId'], how='left', maintain_order='left')
        .with_columns(
            pl.when(valid).then(eaf).otherwise(pl.col('_gnomad')).cast(pl.Float32).alias('effectAlleleFrequency'),
            pl.concat_list('qualityControls', invalid, origin).list.drop_nulls().alias('qualityControls'),
        )
        .select(credible_set_variant.collect_schema().names())
    )


def annotation(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Annotate the validated credible sets and write the published datasets.

    Args:
        source: The validated sets under `credible_set`, the validated variants under
            `credible_set_variant`, and the study, variant, target and disease indexes under `study`,
            `variant`, `target` and `disease`.
        destination: Published paths under `credible_set` and `credible_set_variant`.
        settings: The distance from the gene TSS beyond which a QTL variant is trans, under
            `trans_qtl_threshold`.
        config: Pipeline configuration.
    """
    sources = AnnotationSource.model_validate(source)
    destinations = AnnotationDestination.model_validate(destination)
    options = AnnotationSettings.model_validate(settings)
    validated_set = scan_dataset(sources.credible_set)
    validated_variant = scan_dataset(sources.credible_set_variant)
    study = scan_dataset(sources.study)
    target = scan_dataset(sources.target)
    disease = scan_dataset(sources.disease).select('id', 'therapeuticAreas')
    trans_set = trans_qtls(validated_set, validated_variant, study, target, options.trans_qtl_threshold)
    replicated = replicated_sets(validated_set, study, disease)
    credible_set = (
        validated_set
        .pipe(assign_confidence)
        .pipe(flag_matching_ids, 'id', trans_set, CredibleSetFlag.TRANS_QTL)
        .pipe(flag_matching_ids, 'id', replicated, CredibleSetFlag.REPLICATED)
    )
    credible_set_variant = validated_variant.pipe(
        effect_allele_frequency, validated_set, study, scan_dataset(sources.variant)
    ).pipe(rescaled_effect_size, validated_set, study)
    logger.info(
        'Writing credible sets to {} and variants to {}', destinations.credible_set, destinations.credible_set_variant
    )
    pl.collect_all([
        sink_dataset(credible_set, destinations.credible_set),
        sink_dataset(credible_set_variant, destinations.credible_set_variant),
    ])
