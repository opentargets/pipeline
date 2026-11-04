"""The effect allele frequency of credible set variants, from the source or from gnomAD."""

from __future__ import annotations

import polars as pl

from pts.transformers.credible_set.flags import CredibleSetVariantFlag
from pts.transformers.credible_set.study_lookup import study_of_set


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
