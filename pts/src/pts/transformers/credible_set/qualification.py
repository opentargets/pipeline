"""The QUALIFIED rules: credible sets of qualified GWAS whose lead has the power and effect size to analyse.

A set of a qualified disease or measurement GWAS (the study's QUALIFIED_DISEASE_STUDY or
QUALIFIED_MEASUREMENT_STUDY flag) is qualified when its lead variant has:

- an expected minor allele count `2 * MAF * N` of at least `MIN_EXPECTED_MAC`, with the MAF from the
  lead's `effectAlleleFrequency` and N the study `nCases` for a disease, `nSamples` for a measurement;
- a rescaled effect size `rescaledAbsBeta` of at most `MAX_RESCALED_ABS_BETA`.
"""

from __future__ import annotations

import polars as pl

from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.study import QualityFlag as StudyQualityFlag
from pts.transformers.utils.quality_flags import flag_matching_ids

MIN_EXPECTED_MAC = 20
"""The smallest expected minor allele count of the lead variant of a qualified set."""


MAX_RESCALED_ABS_BETA = 3
"""The largest rescaled effect size of the lead variant of a qualified set."""


_QUALIFIED_SETS = (
    (StudyQualityFlag.QUALIFIED_DISEASE_STUDY, 'nCases', CredibleSetFlag.QUALIFIED_DISEASE_SET),
    (StudyQualityFlag.QUALIFIED_MEASUREMENT_STUDY, 'nSamples', CredibleSetFlag.QUALIFIED_MEASUREMENT_SET),
)
"""The study flag, the sample size of the expected minor allele count, and the set flag of each kind of study."""


def flag_qualified_sets(
    credible_set: pl.LazyFrame, credible_set_variant: pl.LazyFrame, study: pl.LazyFrame
) -> pl.LazyFrame:
    """Flag the qualified sets of qualified disease and measurement GWAS, keeping the rows and their order.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['a', 'b', 'c', 'd'],
        ...     'studyId': ['disease', 'disease', 'disease', 'measurement'],
        ...     'leadVariantId': ['v', 'rare', 'large', 'rare'],
        ...     'qualityControls': [[], [], [], []],
        ... })
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'b', 'c', 'd'],
        ...     'variantId': ['v', 'rare', 'large', 'rare'],
        ...     'effectAlleleFrequency': [0.6, 0.001, 0.2, 0.001],
        ...     'rescaledAbsBeta': [0.1, 0.1, 3.5, 0.1],
        ... })
        >>> study = pl.LazyFrame({
        ...     'studyId': ['disease', 'measurement'],
        ...     'nCases': [5_000, None],
        ...     'nSamples': [50_000, 50_000],
        ...     'qualityControls': [['Qualified disease GWAS'], ['Qualified measurement GWAS']],
        ... })
        >>> flag_qualified_sets(credible_set, credible_set_variant, study).collect()['qualityControls'].to_list()
        [['Qualified credible set of a disease GWAS'], [], [], ['Qualified credible set of a measurement GWAS']]
    """
    lead = credible_set_variant.select(
        pl.col('credibleSetId').alias('id'),
        pl.col('variantId').alias('leadVariantId'),
        pl.col('effectAlleleFrequency').cast(pl.Float64).alias('_eaf'),
        pl.col('rescaledAbsBeta').cast(pl.Float64).alias('_beta'),
    )
    leads = credible_set.select('id', 'studyId', 'leadVariantId').join(lead, on=['id', 'leadVariantId'])
    maf = pl.min_horizontal(pl.col('_eaf'), 1 - pl.col('_eaf'))
    for study_flag, sample_size, set_flag in _QUALIFIED_SETS:
        qualified_study = study.filter(pl.col('qualityControls').list.contains(study_flag.value)).select(
            'studyId', pl.col(sample_size).cast(pl.Float64).alias('_n')
        )
        qualified = leads.join(qualified_study, on='studyId').filter(
            (2 * maf * pl.col('_n') >= MIN_EXPECTED_MAC) & (pl.col('_beta') <= MAX_RESCALED_ABS_BETA)
        )
        credible_set = credible_set.pipe(flag_matching_ids, 'id', qualified.select('id'), set_flag)
    return credible_set
