"""The rescaled effect size of credible set variants, from their p-value, frequency and study design."""

from __future__ import annotations

import polars as pl
from scipy.stats import chi2

from pts.transformers.credible_set.flags import CredibleSetVariantFlag
from pts.transformers.credible_set.study_lookup import study_of_set
from pts.transformers.study import QualityFlag as StudyQualityFlag
from pts.transformers.utils.quality_flags import flag_rows

CHI2_EXACT_MIN_EXPONENT = -300
"""The smallest p-value exponent with an exact chi-squared statistic; below it the p-value underflows."""
CHI2_APPROXIMATION_SLOPE = 4.596
"""The slope of the linear approximation of the chi-squared statistic from -log10(p), a constant."""
CHI2_APPROXIMATION_INTERCEPT = -5.367
"""The intercept of the linear approximation of the chi-squared statistic from -log10(p), a constant."""


def _finite(value: pl.Expr) -> pl.Expr:
    """`value` as a float, null when it is missing, NaN or infinite."""
    value = value.cast(pl.Float64)
    return pl.when(value.is_finite()).then(value)


def _signed(beta: pl.Expr) -> pl.Expr:
    """`beta` as a float, null when it has no direction: missing, NaN, infinite or zero."""
    beta = _finite(beta)
    return pl.when(beta != 0).then(beta)


def _chi2_isf(p_value: pl.Series) -> pl.Series:
    """The exact chi-squared statistic (1 degree of freedom) of each p-value, null when it is not finite."""
    statistic = pl.Series(chi2.isf(p_value.to_numpy(), df=1))
    return statistic.set(~statistic.is_finite(), None)


def chi2_from_pvalue(mantissa: pl.Expr, exponent: pl.Expr) -> pl.Expr:
    """The chi-squared statistic (1 degree of freedom) of a p-value given as mantissa and exponent.

    From `CHI2_EXACT_MIN_EXPONENT` up it is the inverse survival function of the chi-squared
    distribution. Below it, where the p-value underflows, it is the linear approximation
    `CHI2_APPROXIMATION_SLOPE * -log10(p) + CHI2_APPROXIMATION_INTERCEPT`, whose slope 4.596 and
    intercept -5.367 are constants. It is null when the p-value is missing or invalid.

    Examples:
        >>> frame = pl.DataFrame({'mantissa': [5.0, 9.0, 9.0, None], 'exponent': [-8, -300, -301, -8]})
        >>> frame.select(chi2_from_pvalue(pl.col('mantissa'), pl.col('exponent')).round(2))['chi2'].to_list()
        [29.72, 1369.48, 1373.64, None]
    """
    neglog = -(mantissa.cast(pl.Float64).log10() + exponent)
    approximation = neglog * CHI2_APPROXIMATION_SLOPE + CHI2_APPROXIMATION_INTERCEPT
    p_value = mantissa.cast(pl.Float64) * pl.lit(10.0).pow(exponent)
    # the Python function appears once in the plan: the optimiser does not share it between uses
    exact = p_value.map_batches(_chi2_isf, return_dtype=pl.Float64, is_elementwise=True)
    return pl.when(exponent < CHI2_EXACT_MIN_EXPONENT).then(_finite(approximation)).otherwise(exact).alias('chi2')


def rescaled_effect_size(
    credible_set_variant: pl.LazyFrame, credible_set: pl.LazyFrame, study: pl.LazyFrame
) -> pl.LazyFrame:
    """Set the rescaled effect size of every variant from its p-value, frequency and study design.

    `|z|` is the square root of `chi2_from_pvalue`, `varG = 2 * AF * (1 - AF)` and N is the study
    `nSamples`. The trait class comes from the study design flag:

    - MEASUREMENT_STUDY_DESIGN, quantitative: `SE = sqrt(1 / (varG * N))`;
    - CASE_CONTROL_STUDY_DESIGN, binary, with the prevalence `nCases / nSamples`:
      `SE = sqrt(1 / (varG * N * prev * (1 - prev)))`.

    `rescaledAbsBeta = |z| * SE`, `rescaledStandardError = SE`, and `rescaledDirection` is the sign
    of `conditionalBeta`, otherwise of `reportedBeta`, relative to the effect allele; a NaN,
    infinite or zero beta has no direction, so it counts as missing. A rescaled row gets
    RESCALED_FROM_CONDITIONAL or RESCALED_FROM_MARGINAL after the origin of its direction. The
    rescaled columns are null, and no flag is set, without a p-value, a finite non-zero beta, a
    frequency in (0, 1), a supported design or a sample size. The rows and their order are unchanged.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['cs', 'cs'], 'pValueMantissa': [1.0, 1.0], 'pValueExponent': [-10, -10],
        ...     'effectAlleleFrequency': [0.2, None], 'reportedBeta': [0.2, 0.2], 'conditionalBeta': [None, None],
        ...     'rescaledAbsBeta': [None, None],
        ...     'rescaledStandardError': [None, None],
        ...     'rescaledDirection': [None, None],
        ...     'qualityControls': [[], []],
        ... })
        >>> credible_set = pl.LazyFrame({'id': ['cs'], 'studyId': ['s']})
        >>> study = pl.LazyFrame({
        ...     'studyId': ['s'], 'nSamples': [100_000], 'nCases': [None],
        ...     'qualityControls': [[StudyQualityFlag.MEASUREMENT_STUDY_DESIGN.value]],
        ... })
        >>> rescaled = rescaled_effect_size(credible_set_variant, credible_set, study).collect()
        >>> rounded = pl.col('rescaledAbsBeta', 'rescaledStandardError').cast(pl.Float64).round(5)
        >>> rescaled.select(rounded, 'rescaledDirection').rows()
        [(0.03615, 0.00559, 1), (None, None, None)]
    """
    study_flag = pl.col('qualityControls')
    study_design = study.select(
        'studyId',
        pl.col('nSamples').cast(pl.Float64).alias('_n'),
        (pl.col('nCases').cast(pl.Float64) / pl.col('nSamples').cast(pl.Float64)).alias('_prevalence'),
        study_flag.list.contains(StudyQualityFlag.MEASUREMENT_STUDY_DESIGN.value).alias('_quantitative'),
        study_flag.list.contains(StudyQualityFlag.CASE_CONTROL_STUDY_DESIGN.value).alias('_binary'),
    )
    af = pl.col('effectAlleleFrequency').cast(pl.Float64)
    var_g = 2 * af * (1 - af)
    n, prevalence = pl.col('_n'), pl.col('_prevalence')
    standard_error = (
        pl
        .when(pl.col('_quantitative'))
        .then((1 / (var_g * n)).sqrt())
        .when(pl.col('_binary'))
        .then((1 / (var_g * n * prevalence * (1 - prevalence))).sqrt())
    )
    conditional_beta, reported_beta = _signed(pl.col('conditionalBeta')), _signed(pl.col('reportedBeta'))
    beta = pl.coalesce(conditional_beta, reported_beta)
    z = chi2_from_pvalue(pl.col('pValueMantissa'), pl.col('pValueExponent')).sqrt()
    rescalable = (
        af.is_between(0, 1, closed='none') & (n > 0) & beta.is_not_null() & z.is_not_null() & standard_error.is_finite()
    ).fill_null(False)
    conditional = conditional_beta.is_not_null()
    return (
        credible_set_variant
        .join(
            study_of_set(credible_set),
            on='credibleSetId',
            how='left',
            maintain_order='left',
        )
        .join(study_design, on='studyId', how='left', maintain_order='left')
        .with_columns(
            pl.when(rescalable).then(z * standard_error).cast(pl.Float32).alias('rescaledAbsBeta'),
            pl.when(rescalable).then(standard_error).cast(pl.Float32).alias('rescaledStandardError'),
            pl.when(rescalable).then(beta.sign()).cast(pl.Int8).alias('rescaledDirection'),
            rescalable.alias('_rescaled'),
        )
        .pipe(flag_rows, pl.col('_rescaled') & conditional, CredibleSetVariantFlag.RESCALED_FROM_CONDITIONAL)
        .pipe(flag_rows, pl.col('_rescaled') & ~conditional, CredibleSetVariantFlag.RESCALED_FROM_MARGINAL)
        .select(credible_set_variant.collect_schema().names())
    )
