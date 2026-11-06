"""The confidence class of a credible set, from its fine-mapping method and its LD and top hit flags."""

from __future__ import annotations

import polars as pl

from pts.transformers.credible_set.enums import FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetConfidenceClasses, CredibleSetFlag


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
