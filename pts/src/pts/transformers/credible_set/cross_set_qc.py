"""The cross-set QC: the flags a credible set gets from the other credible sets of its study.

Two rules compare credible sets across every source, so they run once, over all of them, before
validation decides each source's sets:

- REDUNDANT_PICS_TOP_HIT: a curated top hit is redundant when its study also has PICS sets from
  summary statistics, whose fine-mapped sets replace the curated hits.
- EXPLAINED_BY_SUSIE: a set that is not SuSiE is explained when one of its members lies in the
  locus of a SuSiE set of the same study.

Only a set that validation would keep on its own flags explains another. A summary statistics set
that is itself explained by SuSiE still makes the study's curated hits redundant: the rule is about
the study, which has summary statistics whatever becomes of that set. `flag_across_sets` gives the
flags as a lookup table, one row per flagged id, and `add_cross_set_flags` adds each flag to the
sets of that id its rule applies to: two sources may share an id.
"""

from __future__ import annotations

import polars as pl

from pts.schemas.credible_set import FinemappingMethod, cross_set_flags_schema
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.identifier import variant_position

_TOP_HIT = pl.col('qualityControls').list.contains(CredibleSetFlag.TOP_HIT.value)
"""A curated top hit."""

_PICS = pl.col('fineMappingMethod') == FinemappingMethod.PICS.value
"""A set fine-mapped by PICS."""

_SUSIE = pl.col('fineMappingMethod').is_in([method.value for method in FinemappingMethod.susie()])
"""A set fine-mapped by SuSiE or SuSiE-inf."""

_FLAGGED_ROW = {
    CredibleSetFlag.REDUNDANT_PICS_TOP_HIT: _TOP_HIT,
    CredibleSetFlag.EXPLAINED_BY_SUSIE: ~_SUSIE,
}
"""The sets each cross-set flag applies to: the candidates of its rule."""


def flag_across_sets(
    credible_set: pl.LazyFrame, credible_set_variant: pl.LazyFrame, explaining_set: pl.LazyFrame
) -> pl.LazyFrame:
    """The cross-set flags of the credible sets of every source, one row per flagged set.

    The flags compare credible sets across sources, so they are computed once, on the QC'd sets and
    variants of every source. Any set can be flagged, but only a set of `explaining_set`, the sets
    that survive their own flags, explains another: a set that is not published cannot replace
    one. The result is a lookup table, not the sets:
    only flagged ids are listed, with their flags, and each source's validation adds them to its
    own sets (`add_cross_set_flags`). Duplicated ids are decided after validation, among the valid
    sets (`_flag_duplicated_ids`).

    - REDUNDANT_PICS_TOP_HIT: `_redundant_pics_top_hits`.
    - EXPLAINED_BY_SUSIE: `_explained_by_susie`.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['top', 'pics', 'susie'],
        ...     'studyId': ['s', 's', 's'],
        ...     'chromosome': ['1'] * 3,
        ...     'locusStart': [None, None, 1000],
        ...     'locusEnd': [None, None, 2000],
        ...     'fineMappingMethod': ['PICS', 'PICS', 'SuSiE'],
        ...     'qualityControls': [[CredibleSetFlag.TOP_HIT.value], [], []],
        ... }, schema_overrides={'locusStart': pl.Int32, 'locusEnd': pl.Int32})
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['top', 'pics', 'susie'],
        ...     'variantId': ['1_100_A_T', '1_1500_A_T', '1_1500_A_T'],
        ...     'qualityControls': [[], [], []],
        ... })
        >>> flagged = flag_across_sets(credible_set, credible_set_variant, credible_set)
        >>> for row in flagged.sort('id').collect().iter_rows(named=True):
        ...     print(row['id'], sorted(CredibleSetFlag(flag).name for flag in row['qualityControls']))
        pics ['EXPLAINED_BY_SUSIE']
        top ['REDUNDANT_PICS_TOP_HIT']
    """
    flagged = [
        (_redundant_pics_top_hits(credible_set, explaining_set), CredibleSetFlag.REDUNDANT_PICS_TOP_HIT),
        (_explained_by_susie(credible_set, credible_set_variant, explaining_set), CredibleSetFlag.EXPLAINED_BY_SUSIE),
    ]
    return (
        pl
        .concat([ids.with_columns(pl.lit(flag.value).alias('flag')) for ids, flag in flagged])
        .group_by('id')
        .agg(pl.col('flag').unique().sort().alias('qualityControls'))
        .match_to_schema(cross_set_flags_schema)
    )


def add_cross_set_flags(credible_set: pl.LazyFrame, cross_set_flag: pl.LazyFrame) -> pl.LazyFrame:
    """Append the cross-set flags of each set to its `qualityControls`; other sets are unchanged.

    A flag is added only to the sets its rule applies to (`_FLAGGED_ROW`), so the copy of an id in
    another source does not take a flag meant for this one: a summary statistics set that shares
    its id with a curated top hit does not become a redundant top hit.
    """
    set_flag = cross_set_flag.select('id', pl.col('qualityControls').alias('_cross_set_flags'))
    no_flag = pl.lit([], dtype=pl.List(pl.String))
    listed = pl.col('_cross_set_flags').fill_null(no_flag)
    added = pl.concat_list([
        pl.when(listed.list.contains(flag.value) & applies).then(pl.lit([flag.value])).otherwise(no_flag)
        for flag, applies in _FLAGGED_ROW.items()
    ])
    merged = pl.concat_list(pl.col('qualityControls'), added).list.unique().list.sort()
    return (
        credible_set
        .join(set_flag, on='id', how='left')
        .with_columns(
            pl.when(added.list.len() > 0).then(merged).otherwise(pl.col('qualityControls')).alias('qualityControls')
        )
        .drop('_cross_set_flags')
    )


def _redundant_pics_top_hits(credible_set: pl.LazyFrame, explaining_set: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of curated top hits made redundant by PICS sets from summary statistics.

    A set is a redundant top hit when it carries TOP_HIT and its study also has a PICS set without
    TOP_HIT among `explaining_set`: the study has summary statistics, whose fine-mapped sets
    replace the curated hits.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['top', 'sumstats', 'top_alone'],
        ...     'studyId': ['s', 's', 't'],
        ...     'fineMappingMethod': ['PICS', 'PICS', 'PICS'],
        ...     'qualityControls': [[CredibleSetFlag.TOP_HIT.value], [], [CredibleSetFlag.TOP_HIT.value]],
        ... })
        >>> _redundant_pics_top_hits(credible_set, credible_set).collect()['id'].to_list()
        ['top']
    """
    study_with_summary_statistics = explaining_set.filter(_PICS & ~_TOP_HIT).select('studyId').unique()
    return (
        credible_set
        .filter(_TOP_HIT)
        .join(study_with_summary_statistics, on='studyId', how='semi')
        .select('id')
        .unique()
    )


def _explained_by_susie(
    credible_set: pl.LazyFrame, credible_set_variant: pl.LazyFrame, explaining_set: pl.LazyFrame
) -> pl.LazyFrame:
    """The ids of sets that a SuSiE or SuSiE-inf set of the same study explains.

    A set that is not SuSiE or SuSiE-inf is explained when one of its members, without
    EXCEEDS_COVERAGE_95, lies inside `[locusStart, locusEnd]` of a SuSiE or SuSiE-inf set of
    `explaining_set` of the same study and chromosome. A member position is the second field of its
    `variantId`; a member on an alternate contig has none and never counts.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['inside', 'outside', 'susie'],
        ...     'studyId': ['s', 's', 's'],
        ...     'chromosome': ['1', '1', '1'],
        ...     'locusStart': [None, None, 1000],
        ...     'locusEnd': [None, None, 2000],
        ...     'fineMappingMethod': ['PICS', 'PICS', 'SuSiE'],
        ... }, schema_overrides={'locusStart': pl.Int32, 'locusEnd': pl.Int32})
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['inside', 'outside', 'susie'],
        ...     'variantId': ['1_1500_A_T', '1_100_A_T', '1_1500_A_T'],
        ...     'qualityControls': [[], [], []],
        ... })
        >>> _explained_by_susie(credible_set, credible_set_variant, credible_set).collect()['id'].to_list()
        ['inside']
    """
    susie_region = explaining_set.filter(_SUSIE).select('studyId', 'chromosome', 'locusStart', 'locusEnd').unique()
    other_set = credible_set.filter(~_SUSIE).select('id', 'studyId', 'chromosome').unique()
    member = (
        credible_set_variant
        .filter(~pl.col('qualityControls').list.contains(CredibleSetVariantFlag.EXCEEDS_COVERAGE_95.value))
        .select(
            pl.col('credibleSetId').alias('id'),
            variant_position(pl.col('variantId')).alias('position'),
        )
        .drop_nulls('position')
        .join(other_set, on='id', how='inner')
    )
    return (
        member
        .join(susie_region, on=['studyId', 'chromosome'], how='inner')
        .filter(pl.col('position').is_between(pl.col('locusStart'), pl.col('locusEnd')))
        .select('id')
        .unique()
    )
