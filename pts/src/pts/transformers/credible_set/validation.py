"""Split the QC'd credible sets and their variants of every source into valid and excluded datasets.

Validation is the only credible set stage that removes rows. A credible set is excluded with all
of its variants. It first runs the cross-set QC, the flags that compare sets across sources, over
every source; it then validates each source's sets against that source's variants only.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Self

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pts.schemas.credible_set import validated_credible_set_schema
from pts.transformers.credible_set.enums import Chromosome, FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.variant_id import (
    variant_alternate_allele,
    variant_chromosome,
    variant_position,
    variant_reference_allele,
)
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, sink_dataset
from pts.transformers.utils.quality_flags import (
    flag_matching_ids,
    flag_rows,
    has_excluded_flag,
    resolve_flag_names,
    valid_rows,
)

cross_set_flags_schema = pl.Schema({
    'id': pl.String(),
    'qualityControls': pl.List(pl.String()),
})
"""One row per credible set flagged by the cross-set QC: its identifier and its cross-set flags."""


class ValidationSource(BaseModel):
    """The QC'd sets and variants of every source, as two lists in the same source order."""

    model_config = ConfigDict(extra='forbid')

    credible_set: list[str]
    credible_set_variant: list[str]


def pair_sources(sources: ValidationSource, config: Config) -> list[tuple[str, str, str]]:
    """Pair each source's sets with its variants, and name the source.

    A source is the folder that holds its datasets (`intermediate/qc/<kind>/<source>/<dataset>`),
    as PIS and the QC explode lay it out. The folders are compared relative to the release root,
    through `StorageHandle`, so local and `gs://` paths are handled alike.

    Returns:
        The release-relative source folder, the set path and the variant path of every source.

    Raises:
        ValueError: if the two lists do not name the same sources in the same order.
    """

    def folder(path: str) -> str:
        return str(PurePosixPath(StorageHandle(path, config).relative).parent)

    credible_set = [folder(path) for path in sources.credible_set]
    credible_set_variant = [folder(path) for path in sources.credible_set_variant]
    if credible_set != credible_set_variant:
        raise ValueError(
            f'credible sets and variants must pair up by source, in order: {credible_set} != {credible_set_variant}'
        )
    return list(zip(credible_set, sources.credible_set, sources.credible_set_variant, strict=True))


class ValidationDestination(BaseModel):
    """Output dataset paths for valid and excluded credible sets and variants."""

    model_config = ConfigDict(extra='forbid')

    valid_set: str
    valid_variant: str
    excluded_set: str
    excluded_variant: str


STRUCTURAL_FLAGS = {CredibleSetFlag.INVALID_VARIANT_IN_SET, CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED}
"""Flags that validation raises itself and always applies; they cannot be listed in the settings."""


class ValidationSettings(BaseModel):
    """Which quality flags exclude a credible set."""

    model_config = ConfigDict(extra='forbid')

    cs_invalid_qc_reasons: list[CredibleSetFlag] = Field(default_factory=list)
    """Credible set flags that exclude the set."""
    csv_set_invalid_qc_reasons: list[CredibleSetVariantFlag] = Field(default_factory=list)
    """Variant flags that exclude the whole set."""
    csv_row_invalid_qc_reasons: list[CredibleSetVariantFlag] = Field(default_factory=list)
    """Variant flags that exclude only the variant row."""

    @field_validator('cs_invalid_qc_reasons', mode='before')
    @classmethod
    def resolve_cs_names(cls, value: Any) -> Any:
        """Resolve configured enum names before validating the credible set flags."""
        return resolve_flag_names(CredibleSetFlag, value)

    @field_validator('csv_set_invalid_qc_reasons', 'csv_row_invalid_qc_reasons', mode='before')
    @classmethod
    def resolve_csv_names(cls, value: Any) -> Any:
        """Resolve configured enum names before validating the variant flags."""
        return resolve_flag_names(CredibleSetVariantFlag, value)

    @model_validator(mode='after')
    def no_structural_flags(self) -> Self:
        """Reject the structural flags, which validation raises and always applies."""
        listed = {*self.cs_invalid_qc_reasons, *self.csv_set_invalid_qc_reasons, *self.csv_row_invalid_qc_reasons}
        if structural := sorted(flag.name for flag in listed & STRUCTURAL_FLAGS):
            raise ValueError(f'structural flags are always applied and cannot be configured: {structural}')
        return self


def redundant_pics_top_hits(credible_set: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of curated top hits made redundant by PICS sets from summary statistics.

    A set is a redundant top hit when it carries TOP_HIT and its study also has a PICS set without
    TOP_HIT: the study has summary statistics, whose fine-mapped sets replace the curated hits.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['top', 'sumstats', 'top_alone'],
        ...     'studyId': ['s', 's', 't'],
        ...     'fineMappingMethod': ['PICS', 'PICS', 'PICS'],
        ...     'qualityControls': [[CredibleSetFlag.TOP_HIT.value], [], [CredibleSetFlag.TOP_HIT.value]],
        ... })
        >>> redundant_pics_top_hits(credible_set).collect()['id'].to_list()
        ['top']
    """
    top_hit = pl.col('qualityControls').list.contains(CredibleSetFlag.TOP_HIT.value)
    pics = pl.col('fineMappingMethod') == FinemappingMethod.PICS.value
    study_with_summary_statistics = credible_set.filter(pics & ~top_hit).select('studyId').unique()
    return (
        credible_set.filter(top_hit).join(study_with_summary_statistics, on='studyId', how='semi').select('id').unique()
    )


def explained_by_susie(credible_set: pl.LazyFrame, credible_set_variant: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of sets that a SuSiE or SuSiE-inf set of the same study explains.

    A set that is not SuSiE or SuSiE-inf is explained when one of its members, without
    EXCEEDS_COVERAGE_95, lies inside `[locusStart, locusEnd]` of a SuSiE or SuSiE-inf set of the
    same study and chromosome. A member position is the second field of its `variantId`; a member on
    an alternate contig has none and never counts.

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
        >>> explained_by_susie(credible_set, credible_set_variant).collect()['id'].to_list()
        ['inside']
    """
    susie = pl.col('fineMappingMethod').is_in([method.value for method in FinemappingMethod.susie()])
    susie_region = credible_set.filter(susie).select('studyId', 'chromosome', 'locusStart', 'locusEnd').unique()
    other_set = credible_set.filter(~susie).select('id', 'studyId', 'chromosome').unique()
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


def flag_across_sets(credible_set: pl.LazyFrame, credible_set_variant: pl.LazyFrame) -> pl.LazyFrame:
    """The cross-set flags of the credible sets of every source, one row per flagged set.

    The flags compare credible sets across sources, so they are computed once, on the QC'd sets and
    variants of every source, before any exclusion. The result is a lookup table, not the sets:
    only flagged ids are listed, with their flags, and each source's validation adds them to its
    own sets (`add_cross_set_flags`). Duplicated ids are decided after validation, among the valid
    sets (`flag_duplicated_ids`).

    - REDUNDANT_PICS_TOP_HIT: `redundant_pics_top_hits`.
    - EXPLAINED_BY_SUSIE: `explained_by_susie`.

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
        >>> for row in flag_across_sets(credible_set, credible_set_variant).sort('id').collect().iter_rows(named=True):
        ...     print(row['id'], sorted(CredibleSetFlag(flag).name for flag in row['qualityControls']))
        pics ['EXPLAINED_BY_SUSIE']
        top ['REDUNDANT_PICS_TOP_HIT']
    """
    flagged = [
        (redundant_pics_top_hits(credible_set), CredibleSetFlag.REDUNDANT_PICS_TOP_HIT),
        (explained_by_susie(credible_set, credible_set_variant), CredibleSetFlag.EXPLAINED_BY_SUSIE),
    ]
    return (
        pl
        .concat([ids.with_columns(pl.lit(flag.value).alias('flag')) for ids, flag in flagged])
        .group_by('id')
        .agg(pl.col('flag').unique().sort().alias('qualityControls'))
        .match_to_schema(cross_set_flags_schema)
    )


def add_cross_set_flags(credible_set: pl.LazyFrame, cross_set_flag: pl.LazyFrame) -> pl.LazyFrame:
    """Append the cross-set flags of each set to its `qualityControls`; other sets are unchanged."""
    set_flag = cross_set_flag.select('id', pl.col('qualityControls').alias('_cross_set_flags'))
    merged = pl.concat_list(pl.col('qualityControls'), pl.col('_cross_set_flags')).list.unique().list.sort()
    return (
        credible_set
        .join(set_flag, on='id', how='left')
        .with_columns(
            pl
            .when(pl.col('_cross_set_flags').is_not_null())
            .then(merged)
            .otherwise(pl.col('qualityControls'))
            .alias('qualityControls')
        )
        .drop('_cross_set_flags')
    )


def type_valid_sets(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Cast the columns of a fixed vocabulary of valid sets to their enums.

    The cast is strict: a value outside the enum is one that quality control should have excluded,
    so it raises instead of being published.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'chromosome': ['10', '2', 'X'],
        ...     'studyType': ['gwas', 'eqtl', 'pqtl'],
        ...     'fineMappingMethod': ['PICS', 'SuSiE', 'SuSiE-inf'],
        ... })
        >>> typed = type_valid_sets(credible_set).collect()
        >>> typed.schema['chromosome'] == validated_credible_set_schema['chromosome']
        True

        Chromosomes sort in genome order, and a value outside the enum raises:

        >>> typed.sort('chromosome')['chromosome'].to_list()
        ['2', '10', 'X']
        >>> outside = pl.LazyFrame({'chromosome': ['23'], 'studyType': ['gwas'], 'fineMappingMethod': ['PICS']})
        >>> type_valid_sets(outside).collect()
        Traceback (most recent call last):
        ...
        polars.exceptions.InvalidOperationError: ...
    """
    enums = pl.Schema({
        name: dtype for name, dtype in validated_credible_set_schema.items() if isinstance(dtype, pl.Enum)
    })
    return frame.cast(enums, strict=True)


def flag_orphan_variants(frame: pl.LazyFrame, credible_set: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variants whose credible set is not among the credible sets of their source.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({'credibleSetId': ['a', 'gone'], 'qualityControls': [[], []]})
        >>> credible_set = pl.LazyFrame({'id': ['a']})
        >>> flagged = flag_orphan_variants(credible_set_variant, credible_set).sort('credibleSetId').collect()
        >>> flagged['qualityControls'].list.len().to_list()
        [0, 1]
    """
    known = credible_set.select(pl.col('id').alias('credibleSetId'), pl.lit(True).alias('_known')).unique()
    return (
        frame
        .join(known, on='credibleSetId', how='left')
        .pipe(flag_rows, pl.col('_known').is_null(), CredibleSetVariantFlag.ORPHAN_VARIANT)
        .drop('_known')
    )


def flag_variant_chromosome_mismatch(frame: pl.LazyFrame, credible_set: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variants whose chromosome, the first field of `variantId`, differs from their set's.

    An orphan variant has no set to compare with, and is not flagged.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'a'],
        ...     'variantId': ['1_1_A_T', '2_5_A_T'],
        ...     'qualityControls': [[], []],
        ... })
        >>> credible_set = pl.LazyFrame({'id': ['a'], 'chromosome': ['1']})
        >>> flagged = flag_variant_chromosome_mismatch(credible_set_variant, credible_set).sort('variantId').collect()
        >>> flagged['qualityControls'].list.len().to_list()
        [0, 1]
    """
    set_chromosome = credible_set.select(
        pl.col('id').alias('credibleSetId'), pl.col('chromosome').alias('_set_chromosome')
    ).unique()
    mismatch = variant_chromosome(pl.col('variantId')) != pl.col('_set_chromosome')
    return (
        frame
        .join(set_chromosome, on='credibleSetId', how='left')
        .pipe(flag_rows, mismatch, CredibleSetVariantFlag.VARIANT_CHROMOSOME_MISMATCH)
        .drop('_set_chromosome')
    )


def flag_lead_variant_missing(frame: pl.LazyFrame, credible_set_variant: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets whose lead variant has no row among the variants of their source.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['a', 'b'],
        ...     'leadVariantId': ['1_1_A_T', '1_9_A_T'],
        ...     'qualityControls': [[], []],
        ... })
        >>> credible_set_variant = pl.LazyFrame({'credibleSetId': ['a', 'b'], 'variantId': ['1_1_A_T', '1_2_A_T']})
        >>> flagged = flag_lead_variant_missing(credible_set, credible_set_variant).sort('id').collect()
        >>> flagged['qualityControls'].list.len().to_list()
        [0, 1]
    """
    lead = credible_set_variant.select(
        pl.col('credibleSetId').alias('id'), pl.col('variantId').alias('leadVariantId'), pl.lit(True).alias('_lead')
    ).unique()
    return (
        frame
        .join(lead, on=['id', 'leadVariantId'], how='left')
        .pipe(flag_rows, pl.col('_lead').is_null(), CredibleSetFlag.LEAD_VARIANT_MISSING)
        .drop('_lead')
    )


def validate(
    credible_set: pl.LazyFrame,
    credible_set_variant: pl.LazyFrame,
    cross_set_flag: pl.LazyFrame,
    settings: ValidationSettings,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Split the credible sets and variants of one source into valid and excluded rows.

    Variant rows first:

    1. Flag `ORPHAN_VARIANT` on a variant whose set is not among this source's sets, and
       `VARIANT_CHROMOSOME_MISMATCH` on a variant on another chromosome than its set.
    2. Exclude the variants with a flag of `csv_row_invalid_qc_reasons` (by default
       EXCEEDS_COVERAGE_95 and ORPHAN_VARIANT). Only the row leaves; its set stays.

    Then whole sets:

    3. Add the cross-set flags (REDUNDANT_PICS_TOP_HIT, EXPLAINED_BY_SUSIE). Duplicated ids are
       decided after every source is validated (`flag_duplicated_ids`).
    4. Flag `INVALID_VARIANT_IN_SET` on a set with any variant flagged in
       `csv_set_invalid_qc_reasons`.
    5. Flag `LEAD_VARIANT_MISSING` on a set whose lead variant is not among its remaining
       variants, either because it was never staged or because step 2 removed it.
    6. Exclude the sets with a flag of `cs_invalid_qc_reasons`, or `INVALID_VARIANT_IN_SET`,
       together with their remaining variants, which get `CREDIBLE_SET_EXCLUDED`.

    Every excluded row carries the flag that explains why it left. `INVALID_VARIANT_IN_SET` and
    `CREDIBLE_SET_EXCLUDED` are structural, so they are not configurable.

    Flags are appended to `qualityControls`; a set with several invalid variants is flagged once.

    Args:
        credible_set: The source's credible sets after set QC.
        credible_set_variant: The source's credible set variants after variant QC.
        cross_set_flag: The cross-set flags of every source, one row per flagged set.
        settings: The flags that exclude a set.

    Returns:
        The valid sets, valid variants, excluded sets and excluded variants.

    Examples:
        Set `a` is clean, set `b` is in the MHC, and set `c` has a member on an alternate contig:

        >>> from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['a', 'b', 'c'],
        ...     'chromosome': ['1', '6', '2'],
        ...     'leadVariantId': ['1_10_A_T', '6_30000000_A_T', '2_10_A_T'],
        ...     'qualityControls': [[], [CredibleSetFlag.IN_MHC.value], []],
        ... })
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'b', 'c', 'c'],
        ...     'variantId': ['1_10_A_T', '6_30000000_A_T', '2_10_A_T', '2_KI270846v1_alt_10_A_T'],
        ...     'qualityControls': [[], [], [], [CredibleSetVariantFlag.INVALID_VARIANT_ID.value]],
        ... })
        >>> settings = ValidationSettings(
        ...     cs_invalid_qc_reasons=['IN_MHC'], csv_set_invalid_qc_reasons=['INVALID_VARIANT_ID']
        ... )
        >>> no_cross_set_flags = pl.LazyFrame(schema=cross_set_flags_schema)
        >>> valid_set, valid_variant, excluded_set, excluded_variant = validate(
        ...     credible_set, credible_set_variant, no_cross_set_flags, settings
        ... )

        Only the clean set and its variant are valid:

        >>> valid_set.collect()['id'].to_list()
        ['a']
        >>> valid_variant.collect()['variantId'].to_list()
        ['1_10_A_T']

        The excluded rows carry the flag that explains their exclusion:

        >>> def names(set_flag, enum):
        ...     return sorted(enum(flag).name for flag in set_flag)
        >>> for row in excluded_set.sort('id').collect().iter_rows(named=True):
        ...     print(row['id'], names(row['qualityControls'], CredibleSetFlag))
        b ['IN_MHC']
        c ['INVALID_VARIANT_IN_SET']
        >>> for row in excluded_variant.sort('variantId').collect().iter_rows(named=True):
        ...     print(row['variantId'], names(row['qualityControls'], CredibleSetVariantFlag))
        2_10_A_T ['CREDIBLE_SET_EXCLUDED']
        2_KI270846v1_alt_10_A_T ['CREDIBLE_SET_EXCLUDED', 'INVALID_VARIANT_ID']
        6_30000000_A_T ['CREDIBLE_SET_EXCLUDED']
    """
    # Variant rows: 1. pairing flags, 2. row exclusion.
    credible_set_variant = flag_orphan_variants(credible_set_variant, credible_set).pipe(
        flag_variant_chromosome_mismatch, credible_set
    )
    row_split = valid_rows(credible_set_variant, settings.csv_row_invalid_qc_reasons)
    # Whole sets: 3. cross-set flags, 4. invalid variant in set, 5. missing lead, 6. exclusion.
    flagged_set = add_cross_set_flags(credible_set, cross_set_flag)
    sets_with_invalid_variant = credible_set_variant.filter(
        has_excluded_flag(settings.csv_set_invalid_qc_reasons)
    ).select(pl.col('credibleSetId').alias('id'))
    flagged_set = flag_matching_ids(
        flagged_set, 'id', sets_with_invalid_variant, CredibleSetFlag.INVALID_VARIANT_IN_SET
    )
    flagged_set = flag_lead_variant_missing(flagged_set, row_split.valid)
    set_split = valid_rows(flagged_set, [*settings.cs_invalid_qc_reasons, CredibleSetFlag.INVALID_VARIANT_IN_SET])
    kept_variant = flag_matching_ids(
        row_split.valid, 'credibleSetId', set_split.invalid, CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED
    )
    variant_split = valid_rows(kept_variant, [CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED])
    excluded_credible_set_variant = pl.concat([row_split.invalid, variant_split.invalid])
    return set_split.valid, variant_split.valid, set_split.invalid, excluded_credible_set_variant


def flag_duplicated_ids(
    split: list[tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]], settings: ValidationSettings
) -> list[tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]]:
    """Flag DUPLICATED_STUDYLOCUS_ID on all but one copy of an id among the valid sets of every source.

    As gentropy's credible set validation does, only the sets that pass the other checks are
    compared, and the copy with the fewest valid variants is kept. Ties go to the copy of the
    earlier source, then to the smaller hash of its row, so the kept copy does not depend on row
    order. When the flag is in `cs_invalid_qc_reasons`, the flagged copies leave the valid sets of
    their source, with their variants flagged `CREDIBLE_SET_EXCLUDED` unless the source keeps
    another copy of the id.

    Args:
        split: The (valid sets, valid variants, excluded sets, excluded variants) of each source.
        settings: The flags that exclude a set.

    Returns:
        The same four datasets of each source, in the same order.

    Examples:
        >>> def source(size):
        ...     quality_control = pl.Series([[]], dtype=pl.List(pl.String))
        ...     credible_set = pl.LazyFrame({'id': ['dup'], 'qualityControls': quality_control})
        ...     variant_id = [f'1_{position}_A_T' for position in range(size)]
        ...     variant = pl.LazyFrame({'credibleSetId': ['dup'] * size, 'variantId': variant_id})
        ...     return credible_set, variant, credible_set.clear(), variant.clear()
        >>> settings = ValidationSettings(cs_invalid_qc_reasons=['DUPLICATED_STUDYLOCUS_ID'])
        >>> larger, smaller = flag_duplicated_ids([source(3), source(2)], settings)
        >>> (a_set, _, a_excluded, _), (b_set, _, _, _) = larger, smaller
        >>> a_set.collect().height, a_excluded.collect().height, b_set.collect().height
        (0, 1, 1)
    """
    duplicated = CredibleSetFlag.DUPLICATED_STUDYLOCUS_ID
    source = pl.col('_source')
    valid_set = pl.concat([
        credible_set.with_columns(pl.lit(index, pl.UInt32).alias('_source'))
        for index, (credible_set, _, _, _) in enumerate(split)
    ])
    size = pl.concat([
        credible_set_variant
        .group_by(pl.col('credibleSetId').alias('id'))
        .len('_size')
        .with_columns(pl.lit(index, pl.UInt32).alias('_source'))
        for index, (_, credible_set_variant, _, _) in enumerate(split)
    ])
    ranked = (
        valid_set
        .join(size, on=['_source', 'id'], how='left')
        .with_columns(pl.struct(pl.exclude('_source', '_size')).hash(seed=0).alias('_hash'))
        .sort('id', '_size', '_source', '_hash', nulls_last=True)
        .pipe(flag_rows, pl.int_range(pl.len()).over('id') > 0, duplicated)
        .drop('_size', '_hash')
    )
    flagged_split = []
    for index, (_, credible_set_variant, excluded_set, excluded_variant) in enumerate(split):
        flagged_set = ranked.filter(source == index).drop('_source')
        if duplicated not in settings.cs_invalid_qc_reasons:
            flagged_split.append((flagged_set, credible_set_variant, excluded_set, excluded_variant))
            continue
        set_split = valid_rows(flagged_set, [duplicated])
        removed_id = set_split.invalid.select('id').join(set_split.valid.select('id'), on='id', how='anti')
        variant_split = valid_rows(
            flag_matching_ids(
                credible_set_variant, 'credibleSetId', removed_id, CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED
            ),
            [CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED],
        )
        flagged_split.append((
            set_split.valid,
            variant_split.valid,
            pl.concat([excluded_set, set_split.invalid]),
            pl.concat([excluded_variant, variant_split.invalid]),
        ))
    return flagged_split


def integrity_checks(
    source: str,
    credible_set: pl.LazyFrame,
    credible_set_variant: pl.LazyFrame,
    valid_set: pl.LazyFrame,
    valid_variant: pl.LazyFrame,
    excluded_set: pl.LazyFrame,
    excluded_variant: pl.LazyFrame,
) -> pl.LazyFrame:
    """The row counts of one source's validation, as one row: input, valid and excluded, for sets and variants.

    `integrity_errors` checks that every input row is either valid or excluded. The counts are
    aggregates, so the checks of every source are computed together, in one pass, before anything
    is written.

    Examples:
        >>> credible_set = pl.LazyFrame({'id': ['a', 'b']})
        >>> credible_set_variant = pl.LazyFrame({'variantId': ['v', 'w', 'x']})
        >>> valid = (credible_set.head(1), credible_set_variant.head(2))
        >>> excluded = (credible_set.tail(1), credible_set_variant.tail(1))
        >>> integrity_count = integrity_checks('gwas/s', credible_set, credible_set_variant, *valid, *excluded)
        >>> integrity_count = integrity_count.collect()
        >>> integrity_count.select(pl.exclude('source')).row(0)
        (2, 1, 1, 3, 2, 1)
    """
    count = pl.len()
    return pl.concat(
        [
            credible_set.select(count.alias('sets_in')),
            valid_set.select(count.alias('valid_sets')),
            excluded_set.select(count.alias('excluded_sets')),
            credible_set_variant.select(count.alias('variants_in')),
            valid_variant.select(count.alias('valid_variants')),
            excluded_variant.select(count.alias('excluded_variants')),
        ],
        how='horizontal',
    ).select(pl.lit(source).alias('source'), pl.all())


def integrity_errors(integrity_count: pl.DataFrame) -> list[str]:
    """One error per source and dataset whose input rows are not exactly its valid plus excluded rows.

    Examples:
        >>> integrity_count = pl.DataFrame({
        ...     'source': ['gwas/s'], 'sets_in': [2], 'valid_sets': [1], 'excluded_sets': [1],
        ...     'variants_in': [3], 'valid_variants': [1], 'excluded_variants': [1],
        ... })
        >>> integrity_errors(integrity_count)
        ['gwas/s: 3 input variants != 1 valid + 1 excluded variants']
    """
    error_message = []
    for row in integrity_count.iter_rows(named=True):
        for dataset in ('sets', 'variants'):
            rows_in, valid, excluded = row[f'{dataset}_in'], row[f'valid_{dataset}'], row[f'excluded_{dataset}']
            if rows_in != valid + excluded:
                error_message.append(
                    f'{row["source"]}: {rows_in} input {dataset} != {valid} valid + {excluded} excluded {dataset}'
                )
    return error_message


def _chromosome_rank(chromosome: pl.Expr) -> pl.Expr:
    """The rank of a chromosome in genome order, as in `Chromosome`; any other chromosome ranks last."""
    order = {member.value: rank for rank, member in enumerate(Chromosome)}
    return chromosome.cast(pl.String).replace_strict(order, default=len(order), return_dtype=pl.Int32)


def sort_sets(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Sort credible sets by chromosome in genome order, then `locusStart`, `locusEnd` and `leadVariantId`.

    A chromosome outside `Chromosome` sorts last.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'chromosome': ['X', '10', '2', '2'],
        ...     'locusStart': [1, 1, 9, 1],
        ...     'locusEnd': [1, 1, 9, 5],
        ...     'leadVariantId': ['x', 'c', 'b', 'a'],
        ... })
        >>> sort_sets(credible_set).collect()['leadVariantId'].to_list()
        ['a', 'b', 'c', 'x']
    """
    rank = _chromosome_rank(pl.col('chromosome'))
    return frame.sort(rank, 'locusStart', 'locusEnd', 'leadVariantId', nulls_last=True)


def sort_variants(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Sort credible set variants in genome order, by the fields of `variantId`, then by `credibleSetId`.

    The chromosome ranks as in `Chromosome`, the position sorts as a number, then the reference and
    alternate alleles. A variant whose position is not a number sorts last on its chromosome. The
    variants of one chromosome are contiguous, for the join to the variant index.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['b', 'a', 'a', 'a', 'a'],
        ...     'variantId': ['1_100_A_T', '10_1_A_T', '1_100_A_C', '1_20_G_A', '1_100_A_T'],
        ... })
        >>> sort_variants(credible_set_variant).collect()['variantId'].to_list()
        ['1_20_G_A', '1_100_A_C', '1_100_A_T', '1_100_A_T', '10_1_A_T']
    """
    variant_id = pl.col('variantId')
    return frame.sort(
        _chromosome_rank(variant_chromosome(variant_id)),
        variant_position(variant_id),
        variant_reference_allele(variant_id),
        variant_alternate_allele(variant_id),
        'credibleSetId',
        nulls_last=True,
    )


def validation(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Run the cross-set QC over every source, validate each source, and write the combined datasets.

    Args:
        source: The QC'd sets of every source under `credible_set` and their variants under
            `credible_set_variant`, in the same source order.
        destination: Paths under `valid_set`, `valid_variant`, `excluded_set` and `excluded_variant`.
        settings: Flag names under `cs_invalid_qc_reasons`, `csv_set_invalid_qc_reasons` and
            `csv_row_invalid_qc_reasons`.
        config: Pipeline configuration.

    Raises:
        ValueError: if a source fails an integrity check; nothing is written then.
    """
    sources = ValidationSource.model_validate(source)
    destinations = ValidationDestination.model_validate(destination)
    options = ValidationSettings.model_validate(settings)
    source_pair = pair_sources(sources, config)
    cross_set_flag = flag_across_sets(
        scan_datasets(sources.credible_set), scan_datasets(sources.credible_set_variant)
    ).collect()
    logger.info('Cross-set QC flagged {} credible sets', cross_set_flag.height)
    source_input: dict[str, tuple[pl.LazyFrame, pl.LazyFrame]] = {}
    validated = []
    for source_name, set_path, variant_path in source_pair:
        # one scan per dataset, shared by the integrity counts and the validation plan
        source_input[source_name] = (scan_dataset(set_path), scan_dataset(variant_path))
        validated.append(validate(*source_input[source_name], cross_set_flag.lazy(), options))
    source_split = {
        source_name: (*source_input[source_name], *split)
        for source_name, split in zip(source_input, flag_duplicated_ids(validated, options), strict=True)
    }
    integrity_count = pl.concat([integrity_checks(source, *split) for source, split in source_split.items()]).collect()
    if error_message := integrity_errors(integrity_count):
        raise ValueError('credible set validation failed its integrity checks:\n' + '\n'.join(error_message))
    valid_set, valid_variant, excluded_set, excluded_variant = (
        pl.concat(part) for part in zip(*(split[2:] for split in source_split.values()), strict=True)
    )
    logger.info('Validated the credible sets of {}', ', '.join(source_split))
    pl.collect_all([
        sink_dataset(sort_sets(type_valid_sets(valid_set)), destinations.valid_set),
        sink_dataset(sort_variants(valid_variant), destinations.valid_variant),
        sink_dataset(sort_sets(excluded_set), destinations.excluded_set),
        sink_dataset(sort_variants(excluded_variant), destinations.excluded_variant),
    ])
