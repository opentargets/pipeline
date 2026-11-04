"""Split the QC'd credible sets and their variants of every source into valid and excluded datasets.

Validation is the only credible set stage that removes rows. A credible set is excluded with all
of its variants. It first runs the cross-set QC, the flags that compare sets across sources, over
every source; it then validates each source's sets against that source's variants only.
"""

from pathlib import PurePosixPath
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pts.schemas.credible_set import validated_credible_set_schema
from pts.transformers.credible_set.enums import Chromosome, FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.set_qc import flag_lead_variant_missing
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows, has_excluded_flag, resolve_flag_names, valid_rows

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

    sets = [folder(path) for path in sources.credible_set]
    variants = [folder(path) for path in sources.credible_set_variant]
    if sets != variants:
        raise ValueError(f'credible sets and variants must pair up by source, in order: {sets} != {variants}')
    return list(zip(sets, sources.credible_set, sources.credible_set_variant, strict=True))


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
    def no_structural_flags(self) -> 'ValidationSettings':
        """Reject the structural flags, which validation raises and always applies."""
        listed = {*self.cs_invalid_qc_reasons, *self.csv_set_invalid_qc_reasons, *self.csv_row_invalid_qc_reasons}
        if structural := sorted(flag.name for flag in listed & STRUCTURAL_FLAGS):
            raise ValueError(f'structural flags are always applied and cannot be configured: {structural}')
        return self


def _flag_matching_ids(
    frame: pl.LazyFrame, key: str, ids: pl.LazyFrame, flag: CredibleSetFlag | CredibleSetVariantFlag
) -> pl.LazyFrame:
    """Flag the rows of `frame` whose `key` is among the `id` values of `ids`."""
    marked = ids.select(pl.col('id').alias(key), pl.lit(True).alias('_matched')).unique()
    return frame.join(marked, on=key, how='left').pipe(flag_rows, pl.col('_matched'), flag).drop('_matched')


def redundant_pics_top_hits(sets: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of curated top hits made redundant by PICS sets from summary statistics.

    A set is a redundant top hit when it carries TOP_HIT and its study also has a PICS set without
    TOP_HIT: the study has summary statistics, whose fine-mapped sets replace the curated hits.

    Examples:
        >>> sets = pl.LazyFrame({
        ...     'id': ['top', 'sumstats', 'top_alone'],
        ...     'studyId': ['s', 's', 't'],
        ...     'fineMappingMethod': ['PICS', 'PICS', 'PICS'],
        ...     'qualityControls': [[CredibleSetFlag.TOP_HIT.value], [], [CredibleSetFlag.TOP_HIT.value]],
        ... })
        >>> redundant_pics_top_hits(sets).collect()['id'].to_list()
        ['top']
    """
    top_hit = pl.col('qualityControls').list.contains(CredibleSetFlag.TOP_HIT.value)
    pics = pl.col('fineMappingMethod') == FinemappingMethod.PICS.value
    studies_with_summary_statistics = sets.filter(pics & ~top_hit).select('studyId').unique()
    return sets.filter(top_hit).join(studies_with_summary_statistics, on='studyId', how='semi').select('id').unique()


def duplicated_ids(sets: pl.LazyFrame) -> pl.LazyFrame:
    """The ids that occur more than once; every copy of such an id is flagged.

    Examples:
        >>> sets = pl.LazyFrame({'id': ['dup', 'dup', 'unique']})
        >>> duplicated_ids(sets).collect()['id'].to_list()
        ['dup']
    """
    return sets.group_by('id').len().filter(pl.col('len') > 1).select('id')


def explained_by_susie(sets: pl.LazyFrame, variants: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of sets that a SuSiE or SuSiE-inf set of the same study explains.

    A set that is not SuSiE or SuSiE-inf is explained when one of its members, without
    EXCEEDS_COVERAGE_95, lies inside `[locusStart, locusEnd]` of a SuSiE or SuSiE-inf set of the
    same study and chromosome. A member position is the second field of its `variantId`; a member on
    an alternate contig has none and never counts.

    Examples:
        >>> sets = pl.LazyFrame({
        ...     'id': ['inside', 'outside', 'susie'],
        ...     'studyId': ['s', 's', 's'],
        ...     'chromosome': ['1', '1', '1'],
        ...     'locusStart': [None, None, 1000],
        ...     'locusEnd': [None, None, 2000],
        ...     'fineMappingMethod': ['PICS', 'PICS', 'SuSiE'],
        ... }, schema_overrides={'locusStart': pl.Int32, 'locusEnd': pl.Int32})
        >>> variants = pl.LazyFrame({
        ...     'credibleSetId': ['inside', 'outside', 'susie'],
        ...     'variantId': ['1_1500_A_T', '1_100_A_T', '1_1500_A_T'],
        ...     'qualityControls': [[], [], []],
        ... })
        >>> explained_by_susie(sets, variants).collect()['id'].to_list()
        ['inside']
    """
    susie = pl.col('fineMappingMethod').is_in([method.value for method in FinemappingMethod.susie()])
    susie_regions = sets.filter(susie).select('studyId', 'chromosome', 'locusStart', 'locusEnd').unique()
    other_sets = sets.filter(~susie).select('id', 'studyId', 'chromosome').unique()
    members = (
        variants
        .filter(~pl.col('qualityControls').list.contains(CredibleSetVariantFlag.EXCEEDS_COVERAGE_95.value))
        .select(
            pl.col('credibleSetId').alias('id'),
            pl.col('variantId').str.split('_').list.get(1).cast(pl.Int64, strict=False).alias('position'),
        )
        .drop_nulls('position')
        .join(other_sets, on='id', how='inner')
    )
    return (
        members
        .join(susie_regions, on=['studyId', 'chromosome'], how='inner')
        .filter(pl.col('position').is_between(pl.col('locusStart'), pl.col('locusEnd')))
        .select('id')
        .unique()
    )


def flag_across_sets(sets: pl.LazyFrame, variants: pl.LazyFrame) -> pl.LazyFrame:
    """The cross-set flags of the credible sets of every source, one row per flagged set.

    The flags compare credible sets across sources, so they are computed once, on the QC'd sets and
    variants of every source, before any exclusion. The result is a lookup table, not the sets:
    only flagged ids are listed, with their flags, and each source's validation adds them to its
    own sets (`add_cross_set_flags`). An id that occurs twice is listed once, and the join gives
    the flag to both copies.

    - REDUNDANT_PICS_TOP_HIT: `redundant_pics_top_hits`.
    - DUPLICATED_STUDYLOCUS_ID: `duplicated_ids`.
    - EXPLAINED_BY_SUSIE: `explained_by_susie`.

    Examples:
        >>> sets = pl.LazyFrame({
        ...     'id': ['top', 'pics', 'susie', 'dup', 'dup'],
        ...     'studyId': ['s', 's', 's', 't', 'u'],
        ...     'chromosome': ['1'] * 5,
        ...     'locusStart': [None, None, 1000, None, None],
        ...     'locusEnd': [None, None, 2000, None, None],
        ...     'fineMappingMethod': ['PICS', 'PICS', 'SuSiE', 'PICS', 'PICS'],
        ...     'qualityControls': [[CredibleSetFlag.TOP_HIT.value], [], [], [], []],
        ... }, schema_overrides={'locusStart': pl.Int32, 'locusEnd': pl.Int32})
        >>> variants = pl.LazyFrame({
        ...     'credibleSetId': ['top', 'pics', 'susie'],
        ...     'variantId': ['1_100_A_T', '1_1500_A_T', '1_1500_A_T'],
        ...     'qualityControls': [[], [], []],
        ... })
        >>> for row in flag_across_sets(sets, variants).sort('id').collect().iter_rows(named=True):
        ...     print(row['id'], sorted(CredibleSetFlag(flag).name for flag in row['qualityControls']))
        dup ['DUPLICATED_STUDYLOCUS_ID']
        pics ['EXPLAINED_BY_SUSIE']
        top ['REDUNDANT_PICS_TOP_HIT']
    """
    flagged = [
        (redundant_pics_top_hits(sets), CredibleSetFlag.REDUNDANT_PICS_TOP_HIT),
        (duplicated_ids(sets), CredibleSetFlag.DUPLICATED_STUDYLOCUS_ID),
        (explained_by_susie(sets, variants), CredibleSetFlag.EXPLAINED_BY_SUSIE),
    ]
    return (
        pl
        .concat([ids.with_columns(pl.lit(flag.value).alias('flag')) for ids, flag in flagged])
        .group_by('id')
        .agg(pl.col('flag').unique().sort().alias('qualityControls'))
        .match_to_schema(cross_set_flags_schema)
    )


def add_cross_set_flags(sets: pl.LazyFrame, cross_set_flags: pl.LazyFrame) -> pl.LazyFrame:
    """Append the cross-set flags of each set to its `qualityControls`; other sets are unchanged."""
    flags = cross_set_flags.select('id', pl.col('qualityControls').alias('_cross_set_flags'))
    merged = pl.concat_list(pl.col('qualityControls'), pl.col('_cross_set_flags')).list.unique().list.sort()
    return (
        sets
        .join(flags, on='id', how='left')
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
        >>> sets = pl.LazyFrame({
        ...     'chromosome': ['10', '2', 'X'],
        ...     'studyType': ['gwas', 'eqtl', 'pqtl'],
        ...     'fineMappingMethod': ['PICS', 'SuSiE', 'SuSiE-inf'],
        ... })
        >>> typed = type_valid_sets(sets).collect()
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


def validate(
    sets: pl.LazyFrame,
    variants: pl.LazyFrame,
    cross_set_flags: pl.LazyFrame,
    settings: ValidationSettings,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Split the credible sets and variants of one source into valid and excluded rows.

    A credible set is excluded together with all of its variants. The steps are:

    0. The flags that cross-set QC computed over every source are added to the sets they belong to.
    1. The variants carrying a flag of `csv_row_invalid_qc_reasons` (by default EXCEEDS_COVERAGE_95
       and ORPHAN_VARIANT) are excluded on their own; the other variants are kept.
    2. The sets with at least one variant carrying a flag of `csv_set_invalid_qc_reasons` get
       `INVALID_VARIANT_IN_SET`: one invalid member invalidates its set.
    3. The sets whose lead has no row among the kept variants get `LEAD_VARIANT_MISSING`, so a lead
       removed in step 1 removes its set.
    4. The sets carrying a flag of `cs_invalid_qc_reasons`, or `INVALID_VARIANT_IN_SET`, are
       excluded. `INVALID_VARIANT_IN_SET` is structural, so it is not configurable.
    5. Every kept variant of an excluded set gets `CREDIBLE_SET_EXCLUDED` and is excluded with it, so
       every excluded row carries the reason it left.

    Flags are appended to `qualityControls`; a set with several invalid variants is flagged once.

    Args:
        sets: The source's credible sets after set QC.
        variants: The source's credible set variants after variant QC.
        cross_set_flags: The cross-set flags of every source, one row per flagged set.
        settings: The flags that exclude a set.

    Returns:
        The valid sets, valid variants, excluded sets and excluded variants.

    Examples:
        Set `a` is clean, set `b` is in the MHC, and set `c` has a member on an alternate contig:

        >>> from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
        >>> sets = pl.LazyFrame({
        ...     'id': ['a', 'b', 'c'],
        ...     'leadVariantId': ['1_10_A_T', '6_30000000_A_T', '2_10_A_T'],
        ...     'qualityControls': [[], [CredibleSetFlag.IN_MHC.value], []],
        ... })
        >>> variants = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'b', 'c', 'c'],
        ...     'variantId': ['1_10_A_T', '6_30000000_A_T', '2_10_A_T', '2_KI270846v1_alt_10_A_T'],
        ...     'qualityControls': [[], [], [], [CredibleSetVariantFlag.INVALID_VARIANT_ID.value]],
        ... })
        >>> settings = ValidationSettings(
        ...     cs_invalid_qc_reasons=['IN_MHC'], csv_set_invalid_qc_reasons=['INVALID_VARIANT_ID']
        ... )
        >>> no_cross_set_flags = pl.LazyFrame(schema=cross_set_flags_schema)
        >>> valid_set, valid_variant, excluded_set, excluded_variant = validate(
        ...     sets, variants, no_cross_set_flags, settings
        ... )

        Only the clean set and its variant are valid:

        >>> valid_set.collect()['id'].to_list()
        ['a']
        >>> valid_variant.collect()['variantId'].to_list()
        ['1_10_A_T']

        The excluded rows carry the flag that explains their exclusion:

        >>> def names(flags, enum):
        ...     return sorted(enum(flag).name for flag in flags)
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
    sets = add_cross_set_flags(sets, cross_set_flags)
    row_split = valid_rows(variants, settings.csv_row_invalid_qc_reasons)
    sets_with_invalid_variant = variants.filter(has_excluded_flag(settings.csv_set_invalid_qc_reasons)).select(
        pl.col('credibleSetId').alias('id')
    )
    flagged_sets = _flag_matching_ids(sets, 'id', sets_with_invalid_variant, CredibleSetFlag.INVALID_VARIANT_IN_SET)
    flagged_sets = flag_lead_variant_missing(flagged_sets, row_split.valid)
    set_split = valid_rows(flagged_sets, [*settings.cs_invalid_qc_reasons, CredibleSetFlag.INVALID_VARIANT_IN_SET])
    kept_variants = _flag_matching_ids(
        row_split.valid, 'credibleSetId', set_split.invalid, CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED
    )
    variant_split = valid_rows(kept_variants, [CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED])
    excluded_variants = pl.concat([row_split.invalid, variant_split.invalid])
    return set_split.valid, variant_split.valid, set_split.invalid, excluded_variants


def integrity_checks(
    source: str,
    sets: pl.LazyFrame,
    variants: pl.LazyFrame,
    valid_set: pl.LazyFrame,
    valid_variant: pl.LazyFrame,
    excluded_set: pl.LazyFrame,
    excluded_variant: pl.LazyFrame,
) -> pl.LazyFrame:
    """The counts the integrity of one source's validation is checked on, as one row.

    `integrity_errors` turns them into errors. The counts are aggregates, so the checks of every
    source are computed together, in one pass, before anything is written.
    """
    count = pl.len()
    valid_ids = valid_set.select('id').unique()
    lead_rows = valid_set.select('id', 'leadVariantId').join(
        valid_variant.select(
            pl.col('credibleSetId').alias('id'), pl.col('variantId').alias('leadVariantId'), pl.lit(1).alias('_lead')
        ),
        on=['id', 'leadVariantId'],
        how='left',
    )
    return pl.concat(
        [
            sets.select(count.alias('sets_in')),
            valid_set.select(count.alias('valid_sets'), pl.col('studyType').is_null().sum().alias('null_study_type')),
            excluded_set.select(count.alias('excluded_sets')),
            variants.select(count.alias('variants_in')),
            valid_variant.select(count.alias('valid_variants')),
            excluded_variant.select(count.alias('excluded_variants')),
            valid_variant.join(valid_ids, left_on='credibleSetId', right_on='id', how='anti').select(
                count.alias('variants_without_valid_set')
            ),
            lead_rows
            .group_by('id')
            .agg(pl.col('_lead').sum().alias('_leads'))
            .select((pl.col('_leads') != 1).sum().alias('sets_without_one_lead')),
        ],
        how='horizontal',
    ).select(pl.lit(source).alias('source'), pl.all())


def integrity_errors(checks: pl.DataFrame) -> list[str]:
    """The integrity errors in the counts of `integrity_checks`, one per failed check and source."""
    errors = []
    for row in checks.iter_rows(named=True):
        source = row['source']
        if row['sets_in'] != row['valid_sets'] + row['excluded_sets']:
            errors.append(
                f'{source}: {row["sets_in"]} input sets != '
                f'{row["valid_sets"]} valid + {row["excluded_sets"]} excluded sets'
            )
        if row['variants_in'] != row['valid_variants'] + row['excluded_variants']:
            errors.append(
                f'{source}: {row["variants_in"]} input variants != '
                f'{row["valid_variants"]} valid + {row["excluded_variants"]} excluded variants'
            )
        if row['variants_without_valid_set']:
            errors.append(f'{source}: {row["variants_without_valid_set"]} valid variants without a valid set')
        if row['sets_without_one_lead']:
            errors.append(f'{source}: {row["sets_without_one_lead"]} valid sets without exactly one lead row')
        if row['null_study_type']:
            errors.append(f'{source}: {row["null_study_type"]} valid sets without a studyType')
    return errors


def exclusion_counts(
    source: str, excluded_set: pl.LazyFrame, excluded_variant: pl.LazyFrame, settings: ValidationSettings
) -> pl.LazyFrame:
    """The number of excluded sets and variants per exclusion reason, for one source.

    A reason is a flag of the exclusion lists or a structural flag; a row excluded for several
    reasons counts once for each.
    """
    set_reasons = {
        flag.value: flag.name for flag in [*settings.cs_invalid_qc_reasons, CredibleSetFlag.INVALID_VARIANT_IN_SET]
    }
    variant_reasons = {
        flag.value: flag.name
        for flag in [
            *settings.csv_set_invalid_qc_reasons,
            *settings.csv_row_invalid_qc_reasons,
            CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED,
        ]
    }

    def per_reason(frame: pl.LazyFrame, dataset: str, reasons: dict[str, str]) -> pl.LazyFrame:
        return (
            frame
            .select(pl.col('qualityControls').explode().alias('flag'))
            .filter(pl.col('flag').is_in(list(reasons)))
            .group_by('flag')
            .len('count')
            .select(
                pl.lit(source).alias('source'),
                pl.lit(dataset).alias('dataset'),
                pl.col('flag').replace_strict(reasons),
                pl.col('count').cast(pl.Int64),
            )
        )

    return pl.concat([
        per_reason(excluded_set, 'credible_set', set_reasons),
        per_reason(excluded_variant, 'credible_set_variant', variant_reasons),
    ])


def sort_sets(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Sort credible sets by chromosome in genome order, then `locusStart`, `locusEnd` and `leadVariantId`.

    A chromosome outside `Chromosome` sorts last.
    """
    order = {chromosome.value: rank for rank, chromosome in enumerate(Chromosome)}
    rank = pl.col('chromosome').cast(pl.String).replace_strict(order, default=len(order), return_dtype=pl.Int32)
    return frame.sort(rank, 'locusStart', 'locusEnd', 'leadVariantId', nulls_last=True)


def sort_variants(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Sort credible set variants by `credibleSetId`, then `variantId`."""
    return frame.sort('credibleSetId', 'variantId')


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
    pairs = pair_sources(sources, config)
    cross_set_flags = flag_across_sets(
        scan_datasets(sources.credible_set), scan_datasets(sources.credible_set_variant)
    ).collect()
    logger.info('Cross-set QC flagged {} credible sets', cross_set_flags.height)
    splits = {
        source: (
            scan_dataset(sets),
            scan_dataset(variants),
            *validate(scan_dataset(sets), scan_dataset(variants), cross_set_flags.lazy(), options),
        )
        for source, sets, variants in pairs
    }
    checks, counts = pl.collect_all([
        pl.concat([integrity_checks(source, *split) for source, split in splits.items()]),
        pl.concat([
            exclusion_counts(source, excluded_set, excluded_variant, options)
            for source, (_, _, _, _, excluded_set, excluded_variant) in splits.items()
        ]),
    ])
    for name, dataset, flag, count in counts.sort('source', 'dataset', 'flag').iter_rows():
        logger.info('{}: {} excluded {} with {}', name, count, dataset, flag)
    if errors := integrity_errors(checks):
        raise ValueError('credible set validation failed its integrity checks:\n' + '\n'.join(errors))
    valid_set, valid_variant, excluded_set, excluded_variant = (
        pl.concat(parts) for parts in zip(*(split[2:] for split in splits.values()), strict=True)
    )
    logger.info('Validated the credible sets of {}', ', '.join(splits))
    pl.collect_all([
        sink_dataset(sort_sets(type_valid_sets(valid_set)), destinations.valid_set),
        sink_dataset(sort_variants(valid_variant), destinations.valid_variant),
        sink_dataset(sort_sets(excluded_set), destinations.excluded_set),
        sink_dataset(sort_variants(excluded_variant), destinations.excluded_variant),
    ])
