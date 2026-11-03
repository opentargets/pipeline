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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pts.schemas.credible_set import validated_credible_set_schema
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows, has_excluded_flag, resolve_flag_names, valid_rows

cross_set_flags_schema = pl.Schema({
    'id': pl.String(),
    'qualityControls': pl.List(pl.String()),
})
"""One row per credible set flagged by the cross-set QC: its identifier and its cross-set flags."""


class ValidationSource(BaseModel):
    """The QC'd sets and variants of every source, as two lists in the same source order.

    A source is the folder that holds its datasets (`intermediate/qc/<kind>/<source>/<dataset>`),
    as PIS and the QC explode lay it out.
    """

    model_config = ConfigDict(extra='forbid')

    credible_set: list[str]
    credible_set_variant: list[str]

    @model_validator(mode='after')
    def sets_and_variants_pair_up(self) -> 'ValidationSource':
        """Require the two lists to name the same sources in the same order."""
        sets = [str(PurePosixPath(path).parent) for path in self.credible_set]
        variants = [str(PurePosixPath(path).parent) for path in self.credible_set_variant]
        if sets != variants:
            raise ValueError(f'credible sets and variants must pair up by source, in order: {sets} != {variants}')
        return self

    def pairs(self) -> list[tuple[str, str, str]]:
        """The source name, set path and variant path of every source."""
        return [
            (PurePosixPath(sets).parent.name, sets, variants)
            for sets, variants in zip(self.credible_set, self.credible_set_variant, strict=True)
        ]


class ValidationDestination(BaseModel):
    """Output dataset paths for valid and excluded credible sets and variants."""

    model_config = ConfigDict(extra='forbid')

    valid_set: str
    valid_variant: str
    excluded_set: str
    excluded_variant: str


class ValidationSettings(BaseModel):
    """Which quality flags exclude a credible set."""

    model_config = ConfigDict(extra='forbid')

    cs_invalid_qc_reasons: list[CredibleSetFlag] = Field(default_factory=list)
    """Credible set flags that exclude the set."""
    csv_set_invalid_qc_reasons: list[CredibleSetVariantFlag] = Field(default_factory=list)
    """Variant flags that exclude the whole set."""

    @field_validator('cs_invalid_qc_reasons', mode='before')
    @classmethod
    def resolve_cs_names(cls, value: Any) -> Any:
        """Resolve configured enum names before validating the credible set flags."""
        return resolve_flag_names(CredibleSetFlag, value)

    @field_validator('csv_set_invalid_qc_reasons', mode='before')
    @classmethod
    def resolve_csv_names(cls, value: Any) -> Any:
        """Resolve configured enum names before validating the variant flags."""
        return resolve_flag_names(CredibleSetVariantFlag, value)


def _flag_matching_ids(
    frame: pl.LazyFrame, key: str, ids: pl.LazyFrame, flag: CredibleSetFlag | CredibleSetVariantFlag
) -> pl.LazyFrame:
    """Flag the rows of `frame` whose `key` is among the `id` values of `ids`."""
    marked = ids.select(pl.col('id').alias(key), pl.lit(True).alias('_matched')).unique()
    return frame.join(marked, on=key, how='left').pipe(flag_rows, pl.col('_matched'), flag).drop('_matched')


def flag_across_sets(sets: pl.LazyFrame, variants: pl.LazyFrame) -> pl.LazyFrame:
    """The cross-set flags of the credible sets of every source, one row per flagged set.

    REDUNDANT_PICS_TOP_HIT, EXPLAINED_BY_SUSIE and DUPLICATED_STUDYLOCUS_ID compare sets across
    sources. Their rules come with the set QC slice for cross-set flags; until then no set is flagged.
    """
    return pl.LazyFrame(schema=cross_set_flags_schema)


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
        >>> sets = pl.LazyFrame({'chromosome': ['10', '2', 'X'], 'fineMappingMethod': ['PICS', 'SuSiE', 'SuSiE-inf']})
        >>> typed = type_valid_sets(sets).collect()
        >>> typed.schema['chromosome'] == validated_credible_set_schema['chromosome']
        True

        Chromosomes sort in genome order, and a value outside the enum raises:

        >>> typed.sort('chromosome')['chromosome'].to_list()
        ['2', '10', 'X']
        >>> type_valid_sets(pl.LazyFrame({'chromosome': ['23'], 'fineMappingMethod': ['PICS']})).collect()
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
    1. The sets with at least one variant carrying a flag of `csv_set_invalid_qc_reasons` get
       `INVALID_VARIANT_IN_SET`: one invalid member invalidates its set.
    2. The sets carrying a flag of `cs_invalid_qc_reasons`, or `INVALID_VARIANT_IN_SET`, are
       excluded. `INVALID_VARIANT_IN_SET` is structural, so it is not configurable.
    3. Every variant of an excluded set gets `CREDIBLE_SET_EXCLUDED` and is excluded with it, so
       every excluded row carries the reason it left.

    Flags are appended to `qualityControls`; a set with several invalid variants is flagged once.
    Orphan variants, row exclusion and the lead variant check come with later slices.

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
        >>> sets = pl.LazyFrame({'id': ['a', 'b', 'c'], 'qualityControls': [[], [CredibleSetFlag.IN_MHC.value], []]})
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
    sets_with_invalid_variant = variants.filter(has_excluded_flag(settings.csv_set_invalid_qc_reasons)).select(
        pl.col('credibleSetId').alias('id')
    )
    flagged_sets = _flag_matching_ids(sets, 'id', sets_with_invalid_variant, CredibleSetFlag.INVALID_VARIANT_IN_SET)
    set_split = valid_rows(flagged_sets, [*settings.cs_invalid_qc_reasons, CredibleSetFlag.INVALID_VARIANT_IN_SET])
    flagged_variants = _flag_matching_ids(
        variants, 'credibleSetId', set_split.invalid, CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED
    )
    variant_split = valid_rows(flagged_variants, [CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED])
    return set_split.valid, variant_split.valid, set_split.invalid, variant_split.invalid


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
        settings: Flag names under `cs_invalid_qc_reasons` and `csv_set_invalid_qc_reasons`.
        config: Pipeline configuration.
    """
    sources = ValidationSource.model_validate(source)
    destinations = ValidationDestination.model_validate(destination)
    options = ValidationSettings.model_validate(settings)
    cross_set_flags = flag_across_sets(
        scan_datasets(sources.credible_set), scan_datasets(sources.credible_set_variant)
    ).collect()
    logger.info('Cross-set QC flagged {} credible sets', cross_set_flags.height)
    splits = [
        validate(scan_dataset(sets), scan_dataset(variants), cross_set_flags.lazy(), options)
        for _, sets, variants in sources.pairs()
    ]
    valid_set, valid_variant, excluded_set, excluded_variant = (pl.concat(parts) for parts in zip(*splits, strict=True))
    logger.info('Validated the credible sets of {}', ', '.join(name for name, _, _ in sources.pairs()))
    pl.collect_all([
        sink_dataset(type_valid_sets(valid_set), destinations.valid_set),
        sink_dataset(valid_variant, destinations.valid_variant),
        sink_dataset(excluded_set, destinations.excluded_set),
        sink_dataset(excluded_variant, destinations.excluded_variant),
    ])
