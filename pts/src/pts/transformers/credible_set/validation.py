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

from pts.schemas.credible_set import cross_set_flags_schema, validated_credible_set_schema
from pts.transformers.credible_set.enums import FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.variant_id import (
    variant_chromosome,
    variant_position,
)
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, sink_dataset
from pts.transformers.utils.quality_flags import (
    ValidationResult,
    flag_rows,
    has_excluded_flag,
    resolve_flag_names,
    valid_rows,
)


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


STRUCTURAL_FLAGS = {
    CredibleSetFlag.DUPLICATED_CREDIBLE_SET_ID,
    CredibleSetFlag.INVALID_VARIANT_IN_SET,
    CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED,
}
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


def redundant_pics_top_hits(credible_set: pl.LazyFrame, explaining_set: pl.LazyFrame) -> pl.LazyFrame:
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
        >>> redundant_pics_top_hits(credible_set, credible_set).collect()['id'].to_list()
        ['top']
    """
    top_hit = pl.col('qualityControls').list.contains(CredibleSetFlag.TOP_HIT.value)
    pics = pl.col('fineMappingMethod') == FinemappingMethod.PICS.value
    study_with_summary_statistics = explaining_set.filter(pics & ~top_hit).select('studyId').unique()
    return (
        credible_set.filter(top_hit).join(study_with_summary_statistics, on='studyId', how='semi').select('id').unique()
    )


def explained_by_susie(
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
        >>> explained_by_susie(credible_set, credible_set_variant, credible_set).collect()['id'].to_list()
        ['inside']
    """
    susie = pl.col('fineMappingMethod').is_in([method.value for method in FinemappingMethod.susie()])
    susie_region = explaining_set.filter(susie).select('studyId', 'chromosome', 'locusStart', 'locusEnd').unique()
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
        >>> flagged = flag_across_sets(credible_set, credible_set_variant, credible_set)
        >>> for row in flagged.sort('id').collect().iter_rows(named=True):
        ...     print(row['id'], sorted(CredibleSetFlag(flag).name for flag in row['qualityControls']))
        pics ['EXPLAINED_BY_SUSIE']
        top ['REDUNDANT_PICS_TOP_HIT']
    """
    flagged = [
        (redundant_pics_top_hits(credible_set, explaining_set), CredibleSetFlag.REDUNDANT_PICS_TOP_HIT),
        (explained_by_susie(credible_set, credible_set_variant, explaining_set), CredibleSetFlag.EXPLAINED_BY_SUSIE),
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


def set_keys(credible_set: pl.LazyFrame, excluded_set: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """One row per set id of a source, with what its variants are checked against.

    The chromosome and the lead of the set, and whether the set is excluded: a set whose id is
    among `excluded_set`, which is unknown until the sets are decided.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['a', 'b'], 'chromosome': ['1', '2'], 'leadVariantId': ['1_1_A_T', '2_5_A_T']
        ... })
        >>> set_keys(credible_set, pl.LazyFrame({'id': ['b']})).sort('credibleSetId').collect().rows()
        [('a', '1', '1_1_A_T', True, False), ('b', '2', '2_5_A_T', True, True)]
    """
    key = credible_set.select(
        pl.col('id').alias('credibleSetId'),
        pl.col('chromosome').alias('_set_chromosome'),
        pl.col('leadVariantId').alias('_set_lead'),
        pl.lit(True).alias('_known'),
    ).unique()
    if excluded_set is None:
        return key.with_columns(pl.lit(False).alias('_set_excluded'))
    excluded = excluded_set.select(pl.col('id').alias('credibleSetId'), pl.lit(True).alias('_set_excluded')).unique()
    return key.join(excluded, on='credibleSetId', how='left').with_columns(pl.col('_set_excluded').fill_null(False))


def _pair_with_sets(
    credible_set_variant: pl.LazyFrame, set_key: pl.LazyFrame, settings: ValidationSettings
) -> pl.LazyFrame:
    """Join each variant to its set once and flag what the pairing shows.

    - ORPHAN_VARIANT: the variant's set is not among the sets of its source.
    - VARIANT_CHROMOSOME_MISMATCH: the variant is on another chromosome than its set; an orphan
      has no set to compare with, and is not flagged.

    It adds `_kept`, whether the row survives `csv_row_invalid_qc_reasons`, `_lead`, whether the
    row is its set's lead, and `_set_excluded`, whether its set is excluded. The joined rows are
    cached: the outputs that branch off them share one read and one join, which polars does not
    share across the column changes that follow.

    Examples:
        >>> credible_set = pl.LazyFrame({'id': ['a'], 'chromosome': ['1'], 'leadVariantId': ['1_1_A_T']})
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'a', 'gone'],
        ...     'variantId': ['1_1_A_T', '2_5_A_T', '1_9_A_T'],
        ...     'qualityControls': [[], [], []],
        ... })
        >>> settings = ValidationSettings(csv_row_invalid_qc_reasons=['ORPHAN_VARIANT'])
        >>> set_key = set_keys(credible_set)
        >>> flagged = _pair_with_sets(credible_set_variant, set_key, settings).sort('variantId').collect()
        >>> for row in flagged.iter_rows(named=True):
        ...     flag_name = sorted(CredibleSetVariantFlag(flag).name for flag in row['qualityControls'])
        ...     print(row['variantId'], flag_name, row['_kept'], row['_lead'])
        1_1_A_T [] True True
        1_9_A_T ['ORPHAN_VARIANT'] False False
        2_5_A_T ['VARIANT_CHROMOSOME_MISMATCH'] True False
    """
    mismatch = variant_chromosome(pl.col('variantId')) != pl.col('_set_chromosome')
    return (
        credible_set_variant
        .join(set_key, on='credibleSetId', how='left')
        .cache()
        .pipe(flag_rows, pl.col('_known').is_null(), CredibleSetVariantFlag.ORPHAN_VARIANT)
        .pipe(flag_rows, mismatch, CredibleSetVariantFlag.VARIANT_CHROMOSOME_MISMATCH)
        .with_columns(
            (~has_excluded_flag(settings.csv_row_invalid_qc_reasons)).alias('_kept'),
            (pl.col('variantId') == pl.col('_set_lead')).fill_null(False).alias('_lead'),
            pl.col('_set_excluded').fill_null(False),
        )
        .drop('_known', '_set_chromosome', '_set_lead')
    )


def set_facts(flagged_variant: pl.LazyFrame, settings: ValidationSettings) -> pl.LazyFrame:
    """What the sets of a source need from their kept variants, one row per set id.

    `_size`, the number of kept variants; `_invalid_variant`, whether one of them carries a flag
    of `csv_set_invalid_qc_reasons`; `_lead`, whether one of them is the set's lead.

    Examples:
        >>> flagged_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'a', 'b', 'b'],
        ...     'qualityControls': [[], [CredibleSetVariantFlag.INVALID_VARIANT_ID.value], [], []],
        ...     '_kept': [True, True, True, False],
        ...     '_lead': [True, False, False, True],
        ... })
        >>> settings = ValidationSettings(csv_set_invalid_qc_reasons=['INVALID_VARIANT_ID'])
        >>> set_facts(flagged_variant, settings).sort('id').collect().rows()
        [('a', 2, True, True), ('b', 1, False, False)]
    """
    return (
        flagged_variant
        .filter('_kept')
        .group_by(pl.col('credibleSetId').alias('id'))
        .agg(
            pl.len().alias('_size'),
            has_excluded_flag(settings.csv_set_invalid_qc_reasons).any().alias('_invalid_variant'),
            pl.col('_lead').any(),
        )
    )


def decide_sets(
    credible_set: pl.LazyFrame, set_fact: pl.LazyFrame, cross_set_flag: pl.LazyFrame, settings: ValidationSettings
) -> ValidationResult:
    """Flag the sets of a source from the cross-set flags and their variants' facts, and split them.

    DUPLICATED_CREDIBLE_SET_ID on every set whose id the source repeats: the variants of the copies
    share that id, so none of them can be told to belong to one copy. INVALID_VARIANT_IN_SET on a
    set with a kept variant flagged in `csv_set_invalid_qc_reasons`, LEAD_VARIANT_MISSING on a set
    without its lead among its kept variants. The sets with a flag of `cs_invalid_qc_reasons`,
    DUPLICATED_CREDIBLE_SET_ID or INVALID_VARIANT_IN_SET are excluded.
    """
    fact = set_fact.select('id', '_invalid_variant', pl.col('_lead').alias('_has_lead'))
    flagged = (
        add_cross_set_flags(credible_set, cross_set_flag)
        .pipe(flag_rows, pl.len().over('id') > 1, CredibleSetFlag.DUPLICATED_CREDIBLE_SET_ID)
        .join(fact, on='id', how='left')
        .pipe(flag_rows, pl.col('_invalid_variant').fill_null(False), CredibleSetFlag.INVALID_VARIANT_IN_SET)
        .pipe(flag_rows, ~pl.col('_has_lead').fill_null(False), CredibleSetFlag.LEAD_VARIANT_MISSING)
        .drop('_invalid_variant', '_has_lead')
    )
    structural = [CredibleSetFlag.DUPLICATED_CREDIBLE_SET_ID, CredibleSetFlag.INVALID_VARIANT_IN_SET]
    return valid_rows(flagged, [*settings.cs_invalid_qc_reasons, *structural])


def split_variants(flagged_variant: pl.LazyFrame) -> ValidationResult:
    """Split variants into valid and excluded; the kept rows of excluded sets get CREDIBLE_SET_EXCLUDED.

    Valid variants are the kept rows of sets that are not excluded.
    """
    set_excluded = pl.col('_kept') & pl.col('_set_excluded')
    flagged = flagged_variant.pipe(flag_rows, set_excluded, CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED)
    valid = pl.col('_kept') & ~pl.col('_set_excluded')
    helper = ['_kept', '_lead', '_set_excluded']
    return ValidationResult(valid=flagged.filter(valid).drop(helper), invalid=flagged.filter(~valid).drop(helper))


def flag_duplicated_ids(
    decided: list[tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]], settings: ValidationSettings
) -> list[tuple[pl.LazyFrame, pl.LazyFrame]]:
    """Flag DUPLICATED_CREDIBLE_SET_ID_ACROSS_SOURCES on the copies of an id that another source keeps.

    Within a source an id is unique by then, as `decide_sets` excludes every copy of a repeated one.
    Across sources the variants of a copy are its own source's, so one copy can be kept. As
    gentropy's credible set validation does, only the valid sets are compared, and the copy with
    the fewest valid variants is kept. Ties go to the copy of the earlier source, then to the
    smaller hash of its row, so the kept copy does not depend on row order. When the flag is in
    `cs_invalid_qc_reasons`, the other copies are excluded. The ranking is collected once, and
    each source takes its sets from it.

    Args:
        decided: The valid sets, excluded sets and set facts (`set_facts`) of each source.
        settings: The flags that exclude a set.

    Returns:
        The valid and excluded sets of each source, in the same order.

    Examples:
        >>> def source(size):
        ...     quality_control = pl.Series([[]], dtype=pl.List(pl.String))
        ...     valid = pl.LazyFrame({'id': ['dup'], 'qualityControls': quality_control})
        ...     return valid, valid.clear(), pl.LazyFrame({'id': ['dup'], '_size': [size]})
        >>> settings = ValidationSettings(cs_invalid_qc_reasons=['DUPLICATED_CREDIBLE_SET_ID_ACROSS_SOURCES'])
        >>> (larger_valid, larger_excluded), (smaller_valid, _) = flag_duplicated_ids([source(3), source(2)], settings)
        >>> larger_valid.collect().height, larger_excluded.collect().height, smaller_valid.collect().height
        (0, 1, 1)
    """
    duplicated = CredibleSetFlag.DUPLICATED_CREDIBLE_SET_ID_ACROSS_SOURCES
    source = pl.col('_source')
    valid_set = pl.concat([
        valid.with_columns(pl.lit(index, pl.UInt32).alias('_source')) for index, (valid, _, _) in enumerate(decided)
    ])
    size = pl.concat([
        set_fact.select('id', pl.col('_size').cast(pl.UInt32)).with_columns(pl.lit(index, pl.UInt32).alias('_source'))
        for index, (_, _, set_fact) in enumerate(decided)
    ])
    ranked = (
        valid_set
        .join(size, on=['_source', 'id'], how='left')
        .with_columns(pl.struct(pl.exclude('_source', '_size')).hash(seed=0).alias('_hash'))
        .sort('id', '_size', '_source', '_hash', nulls_last=True)
        .pipe(flag_rows, source != source.first().over('id'), duplicated)
        .drop('_size', '_hash')
        .collect()
    )
    excluding = [duplicated] if duplicated in settings.cs_invalid_qc_reasons else []
    flagged_split = []
    for index, (_, excluded, _) in enumerate(decided):
        set_split = valid_rows(ranked.lazy().filter(source == index).drop('_source'), excluding)
        flagged_split.append((set_split.valid, pl.concat([excluded, set_split.invalid])))
    return flagged_split


def validation(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Split the QC'd credible sets and variants of every source into valid and excluded rows, and write them.

    Variant rows first:

    1. Flag `ORPHAN_VARIANT` on a variant whose set is not among its source's sets, and
       `VARIANT_CHROMOSOME_MISMATCH` on a variant on another chromosome than its set.
    2. Exclude the variants with a flag of `csv_row_invalid_qc_reasons` (by default
       EXCEEDS_COVERAGE_95 and ORPHAN_VARIANT). Only the row leaves; its set stays.

    Then whole sets:

    3. Add the cross-set flags of every source (`flag_across_sets`: REDUNDANT_PICS_TOP_HIT,
       EXPLAINED_BY_SUSIE). Only a set that its own flags (steps 4 to 7) leave valid explains
       another: one that is not published cannot replace it.
    4. Flag `INVALID_VARIANT_IN_SET` on a set with any remaining variant flagged in
       `csv_set_invalid_qc_reasons`; a flagged variant removed by step 2 does not count.
    5. Flag `LEAD_VARIANT_MISSING` on a set whose lead variant is not among its remaining
       variants, either because it was never staged or because step 2 removed it.
    6. Flag `DUPLICATED_CREDIBLE_SET_ID` on every copy of an id its source repeats: their variants
       share the id, so none can be kept.
    7. Exclude the sets with a flag of `cs_invalid_qc_reasons`, `DUPLICATED_CREDIBLE_SET_ID` or
       `INVALID_VARIANT_IN_SET`.
    8. Among the sets left valid in every source, flag `DUPLICATED_CREDIBLE_SET_ID_ACROSS_SOURCES`
       on all but one copy of an id (`flag_duplicated_ids`).
    9. The remaining variants of an excluded set leave with it, and get `CREDIBLE_SET_EXCLUDED`.

    Every excluded row carries the flag that explains why it left. `DUPLICATED_CREDIBLE_SET_ID`,
    `INVALID_VARIANT_IN_SET` and `CREDIBLE_SET_EXCLUDED` are structural, so they are not
    configurable.

    The sets of every source are read once and kept in memory. The variants are read three times:
    for the cross-set QC, for the facts the sets are decided on (`set_facts`), and, with the
    decision, to split them (`split_variants`). Every join is built on the sets, never on the
    variants.

    Args:
        source: The QC'd sets of every source under `credible_set` and their variants under
            `credible_set_variant`, in the same source order.
        destination: Paths under `valid_set`, `valid_variant`, `excluded_set` and `excluded_variant`.
        settings: Flag names under `cs_invalid_qc_reasons`, `csv_set_invalid_qc_reasons` and
            `csv_row_invalid_qc_reasons`.
        config: Pipeline configuration.
    """
    sources = ValidationSource.model_validate(source)
    destinations = ValidationDestination.model_validate(destination)
    options = ValidationSettings.model_validate(settings)
    source_pair = pair_sources(sources, config)
    source_input = [(scan_dataset(set_path), scan_dataset(variant_path)) for _, set_path, variant_path in source_pair]
    set_frames = pl.collect_all([credible_set for credible_set, _ in source_input])
    # 1. the facts every set is decided on, from one pass over each source's variants
    set_fact = pl.collect_all([
        set_facts(_pair_with_sets(credible_set_variant, set_keys(credible_set.lazy()), options), options)
        for credible_set, (_, credible_set_variant) in zip(set_frames, source_input, strict=True)
    ])
    # 2. the cross-set flags, explained only by the sets that survive their own flags
    no_cross_set_flag = pl.LazyFrame(schema=cross_set_flags_schema)
    explaining_set = pl.concat(
        [
            decide_sets(credible_set.lazy(), fact.lazy(), no_cross_set_flag, options).valid
            for credible_set, fact in zip(set_frames, set_fact, strict=True)
        ],
        how='diagonal_relaxed',
    )
    cross_set_flag = flag_across_sets(
        pl.concat([credible_set.lazy() for credible_set in set_frames], how='diagonal_relaxed'),
        pl.concat([credible_set_variant for _, credible_set_variant in source_input], how='diagonal_relaxed'),
        explaining_set,
    ).collect()
    logger.info('Cross-set QC flagged {} credible sets', cross_set_flag.height)
    # 3. the sets of every source, decided in memory
    decided = []
    for credible_set, fact in zip(set_frames, set_fact, strict=True):
        set_split = decide_sets(credible_set.lazy(), fact.lazy(), cross_set_flag.lazy(), options)
        decided.append((set_split.valid, set_split.invalid, fact.lazy()))
    deduplicated = [pl.collect_all(frames) for frames in flag_duplicated_ids(decided, options)]
    # 4. the variants of every source, split by the decision on their set in one more pass
    variant_split = [
        split_variants(_pair_with_sets(credible_set_variant, set_keys(credible_set.lazy(), excluded.lazy()), options))
        for credible_set, (_, credible_set_variant), (_, excluded) in zip(
            set_frames, source_input, deduplicated, strict=True
        )
    ]
    valid_set = pl.concat([valid.lazy() for valid, _ in deduplicated])
    excluded_set = pl.concat([excluded.lazy() for _, excluded in deduplicated])
    valid_variant = pl.concat([split.valid for split in variant_split])
    excluded_variant = pl.concat([split.invalid for split in variant_split])
    logger.info('Validated the credible sets of {}', ', '.join(source_name for source_name, _, _ in source_pair))
    pl.collect_all([
        sink_dataset(type_valid_sets(valid_set), destinations.valid_set),
        sink_dataset(valid_variant, destinations.valid_variant),
        sink_dataset(excluded_set, destinations.excluded_set),
        sink_dataset(excluded_variant, destinations.excluded_variant),
    ])


def validation_check(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Check that every input row of `validation` was written, either valid or excluded.

    Runs after `validation` has written its datasets, on the same arguments. The written datasets
    hold every source together, so the rows of every source are counted together; the counts of
    parquet come from the file footers.

    Raises:
        ValueError: if the written sets or variants are not exactly the input ones.
    """
    sources = ValidationSource.model_validate(source)
    destinations = ValidationDestination.model_validate(destination)
    error_message = []
    for dataset, input_path, valid_path, excluded_path in [
        ('sets', sources.credible_set, destinations.valid_set, destinations.excluded_set),
        ('variants', sources.credible_set_variant, destinations.valid_variant, destinations.excluded_variant),
    ]:
        rows_in, valid, excluded = (
            frame.select(pl.len()).collect().item()
            for frame in (scan_datasets(input_path), scan_dataset(valid_path), scan_dataset(excluded_path))
        )
        if rows_in != valid + excluded:
            error_message.append(f'{rows_in} input {dataset} != {valid} valid + {excluded} excluded {dataset}')
    if error_message:
        raise ValueError('credible set validation failed its integrity checks:\n' + '\n'.join(error_message))
