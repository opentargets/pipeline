"""Parse the curated L2G gold standard and annotate it with feature-matrix rows.

gentropy's `_parse_gold_standard` has three branches. On the curated file this pipeline
actually consumes -- `gs://otar001-core/l2g/goldStandard/<version>` -- only the second fires:
the file carries one unexpected column, `diseaseIds`, which is dropped. The OTG-curation branch
needs `find_overlaps`, the variant index and the PPI dataset, and has never run on this data, so
it is NOT ported. A file in that format must fail loudly rather than be mis-parsed.
"""

from collections.abc import Sequence

import polars as pl

from pts.transformers.l2g.features import impute_and_cast

CURATED_COLUMNS: tuple[str, ...] = (
    'studyLocusId',
    'variantId',
    'studyId',
    'geneId',
    'goldStandardSet',
)
"""The columns `L2GGoldStandard` declares as mandatory, in schema order."""

OTG_CURATION_COLUMNS: frozenset[str] = frozenset({
    'association_info',
    'gold_standard_info',
    'metadata',
    'sentinel_variant',
    'trait_info',
})
"""Columns that identify the legacy OTG curation format, which is not supported here."""

LABEL_COLUMN = 'goldStandardSet'


def parse_gold_standard(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Reduce the curated gold standard to its mandatory columns.

    Args:
        frame: the gold-standard file as read, including any extra columns.

    Returns:
        LazyFrame of `CURATED_COLUMNS`.

    Raises:
        ValueError: if the frame is in the unported OTG curation format, or is missing a
            mandatory column.
    """
    columns = set(frame.collect_schema().names())

    if present := OTG_CURATION_COLUMNS & columns:
        msg = (
            f'gold standard is in the OTG curation format (found {sorted(present)}), which is not '
            'supported. Port `L2GGoldStandard.from_otg_curation` before using such a file.'
        )
        raise ValueError(msg)

    if missing := [name for name in CURATED_COLUMNS if name not in columns]:
        msg = f'gold standard is missing mandatory columns {missing}'
        raise ValueError(msg)

    return frame.select(CURATED_COLUMNS)


def annotate(
    feature_matrix: pl.LazyFrame,
    credible_set: pl.LazyFrame,
    gold_standard: pl.LazyFrame,
    features: Sequence[str],
) -> pl.LazyFrame:
    """Build the feature matrix restricted to gold-standard (study locus, gene) pairs.

    Mirrors `L2GGoldStandard.build_feature_matrix`: the credible set supplies `variantId` and
    `studyId` for each study locus, the gold standard is joined on all three keys, non
    protein-coding rows are dropped, and the result is deduplicated and imputed.

    Args:
        feature_matrix: the full feature matrix, including `isProteinCoding`.
        credible_set: credible sets, supplying `studyLocusId`, `variantId` and `studyId`.
        gold_standard: output of `parse_gold_standard`.
        features: feature names, in fitted order.

    Returns:
        LazyFrame of `studyLocusId, geneId, goldStandardSet` followed by the features.
    """
    keyed = feature_matrix.join(
        credible_set.select('studyLocusId', 'variantId', 'studyId'),
        on='studyLocusId',
        how='left',
    )

    matched = (
        # The gold standard's own `studyLocusId` is dropped before the join, as gentropy does
        # (`L2GGoldStandard.build_feature_matrix` broadcasts `self.df.drop("studyLocusId", ...)`).
        # Kept, it would arrive as `studyLocusId_right` and the `unique` below would see it: two
        # curated rows agreeing on (studyId, variantId, geneId, goldStandardSet) but assigned to
        # different credible sets would survive as two identical-looking training rows, inflating
        # the fit, the split and every split statistic. Today's curation file has no such pair.
        keyed.join(gold_standard.drop('studyLocusId'), on=['studyId', 'variantId', 'geneId'], how='inner')
        .filter(pl.col('isProteinCoding') == 1)
        .drop('studyId', 'variantId', 'isProteinCoding')
        .unique(maintain_order=True)
    )

    return impute_and_cast(matched, features, keep=[LABEL_COLUMN])
