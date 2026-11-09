"""Parse the L2G gold standard and annotate it with feature-matrix rows.

The gold standard is the L2G training set (`intermediate/l2g_training_set`): one row per
credible set and gene, labelled `positive` or `negative`.
"""

from collections.abc import Sequence

import polars as pl

from pts.transformers.l2g.features import impute_and_cast

CURATED_COLUMNS: tuple[str, ...] = (
    'credibleSetId',
    'variantId',
    'studyId',
    'geneId',
    'goldStandardSet',
)
"""The columns `L2GGoldStandard` declares as mandatory, in schema order.

Deliberately NOT including `traitFromSourceMappedId` or `sources`, which gentropy's schema also
allows and the training set does not carry. A future file that did carry it would have it dropped here,
and two curated rows differing only in that column would then collapse in `annotate`'s `unique`.
gentropy's `n_unique_positive_gene_disease_pairs` split statistic exists precisely to count that
column, and is absent from `split_stats` for the same reason -- add the column here and add the
statistic there together, or neither.
"""

LABEL_COLUMN = 'goldStandardSet'


def parse_gold_standard(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Reduce the curated gold standard to its mandatory columns.

    Args:
        frame: the gold-standard file as read, including any extra columns.

    Returns:
        LazyFrame of `CURATED_COLUMNS`.

    Raises:
        ValueError: if the frame is missing a mandatory column.
    """
    columns = set(frame.collect_schema().names())

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
    """Build the feature matrix restricted to gold-standard (credible set, gene) pairs.

    Mirrors gentropy v4 `L2GGoldStandard.build_feature_matrix`: the credible set supplies the lead
    variant and `studyId` for each credible set, the gold standard is joined on study, variant and
    gene, non protein-coding rows are dropped, and the result is deduplicated and imputed.

    Args:
        feature_matrix: the full feature matrix, keyed by `credibleSetId`, including `isProteinCoding`.
        credible_set: v4 credible sets, supplying `id`, `leadVariantId` and `studyId`.
        gold_standard: output of `parse_gold_standard`.
        features: feature names, in fitted order.

    Returns:
        LazyFrame of `credibleSetId, geneId, goldStandardSet` followed by the features.
    """
    keyed = feature_matrix.join(
        credible_set.select(
            pl.col('id').alias('credibleSetId'),
            pl.col('leadVariantId').alias('variantId'),
            'studyId',
        ),
        on='credibleSetId',
        how='left',
    )

    matched = (
        # drop the gold standard's own credibleSetId before the join, as gentropy does
        keyed.join(gold_standard.drop('credibleSetId'), on=['studyId', 'variantId', 'geneId'], how='inner')
        .filter(pl.col('isProteinCoding') == 1)
        .drop('studyId', 'variantId', 'isProteinCoding')
        .unique(maintain_order=True)
    )

    return impute_and_cast(matched, features, keep=[LABEL_COLUMN])
