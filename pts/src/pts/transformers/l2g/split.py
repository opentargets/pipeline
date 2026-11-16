"""Split the L2G training rows into train and test sets.

We expect a predefined set of test genes, which limits the space of possible test rows. Only the
positive genes of that set are used, so it stays valid when credible set ids change.
"""

from typing import Any

import polars as pl

from pts.transformers.l2g.gold_standard import LABEL_COLUMN

POSITIVE = 'positive'
NEGATIVE = 'negative'


def encode_labels(frame: pl.LazyFrame, column: str = LABEL_COLUMN) -> pl.LazyFrame:
    """Encode `negative` as 0 and `positive` as 1, tolerating already-encoded values.

    Args:
        frame: frame carrying the label column.
        column: name of the label column.

    Returns:
        The frame with `column` as `Int32`.
    """
    as_string = pl.col(column).cast(pl.String)
    return frame.with_columns(
        pl.when(as_string == NEGATIVE)
        .then(0)
        .when(as_string == POSITIVE)
        .then(1)
        .otherwise(as_string.cast(pl.Int32, strict=False))
        .cast(pl.Int32)
        .alias(column)
    )


def derive_splits(
    annotated: pl.LazyFrame,
    predefined_test: pl.LazyFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the annotated gold-standard matrix on the positive genes of a pinned test set.

    Only the positive genes of the predefined test set are used; its credible sets, negatives and
    features are rebuilt from the current matrix, so the pinned set does not need this release's ids.

    A credible set goes to the test set when all its positives are test genes, and to the training
    set when it contains no test gene at all. Every other credible set is dropped: those with
    positives on both sides, and those whose only test genes are negatives.

    Args:
        annotated: output of `gold_standard.annotate`, string or integer labels.
        predefined_test: the pinned test set; only `geneId` and the label are used.

    Returns:
        `(train, test)` as eager DataFrames with `Int32` labels.
    """
    encoded = encode_labels(annotated)
    test_genes = (
        encode_labels(predefined_test.select('geneId', LABEL_COLUMN))
        .filter(pl.col(LABEL_COLUMN) == 1)
        .select('geneId')
        .unique()
        .with_columns(pl.lit(True).alias('_is_test_gene'))
    )

    is_positive = pl.col(LABEL_COLUMN) == 1
    is_test_gene = pl.col('_is_test_gene')
    loci = (
        encoded.join(test_genes, on='geneId', how='left')
        .with_columns(is_test_gene.fill_null(False))
        .group_by('credibleSetId')
        .agg(
            is_test_gene.any().alias('has_test_gene'),
            (is_positive & is_test_gene).any().alias('has_test_positive'),
            (is_positive & ~is_test_gene).any().alias('has_train_positive'),
        )
    )
    train_loci = loci.filter(~pl.col('has_test_gene')).select('credibleSetId')
    test_loci = loci.filter(pl.col('has_test_positive') & ~pl.col('has_train_positive')).select('credibleSetId')

    train = encoded.join(train_loci, on='credibleSetId', how='semi')
    test = encoded.join(test_loci, on='credibleSetId', how='semi')
    return train.collect(), test.collect()


def count_test_genes(predefined_test: pl.LazyFrame) -> int:
    """Count the distinct positive genes of the predefined test set.

    Args:
        predefined_test: the pinned test set.

    Returns:
        The number of genes `derive_splits` splits on.
    """
    return (
        encode_labels(predefined_test.select('geneId', LABEL_COLUMN))
        .filter(pl.col(LABEL_COLUMN) == 1)
        .select(pl.col('geneId').n_unique())
        .collect()
        .item()
    )


def _set_stats(frame: pl.DataFrame) -> dict[str, int]:
    """Descriptive statistics for one partition.

    Args:
        frame: a train or test partition with integer labels.

    Returns:
        Counts of positives, negatives, distinct loci and distinct positive genes.
    """
    positive = frame.filter(pl.col(LABEL_COLUMN) == 1)
    return {
        'n_positive': positive.height,
        'n_negative': frame.filter(pl.col(LABEL_COLUMN) == 0).height,
        'n_unique_loci': frame['credibleSetId'].n_unique(),
        'n_unique_positive_genes': positive['geneId'].n_unique(),
    }


def split_stats(
    annotated_total: int,
    predefined_test_genes: int,
    train: pl.DataFrame,
    test: pl.DataFrame,
) -> dict[str, Any]:
    """Build the split statistics document, matching gentropy v4's key set for a predefined split.

    Args:
        annotated_total: rows in the annotated gold-standard matrix.
        predefined_test_genes: distinct positive genes in the pinned test set.
        train: the training partition.
        test: the test partition.

    Returns:
        A JSON-serialisable dict.
    """
    return {
        'n_original_total': annotated_total,
        'n_predefined_test_genes': predefined_test_genes,
        'n_train': train.height,
        'n_test': test.height,
        'n_lost_total': annotated_total - train.height - test.height,
        'train': _set_stats(train),
        'test': _set_stats(test),
    }
