"""Derive the train and test splits from the pinned, predefined test set.

The test set is pinned across releases rather than resampled, so models stay comparable. This
module implements only that path: gentropy's fresh hierarchical split is not ported, because the
pipeline always supplies `predefined_test_parquet_path` and a second split would defeat the point
of pinning the first.
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
    """Split the annotated gold-standard matrix against a pinned test set.

    The test partition is re-derived from the current annotated matrix using the pinned
    (study locus, gene) pairs, so its features track the release. The training partition drops
    every study locus that contains a gene which is positive anywhere in the test set.

    The row's own label is deliberately NOT consulted when finding contamination: a locus whose
    gene label flipped from positive to negative between releases is still in the test set via
    the pair join, and must stay out of training.

    Args:
        annotated: output of `gold_standard.annotate`, string or integer labels.
        predefined_test: the pinned test set; only `studyLocusId`, `geneId` and the label are used.

    Returns:
        `(train, test)` as eager DataFrames with `Int32` labels.
    """
    encoded = encode_labels(annotated)
    pinned = encode_labels(predefined_test.select('studyLocusId', 'geneId', LABEL_COLUMN))

    test_positive_genes = pinned.filter(pl.col(LABEL_COLUMN) == 1).select('geneId').unique()

    contaminated = (
        encoded.join(test_positive_genes, on='geneId', how='inner').select('studyLocusId').unique()
    )

    train = encoded.join(contaminated, on='studyLocusId', how='anti')
    test = encoded.join(pinned.select('studyLocusId', 'geneId'), on=['studyLocusId', 'geneId'], how='inner')

    return train.collect(), test.collect()


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
        'n_unique_loci': frame['studyLocusId'].n_unique(),
        'n_unique_positive_genes': positive['geneId'].n_unique(),
    }


def split_stats(
    annotated_total: int,
    predefined_total: int,
    train: pl.DataFrame,
    test: pl.DataFrame,
) -> dict[str, Any]:
    """Build the split statistics document, matching gentropy's key set.

    Args:
        annotated_total: rows in the annotated gold-standard matrix.
        predefined_total: rows in the pinned test set as staged.
        train: the training partition.
        test: the test partition.

    Returns:
        A JSON-serialisable dict.
    """
    return {
        'n_original_total': annotated_total,
        'n_original_test': predefined_total,
        'n_test_new': test.height,
        'n_lost_test': predefined_total - test.height,
        'n_train': train.height,
        'n_lost_total': annotated_total - test.height - train.height,
        'train': _set_stats(train),
        'test': _set_stats(test),
    }
