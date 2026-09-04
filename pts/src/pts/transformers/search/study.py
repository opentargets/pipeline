"""The study search index."""

from __future__ import annotations

import polars as pl

from pts.transformers.search.helpers import as_list, flatten_cat, search_index
from pts.transformers.search.ranks import composite_global_rank


def credible_set_counts(credible_sets: pl.LazyFrame) -> pl.LazyFrame:
    """Count credible sets per study.

    Args:
        credible_sets: the `credible_set` dataset; only `studyId` is read.

    Returns:
        LazyFrame with `studyId` and a `Float64` `credibleSetCount`.
    """
    return (
        credible_sets.select('studyId').group_by('studyId').agg(pl.len().cast(pl.Float64).alias('credibleSetCount'))
    )


def build_study_index(
    studies: pl.LazyFrame,
    targets: pl.LazyFrame,
    credible_sets: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build the search index for studies.

    Ports `Transformers.Implicits.setIdAndSelectFromStudies`.

    The multiplier scales linearly from 2.0 for the best-ranked study to 1.0 for the worst,
    over a GLOBAL rank on `(credibleSetCount desc, nSamples desc)`. Computing `max_rank`
    forces a materialisation -- there is no way around it, the value is needed as a scalar.

    Args:
        studies: the `study` dataset.
        targets: frame with `targetId` and `approvedSymbol`.
        credible_sets: the `credible_set` dataset.

    Returns:
        Search index LazyFrame, one row per study.
    """
    ranked = composite_global_rank(
        studies.rename({'geneId': 'targetId'})
        .join(targets.select('targetId', 'approvedSymbol'), on='targetId', how='left')
        .join(credible_set_counts(credible_sets), on='studyId', how='left'),
        ['credibleSetCount', 'nSamples'],
    )

    max_rank = ranked.select(pl.col('rank').max()).collect().item()
    if max_rank is None or max_rank <= 1:
        # A single study (or none) would divide by zero on `max_rank - 1`.
        multiplier = pl.lit(1.0, dtype=pl.Float64)
    else:
        multiplier = 1.0 + (pl.lit(float(max_rank)) - pl.col('rank')) / pl.lit(float(max_rank - 1))

    terms = flatten_cat(
        as_list(pl.col('traitFromSource')),
        pl.col('diseaseIds'),
        as_list(pl.col('approvedSymbol')),
        as_list(pl.col('targetId')),
    )
    identifiers = flatten_cat(
        as_list(pl.col('studyId')),
        as_list(pl.col('pubmedId')),
        as_list(pl.col('publicationFirstAuthor')),
    )

    return search_index(
        ranked,
        id_col=pl.col('studyId'),
        name_col=pl.col('studyId'),
        entity='study',
        category_col=as_list(pl.lit('study')),
        keywords_col=identifiers,
        prefixes_col=identifiers,
        ngrams_col=flatten_cat(as_list(pl.col('studyId'))),
        terms_col=terms,
        terms25_col=terms,
        terms5_col=terms,
        multiplier_col=multiplier,
    )
