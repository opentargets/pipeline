"""The target search index."""

from __future__ import annotations

import polars as pl

from pts.transformers.search.helpers import (
    EMPTY_LIST,
    as_list,
    flatten_cat,
    list_struct_field,
    search_index,
    tier,
)
from pts.transformers.search.ranks import partitioned_rank
from pts.transformers.search.variant import variant_labels

_TOP50 = 50
_TOP25 = 25
_TOP5 = 5


def _hgnc_identifiers() -> pl.Expr:
    """The target's HGNC xrefs, each prefixed with `HGNC:`.

    The pyspark job used `explode_outer` here and inner-joined the result back onto the target
    frame, so a target with two HGNC xrefs would have produced TWO search documents. That never
    fires on current data -- every target has 0 or 1 HGNC xref -- so collecting them into one
    list instead is zero-diff today, and makes the fan-out structurally impossible.
    """
    return pl.col('dbXrefs').list.eval(
        pl.element()
        .filter(pl.element().struct.field('source') == 'HGNC')
        .struct.field('id')
        .pipe(lambda ids: pl.concat_str([pl.lit('HGNC:'), ids]))
    )


def variant_labels_by_target(variants: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate variant labels per target, keeping the 50 closest transcripts.

    The labels are a pure function of the variant row, so they are computed ONCE over the 8.17M
    variants and joined back after the cut, rather than carried through the 290M-row explode of
    `transcriptConsequences`. That is the difference between a ~12 GB intermediate and a 40 GB+
    one, and it is exactly equivalent.

    `transcriptScore` is null whenever either factor is null, and spark ranks those FIRST
    (`ASC NULLS FIRST`), so they survive the cut -- `partitioned_rank` reproduces that.

    Args:
        variants: frame with `variantId`, `transcriptConsequences` and the label source columns.

    Returns:
        LazyFrame with `targetId`, `variant_labels`, `variant_labels_25`, `variant_labels_5`.
    """
    consequence = pl.col('transcriptConsequences')
    ranked = (
        variants.select('variantId', 'transcriptConsequences')
        # spark's `explode` drops null AND empty arrays; polars' would emit a null row
        .filter(consequence.list.len() > 0)
        .explode('transcriptConsequences', empty_as_null=False)
        .select(
            'variantId',
            consequence.struct.field('targetId').alias('targetId'),
            (
                (consequence.struct.field('consequenceScore').cast(pl.Float64) + 1)
                * consequence.struct.field('distanceFromFootprint').cast(pl.Float64)
            ).alias('transcriptScore'),
        )
        .with_columns(
            partitioned_rank(pl.col('transcriptScore'), by='targetId', descending=False).alias('variantTargetRank')
        )
        .filter(pl.col('variantTargetRank') <= _TOP50)
    )

    return (
        ranked.join(variant_labels(variants), on='variantId', how='inner')
        .group_by('targetId')
        .agg(
            tier('variant_labels', 'variantTargetRank', _TOP50).alias('variant_labels'),
            tier('variant_labels', 'variantTargetRank', _TOP25).alias('variant_labels_25'),
            tier('variant_labels', 'variantTargetRank', _TOP5).alias('variant_labels_5'),
        )
    )


def build_target_index(
    targets: pl.LazyFrame,
    associations: pl.LazyFrame,
    d_lut: pl.LazyFrame,
    variants: pl.LazyFrame,
    scored_drugs: pl.LazyFrame,
    dr_lut: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build the search index for targets.

    Ports `Transformers.Implicits.setIdAndSelectFromTargets`.

    Args:
        targets: target frame.
        associations: output of `association_scores`.
        d_lut: `diseaseId` and `disease_labels`.
        variants: variant frame.
        scored_drugs: output of `scored_drug_associations`.
        dr_lut: `drugId` and `drug_labels`.

    Returns:
        Search index LazyFrame, one row per target.
    """
    drug_by_association = (
        scored_drugs.join(dr_lut, on='drugId', how='inner')
        .group_by('associationId')
        .agg(
            pl.col('drug_labels')
            .drop_nulls()
            .list.explode(keep_nulls=False, empty_as_null=False)
            .unique(maintain_order=True)
            .alias('drug_labels')
        )
    )

    ranked = (
        associations.join(drug_by_association, on='associationId', how='left')
        .with_columns(partitioned_rank(pl.col('score'), by='targetId', descending=True).alias('rank'))
        .filter(pl.col('rank') <= _TOP50)
        .join(d_lut, on='diseaseId', how='inner')
        .group_by('targetId')
        .agg(
            tier('disease_labels', 'rank', _TOP50).alias('disease_labels'),
            tier('disease_labels', 'rank', _TOP25).alias('disease_labels_25'),
            tier('disease_labels', 'rank', _TOP5).alias('disease_labels_5'),
            tier('drug_labels', 'rank', _TOP50).alias('drug_labels'),
            tier('drug_labels', 'rank', _TOP25).alias('drug_labels_25'),
            tier('drug_labels', 'rank', _TOP5).alias('drug_labels_5'),
            pl.col('score').mean().alias('target_relevance'),
        )
    )

    # Variant labels are joined onto the association aggregate BEFORE that aggregate is joined
    # onto targets -- not onto targets directly. `ranked` has a row only for targets that appear
    # in an association, so a target with variants but no association must NOT receive variant
    # labels; that is what the pyspark original's `assocs_with_labels.join(variant_labels_df,
    # ...)` followed by `targets.join(assocs_with_variants, ...)` chain does, and what this
    # mirrors.
    ranked_with_variants = ranked.join(variant_labels_by_target(variants), on='targetId', how='left')

    frame = (
        targets.with_columns(_hgnc_identifiers().alias('hgncIds'))
        .join(ranked_with_variants, on='targetId', how='left')
        .with_columns(
            pl.col('disease_labels').fill_null(EMPTY_LIST),
            pl.col('disease_labels_25').fill_null(EMPTY_LIST),
            pl.col('disease_labels_5').fill_null(EMPTY_LIST),
            pl.col('variant_labels').fill_null(EMPTY_LIST),
            pl.col('variant_labels_25').fill_null(EMPTY_LIST),
            pl.col('variant_labels_5').fill_null(EMPTY_LIST),
            pl.col('drug_labels').fill_null(EMPTY_LIST),
            pl.col('drug_labels_25').fill_null(EMPTY_LIST),
            pl.col('drug_labels_5').fill_null(EMPTY_LIST),
        )
    )

    return search_index(
        frame,
        id_col=pl.col('targetId'),
        name_col=pl.col('approvedSymbol'),
        description_col=pl.col('approvedName'),
        entity='target',
        category_col=as_list(pl.col('biotype')),
        keywords_col=flatten_cat(
            list_struct_field('synonyms', 'label'),
            list_struct_field('proteinIds', 'id'),
            as_list(pl.col('approvedName')),
            as_list(pl.col('approvedSymbol')),
            pl.col('hgncIds'),
            as_list(pl.col('targetId')),
        ),
        prefixes_col=flatten_cat(
            list_struct_field('synonyms', 'label'),
            list_struct_field('proteinIds', 'id'),
            as_list(pl.col('approvedName')),
            as_list(pl.col('approvedSymbol')),
        ),
        ngrams_col=flatten_cat(
            list_struct_field('proteinIds', 'id'),
            list_struct_field('synonyms', 'label'),
            as_list(pl.col('approvedName')),
            as_list(pl.col('approvedSymbol')),
        ),
        terms_col=flatten_cat(pl.col('disease_labels'), pl.col('drug_labels'), pl.col('variant_labels')),
        terms25_col=flatten_cat(pl.col('disease_labels_25'), pl.col('drug_labels_25'), pl.col('variant_labels_25')),
        terms5_col=flatten_cat(pl.col('disease_labels_5'), pl.col('drug_labels_5'), pl.col('variant_labels_5')),
        multiplier_col=pl.when(pl.col('target_relevance').is_not_null())
        .then(pl.col('target_relevance').log1p() + 1.0)
        .otherwise(0.01),
    )
