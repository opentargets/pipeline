"""The drug search index."""

from __future__ import annotations

import polars as pl

from pts.transformers.search.helpers import (
    EMPTY_LIST,
    as_list,
    flatten_cat,
    list_struct_field,
    search_index,
)


def _flat_list(column: str) -> pl.Expr:
    """`flatten(collect_list(column))` for use inside an aggregation.

    Deliberately NOT deduplicated: the pyspark original uses `flatten(collect_list(...))` here
    with no `array_distinct`, unlike the tiered aggregates elsewhere which do dedupe. The final
    `flatten_cat` dedupes anyway, so the result is identical either way.
    """
    return pl.col(column).drop_nulls().list.explode(keep_nulls=False, empty_as_null=False)


def _explode_ids(frame: pl.LazyFrame, column: str, alias: str) -> pl.LazyFrame:
    """Explode a list column into one row per element, dropping null and empty lists."""
    return (
        frame.filter(pl.col(column).list.len() > 0)
        .explode(column, empty_as_null=False)
        .rename({column: alias})
    )


def build_drug_index(
    drugs: pl.LazyFrame,
    drug_assocs: pl.LazyFrame,
    t_lut: pl.LazyFrame,
    d_lut: pl.LazyFrame,
    nct_by_drug: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build the search index for drugs.

    Ports `Transformers.Implicits.setIdAndSelectFromDrugs`.

    Args:
        drugs: drug frame carrying the `rows` (mechanism) and `indications` columns.
        drug_assocs: output of `drug_associations`.
        t_lut: `targetId` and `target_labels`.
        d_lut: `diseaseId`, `disease_labels`, `disease_name`, `therapeutic_labels`.
        nct_by_drug: `drugId` and `nctIds`.

    Returns:
        Search index LazyFrame, one row per drug.
    """
    targets_by_drug = (
        t_lut.join(_explode_ids(drug_assocs, 'targetIds', 'targetId'), on='targetId', how='inner')
        .group_by('drugId')
        .agg(_flat_list('target_labels').alias('target_labels'))
    )
    diseases_by_drug = (
        d_lut.join(_explode_ids(drug_assocs, 'diseaseIds', 'diseaseId'), on='diseaseId', how='inner')
        .group_by('drugId')
        .agg(
            _flat_list('disease_labels').alias('disease_labels'),
            _flat_list('therapeutic_labels').alias('therapeutic_labels'),
        )
    )
    enriched = targets_by_drug.join(diseases_by_drug, on='drugId', how='full', coalesce=True)

    indication_labels = (
        _explode_ids(drugs.select('drugId', 'indications'), 'indications', 'indicationId')
        .join(d_lut.select(pl.col('diseaseId').alias('indicationId'), 'disease_name'), on='indicationId', how='inner')
        .group_by('drugId')
        .agg(pl.col('disease_name').unique(maintain_order=True).alias('indicationLabels'))
    )

    frame = (
        drugs.join(nct_by_drug, on='drugId', how='left')
        .with_columns(pl.col('nctIds').fill_null(EMPTY_LIST))
        .join(drug_assocs, on='drugId', how='left')
        .join(enriched, on='drugId', how='left')
        .join(indication_labels, on='drugId', how='left')
        .with_columns(
            list_struct_field('rows', 'mechanismOfAction').alias('descriptions'),
            pl.col('target_labels').fill_null(EMPTY_LIST),
            pl.col('disease_labels').fill_null(EMPTY_LIST),
            # sort_array(array_distinct(flatten(transform(crossReferences, x -> x.ids))))
            pl.col('crossReferences')
            .list.eval(pl.element().struct.field('ids').list.explode(keep_nulls=False, empty_as_null=False))
            .list.unique(maintain_order=True)
            .list.sort()
            .alias('crossReferences'),
        )
    )

    return search_index(
        frame,
        id_col=pl.col('drugId'),
        name_col=pl.col('name'),
        description_col=pl.col('description'),
        entity='drug',
        category_col=as_list(pl.col('drugType')),
        keywords_col=flatten_cat(
            list_struct_field('synonyms', 'label'),
            list_struct_field('tradeNames', 'label'),
            as_list(pl.col('name')),
            as_list(pl.col('drugId')),
            pl.col('crossReferences'),
            pl.col('nctIds'),
        ),
        prefixes_col=flatten_cat(
            list_struct_field('synonyms', 'label'),
            list_struct_field('tradeNames', 'label'),
            as_list(pl.col('name')),
            pl.col('descriptions'),
        ),
        ngrams_col=flatten_cat(
            as_list(pl.col('name')),
            list_struct_field('synonyms', 'label'),
            list_struct_field('tradeNames', 'label'),
            pl.col('descriptions'),
        ),
        terms_col=flatten_cat(
            pl.col('disease_labels'),
            pl.col('target_labels'),
            pl.col('indicationLabels'),
            pl.col('therapeutic_labels'),
            pl.col('childChemblIds'),
        ),
        multiplier_col=pl.when(pl.col('drug_relevance').is_not_null())
        .then(pl.col('drug_relevance').log1p() + 1.0)
        .otherwise(0.01),
    )
