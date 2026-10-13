"""The disease search index."""

from __future__ import annotations

import polars as pl

from pts.transformers.search.helpers import (
    EMPTY_LIST,
    SYNONYM_FLAVOURS,
    as_list,
    flatten_cat,
    search_index,
    synonym_field,
    tier,
)
from pts.transformers.search.lookups import drug_labels_by_association
from pts.transformers.search.ranks import partitioned_rank

_TOP50 = 50
_TOP25 = 25
_TOP5 = 5


def build_disease_index(
    diseases: pl.LazyFrame,
    phenotypes: pl.LazyFrame,
    associations: pl.LazyFrame,
    scored_drugs: pl.LazyFrame,
    t_lut: pl.LazyFrame,
    dr_lut: pl.LazyFrame,
    studies: pl.LazyFrame,
    nct_by_disease: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build the search index for diseases.

    Ports `Transformers.Implicits.setIdAndSelectFromDiseases`.

    Args:
        diseases: disease frame carrying `therapeutic_labels`.
        phenotypes: `diseaseId` and `phenotype_labels`.
        associations: output of `association_scores`.
        scored_drugs: output of `scored_drug_associations`.
        t_lut: `targetId` and `target_labels`.
        dr_lut: `drugId` and `drug_labels`.
        studies: `studyId` and `diseaseIds`.
        nct_by_disease: `diseaseId` and `nctIds`.

    Returns:
        Search index LazyFrame, one row per disease.
    """
    drug_by_association = drug_labels_by_association(scored_drugs, dr_lut)

    studies_by_disease = (
        studies.select('studyId', 'diseaseIds')
        .filter(pl.col('diseaseIds').list.len() > 0)
        .explode('diseaseIds', empty_as_null=False)
        .rename({'diseaseIds': 'diseaseId'})
        .group_by('diseaseId')
        .agg(pl.col('studyId').alias('studyIds'))
    )

    ranked = (
        associations.join(drug_by_association, on='associationId', how='full', coalesce=True)
        .with_columns(partitioned_rank(pl.col('score'), by='diseaseId', descending=True).alias('rank'))
        .filter(pl.col('rank') <= _TOP50)
        .join(t_lut, on='targetId', how='inner')
        .group_by('diseaseId')
        .agg(
            tier('target_labels', 'rank', _TOP50).alias('target_labels'),
            tier('drug_labels', 'rank', _TOP50).alias('drug_labels'),
            tier('target_labels', 'rank', _TOP25).alias('target_labels_25'),
            tier('drug_labels', 'rank', _TOP25).alias('drug_labels_25'),
            tier('target_labels', 'rank', _TOP5).alias('target_labels_5'),
            tier('drug_labels', 'rank', _TOP5).alias('drug_labels_5'),
            pl.col('score').mean().alias('disease_relevance'),
        )
    )

    frame = (
        diseases.join(phenotypes, on='diseaseId', how='left')
        .join(ranked, on='diseaseId', how='left')
        .join(studies_by_disease, on='diseaseId', how='left')
        .join(nct_by_disease, on='diseaseId', how='left')
        .with_columns(
            pl.col('nctIds').fill_null(EMPTY_LIST),
            pl.col('phenotype_labels').fill_null(EMPTY_LIST),
            pl.col('target_labels').fill_null(EMPTY_LIST),
            pl.col('drug_labels').fill_null(EMPTY_LIST),
            pl.col('studyIds').fill_null(EMPTY_LIST),
        )
    )

    synonyms = tuple(synonym_field(flavour) for flavour in SYNONYM_FLAVOURS)

    return search_index(
        frame,
        id_col=pl.col('diseaseId'),
        name_col=pl.col('name'),
        description_col=pl.col('description'),
        entity='disease',
        # Deliberately NOT coalesced: a disease with no therapeutic area keeps a null category,
        # which is what the release holds.
        category_col=pl.col('therapeutic_labels'),
        keywords_col=flatten_cat(
            as_list(pl.col('name')),
            as_list(pl.col('diseaseId')),
            *synonyms,
            pl.col('nctIds'),
        ),
        prefixes_col=flatten_cat(as_list(pl.col('name')), *synonyms),
        ngrams_col=flatten_cat(as_list(pl.col('name')), *synonyms, pl.col('phenotype_labels')),
        terms_col=flatten_cat(pl.col('target_labels'), pl.col('drug_labels'), pl.col('studyIds')),
        terms25_col=flatten_cat(pl.col('target_labels_25'), pl.col('drug_labels_25'), pl.col('studyIds')),
        terms5_col=flatten_cat(pl.col('target_labels_5'), pl.col('drug_labels_5'), pl.col('studyIds')),
        multiplier_col=pl.when(pl.col('disease_relevance').is_not_null())
        .then(pl.col('disease_relevance').log1p() + 1.0)
        .otherwise(0.01),
    )
