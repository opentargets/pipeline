"""Lookup tables and association aggregates shared by the five index builders."""

from __future__ import annotations

import polars as pl

from pts.transformers.search.helpers import (
    SYNONYM_FLAVOURS,
    as_list,
    flatten_cat,
    list_struct_field,
    synonym_field,
)


def resolve_ta_labels(diseases: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve therapeutic-area ids to names and attach them as `therapeutic_labels`.

    Ports `Transformers.Implicits.resolveTALabels`: a self-join of the disease frame on its own
    `therapeuticAreas`.

    A disease with no therapeutic areas keeps a NULL `therapeutic_labels`, not an empty list.
    Spark's `explode` drops the empty array so no row reaches the group-by, and the left join
    leaves null -- which is what makes the disease index's `category` nullable. Do not coalesce.

    Args:
        diseases: frame with `diseaseId`, `name` and `therapeuticAreas`.

    Returns:
        `diseases` with a nullable `therapeutic_labels` list column added.
    """
    area_names = diseases.select(
        pl.col('diseaseId').alias('therapeuticAreaId'),
        pl.col('name').alias('therapeuticAreaLabel'),
    )
    labels = (
        diseases.select('diseaseId', 'therapeuticAreas')
        .filter(pl.col('therapeuticAreas').list.len() > 0)
        .explode('therapeuticAreas', empty_as_null=False)
        .rename({'therapeuticAreas': 'therapeuticAreaId'})
        .join(area_names, on='therapeuticAreaId', how='inner')
        .group_by('diseaseId')
        .agg(pl.col('therapeuticAreaLabel').unique().alias('therapeutic_labels'))
    )
    return diseases.join(labels, on='diseaseId', how='left')


def phenotype_names(disease_phenotype: pl.LazyFrame, hpo: pl.LazyFrame) -> pl.LazyFrame:
    """Collect HPO phenotype labels per disease.

    Args:
        disease_phenotype: frame with `disease` and `phenotype` (the `disease_phenotype` dataset).
        hpo: frame with `id` and `name` (the `disease_hpo` dataset).

    Returns:
        LazyFrame with `diseaseId` and `phenotype_labels`.
    """
    return (
        disease_phenotype.select('disease', 'phenotype')
        .join(hpo.select(pl.col('id').alias('phenotype'), 'name'), on='phenotype', how='inner')
        .group_by('disease')
        .agg(pl.col('name').unique().alias('phenotype_labels'))
        .rename({'disease': 'diseaseId'})
    )


def disease_lut(diseases: pl.LazyFrame) -> pl.LazyFrame:
    """Disease id to its searchable labels, display name and therapeutic areas.

    Args:
        diseases: frame already carrying `therapeutic_labels` from `resolve_ta_labels`.

    Returns:
        LazyFrame with `diseaseId`, `disease_labels`, `disease_name`, `therapeutic_labels`.
    """
    return diseases.select(
        'diseaseId',
        flatten_cat(
            as_list(pl.col('name')),
            *(synonym_field(flavour) for flavour in SYNONYM_FLAVOURS),
        ).alias('disease_labels'),
        pl.col('name').alias('disease_name'),
        'therapeutic_labels',
    )


def target_lut(targets: pl.LazyFrame) -> pl.LazyFrame:
    """Target id to its searchable labels.

    Args:
        targets: frame with `targetId`, `synonyms`, `approvedName`, `approvedSymbol`.

    Returns:
        LazyFrame with `targetId` and `target_labels`.
    """
    return targets.select(
        'targetId',
        flatten_cat(
            list_struct_field('synonyms', 'label'),
            as_list(pl.col('approvedName')),
            as_list(pl.col('approvedSymbol')),
        ).alias('target_labels'),
    )


def drug_lut(drugs: pl.LazyFrame) -> pl.LazyFrame:
    """Drug id to its searchable labels.

    Args:
        drugs: frame with `drugId`, `synonyms`, `tradeNames`, `name` and the `rows` column
            produced by joining the mechanism-of-action dataset.

    Returns:
        LazyFrame with `drugId` and `drug_labels`.
    """
    return drugs.select(
        'drugId',
        flatten_cat(
            list_struct_field('synonyms', 'label'),
            list_struct_field('tradeNames', 'label'),
            as_list(pl.col('name')),
            list_struct_field('rows', 'mechanismOfAction'),
        ).alias('drug_labels'),
    )


def association_scores(associations: pl.LazyFrame) -> pl.LazyFrame:
    """Project the association dataset onto its search-relevant columns.

    `associationId` is `concat_ws('-', diseaseId, targetId)`, which SKIPS a null component
    rather than propagating it or leaving a dangling separator.

    Args:
        associations: the `association_overall_indirect` dataset.

    Returns:
        LazyFrame with `associationId`, `targetId`, `diseaseId`, `score`.
    """
    return associations.select(
        pl.concat_str([pl.col('diseaseId'), pl.col('targetId')], separator='-', ignore_nulls=True).alias(
            'associationId'
        ),
        'targetId',
        'diseaseId',
        pl.col('associationScore').alias('score'),
    )


def drug_associations_from_evidence(evidence: pl.LazyFrame) -> pl.LazyFrame:
    """Group drug-bearing evidence into one row per association.

    Split out from `scored_drug_associations` because its ROW COUNT is the `drug_relevance`
    denominator, and spark takes that count HERE -- before the inner join with the association
    scores. Counting after the join instead would silently exclude every association that
    appears in evidence but not in the association dataset, changing the multiplier on every
    drug in the release.

    Args:
        evidence: union of the evidence datasets, with `drugId`, `targetId`, `diseaseId`.

    Returns:
        LazyFrame with `associationId`, `drugIds`, `targetId`, `diseaseId`.
    """
    return (
        evidence.filter(pl.col('drugId').is_not_null())
        .select('drugId', 'targetId', 'diseaseId')
        .with_columns(
            pl.concat_str([pl.col('diseaseId'), pl.col('targetId')], separator='-', ignore_nulls=True).alias(
                'associationId'
            )
        )
        .group_by('associationId')
        .agg(
            pl.col('drugId').unique().alias('drugIds'),
            # `associationId` is `diseaseId-targetId`, so both are constant within a group and
            # spark's unordered `first()` was deterministic despite appearances.
            pl.col('targetId').first().alias('targetId'),
            pl.col('diseaseId').first().alias('diseaseId'),
        )
    )


def scored_drug_associations(from_evidence: pl.LazyFrame, scores: pl.LazyFrame) -> pl.LazyFrame:
    """One row per (association, drug) pair, carrying the association score.

    Args:
        from_evidence: output of `drug_associations_from_evidence`.
        scores: output of `association_scores`.

    Returns:
        LazyFrame with `associationId`, `drugId`, `drugIds`, `targetId`, `diseaseId`, `score`.
    """
    return (
        from_evidence.join(scores.select('associationId', 'score'), on='associationId', how='inner')
        # `drugIds` survives the explode in the spark original, so duplicate it first rather
        # than reconstructing it afterwards with a window.
        .with_columns(pl.col('drugIds').alias('drugId'))
        .explode('drugId', empty_as_null=False)
        .select('associationId', 'drugId', 'drugIds', 'targetId', 'diseaseId', 'score')
    )


def drug_associations(scored: pl.LazyFrame, total: int) -> pl.LazyFrame:
    """Aggregate scored drug associations per drug.

    Args:
        scored: output of `scored_drug_associations`.
        total: the number of drug-bearing associations, the `drug_relevance` denominator.

    Returns:
        LazyFrame with `drugId`, `targetIds`, `diseaseIds`, `meanScore`, `drug_relevance`.
    """
    return scored.group_by('drugId').agg(
        pl.col('targetId').unique().alias('targetIds'),
        pl.col('diseaseId').unique().alias('diseaseIds'),
        pl.col('score').mean().alias('meanScore'),
        (pl.len().cast(pl.Float64) / pl.lit(float(total))).alias('drug_relevance'),
    )


def nct_map(indication: pl.LazyFrame) -> pl.LazyFrame:
    """Clinical-trial ids with their disease and drug, for indications that have any.

    Spark's `size(null)` is `-1`, so `size(nctIds) > 0` drops a row whose report list is null.
    polars' `list.len()` on a null list is null, which the same filter also drops -- matching
    behaviour, reached differently.

    Args:
        indication: the `clinical_indication` dataset.

    Returns:
        LazyFrame with `nctIds`, `drugId`, `diseaseId`.
    """
    return indication.select(
        pl.col('clinicalReportIds')
        .list.eval(pl.element().filter(pl.element().str.starts_with('nct')))
        .alias('nctIds'),
        'drugId',
        'diseaseId',
    ).filter(pl.col('nctIds').list.len() > 0)


def nct_by(indication: pl.LazyFrame, key: str) -> pl.LazyFrame:
    """Merge every clinical-trial id recorded against one key.

    Args:
        indication: the `clinical_indication` dataset.
        key: `'diseaseId'` or `'drugId'`.

    Returns:
        LazyFrame with `key` and a deduplicated `nctIds`.
    """
    return (
        nct_map(indication)
        .group_by(key)
        .agg(
            pl.col('nctIds')
            .list.explode(keep_nulls=False, empty_as_null=False)
            .unique(maintain_order=True)
            .alias('nctIds')
        )
    )
