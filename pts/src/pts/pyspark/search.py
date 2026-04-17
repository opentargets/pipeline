"""Search index generation for diseases, targets, drugs, variants and studies.

Ported from platform-etl-backend Search step. Builds search-index documents
with ranked terms, keywords, prefixes and ngrams for each entity type.

Scala source ported:
    - Search.scala (object Search + object Transformers + case class SearchIndex)
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql.types import DoubleType

from pts.pyspark.common.session import Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_cat(*col_exprs: str) -> Column:
    """Build a deduplicated, null-filtered, comma-stripped array from column expressions.

    Ports Helpers.flattenCat from Scala. Combines multiple array or scalar
    columns, removes nulls, strips commas from values, and deduplicates.

    Args:
        *col_exprs: Column expressions (e.g. ``'array(name)'``, ``'synonyms.label'``).

    Returns:
        A PySpark Column containing the merged array.
    """
    cols = ', '.join(col_exprs)
    return f.expr(f"""filter(array_distinct(
        transform(
            flatten(
                filter(array({cols}),
                    x -> isnotnull(x)
                )
            ),
            s -> replace(trim(s), ',', '')
        )
    ),
    t -> isnotnull(t))""")


def _search_index(
    df: DataFrame,
    *,
    id_col: Column,
    name_col: Column,
    entity_col: Column,
    category_col: Column,
    keywords_col: Column,
    prefixes_col: Column,
    ngrams_col: Column,
    description_col: Column | None = None,
    terms_col: Column | None = None,
    terms25_col: Column | None = None,
    terms5_col: Column | None = None,
    multiplier_col: Column | None = None,
) -> DataFrame:
    """Select search-index columns with standardised names.

    Ports the SearchIndex case class from Scala.

    Args:
        df: Source DataFrame to select from.
        id_col: Column or expression for the entity ID.
        name_col: Column or expression for the display name.
        entity_col: Column or expression for entity type string.
        category_col: Column or expression for category array.
        keywords_col: Column or expression for keywords array.
        prefixes_col: Column or expression for prefixes array.
        ngrams_col: Column or expression for ngrams array.
        description_col: Column or expression for description.
        terms_col: Column for top-50 terms.
        terms25_col: Column for top-25 terms.
        terms5_col: Column for top-5 terms.
        multiplier_col: Column for the ranking multiplier.

    Returns:
        DataFrame with standardised search-index column names.
    """
    if description_col is None:
        description_col = f.lit(None).cast('string')
    if terms_col is None:
        terms_col = f.array().cast('array<string>')
    if terms25_col is None:
        terms25_col = f.array().cast('array<string>')
    if terms5_col is None:
        terms5_col = f.array().cast('array<string>')
    if multiplier_col is None:
        multiplier_col = f.lit(0.01)
    return df.select(
        id_col.alias('id'),
        name_col.alias('name'),
        description_col.alias('description'),
        entity_col.alias('entity'),
        category_col.alias('category'),
        keywords_col.alias('keywords'),
        prefixes_col.alias('prefixes'),
        ngrams_col.alias('ngrams'),
        terms_col.alias('terms'),
        terms25_col.alias('terms25'),
        terms5_col.alias('terms5'),
        multiplier_col.alias('multiplier'),
    )


def _resolve_ta_labels(df: DataFrame, id_col: str, output_col: str) -> DataFrame:
    """Resolve therapeutic-area IDs to names and attach as an array column.

    Ports Transformers.Implicits.resolveTALabels from Scala.

    Args:
        df: Disease DataFrame with ``therapeuticAreas`` array column.
        id_col: Name of the entity ID column (e.g. ``'diseaseId'``).
        output_col: Name for the resulting therapeutic-area label column.

    Returns:
        ``df`` with an additional array<string> column ``output_col``.
    """
    ta_labels = (
        df
        .select(id_col, 'therapeuticAreas')
        .withColumn('therapeuticAreaId', f.explode(f.col('therapeuticAreas')))
        .join(
            df.selectExpr(f'{id_col} as therapeuticAreaId', 'name as therapeuticAreaLabel'),
            'therapeuticAreaId',
            'inner',
        )
        .drop('therapeuticAreaId', 'therapeuticAreas')
        .groupBy(id_col)
        .agg(f.collect_set('therapeuticAreaLabel').alias(output_col))
    )
    return df.join(ta_labels, id_col, 'left_outer')


# ---------------------------------------------------------------------------
# Entity index builders
# ---------------------------------------------------------------------------


def _build_disease_index(
    diseases: DataFrame,
    phenotype_names: DataFrame,
    associations: DataFrame,
    assoc_drugs: DataFrame,
    t_lut: DataFrame,
    dr_lut: DataFrame,
    studies: DataFrame,
) -> DataFrame:
    """Build the search index for diseases.

    Ports Transformers.Implicits.setIdAndSelectFromDiseases.

    Args:
        diseases: Disease DataFrame (with diseaseId, name, etc).
        phenotype_names: DataFrame with diseaseId and phenotype_labels.
        associations: Association scores (associationId, diseaseId, targetId, score).
        assoc_drugs: Associations expanded with drug IDs (associationId, drugId, …, score).
        t_lut: Target lookup with targetId and target_labels.
        dr_lut: Drug lookup with drugId and drug_labels.
        studies: Study DataFrame (studyId, diseaseIds).

    Returns:
        Search index DataFrame for diseases.
    """
    top50 = 50
    top25 = 25
    top5 = 5
    window = Window.partitionBy('diseaseId').orderBy(f.col('score').desc())

    drug_by_disease = (
        assoc_drugs
        .join(dr_lut, 'drugId', 'inner')
        .groupBy('associationId')
        .agg(f.array_distinct(f.flatten(f.collect_list('drug_labels'))).alias('drug_labels'))
    )

    studies_by_disease = (
        studies
        .select('studyId', f.explode('diseaseIds').alias('diseaseId'))
        .groupBy('diseaseId')
        .agg(f.collect_list('studyId').alias('studyIds'))
    )

    assocs_with_labels = (
        associations
        .join(drug_by_disease.drop('diseaseId', 'targetId'), 'associationId', 'full_outer')
        .withColumn('rank', f.rank().over(window))
        .where(f.col('rank') <= top50)
        .join(t_lut, 'targetId', 'inner')
        .groupBy('diseaseId')
        .agg(
            f.array_distinct(f.flatten(f.collect_list('target_labels'))).alias('target_labels'),
            f.array_distinct(f.flatten(f.collect_list('drug_labels'))).alias('drug_labels'),
            f.array_distinct(f.flatten(f.collect_list(f.when(f.col('rank') <= top25, f.col('target_labels'))))).alias(
                'target_labels_25'
            ),
            f.array_distinct(f.flatten(f.collect_list(f.when(f.col('rank') <= top25, f.col('drug_labels'))))).alias(
                'drug_labels_25'
            ),
            f.array_distinct(f.flatten(f.collect_list(f.when(f.col('rank') <= top5, f.col('target_labels'))))).alias(
                'target_labels_5'
            ),
            f.array_distinct(f.flatten(f.collect_list(f.when(f.col('rank') <= top5, f.col('drug_labels'))))).alias(
                'drug_labels_5'
            ),
            f.mean('score').alias('disease_relevance'),
        )
    )

    disease = (
        diseases
        .join(phenotype_names, 'diseaseId', 'left_outer')
        .join(assocs_with_labels, 'diseaseId', 'left_outer')
        .join(studies_by_disease, 'diseaseId', 'left_outer')
        .withColumn('phenotype_labels', f.coalesce('phenotype_labels', f.array()))
        .withColumn('target_labels', f.coalesce('target_labels', f.array()))
        .withColumn('drug_labels', f.coalesce('drug_labels', f.array()))
        .withColumn('studyIds', f.coalesce('studyIds', f.array()))
    )

    return _search_index(
        disease,
        id_col=f.col('diseaseId'),
        name_col=f.col('name'),
        description_col=f.col('description'),
        entity_col=f.lit('disease'),
        category_col=f.col('therapeutic_labels'),
        keywords_col=_flatten_cat(
            'array(name)',
            'array(diseaseId)',
            'synonyms.hasBroadSynonym',
            'synonyms.hasExactSynonym',
            'synonyms.hasNarrowSynonym',
            'synonyms.hasRelatedSynonym',
        ),
        prefixes_col=_flatten_cat(
            'array(name)',
            'synonyms.hasBroadSynonym',
            'synonyms.hasExactSynonym',
            'synonyms.hasNarrowSynonym',
            'synonyms.hasRelatedSynonym',
        ),
        ngrams_col=_flatten_cat(
            'array(name)',
            'synonyms.hasBroadSynonym',
            'synonyms.hasExactSynonym',
            'synonyms.hasNarrowSynonym',
            'synonyms.hasRelatedSynonym',
            'phenotype_labels',
        ),
        terms_col=_flatten_cat('target_labels', 'drug_labels', 'studyIds'),
        terms25_col=_flatten_cat('target_labels_25', 'drug_labels_25', 'studyIds'),
        terms5_col=_flatten_cat('target_labels_5', 'drug_labels_5', 'studyIds'),
        multiplier_col=f.when(
            f.col('disease_relevance').isNotNull(),
            f.log1p(f.col('disease_relevance')) + f.lit(1.0),
        ).otherwise(0.01),
    )


def _build_target_index(
    targets: DataFrame,
    associations: DataFrame,
    d_lut: DataFrame,
    dr_lut: DataFrame,
    variants: DataFrame,
) -> DataFrame:
    """Build the search index for targets.

    Ports Transformers.Implicits.setIdAndSelectFromTargets.

    Args:
        targets: Target DataFrame (targetId, approvedSymbol, approvedName, …).
        associations: Association scores (associationId, targetId, diseaseId, score).
        d_lut: Disease lookup with diseaseId and disease_labels.
        dr_lut: Drug lookup with drugId and drug_labels.
        variants: Variant DataFrame with transcriptConsequences.

    Returns:
        Search index DataFrame for targets.
    """
    top50 = 50
    top25 = 25
    top5 = 5
    window = Window.partitionBy('targetId').orderBy(f.col('score').desc())
    variant_window = Window.partitionBy('targetId').orderBy(f.col('transcriptScore').asc())

    # HGNC IDs
    target_hgnc = (
        targets
        .select(
            'targetId',
            f.filter(f.col('dbXRefs'), lambda c: c.getField('source') == 'HGNC').alias('h'),
        )
        .select('targetId', f.explode_outer(f.col('h.id')).alias('hgncId'))
        .withColumn('hgncId', f.when(f.col('hgncId').isNotNull(), f.concat(f.lit('HGNC:'), f.col('hgncId'))))
        .orderBy('targetId')
    )

    # Drug labels per association
    drug_by_target = (
        (
            associations
            .join(dr_lut, 'drugId', 'inner')
            .groupBy('associationId')
            .agg(f.array_distinct(f.flatten(f.collect_list('drug_labels'))).alias('drug_labels'))
        )
        if 'drugId' in associations.columns
        else (
            associations
            .limit(0)
            .withColumn('drug_labels', f.lit(None).cast('array<string>'))
            .select('associationId', 'drug_labels')
        )
    )

    # Variant labels per target
    variant_labels_df = (
        variants
        .withColumn('transcriptConsequences', f.explode(f.col('transcriptConsequences')))
        .withColumn(
            'consequenceScore',
            f.when(
                f.col('transcriptConsequences.consequenceScore').isNotNull(),
                f.col('transcriptConsequences.consequenceScore'),
            ).otherwise(f.lit(1)),
        )
        .withColumn('targetId', f.col('transcriptConsequences.targetId'))
        .withColumn(
            'transcriptScore',
            (f.col('transcriptConsequences.consequenceScore') + f.lit(1))
            * f.col('transcriptConsequences.distanceFromFootprint'),
        )
        .withColumn(
            'variant_labels',
            _flatten_cat(
                'array(variantId)',
                'array(hgvsId)',
                'dbXrefs.id',
                'rsIds',
            ),
        )
        .withColumn('variantTargetRank', f.rank().over(variant_window))
        .where(f.col('variantTargetRank') <= top50)
        .groupBy('targetId')
        .agg(
            f.array_distinct(
                f.flatten(f.collect_list(f.when(f.col('variantTargetRank') <= top50, f.col('variant_labels'))))
            ).alias('variant_labels'),
            f.array_distinct(
                f.flatten(f.collect_list(f.when(f.col('variantTargetRank') <= top25, f.col('variant_labels'))))
            ).alias('variant_labels_25'),
            f.array_distinct(
                f.flatten(f.collect_list(f.when(f.col('variantTargetRank') <= top5, f.col('variant_labels'))))
            ).alias('variant_labels_5'),
        )
    )

    assocs_with_labels = (
        associations
        .join(drug_by_target, 'associationId', 'left_outer')
        .withColumn('rank', f.rank().over(window))
        .where(f.col('rank') <= top50)
        .join(d_lut, 'diseaseId', 'inner')
        .groupBy('targetId')
        .agg(
            f.array_distinct(f.flatten(f.collect_list('disease_labels'))).alias('disease_labels'),
            f.array_distinct(f.flatten(f.collect_list(f.when(f.col('rank') <= top25, f.col('disease_labels'))))).alias(
                'disease_labels_25'
            ),
            f.array_distinct(f.flatten(f.collect_list(f.when(f.col('rank') <= top5, f.col('disease_labels'))))).alias(
                'disease_labels_5'
            ),
            f.array_distinct(f.flatten(f.collect_list('drug_labels'))).alias('drug_labels'),
            f.array_distinct(f.flatten(f.collect_list(f.when(f.col('rank') <= top25, f.col('drug_labels'))))).alias(
                'drug_labels_25'
            ),
            f.array_distinct(f.flatten(f.collect_list(f.when(f.col('rank') <= top5, f.col('drug_labels'))))).alias(
                'drug_labels_5'
            ),
            f.mean('score').alias('target_relevance'),
        )
    )

    assocs_with_variants = assocs_with_labels.join(variant_labels_df, 'targetId', 'left_outer')

    target_df = (
        targets
        .join(target_hgnc, 'targetId')
        .join(assocs_with_variants, 'targetId', 'left_outer')
        .withColumn('disease_labels', f.coalesce('disease_labels', f.array()))
        .withColumn('drug_labels', f.coalesce('drug_labels', f.array()))
        .withColumn('variant_labels', f.coalesce('variant_labels', f.array()))
        .withColumn('disease_labels_5', f.coalesce('disease_labels_5', f.array()))
        .withColumn('drug_labels_5', f.coalesce('drug_labels_5', f.array()))
        .withColumn('variant_labels_5', f.coalesce('variant_labels_5', f.array()))
        .withColumn('disease_labels_25', f.coalesce('disease_labels_25', f.array()))
        .withColumn('drug_labels_25', f.coalesce('drug_labels_25', f.array()))
        .withColumn('variant_labels_25', f.coalesce('variant_labels_25', f.array()))
    )

    return _search_index(
        target_df,
        id_col=f.col('targetId'),
        name_col=f.col('approvedSymbol'),
        description_col=f.col('approvedName'),
        entity_col=f.lit('target'),
        category_col=f.array(f.col('biotype')),
        keywords_col=_flatten_cat(
            'synonyms.label',
            'proteinIds.id',
            'array(approvedName)',
            'array(approvedSymbol)',
            'array(hgncId)',
            'array(targetId)',
        ),
        prefixes_col=_flatten_cat(
            'synonyms.label',
            'proteinIds.id',
            'array(approvedName)',
            'array(approvedSymbol)',
        ),
        ngrams_col=_flatten_cat(
            'proteinIds.id',
            'synonyms.label',
            'array(approvedName)',
            'array(approvedSymbol)',
        ),
        terms_col=_flatten_cat('disease_labels', 'drug_labels', 'variant_labels'),
        terms25_col=_flatten_cat('disease_labels_25', 'drug_labels_25', 'variant_labels_25'),
        terms5_col=_flatten_cat('disease_labels_5', 'drug_labels_5', 'variant_labels_5'),
        multiplier_col=f.when(
            f.col('target_relevance').isNotNull(),
            f.log1p(f.col('target_relevance')) + f.lit(1.0),
        ).otherwise(0.01),
    )


def _build_drug_index(
    drugs: DataFrame,
    assoc_drugs: DataFrame,
    t_lut: DataFrame,
    d_lut: DataFrame,
) -> DataFrame:
    """Build the search index for drugs.

    Ports Transformers.Implicits.setIdAndSelectFromDrugs.

    Args:
        drugs: Drug DataFrame (with drugId, name, synonyms, tradeNames, …).
            Must have a ``rows`` column containing mechanism-of-action data
            and an ``indications`` array column.
        assoc_drugs: Drug association aggregates (drugId, targetIds, diseaseIds,
            drug_relevance).
        t_lut: Target lookup (targetId, target_labels).
        d_lut: Disease lookup (diseaseId, disease_labels, therapeutic_labels,
            disease_name).

    Returns:
        Search index DataFrame for drugs.
    """
    t_luts = (
        t_lut
        .join(
            assoc_drugs.withColumn('targetId', f.explode('targetIds')),
            'targetId',
            'inner',
        )
        .groupBy('drugId')
        .agg(f.flatten(f.collect_list('target_labels')).alias('target_labels'))
    )

    d_luts = (
        d_lut
        .join(
            assoc_drugs.withColumn('diseaseId', f.explode('diseaseIds')),
            'diseaseId',
            'inner',
        )
        .groupBy('drugId')
        .agg(
            f.flatten(f.collect_list('disease_labels')).alias('disease_labels'),
            f.flatten(f.collect_list('therapeutic_labels')).alias('therapeutic_labels'),
        )
    )

    drug_enriched = t_luts.join(d_luts, 'drugId', 'full_outer').orderBy('drugId')

    # Indication labels
    d_labels = d_lut.selectExpr('diseaseId as indicationId', 'disease_name').orderBy('indicationId')
    indication_labels = (
        drugs
        .withColumn('indicationId', f.explode(f.col('indications')))
        .select('drugId', 'indicationId')
        .join(d_labels, 'indicationId', 'inner')
        .groupBy('drugId')
        .agg(f.collect_set('disease_name').alias('indicationLabels'))
    )

    drug_df = (
        drugs
        .join(assoc_drugs, drugs['drugId'] == assoc_drugs['drugId'], 'left_outer')
        .drop(assoc_drugs['drugId'])
        .withColumn('targetIds', f.coalesce('targetIds', f.array()))
        .withColumn('diseaseIds', f.coalesce('diseaseIds', f.array()))
        .withColumn('descriptions', f.col('rows.mechanismOfAction'))
        .join(drug_enriched, 'drugId', 'left_outer')
        .join(indication_labels, 'drugId', 'left_outer')
        .withColumn('target_labels', f.coalesce('target_labels', f.array()))
        .withColumn('disease_labels', f.coalesce('disease_labels', f.array()))
        .withColumn(
            'crossReferences',
            f.sort_array(
                f.array_distinct(f.flatten(f.transform(f.col('crossReferences'), lambda x: x.getField('ids'))))
            ).alias('crossReferences'),
        )
    )

    return _search_index(
        drug_df,
        id_col=f.col('drugId'),
        name_col=f.col('name'),
        description_col=f.col('description'),
        entity_col=f.lit('drug'),
        category_col=f.array(f.col('drugType')),
        keywords_col=_flatten_cat(
            'synonyms',
            'tradeNames',
            'array(name)',
            'array(drugId)',
            'childChemblIds',
            'crossReferences',
        ),
        prefixes_col=_flatten_cat('synonyms', 'tradeNames', 'array(name)', 'descriptions'),
        ngrams_col=_flatten_cat('array(name)', 'synonyms', 'tradeNames', 'descriptions'),
        terms_col=_flatten_cat('disease_labels', 'target_labels', 'indicationLabels', 'therapeutic_labels'),
        multiplier_col=f.when(
            f.col('drug_relevance').isNotNull(),
            f.log1p(f.col('drug_relevance')) + f.lit(1.0),
        ).otherwise(0.01),
    )


def _build_variant_index(variants: DataFrame) -> DataFrame:
    """Build the search index for variants.

    Ports Transformers.Implicits.setIdAndSelectFromVariants.

    Args:
        variants: Variant DataFrame (variantId, chromosome, position, rsIds,
            hgvsId, dbXrefs, transcriptConsequences).

    Returns:
        Search index DataFrame for variants.
    """
    variant_df = (
        variants
        .withColumn('locationUnderscore', f.concat(f.col('chromosome'), f.lit('_'), f.col('position'), f.lit('_')))
        .withColumn('locationDash', f.concat(f.col('chromosome'), f.lit('-'), f.col('position'), f.lit('-')))
        .withColumn('locationColon', f.concat(f.col('chromosome'), f.lit(':'), f.col('position'), f.lit(':')))
    )
    return _search_index(
        variant_df,
        id_col=f.col('variantId'),
        name_col=f.col('variantId'),
        entity_col=f.lit('variant'),
        category_col=f.array(f.lit('variant')),
        keywords_col=_flatten_cat(
            'array(variantId)',
            'array(hgvsId)',
            'dbXrefs.id',
            'rsIds',
            'array(locationUnderscore)',
            'array(locationDash)',
            'array(locationColon)',
        ),
        prefixes_col=_flatten_cat(
            'array(variantId)',
            'array(hgvsId)',
            'dbXrefs.id',
            'rsIds',
            'array(locationColon)',
        ),
        ngrams_col=_flatten_cat('array(variantId)', 'dbXrefs.id'),
        multiplier_col=f.lit(1.0),
    )


def _build_study_index(
    studies: DataFrame,
    targets: DataFrame,
    credible_sets: DataFrame,
) -> DataFrame:
    """Build the search index for studies.

    Ports Transformers.Implicits.setIdAndSelectFromStudies.

    Args:
        studies: Study DataFrame (studyId, traitFromSource, pubmedId,
            publicationFirstAuthor, diseaseIds, nSamples, geneId).
        targets: Target DataFrame with targetId and approvedSymbol.
        credible_sets: Aggregated credible-set counts per studyId.

    Returns:
        Search index DataFrame for studies.
    """
    studies_with_targets = studies.withColumnRenamed('geneId', 'targetId').join(
        targets.select('targetId', 'approvedSymbol'), 'targetId', 'left_outer'
    )

    window = Window.orderBy(f.col('credibleSetCount').desc(), f.col('nSamples').desc())
    studies_with_cred = studies_with_targets.join(credible_sets, 'studyId', 'left_outer').withColumn(
        'rank', f.rank().over(window)
    )

    max_rank = studies_with_cred.agg(f.max('rank')).first()[0] or 1
    multiplier = f.expr(f'1 + (({max_rank} - rank) / ({max_rank} - 1))') if max_rank > 1 else f.lit(1.0)

    return _search_index(
        studies_with_cred,
        id_col=f.col('studyId'),
        name_col=f.col('studyId'),
        entity_col=f.lit('study'),
        category_col=f.array(f.lit('study')),
        keywords_col=_flatten_cat(
            'array(studyId)',
            'array(pubmedId)',
            'array(publicationFirstAuthor)',
        ),
        prefixes_col=_flatten_cat(
            'array(studyId)',
            'array(pubmedId)',
            'array(publicationFirstAuthor)',
        ),
        ngrams_col=_flatten_cat('array(studyId)'),
        terms_col=_flatten_cat(
            'array(traitFromSource)',
            'diseaseIds',
            'array(approvedSymbol)',
            'array(targetId)',
        ),
        terms25_col=_flatten_cat(
            'array(traitFromSource)',
            'diseaseIds',
            'array(approvedSymbol)',
            'array(targetId)',
        ),
        terms5_col=_flatten_cat(
            'array(traitFromSource)',
            'diseaseIds',
            'array(approvedSymbol)',
            'array(targetId)',
        ),
        multiplier_col=multiplier,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def search(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Run the search index generation pipeline.

    Reads input datasets, computes association labels for drugs, targets and
    diseases, and writes five output search index views.

    Args:
        source: Path mapping with keys: association, drug, evidence, indication,
            mechanism, target, credible_sets, disease, disease_hpo, hpo, studies,
            variants.
        destination: Path mapping with keys: diseases, targets, drugs, variants,
            studies.
        settings: Unused; reserved for future configuration.
        properties: Spark session properties.
    """
    spark = Session(app_name='search', properties=properties).spark

    logger.info('Loading input data')
    disease_raw = spark.read.parquet(source['disease'])
    target_raw = spark.read.parquet(source['target'])
    drug_raw = spark.read.parquet(source['drug'])
    mechanism = spark.read.parquet(source['mechanism'])
    indication = spark.read.parquet(source['indication'])
    association_raw = spark.read.parquet(source['association'])
    evidence_raw = spark.read.parquet(source['evidence'])
    disease_hpo = spark.read.parquet(source['disease_hpo'])
    hpo = spark.read.parquet(source['hpo'])
    studies_raw = spark.read.parquet(source['studies'])
    variants_raw = spark.read.parquet(source['variants'])
    credible_sets_raw = spark.read.parquet(source['credible_sets'])

    logger.info('Processing diseases')
    diseases = (
        disease_raw
        .withColumnRenamed('id', 'diseaseId')
        .transform(lambda df: _resolve_ta_labels(df, 'diseaseId', 'therapeutic_labels'))
        .orderBy('diseaseId')
    )

    logger.info('Processing phenotype names')
    phenotype_names = (
        disease_hpo
        .join(hpo, disease_hpo['phenotype'] == hpo['id'])
        .select('disease', 'phenotype', 'name')
        .groupBy('disease')
        .agg(f.collect_set('name').alias('phenotype_labels'))
        .withColumnRenamed('disease', 'diseaseId')
        .orderBy('diseaseId')
    )

    logger.info('Processing targets')
    targets = target_raw.withColumnRenamed('id', 'targetId').orderBy('targetId')

    logger.info('Processing drugs with mechanism and indication data')
    mech_nested = (
        mechanism
        .withColumn('id', f.explode('chemblIds'))
        .select(
            'id',
            f.struct(
                'mechanismOfAction',
                'references',
                'targetName',
                'targets',
            ).alias('rows'),
            'actionType',
            'targetType',
        )
        .groupBy('id')
        .agg(
            f.collect_list('rows').alias('rows'),
            f.collect_set('actionType').alias('uniqueActionTypes'),
            f.collect_set('targetType').alias('uniqueTargetType'),
        )
    )
    indications_grouped = indication.groupBy(f.col('drugId').alias('id')).agg(
        f.collect_list('diseaseId').alias('indications')
    )
    drugs = (
        drug_raw
        .join(mech_nested, 'id', 'left_outer')
        .join(indications_grouped, 'id', 'left_outer')
        .withColumnRenamed('id', 'drugId')
        .orderBy('drugId')
    )

    logger.info('Processing variants and studies')
    variants = variants_raw.select(
        'variantId',
        'rsIds',
        'hgvsId',
        'dbXrefs',
        'chromosome',
        'position',
        'transcriptConsequences',
    )
    studies = studies_raw.select(
        'studyId',
        'traitFromSource',
        'pubmedId',
        'publicationFirstAuthor',
        'diseaseIds',
        'nSamples',
        'geneId',
    )
    credible_sets = (
        credible_sets_raw
        .select('studyId')
        .groupBy('studyId')
        .agg(f.count('studyId').cast(DoubleType()).alias('credibleSetCount'))
    )

    logger.info('Building lookup tables')
    d_lut = (
        diseases
        .withColumn(
            'disease_labels',
            _flatten_cat(
                'array(name)',
                'synonyms.hasBroadSynonym',
                'synonyms.hasExactSynonym',
                'synonyms.hasNarrowSynonym',
                'synonyms.hasRelatedSynonym',
            ),
        )
        .selectExpr('diseaseId', 'disease_labels', 'name as disease_name', 'therapeutic_labels')
        .orderBy('diseaseId')
    )
    dr_lut = (
        drugs
        .withColumn(
            'drug_labels',
            _flatten_cat(
                'synonyms',
                'tradeNames',
                'array(name)',
                'rows.mechanismOfAction',
            ),
        )
        .select('drugId', 'drug_labels')
        .orderBy('drugId')
    )
    t_lut = (
        targets
        .withColumn(
            'target_labels',
            _flatten_cat(
                'synonyms.label',
                'array(approvedName)',
                'array(approvedSymbol)',
            ),
        )
        .select('targetId', 'target_labels')
        .orderBy('targetId')
    )

    logger.info('Building association scores')
    association_scores = (
        association_raw
        .withColumn('associationId', f.concat_ws('-', f.col('diseaseId'), f.col('targetId')))
        .withColumnRenamed('associationScore', 'score')
        .select('associationId', 'targetId', 'diseaseId', 'score')
    )

    assoc_with_drugs_from_evidence = (
        evidence_raw
        .filter(f.col('drugId').isNotNull())
        .selectExpr('drugId', 'targetId', 'diseaseId')
        .withColumn('associationId', f.concat_ws('-', f.col('diseaseId'), f.col('targetId')))
        .groupBy('associationId')
        .agg(
            f.collect_set('drugId').alias('drugIds'),
            f.first('targetId').alias('targetId'),
            f.first('diseaseId').alias('diseaseId'),
        )
    )

    total_assocs_with_drugs = assoc_with_drugs_from_evidence.count()

    assoc_drugs_with_scores = (
        assoc_with_drugs_from_evidence
        .join(association_scores.select('associationId', 'score'), 'associationId', 'inner')
        .withColumn('drugId', f.explode('drugIds'))
        .select('associationId', 'drugId', 'drugIds', 'targetId', 'diseaseId', 'score')
    )

    assoc_drugs = assoc_drugs_with_scores.groupBy('drugId').agg(
        f.collect_set('targetId').alias('targetIds'),
        f.collect_set('diseaseId').alias('diseaseIds'),
        f.mean('score').alias('meanScore'),
        (f.count('associationId').cast(DoubleType()) / f.lit(float(total_assocs_with_drugs))).alias('drug_relevance'),
    )

    logger.info('Building search index for diseases')
    search_diseases = _build_disease_index(
        diseases,
        phenotype_names,
        association_scores,
        assoc_drugs_with_scores,
        t_lut,
        dr_lut,
        studies,
    )

    logger.info('Building search index for targets')
    search_targets = _build_target_index(
        targets,
        association_scores,
        d_lut,
        dr_lut,
        variants,
    )

    logger.info('Building search index for drugs')
    search_drugs = drugs.transform(lambda df: _build_drug_index(df, assoc_drugs, t_lut, d_lut))

    logger.info('Building search index for variants')
    search_variants = _build_variant_index(variants).repartition(100)

    logger.info('Building search index for studies')
    search_studies = _build_study_index(studies, targets, credible_sets).repartition(100)

    logger.info('Writing outputs')
    search_diseases.write.mode('overwrite').parquet(destination['diseases'])
    search_targets.write.mode('overwrite').parquet(destination['targets'])
    search_drugs.write.mode('overwrite').parquet(destination['drugs'])
    search_variants.write.mode('overwrite').parquet(destination['variants'])
    search_studies.write.mode('overwrite').parquet(destination['studies'])
