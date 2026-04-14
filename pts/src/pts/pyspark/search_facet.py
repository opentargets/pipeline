"""Search facet dataset generation.

Ported from platform-etl-backend searchFacet step. Computes facets for
targets and diseases used by the Open Targets Platform search.

Scala sources ported:
    - SearchFacet.scala (main assembly)
    - TargetFacets.scala
    - DiseaseFacets.scala
    - Helpers.scala
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session

# ---------------------------------------------------------------------------
# Category label defaults (mirrors reference.conf categories block)
# ---------------------------------------------------------------------------

DEFAULT_CATEGORIES: dict[str, str] = {
    'disease_name': 'Disease',
    'therapeutic_area': 'Therapeutic Area',
    'sm': 'Tractability Small Molecule',
    'ab': 'Tractability Antibody',
    'pr': 'Tractability PROTAC',
    'oc': 'Tractability Other Modalities',
    'target_id': 'Target ID',
    'approved_symbol': 'Approved Symbol',
    'approved_name': 'Approved Name',
    'subcellular_location': 'Subcellular Location',
    'target_class': 'ChEMBL Target Class',
    'pathways': 'Reactome',
    'go_f': 'GO:MF',
    'go_p': 'GO:BP',
    'go_c': 'GO:CC',
}


# ---------------------------------------------------------------------------
# Disease facets
# ---------------------------------------------------------------------------


def _compute_disease_name_facets(disease_df: DataFrame, categories: dict[str, str]) -> DataFrame:
    """Compute disease name facets.

    Each disease produces one row with label=name, category='Disease',
    datasourceId=disease id, entityIds=[id].

    Args:
        disease_df: DataFrame with columns id, name.
        categories: category label mapping.

    Returns:
        DataFrame with columns label, category, datasourceId, entityIds.
    """
    logger.info('Computing disease name facets')
    return (
        disease_df
        .select(f.col('id'), f.col('name').alias('label'))
        .withColumn('category', f.lit(categories['disease_name']))
        .withColumn('datasourceId', f.col('id'))
        .groupBy('label', 'category', 'datasourceId')
        .agg(f.collect_set('id').alias('entityIds'))
    )


def _compute_therapeutic_areas_facets(disease_df: DataFrame, categories: dict[str, str]) -> DataFrame:
    """Compute therapeutic area facets.

    Explodes the therapeuticAreas array, joins back to disease names, then groups
    diseases by the therapeutic area they belong to.

    Args:
        disease_df: DataFrame with columns id, name, therapeuticAreas.
        categories: category label mapping.

    Returns:
        DataFrame with columns label, category, datasourceId, entityIds.
    """
    logger.info('Computing therapeutic areas facets')
    disease_names = disease_df.select(f.col('id'), f.col('name'))
    return (
        disease_df
        .where(f.col('therapeuticAreas').isNotNull())
        .select(f.col('id').alias('diseaseId'), f.explode('therapeuticAreas').alias('tId'))
        .join(disease_names, f.col('tId') == f.col('id'))
        .select(
            f.col('name').alias('label'),
            f.lit(categories['therapeutic_area']).alias('category'),
            f.col('diseaseId'),
            f.col('tId').alias('datasourceId'),
        )
        .groupBy('label', 'category', 'datasourceId')
        .agg(f.collect_set('diseaseId').alias('entityIds'))
    )


# ---------------------------------------------------------------------------
# Target facets
# ---------------------------------------------------------------------------


def _compute_tractability_facets(target_df: DataFrame, categories: dict[str, str]) -> DataFrame:
    """Compute tractability facets.

    Explodes the tractability array, filters to value=True, maps modality codes
    to human-readable category labels (SM→'Tractability Small Molecule', etc.).

    Args:
        target_df: DataFrame with columns id, tractability (array of structs with
            modality, id, value).
        categories: category label mapping.

    Returns:
        DataFrame with columns label, category, entityIds, datasourceId (null).
    """
    logger.info('Computing tractability facets')
    modality_map = f.create_map(
        f.lit('SM'),
        f.lit(categories['sm']),
        f.lit('AB'),
        f.lit(categories['ab']),
        f.lit('PR'),
        f.lit(categories['pr']),
        f.lit('OC'),
        f.lit(categories['oc']),
    )
    return (
        target_df
        .where(f.col('tractability').isNotNull())
        .select(f.col('id'), f.explode('tractability').alias('t'))
        .select(
            f.col('id').alias('ensemblGeneId'),
            f.col('t.modality').alias('category'),
            f.col('t.id').alias('label'),
            f.col('t.value').alias('value'),
        )
        .where(f.col('value'))
        .groupBy('category', 'label')
        .agg(f.collect_set('ensemblGeneId').alias('entityIds'))
        .withColumn(
            'category',
            f.when(modality_map[f.col('category')].isNotNull(), modality_map[f.col('category')]).otherwise(
                f.col('category')
            ),
        )
        .withColumn('datasourceId', f.lit(None).cast('string'))
    )


def _compute_go_facets(target_df: DataFrame, go_df: DataFrame, categories: dict[str, str]) -> DataFrame:
    """Compute Gene Ontology facets.

    Explodes the go array, joins with the GO reference DataFrame to get labels,
    maps aspect codes (F/P/C) to category labels (GO:MF/GO:BP/GO:CC).

    Args:
        target_df: DataFrame with columns id, go (array of structs with id, aspect).
        go_df: Reference GO DataFrame with columns id, label.
        categories: category label mapping.

    Returns:
        DataFrame with columns label, category, entityIds, datasourceId (GO id).
    """
    logger.info('Computing GO facets')
    aspect_map = f.create_map(
        f.lit('F'),
        f.lit(categories['go_f']),
        f.lit('P'),
        f.lit(categories['go_p']),
        f.lit('C'),
        f.lit(categories['go_c']),
    )
    return (
        target_df
        .where(f.col('go').isNotNull())
        .select(f.col('id').alias('ensemblGeneId'), f.explode('go').alias('g'))
        .select(
            f.col('ensemblGeneId'),
            f.col('g.id').alias('id'),
            f.col('g.aspect').alias('aspect'),
        )
        .join(go_df, 'id', 'left')
        .where(f.col('label').isNotNull())
        .withColumn('datasourceId', f.col('id'))
        .groupBy('label', 'aspect', 'datasourceId')
        .agg(f.collect_set('ensemblGeneId').alias('entityIds'))
        .withColumn(
            'category',
            f.when(aspect_map[f.col('aspect')].isNotNull(), aspect_map[f.col('aspect')]).otherwise(f.col('aspect')),
        )
        .drop('aspect')
    )


def _compute_subcellular_location_facets(target_df: DataFrame, categories: dict[str, str]) -> DataFrame:
    """Compute subcellular location facets.

    Explodes the subcellularLocations array, using location as label and termSl
    as datasourceId.

    Args:
        target_df: DataFrame with columns id, subcellularLocations (array of structs
            with location, termSl).
        categories: category label mapping.

    Returns:
        DataFrame with columns label, category, entityIds, datasourceId.
    """
    logger.info('Computing subcellular location facets')
    return (
        target_df
        .where(f.col('subcellularLocations').isNotNull())
        .select(f.col('id'), f.explode('subcellularLocations').alias('s'))
        .select(
            f.col('id'),
            f.col('s.location').alias('label'),
            f.lit(categories['subcellular_location']).alias('category'),
            f.col('s.termSl').alias('datasourceId'),
        )
        .groupBy('label', 'category', 'datasourceId')
        .agg(f.collect_set('id').alias('entityIds'))
    )


def _compute_target_class_facets(target_df: DataFrame, categories: dict[str, str]) -> DataFrame:
    """Compute target class facets.

    Explodes the targetClass array, using the class label as facet label.

    Args:
        target_df: DataFrame with columns id, targetClass (array of structs with
            id, label, level).
        categories: category label mapping.

    Returns:
        DataFrame with columns label, category, entityIds, datasourceId (null).
    """
    logger.info('Computing target class facets')
    return (
        target_df
        .where(f.col('targetClass').isNotNull())
        .select(f.col('id').alias('ensemblGeneId'), f.explode('targetClass').alias('tc'))
        .select(
            f.col('ensemblGeneId'),
            f.col('tc.label').alias('label'),
            f.lit(categories['target_class']).alias('category'),
        )
        .groupBy('label', 'category')
        .agg(f.collect_set('ensemblGeneId').alias('entityIds'))
        .withColumn('datasourceId', f.lit(None).cast('string'))
    )


def _compute_pathway_facets(target_df: DataFrame, categories: dict[str, str]) -> DataFrame:
    """Compute Reactome pathway facets.

    Explodes the pathways array, using pathway name as label and pathwayId as
    datasourceId.

    Args:
        target_df: DataFrame with columns id, pathways (array of structs with
            pathwayId, pathway, topLevelTerm).
        categories: category label mapping.

    Returns:
        DataFrame with columns label, category, entityIds, datasourceId.
    """
    logger.info('Computing pathway facets')
    return (
        target_df
        .where(f.col('pathways').isNotNull())
        .select(f.col('id').alias('ensemblGeneId'), f.explode('pathways').alias('p'))
        .select(
            f.col('ensemblGeneId'),
            f.col('p.pathway').alias('label'),
            f.lit(categories['pathways']).alias('category'),
            f.col('p.pathwayId').alias('datasourceId'),
        )
        .groupBy('label', 'category', 'datasourceId')
        .agg(f.collect_set('ensemblGeneId').alias('entityIds'))
    )


def _compute_simple_facet(
    df: DataFrame,
    label_field: str,
    category_value: str,
    entity_id_field: str,
) -> DataFrame:
    """Compute a simple facet from a flat column.

    Args:
        df: Source DataFrame.
        label_field: Column to use as facet label.
        category_value: Literal string for the category column.
        entity_id_field: Column to collect as entity IDs.

    Returns:
        DataFrame with columns label, category, entityIds, datasourceId (null).
    """
    return (
        df
        .select(
            f.col(label_field).alias('label'),
            f.lit(category_value).alias('category'),
            f.col(entity_id_field).alias('_eid'),
        )
        .groupBy('label', 'category')
        .agg(f.collect_set('_eid').alias('entityIds'))
        .withColumn('datasourceId', f.lit(None).cast('string'))
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def search_facet(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Compute search facets for targets and diseases.

    Reads disease, target, and GO datasets, computes facets for each, and writes
    results to the configured destination paths.

    Args:
        source: Input paths keyed by 'diseases', 'targets', 'go'.
        destination: Output paths keyed by 'targets', 'diseases'.
        settings: Step settings; may contain a 'categories' sub-dict to override
            the default category labels.
        properties: Spark/GCS properties forwarded to Session.
    """
    spark = Session(app_name='search_facet', properties=properties).spark

    categories = {**DEFAULT_CATEGORIES, **(settings.get('categories') or {})}

    logger.info('Loading disease data from %s', source['diseases'])
    disease_df = spark.read.parquet(source['diseases'])

    logger.info('Loading target data from %s', source['targets'])
    target_df = spark.read.parquet(source['targets'])

    logger.info('Loading GO data from %s', source['go'])
    go_df = spark.read.parquet(source['go'])

    # ---- disease facets ----
    disease_facets = _compute_disease_name_facets(disease_df, categories).unionByName(
        _compute_therapeutic_areas_facets(disease_df, categories)
    )

    # ---- target facets ----
    target_facets = (
        _compute_simple_facet(target_df, 'id', categories['target_id'], 'id')
        .unionByName(_compute_simple_facet(target_df, 'approvedSymbol', categories['approved_symbol'], 'id'))
        .unionByName(_compute_simple_facet(target_df, 'approvedName', categories['approved_name'], 'id'))
        .unionByName(_compute_go_facets(target_df, go_df, categories))
        .unionByName(_compute_subcellular_location_facets(target_df, categories))
        .unionByName(_compute_target_class_facets(target_df, categories))
        .unionByName(_compute_pathway_facets(target_df, categories))
        .unionByName(_compute_tractability_facets(target_df, categories))
    )

    logger.info('Writing target facets to %s', destination['targets'])
    target_facets.coalesce(200).write.mode('overwrite').parquet(destination['targets'])

    logger.info('Writing disease facets to %s', destination['diseases'])
    disease_facets.coalesce(200).write.mode('overwrite').parquet(destination['diseases'])
