"""Association on-the-fly (OTF) enrichment for ClickHouse.

Enriches evidence records with hierarchical facet data from diseases and targets.
Ported from the Scala AssociationOTF class in platform-etl-backend.
"""

from __future__ import annotations

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session


def _compute_facet_classes(df: DataFrame) -> DataFrame:
    """Compute target class facets at L1 and L2 levels.

    Filters the targetClass array for entries at levels "l1" and "l2", groups
    them into structs with {l1, l2} labels, and aggregates per target.

    Args:
        df: DataFrame with `targetId` and `targetClass` columns.

    Returns:
        DataFrame: Original df with `targetClass` replaced by `facetClasses`.
    """
    fc_df = (
        df.select(
            f.col('targetId'),
            f.explode(
                f.filter(
                    f.col('targetClass'),
                    lambda c: (c['level'] == 'l1') | (c['level'] == 'l2'),
                )
            ).alias('fc'),
        )
        .select('targetId', 'fc.*')
        .orderBy('targetId', 'id', 'level')
        .groupBy('targetId', 'id')
        .agg(f.collect_list('label').alias('levels'))
        .withColumn(
            'facetClasses',
            f.struct(
                f.col('levels')[0].alias('l1'),
                f.col('levels')[1].alias('l2'),
            ),
        )
        .groupBy('targetId')
        .agg(f.collect_list('facetClasses').alias('facetClasses'))
        .orderBy('targetId')
    )

    return df.drop('targetClass').join(fc_df, ['targetId'], 'left_outer')


def _compute_facet_therapeutic_areas(df: DataFrame, key_col: str, label_col: str, vec_col: str) -> DataFrame:
    """Compute facet therapeutic areas by resolving IDs to labels.

    Explodes a vector column of IDs, joins each ID with its label from the
    same dataframe, and collects unique labels per key.

    Args:
        df: Input DataFrame.
        key_col: Column name for the grouping key (e.g. "diseaseId").
        label_col: Column name containing the label (e.g. "name").
        vec_col: Column name containing the array of IDs (e.g. "therapeuticAreas").

    Returns:
        DataFrame: With columns (key_col, vec_col) where vec_col contains resolved labels.
    """
    ta_id = f'{vec_col}_tmp'

    tas = df.select(key_col, vec_col).withColumn(ta_id, f.explode_outer(f.col(vec_col))).drop(vec_col)

    labels = df.select(key_col, label_col).withColumnRenamed(key_col, ta_id)

    return (
        tas.join(labels, [ta_id], 'left_outer')
        .groupBy(f.col(key_col))
        .agg(f.collect_set(f.col(label_col)).alias(vec_col))
    )


def _compute_facet_tractability(df: DataFrame) -> DataFrame:
    """Compute tractability facets by modality.

    Filters the tractability array for each modality (SM, AB, PR, OC) where
    value is true, and extracts the corresponding IDs into separate columns.

    Args:
        df: DataFrame with `targetId` and `tractability` columns.

    Returns:
        DataFrame: Original df enriched with facet tractability columns.
    """

    def facet_filter(modality: str):
        return lambda c: (c['value'] == True) & (c['modality'] == modality)  # noqa: E712

    tractability_facets_df = (
        df.select(
            f.col('targetId'),
            f.filter(f.col('tractability'), facet_filter('SM')).alias('sm'),
            f.filter(f.col('tractability'), facet_filter('AB')).alias('ab'),
            f.filter(f.col('tractability'), facet_filter('PR')).alias('pr'),
            f.filter(f.col('tractability'), facet_filter('OC')).alias('oc'),
            f.monotonically_increasing_id().alias('miid'),
        )
        .select(
            f.col('targetId'),
            f.col('sm.id').alias('facetTractabilitySmallmolecule'),
            f.col('ab.id').alias('facetTractabilityAntibody'),
            f.col('pr.id').alias('facetTractabilityProtac'),
            f.col('oc.id').alias('facetTractabilityOthermodalities'),
            f.col('miid'),
        )
        .orderBy('targetId')
    )

    return df.join(tractability_facets_df, ['targetId'], 'left_outer').orderBy('miid').drop('miid')


def association_otf(
    source: dict[str, str], destination: str, settings: dict[str, str] | None, properties: dict[str, str]
) -> None:
    """Enrich evidence with disease and target facet data for ClickHouse.

    Reads diseases, targets, and evidences datasets, computes facet enrichments
    (therapeutic areas, target classes, tractability, reactome pathways), and
    joins them onto evidence records.

    Args:
        source: Dict with keys "diseases", "targets", "evidences" mapping to paths.
        destination: Output parquet path.
        settings: Step-specific settings (unused, kept for interface consistency).
        properties: Spark configuration properties.
    """
    spark = Session(app_name='association_otf', properties=properties)

    logger.info(f'Loading data from {source}')
    diseases_raw = spark.load_data(source['disease'])
    targets_raw = spark.load_data(source['target'])
    evidences_raw = spark.load_data(source['evidence'])
    assocation_direct = spark.load_data(source['association_direct'])
    assocation_indirect = spark.load_data(source['association_indirect'])

    # Process diseases: select relevant columns and compute therapeutic area facets.
    diseases = (
        diseases_raw.select(
            f.col('id').alias('diseaseId'),
            f.concat(f.col('id'), f.lit(' '), f.col('name')).alias('diseaseData'),
            f.col('therapeuticAreas'),
            f.col('name'),
        )
        .orderBy(f.col('diseaseId').asc())
        .persist()
    )

    diseases_facet_tas = _compute_facet_therapeutic_areas(
        diseases, 'diseaseId', 'name', 'therapeuticAreas'
    ).withColumnRenamed('therapeuticAreas', 'facetTherapeuticAreas')

    final_diseases = diseases.join(diseases_facet_tas, ['diseaseId'], 'left_outer').drop('therapeuticAreas', 'name')

    # Process targets: select relevant columns and compute class, tractability,
    # and reactome pathway facets.
    targets = (
        targets_raw.select(
            f.col('id').alias('targetId'),
            f.concat(
                f.col('id'),
                f.lit(' '),
                f.col('approvedName'),
                f.lit(' '),
                f.col('approvedSymbol'),
            ).alias('targetData'),
            f.col('targetClass'),
            f.col('pathways').alias('reactome'),
            f.col('tractability'),
        )
        .orderBy(f.col('targetId').asc())
        .persist()
    )

    targets_facet_reactome = targets.select(
        f.col('targetId'),
        f.transform(
            f.col('reactome'),
            lambda r: f.struct(
                r['topLevelTerm'].alias('l1'),
                r['pathway'].alias('l2'),
            ),
        ).alias('facetReactome'),
    )

    final_targets = (
        targets.transform(_compute_facet_classes)
        .transform(_compute_facet_tractability)
        .join(targets_facet_reactome, ['targetId'], 'left_outer')
        .drop('reactome', 'tractability')
    )

    # Process direct and indirect assocation data:
    novelty_direct = assocation_direct.select('targetId', 'diseaseId', f.col('currentNovelty').alias('noveltyDirect'))

    novelty_indirect = assocation_indirect.select(
        'targetId', 'diseaseId', f.col('currentNovelty').alias('noveltyIndirect')
    )

    # Join evidences with enriched disease and target data.
    evidences = evidences_raw.select(
        f.col('id').alias('rowId'),
        'diseaseId',
        'targetId',
        'datasourceId',
        'datatypeId',
        f.col('score').alias('rowScore'),
    )

    result = (
        evidences.join(final_diseases, ['diseaseId'], 'left_outer')
        .join(final_targets, ['targetId'], 'left_outer')
        .join(novelty_direct, on=['diseaseId', 'targetId'], how='left')
        .join(novelty_indirect, on=['diseaseId', 'targetId'], how='left')
        .select(
            'rowId',
            'diseaseId',
            'targetId',
            'datasourceId',
            'datatypeId',
            'rowScore',
            'diseaseData',
            'targetData',
            'noveltyDirect',
            'noveltyIndirect',
        )
    )

    logger.info(f'Writing association OTF data to {destination}')
    result.write.parquet(destination, mode='overwrite')
