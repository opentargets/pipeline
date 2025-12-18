from __future__ import annotations

from functools import reduce
from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session
from pts.pyspark.expression_utils.expression import (
    split_valid_invalid,
    validate_biosample,
    validate_target,
)
from pts.pyspark.expression_utils.validation_lut import (
    prepare_biosample_lut,
    prepare_target_lut,
)


def _resolve_pride_target_ids(
    spark: SparkSession,
    merged_df: DataFrame,
    target_index_path: str | None,
) -> DataFrame:
    if target_index_path is None:
        logger.warning('target_index not provided in source config. Skipping PRIDE target ID resolution.')
        return merged_df

    if 'datasourceId' not in merged_df.columns:
        logger.warning(
            'Missing datasourceId column in merged baseline expression. Skipping PRIDE target ID resolution.'
        )
        return merged_df

    if 'targetFromSourceId' not in merged_df.columns:
        logger.warning(
            'Missing targetFromSourceId column in merged baseline expression. Skipping PRIDE target ID resolution.'
        )
        return merged_df

    logger.info('Resolving PRIDE UniProt IDs to Ensembl target IDs during merge')

    target_mapping = (
        spark.read
        .parquet(target_index_path)
        .select(
            f.col('id').alias('resolvedTargetId'),
            f.explode(f.col('proteinIds')).alias('proteinId'),
        )
        .select(
            f.col('proteinId.id').alias('proteinId'),
            f.col('resolvedTargetId'),
        )
        .filter(f.col('proteinId').isNotNull())
    )

    pride_rows = merged_df.filter(f.col('datasourceId') == 'PRIDE')
    non_pride_rows = merged_df.filter(f.col('datasourceId') != 'PRIDE')

    pride_rows = (
        pride_rows
        .drop('targetId')
        .join(target_mapping, pride_rows['targetFromSourceId'] == target_mapping['proteinId'], how='left')
        .drop('proteinId')
        .withColumn('targetId', f.coalesce(f.col('resolvedTargetId'), f.col('targetFromSourceId')))
        .drop('resolvedTargetId')
    )

    return non_pride_rows.unionByName(pride_rows, allowMissingColumns=True)


def _override_biosample_id(
    spark: SparkSession,
    df: DataFrame,
    biosample_mapping_reference_path: str,
    key_col: str,
    source_col: str,
    id_col: str,
) -> DataFrame:

    mapping_df = (
        spark.read
        .option('header', True)
        .option('inferSchema', True)
        .option('sep', '\t')
        .csv(biosample_mapping_reference_path)
    ).drop('Count')

    other_cols = [column_name for column_name in mapping_df.columns if column_name != key_col]

    def norm(column_name: str):
        return f.regexp_replace(f.lower(f.col(column_name)), r'[^a-z0-9]', '')

    def pretty(column_name: str):
        value = f.coalesce(f.col(column_name).cast('string'), f.lit(''))
        value = f.regexp_replace(value, r'[_-]+', ' ')
        value = f.regexp_replace(value, r'([A-Z])([A-Z][a-z])', r'\1 \2')
        value = f.regexp_replace(value, r'(?<=[a-z])(?=[A-Z])', ' ')
        value = f.regexp_replace(value, r'(?<=[a-z])(?=\d)', ' ')
        value = f.regexp_replace(value, r'(?<=\d)(?=[a-z])', ' ')
        value = f.regexp_replace(value, r'[.,]', '')
        value = f.regexp_replace(value, r'\s+', ' ')
        value = f.trim(value)

        words = f.split(value, ' ')
        words = f.transform(
            words,
            lambda item: f.when(
                item.rlike(r'^[A-Z0-9]{2,}$') | item.rlike(r'^[A-Z]$'),
                item,
            ).otherwise(f.lower(item)),
        )
        return f.array_join(words, ' ')

    def clean_tokens(column_name: str, delim: str = ';'):
        value = f.coalesce(f.col(column_name).cast('string'), f.lit(''))
        values = f.split(value, delim)
        values = f.transform(values, lambda item: f.trim(item))
        return f.filter(values, lambda item: item.isNotNull() & (item != ''))  # noqa: PLC1901

    long_parts = [
        mapping_df.select(f.col(key_col), f.explode_outer(clean_tokens(column_name)).alias('value'))
        for column_name in other_cols
    ]
    mapping_long = reduce(lambda left, right: left.unionByName(right), long_parts).filter(
        f.col('value').isNotNull() & (f.length(f.col('value')) > 0)
    )

    enriched = (
        mapping_long
        .withColumn('norm', norm('value'))
        .withColumn('pretty', pretty('value'))
        .filter(f.length(f.col('norm')) > 0)
        .cache()
    )
    min_key_by_norm = enriched.groupBy('norm').agg(f.min(f.col(key_col)).alias(key_col))
    mapping_norm = (
        enriched
        .join(min_key_by_norm, on=['norm', key_col], how='inner')
        .select('norm', key_col, 'pretty')
        .dropDuplicates(['norm'])
    )
    enriched.unpersist()

    return (
        df
        .withColumn('norm', norm(source_col))
        .drop(id_col)
        .join(f.broadcast(mapping_norm), on='norm', how='left')
        .withColumn(id_col, f.col(key_col))
        .withColumn(source_col, f.coalesce(f.col('pretty'), f.col(source_col)))
        .drop('norm', key_col, 'pretty')
    )


def _add_parental_biosample_id(
    spark: SparkSession,
    df: DataFrame,
    biosample_index_path: str,
    potential_parents_path: str,
    id_col: str,
) -> DataFrame:
    if id_col == 'tissueBiosampleId':
        parent_col_name = 'tissueBiosampleParentId'
        source_col = 'tissueBiosampleFromSource'
    elif id_col == 'celltypeBiosampleId':
        parent_col_name = 'celltypeBiosampleParentId'
        source_col = 'celltypeBiosampleFromSource'
    else:
        raise ValueError(
            f'Unexpected id_col "{id_col}". '
            'Accepted values: "tissueBiosampleId", "celltypeBiosampleId".'
        )

    if id_col not in df.columns:
        logger.warning(f'Missing {id_col} column in merged baseline expression. Skipping {parent_col_name} resolution.')
        return df

    if source_col not in df.columns:
        logger.warning(
            f'Missing {source_col} column in merged baseline expression. Skipping {parent_col_name} resolution.'
        )
        return df

    biosample_index = spark.read.parquet(biosample_index_path)
    potential_parents_df = (
        spark.read.option('header', True).option('inferSchema', True).option('sep', '\t').csv(potential_parents_path)
    )

    parent_col = potential_parents_df.columns[0]
    potential_parents_list = [row[parent_col] for row in potential_parents_df.select(parent_col).collect()]

    parent_ancestor_counts = (
        biosample_index
        .filter(f.col('biosampleId').isin(potential_parents_list))
        .select('biosampleId', f.size(f.coalesce(f.col('ancestors'), f.array())).alias('ancestor_count'))
        .collect()
    )
    count_map = {row.biosampleId: row.ancestor_count for row in parent_ancestor_counts}
    # Higher ancestor_count => more specific => higher priority (lower priority value).
    potential_parents_list.sort(key=lambda value: count_map.get(value, -1), reverse=True)

    # Build a small (parent_id, priority) lookup DataFrame
    priority_rows = [(pid, idx) for idx, pid in enumerate(potential_parents_list)]
    priority_df = spark.createDataFrame(priority_rows, ['__parent_candidate', '__parent_priority'])

    # For each biosample in the index, expand to (biosampleId, candidate) over self plus ancestors,
    # join to the priority list, and keep the highest-priority match per biosample.
    biosample_to_parent = (
        biosample_index
        .select(
            f.col('biosampleId'),
            f.array_union(
                f.array(f.col('biosampleId')),
                f.coalesce(f.col('ancestors'), f.array().cast('array<string>')),
            ).alias('__candidates'),
        )
        .select('biosampleId', f.explode('__candidates').alias('__parent_candidate'))
        .join(f.broadcast(priority_df), on='__parent_candidate', how='inner')
        .groupBy('biosampleId')
        .agg(f.min(f.struct('__parent_priority', '__parent_candidate')).alias('__best'))
        .select(
            f.col('biosampleId').alias(id_col),
            f.col('__best.__parent_candidate').alias(parent_col_name),
        )
    )

    df_with_parent = df.join(biosample_to_parent, on=id_col, how='left').cache()

    null_parent_df = df_with_parent.filter(f.col(parent_col_name).isNull())
    if null_parent_df.take(1):
        logger.info(f'Using override mapping as fallback for null {parent_col_name} values')
        override_df = _override_biosample_id(
            spark=spark,
            df=null_parent_df,
            biosample_mapping_reference_path=potential_parents_path,
            key_col='BiosampleId',
            source_col=source_col,
            id_col=parent_col_name,
        )
        non_null_parent_df = df_with_parent.filter(f.col(parent_col_name).isNotNull())
        return non_null_parent_df.unionByName(override_df, allowMissingColumns=True)

    logger.info(f'No null {parent_col_name} found, skipping override fallback')
    return df_with_parent


def _merge_baseline_expression(
    spark: SparkSession,
    source: dict[str, str],
    settings: dict[str, Any],
) -> DataFrame:
    aggregation = settings.get('aggregation')

    raw_datasets = settings.get('datasets', [])
    if isinstance(raw_datasets, str):
        raw_datasets = raw_datasets.split(',')
    datasets = [d.strip().strip('/') for d in raw_datasets if d.strip()]

    source_root = source.get('baseline_expression', '').rstrip('/')

    target_index_path = source.get('target_index')

    biosample_index_path = source.get('biosample_index')
    tissue_parents_path = source.get('tissue_parents')
    celltype_parents_path = source.get('celltype_parents')

    input_paths = [f'{source_root}/{aggregation}/{d}/parquet' for d in datasets]

    logger.info(f'Resolved {len(input_paths)} baseline expression parquet inputs')
    for input_path in input_paths:
        logger.info(f'input path: {input_path}')
    logger.info(f'target index path: {target_index_path}')
    if biosample_index_path:
        logger.info(f'biosample index path: {biosample_index_path}')
    if tissue_parents_path:
        logger.info(f'tissue parents path: {tissue_parents_path}')
    if celltype_parents_path:
        logger.info(f'celltype parents path: {celltype_parents_path}')

    merged_df = spark.read.option('mergeSchema', 'true').parquet(*input_paths)
    merged_df = _resolve_pride_target_ids(spark, merged_df, target_index_path)

    if tissue_parents_path and biosample_index_path:
        logger.info('Resolving tissue parental biosample IDs during merge')
        merged_df = _add_parental_biosample_id(
            spark=spark,
            df=merged_df,
            biosample_index_path=biosample_index_path,
            potential_parents_path=tissue_parents_path,
            id_col='tissueBiosampleId',
        )

    if celltype_parents_path and biosample_index_path:
        logger.info('Resolving celltype parental biosample IDs during merge')
        merged_df = _add_parental_biosample_id(
            spark=spark,
            df=merged_df,
            biosample_index_path=biosample_index_path,
            potential_parents_path=celltype_parents_path,
            id_col='celltypeBiosampleId',
        )

    return merged_df


def baseline_expression(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression merge and validate')

    session = Session(app_name='baseline_expression_merge', properties=properties)
    try:
        merged_df = _merge_baseline_expression(session.spark, source, settings)

        target_lut = prepare_target_lut(session.load_data(source['target_index'], format='parquet'))
        biosample_lut = prepare_biosample_lut(session.load_data(source['biosample_index'], format='parquet'))

        validated = validate_target(merged_df, target_lut)
        validated = validate_biosample(validated, biosample_lut, 'tissueBiosampleId', 'tissueBiosampleFromSource')
        validated = validate_biosample(validated, biosample_lut, 'celltypeBiosampleId', 'celltypeBiosampleFromSource')
        validated = validate_biosample(validated, biosample_lut, 'tissueBiosampleParentId', 'tissueBiosampleFromSource')
        validated = validate_biosample(
            validated, biosample_lut, 'celltypeBiosampleParentId', 'celltypeBiosampleFromSource'
        )
        validated = validated.persist()

        try:
            valid, invalid = split_valid_invalid(validated)
            valid.write.mode('overwrite').parquet(destination['valid'])
            invalid.write.mode('overwrite').parquet(destination['failed'])
        finally:
            validated.unpersist()
    finally:
        session.stop()

    logger.info('Baseline expression merge and validate completed successfully')
