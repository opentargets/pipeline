"""Baseline expression dataset generation.

Ported from Expression.scala in platform-etl-backend.
Combines HPA normal tissue protein expression with baseline RNA expression
data to produce a per-gene tissues array.
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session

# Maps from Scala Expression.scala transformNormalTissue
_RELIABILITY_MAP = {
    'Supportive': True,
    'Uncertain': False,
    'Approved': True,
    'Supported': True,
    'Enhanced': True,
}

_LEVEL_MAP = {
    'Not detected': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'N/A': 0,
    'Not representative': 0,
}


def _transform_normal_tissue(df: DataFrame) -> DataFrame:
    """Filter N/A levels, normalise column names, and map reliability/level to bool/int.

    Mirrors transformNormalTissue in Expression.scala.

    Args:
        df: Raw normal tissue DataFrame (TSV columns may contain spaces).

    Returns:
        DataFrame with ReliabilityMap (bool) and LevelMap (int) columns, N/A rows removed.
    """
    # Normalise column names: replace spaces with underscores
    for col in df.columns:
        if ' ' in col:
            df = df.withColumnRenamed(col, col.replace(' ', '_'))

    reliability_map_col = f.create_map(*[item for k, v in _RELIABILITY_MAP.items() for item in (f.lit(k), f.lit(v))])
    level_map_col = f.create_map(*[item for k, v in _LEVEL_MAP.items() for item in (f.lit(k), f.lit(v))])

    return (
        df
        .filter(f.col('Level') != 'N/A')
        .withColumn('ReliabilityMap', reliability_map_col[f.col('Reliability')])
        .withColumn('LevelMap', level_map_col[f.col('Level')])
    )


def _transpose_baseline(df: DataFrame, unit: str) -> DataFrame:
    """Pivot a wide baseline TSV (genes as columns, tissues as rows) into long format.

    Uses explode(array(struct(...))) — same approach as Scala transposeDataframe in Helpers.scala.

    Args:
        df: Wide DataFrame with ID column (tissue names) and gene columns.
        unit: Unit label to attach to all rows (e.g. 'TPM' for RNA, '' for binned/zscore).

    Returns:
        Long-format DataFrame with columns [Gene, Tissue, val, unit].
    """
    id_col = 'ID'
    gene_cols = [c for c in df.columns if c != id_col]
    return df.select(
        f.col(id_col),
        f.explode(
            f.array(*[f.struct(f.lit(c).alias('key'), f.col(c).cast('double').alias('val')) for c in gene_cols])
        ).alias('_kvs'),
    ).select(
        f.col(id_col).alias('Gene'),
        f.col('_kvs.key').alias('Tissue'),
        f.col('_kvs.val').alias('val'),
        f.lit(unit).alias('unit'),
    )


def _efo_tissue_mapping(efomap: DataFrame, hierarchy: DataFrame) -> DataFrame:
    """Join EFO tissue map with expression hierarchy.

    Mirrors efoTissueMapping in Expression.scala.

    Args:
        efomap: EFO map parquet with tissue_id and efo_code/label columns.
        hierarchy: Expression hierarchy TSV with two unnamed columns.

    Returns:
        DataFrame with efoId and labelNew columns added.
    """
    expr_renamed = hierarchy.withColumnRenamed('_c0', 'expressionId').withColumnRenamed('_c1', 'name')
    map_renamed = efomap.withColumnRenamed('tissue_id', 'tissue_internal_id')
    return (
        map_renamed
        .join(expr_renamed, f.col('name') == f.col('tissue_internal_id'), 'full')
        .withColumn('efoId', f.when(f.col('efo_code').isNull(), f.col('name')).otherwise(f.col('efo_code')))
        .withColumn('labelNew', f.when(f.col('label').isNull(), f.col('name')).otherwise(f.col('label')))
        .withColumn('labelLower', f.lower(f.col('labelNew')))
        .withColumn('expressionId', f.lower(f.col('expressionId')))
    )


def expression(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate baseline expression dataset.

    Reads HPA normal tissue, baseline RNA expression TSVs, EFO tissue map, and expression
    hierarchy to produce a per-gene tissues array suitable for the Open Targets Platform.

    Args:
        source: Dict with keys: rna, binned, zscore, exprhierarchy, efomap, tissues.
        destination: Output parquet path.
        settings: Unused settings dict.
        properties: Spark session properties.
    """
    spark = Session(app_name='expression', properties=properties).spark

    logger.info('Reading expression inputs')
    rna = spark.read.option('sep', '\t').option('header', 'true').csv(source['rna'])
    binned = spark.read.option('sep', '\t').option('header', 'true').csv(source['binned'])
    zscore = spark.read.option('sep', '\t').option('header', 'true').csv(source['zscore'])
    hierarchy = spark.read.option('sep', '\t').option('header', 'false').csv(source['exprhierarchy'])
    efomap = spark.read.parquet(source['efomap'])
    tissues = spark.read.option('sep', '\t').option('header', 'true').csv(source['tissues'])

    normal = _transform_normal_tissue(tissues)

    # Build EFO tissue map (mirrors efoTissueMapping in Expression.scala)
    efo_map = _efo_tissue_mapping(efomap, hierarchy)

    # Transpose baseline expression files
    rna_long = _transpose_baseline(rna, 'TPM').withColumnRenamed('val', 'rna')
    binned_long = _transpose_baseline(binned, '').withColumnRenamed('val', 'binned').drop('unit')
    zscore_long = _transpose_baseline(zscore, '').withColumnRenamed('val', 'zscore').drop('unit')

    baseline = rna_long.join(binned_long, ['Gene', 'Tissue'], 'full').join(zscore_long, ['Gene', 'Tissue'], 'full')

    # Join normal tissue with EFO map (mirrors selectTissues + generateBaselineInfo in Expression.scala)
    normal_lower = normal.withColumn('Tissue', f.lower(f.col('Tissue')))

    by_label = normal_lower.join(efo_map, f.col('labelLower') == f.col('Tissue'), 'left')
    by_expr = normal_lower.join(efo_map, f.col('expressionId') == f.col('Tissue'), 'left')
    valid = by_label.unionByName(by_expr, allowMissingColumns=True).dropDuplicates()

    # Join HPA (valid) with RNA baseline on (Gene, Tissue) — matching the Scala
    # string-based join. Then map baseline tissues to EFO codes separately for
    # RNA-only genes that don't appear in HPA.
    baseline_lower = baseline.withColumn('Tissue', f.lower(f.col('Tissue')))

    tissue_base = valid.join(
        baseline_lower,
        (valid['Gene'] == baseline_lower['Gene']) & (valid['Tissue'] == baseline_lower['Tissue']),
        'full',
    ).select(
        f.coalesce(valid['Gene'], baseline_lower['Gene']).alias('Gene'),
        f.coalesce(valid['Tissue'], baseline_lower['Tissue']).alias('Tissue'),
        f.coalesce(f.col('LevelMap'), f.lit(-1)).alias('level'),
        f.col('Cell_type').alias('cell_type'),
        f.coalesce(f.col('ReliabilityMap'), f.lit(False)).alias('reliability'),
        f.coalesce(f.col('rna'), f.lit(0.0)).alias('rna_val'),
        f.coalesce(f.col('binned'), f.lit(-1.0)).alias('binned_val'),
        f.coalesce(f.col('zscore'), f.lit(-1.0)).alias('zscore_val'),
        f.coalesce(f.col('unit'), f.lit('')).alias('unit_val'),
        f.col('efoId'),
        f.col('labelNew').alias('labelDef'),
        f.col('anatomical_systems'),
        f.col('organs'),
    )

    # For rows that came only from baseline (no HPA match), efoId is null.
    # Map their tissue names to EFO codes via the EFO map.
    needs_efo = tissue_base.filter(f.col('efoId').isNull())
    has_efo = tissue_base.filter(f.col('efoId').isNotNull())

    mapped_by_label = needs_efo.join(
        efo_map.select(
            f.col('labelLower').alias('_ll'),
            f.col('efoId').alias('_efoId'),
            f.col('labelNew').alias('_labelNew'),
            f.col('anatomical_systems').alias('_as'),
            f.col('organs').alias('_org'),
        ),
        f.col('_ll') == needs_efo['Tissue'],
        'left',
    )
    mapped_by_expr = needs_efo.join(
        efo_map.select(
            f.col('expressionId').alias('_eid'),
            f.col('efoId').alias('_efoId'),
            f.col('labelNew').alias('_labelNew'),
            f.col('anatomical_systems').alias('_as'),
            f.col('organs').alias('_org'),
        ),
        f.col('_eid') == needs_efo['Tissue'],
        'left',
    )
    recovered = (
        mapped_by_label
        .unionByName(mapped_by_expr, allowMissingColumns=True)
        .filter(f.col('_efoId').isNotNull())
        .withColumn('efoId', f.col('_efoId'))
        .withColumn('labelDef', f.col('_labelNew'))
        .withColumn('anatomical_systems', f.col('_as'))
        .withColumn('organs', f.col('_org'))
        .drop('_ll', '_eid', '_efoId', '_labelNew', '_as', '_org')
        .dropDuplicates()
    )

    tissue_base = has_efo.unionByName(recovered).filter(f.col('efoId').isNotNull())

    protein = (
        tissue_base
        .groupBy('Gene', 'labelDef', 'efoId', 'anatomical_systems', 'organs')
        .agg(
            f.max('reliability').alias('reliability'),
            f.max('level').alias('level'),
            f.struct(
                f.max('rna_val').alias('value'),
                f.max('zscore_val').cast('int').alias('zscore'),
                f.max('binned_val').cast('int').alias('level'),
                f.max('unit_val').alias('unit'),
            ).alias('rna'),
            f.array_distinct(
                f.array_compact(
                    f.collect_list(
                        f.when(
                            f.col('cell_type').isNotNull(),
                            f.struct(
                                f.col('cell_type').alias('name'),
                                f.col('reliability').alias('reliability'),
                                f.col('level').alias('level'),
                            ),
                        )
                    )
                )
            ).alias('cell_type'),
        )
        .withColumn('organs', f.when(f.col('organs').isNull(), f.array()).otherwise(f.col('organs')))
        .withColumn(
            'anatomical_systems',
            f.when(f.col('anatomical_systems').isNull(), f.array()).otherwise(f.col('anatomical_systems')),
        )
    )

    result = (
        protein
        .groupBy('Gene')
        .agg(
            f.collect_set(
                f.struct(
                    f.col('efoId').alias('efo_code'),
                    f.col('labelDef').alias('label'),
                    f.col('organs'),
                    f.col('anatomical_systems'),
                    f.col('rna'),
                    f.struct(
                        f.col('reliability'),
                        f.col('level'),
                        f.col('cell_type'),
                    ).alias('protein'),
                )
            ).alias('tissues')
        )
        .withColumnRenamed('Gene', 'id')
    )

    logger.info(f'Writing expression output to {destination}')
    result.write.mode('overwrite').parquet(destination)
