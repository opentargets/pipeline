"""PTS task for reusable expression-specificity LD-score annotations."""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from gentropy.method.ldsc import (
    build_snp_annotations,
    compute_annotation_ld_scores,
    map_genes_to_variants,
    melt_specificity_matrix,
)
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.storagelevel import StorageLevel

from pts.pyspark.common.session import Session
from pts.pyspark.ldsc_cts_utils import (
    discover_edge_manifest,
    load_edge_manifest,
    normalise_chromosome,
    read_table,
    success_exists,
    variant_positions_from_edges,
)


def ldsc_cts_annotation(
    source: dict[str, str],
    destination: str | dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Build chromosome-partitioned annotation LD scores for one dataset/population.

    The input-specificity matrix and target index are loaded once. Edge exports
    remain chromosome-scoped so a failed or interrupted task can resume without
    rebuilding completed chromosomes.
    """
    effective_properties = {
        'spark.driver.memory': '30g',
        'spark.executor.memory': '30g',
        'spark.driver.memoryOverhead': '3g',
        'spark.executor.memoryOverhead': '3g',
        'spark.sql.shuffle.partitions': '1024',
        'spark.default.parallelism': '16',
        'spark.dynamicAllocation.enabled': 'false',
        'spark.sql.adaptive.enabled': 'true',
        'spark.sql.adaptive.coalescePartitions.enabled': 'false',
        **properties,
    }
    spark_uri = str(settings.get('spark_uri', 'local[4]'))
    session = Session(
        app_name='ldsc_cts_annotation',
        spark_uri=spark_uri,
        properties=effective_properties,
    )
    spark = session.spark

    specificity_path = source['specificity']
    target_path = source['target_index']
    output_root = destination['annotations'] if isinstance(destination, dict) else destination
    chromosomes = list(
        dict.fromkeys(
            normalise_chromosome(c)
            for c in settings.get('chromosomes', range(1, 23))
        )
    )
    ancestry = str(settings['ancestry']).lower()
    specificity_id = str(settings.get('specificity_id', 'specificity'))
    edge_manifest = _resolve_edge_manifest(source, ancestry, chromosomes)
    # Keep the task destination reusable across populations while avoiding
    # collisions between reference panels.
    output_root = f'{output_root.rstrip("/")}/{ancestry}'

    logger.info(
        'Building LDSC-CTS annotations for {} ({}) across chromosomes {}',
        specificity_id,
        ancestry,
        chromosomes,
    )
    specificity = _read_specificity(spark, specificity_path, settings).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    specificity_long = melt_specificity_matrix(
        specificity,
        gene_id_column=str(settings.get('gene_id_column', 'gene')),
        strip_gene_version=bool(settings.get('strip_gene_version', True)),
    ).persist(StorageLevel.MEMORY_AND_DISK)
    gene_locations = _read_gene_locations(
        spark=spark,
        target_path=target_path,
        specificity=specificity,
        gene_id_column=str(settings.get('gene_id_column', 'gene')),
        strip_gene_version=bool(settings.get('strip_gene_version', True)),
    ).persist(StorageLevel.MEMORY_AND_DISK)
    score_variants_all = _read_score_variants(
        spark, source['score_variants'], settings
    ).persist(StorageLevel.MEMORY_AND_DISK)

    for chromosome in chromosomes:
        output = f"{output_root.rstrip('/')}/chr{chromosome}"
        if success_exists(spark, f'{output}/ld_scores') and success_exists(
            spark, f'{output}/m_annot'
        ):
            logger.info('Skipping completed LDSC-CTS annotation chromosome {}', chromosome)
            continue

        edges = _read_edges(spark, edge_manifest[chromosome], chromosome)
        variant_positions = variant_positions_from_edges(edges)
        score_variants = score_variants_all.filter(
            f.col('variantId').startswith(f'{chromosome}_')
        )
        if score_variants.limit(1).count() == 0:
            raise ValueError(f'Scored-variant input contains no variants for chromosome {chromosome}')

        gene_variant_map = map_genes_to_variants(
            gene_locations=gene_locations,
            variant_positions=variant_positions,
            window_kb=int(settings.get('window_kb', 100)),
        )
        annotations = build_snp_annotations(gene_variant_map, specificity_long)
        ld_scores, m_annot = compute_annotation_ld_scores(
            annotations_long=annotations,
            ld_edges=edges,
            score_variants=score_variants,
        )
        (
            ld_scores.withColumn('specificityId', f.lit(specificity_id))
            .write.mode('overwrite')
            .parquet(f'{output}/ld_scores')
        )
        (
            m_annot.withColumn('specificityId', f.lit(specificity_id))
            .write.mode('overwrite')
            .parquet(f'{output}/m_annot')
        )
        logger.info('Completed LDSC-CTS annotation chromosome {}', chromosome)


def _resolve_edge_manifest(
    source: dict[str, str], ancestry: str, chromosomes: list[str]
) -> dict[str, str]:
    requested = [int(chromosome) for chromosome in chromosomes]
    if source.get('edge_manifest'):
        return load_edge_manifest(source['edge_manifest'], ancestry, requested)
    if source.get('edge_work_root'):
        return discover_edge_manifest(source['edge_work_root'], ancestry, requested)
    raise ValueError('Annotation task requires edge_manifest or edge_work_root')


def _read_specificity(spark: SparkSession, path: str, settings: dict[str, Any]) -> DataFrame:
    return read_table(
        spark,
        path,
        fmt=str(settings.get('specificity_format', 'csv')),
        sep=str(settings.get('specificity_sep', ',')),
    )


def _read_score_variants(
    spark: SparkSession, path: str, settings: dict[str, Any]
) -> DataFrame:
    raw = read_table(
        spark,
        path,
        fmt=str(settings.get('score_variant_format', 'csv')),
        sep=str(settings.get('score_variant_sep', '\t')),
    )
    column = str(settings.get('score_variant_column', 'variantId'))
    if column not in raw.columns:
        raise ValueError(f"Scored-variant input is missing '{column}'")
    return (
        raw.select(f.col(column).cast('string').alias('variantId'))
        .filter(f.col('variantId').isNotNull())
        .distinct()
    )


def _read_edges(spark: SparkSession, path: str, chromosome: str) -> DataFrame:
    if not success_exists(spark, path):
        raise ValueError(f'Flat LD edge input is not a completed dataset: {path}')
    edges = spark.read.parquet(path)
    required = {'variantId', 'tagVariantId', 'r'}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f'Flat LD edge input is missing required columns: {sorted(missing)}')
    edges = edges.select(
        f.col('variantId').cast('string').alias('variantId'),
        f.col('tagVariantId').cast('string').alias('tagVariantId'),
        f.col('r').cast('double').alias('r'),
    )
    if edges.filter(
        f.col('variantId').isNull()
        | f.col('tagVariantId').isNull()
        | f.col('r').isNull()
        | f.isnan('r')
        | (f.abs(f.col('r')) > f.lit(1.0000001))
    ).limit(1).count():
        raise ValueError('Flat LD edge input contains null, non-finite, or invalid correlation values')
    prefix = f'{chromosome}_'
    if edges.filter(
        ~f.col('variantId').startswith(prefix)
        | ~f.col('tagVariantId').startswith(prefix)
    ).limit(1).count():
        raise ValueError(f'Flat LD edge input contains an endpoint outside chromosome {chromosome}')
    return edges


def _read_gene_locations(
    spark: SparkSession,
    target_path: str,
    specificity: DataFrame,
    gene_id_column: str,
    strip_gene_version: bool,
) -> DataFrame:
    target = spark.read.parquet(target_path)
    if 'id' in target.columns:
        target_gene = f.col('id')
    elif 'geneId' in target.columns:
        target_gene = f.col('geneId')
    else:
        raise ValueError("Target index must contain 'id' or 'geneId'")
    if 'genomicLocation' in target.columns:
        chromosome = f.col('genomicLocation.chromosome')
        start = f.col('genomicLocation.start')
        end = f.col('genomicLocation.end')
    elif {'chromosome', 'start', 'end'}.issubset(target.columns):
        chromosome, start, end = (f.col(c) for c in ('chromosome', 'start', 'end'))
    else:
        raise ValueError('Target index lacks genomic coordinates')

    gene = f.regexp_replace(target_gene, r'\.\d+$', '') if strip_gene_version else target_gene
    matrix_gene = f.col(gene_id_column)
    if strip_gene_version:
        matrix_gene = f.regexp_replace(matrix_gene, r'\.\d+$', '')
    matrix_genes = specificity.select(matrix_gene.alias('geneId')).distinct()
    return (
        target.select(
            gene.alias('geneId'),
            f.regexp_replace(
                chromosome.cast('string'), r'(?i)^chr', ''
            ).alias('chromosome'),
            start.cast('long').alias('start'),
            end.cast('long').alias('end'),
        )
        .filter(f.col('chromosome').isNotNull())
        .filter(f.col('start').isNotNull() & f.col('end').isNotNull())
        .dropDuplicates(['geneId'])
        .join(matrix_genes, on='geneId', how='inner')
    )
