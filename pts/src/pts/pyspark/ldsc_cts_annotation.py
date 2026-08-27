"""PTS task for reusable expression-specificity LDSC-CTS annotations."""

from __future__ import annotations

from dataclasses import dataclass
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
from pyspark.sql.types import LongType, StringType, StructField, StructType
from pyspark.storagelevel import StorageLevel

from pts.pyspark.common.session import Session
from pts.pyspark.ldsc_cts_utils import (
    discover_edge_manifest,
    join_input_path,
    load_edge_manifest,
    normalise_chromosomes,
    normalise_dataset_registry,
    normalise_populations,
    read_table,
    reference_path,
    success_exists,
    variant_positions_from_edges,
)


@dataclass
class _SpecificityContext:
    """Prepared gene-level data for one specificity dataset."""

    dataset_id: str
    specificity_long: DataFrame
    gene_locations: DataFrame


def ldsc_cts_annotation(
    source: dict[str, str],
    destination: str | dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Build reusable annotations for all configured datasets and populations.

    Specificity matrices are declared once in ``settings['datasets']`` and LD
    populations once in ``settings['populations']``. Large flat edge datasets
    are read one population/chromosome at a time, while every configured
    specificity dataset is scored against that edge set before it is released.
    """
    spark_uri = str(settings.get('spark_uri', 'local[*]'))
    session = Session(
        app_name='ldsc_cts_annotations',
        spark_uri=spark_uri,
        properties=properties,
    )
    spark = session.spark

    datasets = normalise_dataset_registry(settings.get('datasets'))
    populations = normalise_populations(settings.get('populations'))
    chromosomes = normalise_chromosomes(settings.get('chromosomes'))
    specificity_root = source.get('specificity_root')
    if not specificity_root:
        raise ValueError("Annotation source is missing 'specificity_root'")
    target_path = source.get('target_index')
    if not target_path:
        raise ValueError("Annotation source is missing 'target_index'")
    reference_root = source.get('reference_root')
    if not reference_root:
        raise ValueError("Annotation source is missing 'reference_root'")
    output_root = destination['annotations'] if isinstance(destination, dict) else destination
    if not output_root:
        raise ValueError('Annotation destination is missing an annotations path')
    output_root = output_root.rstrip('/')

    target = spark.read.parquet(target_path).persist(StorageLevel.MEMORY_AND_DISK)
    contexts = _prepare_specificity_contexts(
        spark=spark,
        target=target,
        root=specificity_root,
        datasets=datasets,
        settings=settings,
    )
    catalog_records: list[dict[str, Any]] = []

    try:
        for population in populations:
            edge_manifest = _resolve_edge_manifest(source, population, chromosomes)
            score_variant_path = reference_path(
                reference_root,
                population,
                settings,
                'score_variants',
                'baseline_ld_scores.tsv.gz',
            )
            score_variants_all = _read_score_variants(
                spark,
                score_variant_path,
                settings,
            ).persist(StorageLevel.MEMORY_AND_DISK)
            try:
                for chromosome in chromosomes:
                    pending = [
                        context
                        for context in contexts
                        if not _annotation_complete(output_root, context.dataset_id, population, chromosome, spark)
                    ]
                    if not pending:
                        catalog_records.extend(
                            [
                                _catalog_record(
                                    output_root,
                                    context.dataset_id,
                                    population,
                                    chromosome,
                                    spark,
                                )
                                for context in contexts
                            ]
                        )
                        continue

                    edges = _read_edges(spark, edge_manifest[chromosome], chromosome).persist(
                        StorageLevel.MEMORY_AND_DISK
                    )
                    variant_positions = variant_positions_from_edges(edges).persist(
                        StorageLevel.MEMORY_AND_DISK
                    )
                    try:
                        for context in pending:
                            _score_specificity_chromosome(
                                spark=spark,
                                context=context,
                                edges=edges,
                                variant_positions=variant_positions,
                                score_variants_all=score_variants_all,
                                chromosome=chromosome,
                                output_root=output_root,
                                population=population,
                                settings=settings,
                            )
                            catalog_records.append(
                                _catalog_record(
                                    output_root,
                                    context.dataset_id,
                                    population,
                                    chromosome,
                                    spark,
                                )
                            )
                    finally:
                        variant_positions.unpersist()
                        edges.unpersist()
            finally:
                score_variants_all.unpersist()
    finally:
        for context in contexts:
            context.specificity_long.unpersist()
            context.gene_locations.unpersist()
        target.unpersist()

    _write_catalog(spark, f'{output_root}/_catalog', catalog_records)


def _prepare_specificity_contexts(
    spark: SparkSession,
    target: DataFrame,
    root: str,
    datasets: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[_SpecificityContext]:
    contexts: list[_SpecificityContext] = []
    for dataset in datasets:
        dataset_settings = {**settings, **dataset}
        specificity = _read_specificity(
            spark,
            join_input_path(root, dataset['path']),
            dataset_settings,
        )
        gene_id_column = str(dataset_settings.get('gene_id_column', 'gene'))
        strip_gene_version = bool(dataset_settings.get('strip_gene_version', True))
        specificity_long = melt_specificity_matrix(
            specificity,
            gene_id_column=gene_id_column,
            strip_gene_version=strip_gene_version,
        ).persist(StorageLevel.DISK_ONLY)
        gene_locations = _read_gene_locations(
            spark=spark,
            target=target,
            specificity=specificity,
            gene_id_column=gene_id_column,
            strip_gene_version=strip_gene_version,
        ).persist(StorageLevel.DISK_ONLY)
        specificity.unpersist()
        contexts.append(
            _SpecificityContext(
                dataset_id=dataset['id'],
                specificity_long=specificity_long,
                gene_locations=gene_locations,
            )
        )
    return contexts


def _score_specificity_chromosome(
    spark: SparkSession,
    context: _SpecificityContext,
    edges: DataFrame,
    variant_positions: DataFrame,
    score_variants_all: DataFrame,
    chromosome: str,
    output_root: str,
    population: str,
    settings: dict[str, Any],
) -> None:
    output = f'{output_root}/{context.dataset_id}/{population}/chr{chromosome}'
    score_variants = score_variants_all.filter(
        f.col('variantId').startswith(f'{chromosome}_')
    )
    if score_variants.limit(1).count() == 0:
        raise ValueError(
            f'Scored-variant input contains no variants for chromosome {chromosome} '
            f'and population {population}'
        )

    gene_variant_map = map_genes_to_variants(
        gene_locations=context.gene_locations,
        variant_positions=variant_positions,
        window_kb=int(settings.get('window_kb', 100)),
    )
    annotations = build_snp_annotations(gene_variant_map, context.specificity_long)
    ld_scores, m_annot = compute_annotation_ld_scores(
        annotations_long=annotations,
        ld_edges=edges,
        score_variants=score_variants,
    )
    (
        ld_scores.withColumn('specificityId', f.lit(context.dataset_id))
        .write.mode('overwrite')
        .parquet(f'{output}/ld_scores')
    )
    (
        m_annot.withColumn('specificityId', f.lit(context.dataset_id))
        .write.mode('overwrite')
        .parquet(f'{output}/m_annot')
    )
    logger.info(
        'Completed LDSC-CTS annotation {} / {} / chromosome {}',
        context.dataset_id,
        population,
        chromosome,
    )


def _annotation_complete(
    output_root: str,
    dataset_id: str,
    population: str,
    chromosome: str,
    spark: SparkSession,
) -> bool:
    output = f'{output_root}/{dataset_id}/{population}/chr{chromosome}'
    return success_exists(spark, f'{output}/ld_scores') and success_exists(
        spark, f'{output}/m_annot'
    )


def _catalog_record(
    output_root: str,
    dataset_id: str,
    population: str,
    chromosome: str,
    spark: SparkSession,
) -> dict[str, Any]:
    m_path = f'{output_root}/{dataset_id}/{population}/chr{chromosome}/m_annot'
    annotation_count = spark.read.parquet(m_path).select('annotation').distinct().count()
    return {
        'specificityId': dataset_id,
        'ldPopulation': population,
        'chromosome': int(chromosome),
        'ldScoresPath': f'{output_root}/{dataset_id}/{population}/chr{chromosome}/ld_scores',
        'mAnnotPath': m_path,
        'annotationCount': annotation_count,
    }


_CATALOG_SCHEMA = StructType([
    StructField('specificityId', StringType(), False),
    StructField('ldPopulation', StringType(), False),
    StructField('chromosome', LongType(), False),
    StructField('ldScoresPath', StringType(), False),
    StructField('mAnnotPath', StringType(), False),
    StructField('annotationCount', LongType(), False),
])


def _write_catalog(spark: SparkSession, path: str, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError('No completed annotation units were recorded')
    rows = [
        {
            **record,
            'chromosome': int(record['chromosome']),
            'annotationCount': int(record['annotationCount']),
        }
        for record in records
    ]
    spark.createDataFrame(rows, schema=_CATALOG_SCHEMA).dropDuplicates(
        ['specificityId', 'ldPopulation', 'chromosome']
    ).write.mode('overwrite').parquet(path)


def _resolve_edge_manifest(
    source: dict[str, str], population: str, chromosomes: list[str]
) -> dict[str, str]:
    requested = [int(chromosome) for chromosome in chromosomes]
    if source.get('edge_manifest'):
        return load_edge_manifest(source['edge_manifest'], population, requested)
    if source.get('edge_work_root'):
        return discover_edge_manifest(source['edge_work_root'], population, requested)
    raise ValueError('Annotation source requires edge_manifest or edge_work_root')


def _read_specificity(spark: SparkSession, path: str, settings: dict[str, Any]) -> DataFrame:
    return read_table(
        spark,
        path,
        fmt=str(settings.get('format', settings.get('specificity_format', 'csv'))),
        sep=str(settings.get('separator', settings.get('specificity_sep', ','))),
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
    target: DataFrame | str,
    specificity: DataFrame,
    gene_id_column: str,
    strip_gene_version: bool,
) -> DataFrame:
    if isinstance(target, str):
        target = spark.read.parquet(target)
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

    target_gene = target_gene.cast('string')
    gene = f.regexp_replace(target_gene, r'\.\d+$', '') if strip_gene_version else target_gene
    matrix_gene = f.col(gene_id_column).cast('string')
    if strip_gene_version:
        matrix_gene = f.regexp_replace(matrix_gene, r'\.\d+$', '')
    matrix_genes = specificity.select(matrix_gene.alias('geneId')).distinct()
    return (
        target.select(
            gene.alias('geneId'),
            f.regexp_replace(chromosome.cast('string'), r'(?i)^chr', '').alias('chromosome'),
            start.cast('long').alias('start'),
            end.cast('long').alias('end'),
        )
        .filter(f.col('chromosome').isNotNull())
        .filter(f.col('start').isNotNull() & f.col('end').isNotNull())
        .dropDuplicates(['geneId'])
        .join(matrix_genes, on='geneId', how='inner')
    )
