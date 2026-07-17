"""Homologues dataset generation.

Resolves Ensembl compara homology data into a flat, one-row-per-pair
targetId/homologue dataset, validated against output/target (implicit
via inner join on gene symbols).
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import maybe_coalesce

DEFAULT_HGNC_ORTHOLOG_SPECIES = [
    '9606-human',
    '9598-chimpanzee',
    '9544-macaque',
    '10090-mouse',
    '10116-rat',
    '9986-rabbit',
    '10141-guineapig',
    '9615-dog',
    '9823-pig',
    '8364-frog',
    '7955-zebrafish',
    '7227-fly',
    '6239-worm',
]


def homologues(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Build the homologues dataset and write it out.

    Args:
        source: Mapping of logical input names to paths. Expected keys:
            ``homology_dictionary`` (Ensembl vertebrates species dictionary),
            ``homology_coding_proteins`` (Ensembl compara homologies TSVs),
            ``homology_gene_dictionary`` (gene ID → gene name parquet), and
            ``target`` (output/target parquet, used for gene-symbol lookup
            and identifier validation).
        destination: Output path for the homologues parquet dataset.
        settings: Step settings; supports ``hgncOrthologSpecies`` (list[str])
            and ``partition_count`` (int).
        properties: Spark properties passed to :class:`Session`.
    """
    spark = Session(app_name='homologues', properties=properties).spark

    logger.debug(f'loading data from: {source}')
    homology_dict_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['homology_dictionary'])
    homology_coding_proteins_raw = (
        spark.read
        .option('recursiveFileLookup', 'true')
        .option('sep', '\t')
        .option('header', 'true')
        .csv(source['homology_coding_proteins'])
    )
    homology_gene_dict_raw = spark.read.option('recursiveFileLookup', 'true').parquet(
        source['homology_gene_dictionary']
    )
    target_df = spark.read.parquet(source['target']).select('id', 'approvedSymbol')

    hgnc_ortholog_species: list[str] = settings.get('hgncOrthologSpecies', DEFAULT_HGNC_ORTHOLOG_SPECIES)

    logger.info('building homologue/ortholog pairs')
    homology_df = _build_homologues(
        homology_dict_raw,
        homology_coding_proteins_raw,
        homology_gene_dict_raw,
        hgnc_ortholog_species,
    )

    logger.info('resolving target-gene symbols and validating against output/target')
    result = _resolve_homologues(homology_df, target_df)

    partition_count = settings.get('partition_count')
    logger.info(f'writing output data to {destination}.')
    maybe_coalesce(result, partition_count).write.mode('overwrite').parquet(destination)


# ===========================================================================
# Ortholog.scala → _build_homologues
# ===========================================================================


def _build_homologues(
    homology_dict: DataFrame,
    coding_proteins: DataFrame,
    gene_dict: DataFrame,
    target_species: list[str],
) -> DataFrame:
    """Build homologue/ortholog DataFrame.

    Args:
        homology_dict: Ensembl vertebrates species dictionary.
        coding_proteins: Ensembl compara homologies TSV (protein + ncrna).
        gene_dict: Gene ID → gene name mapping (pre-processed parquet).
        target_species: Whitelisted species in format "TAXID-species_name".

    Returns:
        DataFrame with homolog fields including speciesId, speciesName, homologyType, etc.
    """
    # Extract tax IDs from whitelist
    tax_ids = [s.split('-')[0] for s in target_species]
    priority_df_data = [(s.split('-')[0], i) for i, s in enumerate(target_species)]

    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError('no active spark session found')
    priority_df = spark.createDataFrame(priority_df_data, ['speciesId', 'priority']).withColumn(
        'priority', f.col('priority').cast('int')
    )

    homo_dict = homology_dict.select(
        f.col('#name').alias('name'),
        f.col('species').alias('speciesName'),
        f.col('taxonomy_id'),
        f.array(*[f.lit(t) for t in tax_ids]).alias('whitelist'),
    ).filter(f.array_contains(f.col('whitelist'), f.col('taxonomy_id')))

    gene_dict_mapped = gene_dict.select(
        f.col('id').alias('homology_gene_stable_id'),
        f
        .when(f.col('name').isNotNull() & (f.col('name') != ''), f.col('name'))
        .otherwise(f.col('id'))
        .alias('targetGeneSymbol'),
    )

    reference = 'homo_sapiens'

    # homo_sapiens homologies
    homo_sapiens_h = coding_proteins.filter(f.col('species') == reference)

    # paralogs and cross-species
    other_h = (
        coding_proteins.filter(
            (
                (f.col('species') == reference)
                & ((f.col('homology_type') == 'other_paralog') | (f.col('homology_type') == 'within_species_paralog'))
            )
            | ((f.col('species') != reference) & (f.col('homology_species') == reference))
        )
        # swap homo_sapiens ↔ homology columns
        .select(
            f.col('homology_gene_stable_id').alias('gene_stable_id'),
            f.col('homology_protein_stable_id').alias('protein_stable_id'),
            f.col('homology_species').alias('species'),
            f.col('homology_identity').alias('identity'),
            f.col('homology_type'),
            f.col('gene_stable_id').alias('homology_gene_stable_id'),
            f.col('protein_stable_id').alias('homology_protein_stable_id'),
            f.col('species').alias('homology_species'),
            f.col('identity').alias('homology_identity'),
            f.col('dn'),
            f.col('ds'),
            f.col('goc_score'),
            f.col('wga_coverage'),
            f.col('is_high_confidence'),
            f.col('homology_id'),
        )
    )

    all_homologies = homo_sapiens_h.unionByName(other_h)

    return (
        all_homologies
        .join(homo_dict, all_homologies['homology_species'] == homo_dict['speciesName'])
        .join(gene_dict_mapped, 'homology_gene_stable_id', 'left_outer')
        .select(
            f.col('gene_stable_id').alias('id'),
            f.col('taxonomy_id').alias('speciesId'),
            f.col('name').alias('speciesName'),
            f.col('homology_type').alias('homologyType'),
            f.col('homology_gene_stable_id').alias('targetGeneId'),
            f.col('is_high_confidence').alias('isHighConfidence'),
            f.col('targetGeneSymbol'),
            f.col('identity').cast(DoubleType()).alias('queryPercentageIdentity'),
            f.col('homology_identity').cast(DoubleType()).alias('targetPercentageIdentity'),
        )
        .join(f.broadcast(priority_df), 'speciesId', 'left_outer')
    )


def _resolve_homologues(orthologs: DataFrame, target_df: DataFrame) -> DataFrame:
    """Resolve homologue target-gene symbols and validate against output/target.

    Returns a flat DataFrame (one row per target/homologue pair). Rows whose
    ``id`` has no match in ``target_df`` are dropped (implicit validation,
    via inner join).

    Args:
        orthologs: Homology pairs from :func:`_build_homologues`.
        target_df: Target parquet (``id``, ``approvedSymbol`` columns) from
            output/target.

    Returns:
        DataFrame with [targetId, speciesId, speciesName, homologyType,
        targetGeneId, isHighConfidence, targetGeneSymbol,
        queryPercentageIdentity, targetPercentageIdentity, priority].
    """
    gene_symbols = target_df.select('id', 'approvedSymbol')
    paralog_symbols = gene_symbols.withColumnRenamed('approvedSymbol', 'paralogGeneSymbol').withColumnRenamed(
        'id', 'paralogId'
    )

    return (
        orthologs
        .join(f.broadcast(gene_symbols), 'id')
        .join(
            f.broadcast(paralog_symbols),
            f.col('paralogId') == f.col('targetGeneId'),
            'left_outer',
        )
        .withColumn(
            'targetGeneSymbol',
            f.coalesce(
                f.col('paralogGeneSymbol'),
                f.col('targetGeneSymbol'),
                f.col('approvedSymbol'),
            ),
        )
        .select(
            f.col('id').alias('targetId'),
            'speciesId',
            'speciesName',
            'homologyType',
            'targetGeneId',
            'isHighConfidence',
            'targetGeneSymbol',
            'queryPercentageIdentity',
            'targetPercentageIdentity',
            'priority',
        )
    )
