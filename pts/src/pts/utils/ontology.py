"""This module provides functionality to map disease information to EFO using the OnToma."""

import random
import time
from pathlib import Path

from loguru import logger
from numpy import nan
from ontoma.interface import OnToma
from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, StructField, StructType

ONTOMA_MAX_ATTEMPTS = 1


def _simple_retry(func, **kwargs):
    """Simple retry handling for functions. Cannot be a decorator, so that the functions could still be pickled."""
    for attempt in range(1, ONTOMA_MAX_ATTEMPTS + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            # If this is not the last attempt, wait until the next one.
            if attempt != ONTOMA_MAX_ATTEMPTS:
                logger.warning(f'OnToma lookup attempt {attempt} failed: {e}. Retrying...')
                time.sleep(5 + 10 * random.random())
            else:
                logger.error(f'OnToma lookup failed after {ONTOMA_MAX_ATTEMPTS} attempts for {kwargs!r}: {e}')
    return []


def _get_cache_directory(base_cache_dir=None, efo_version=None):
    """Get or create a persistent cache directory for OnToma.

    Args:
        base_cache_dir: Custom base directory for cache (optional)
        efo_version: EFO version string (required, e.g. 'v3.81.0')
    """
    if not efo_version:
        raise ValueError('efo_version is required and cannot be None')

    if base_cache_dir:
        cache_dir = Path(base_cache_dir)
    else:
        # Create cache directory based on explicit version
        cache_dir = Path.home() / '.ontoma_cache' / f'efo_{efo_version}'

    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def _initialize_ontoma_with_cache(ontoma_cache_dir, efo_version):
    """Initialize OnToma with persistent caching, only parsing EFO if not cached.

    Args:
        ontoma_cache_dir: Directory path for OnToma cache
        efo_version: EFO version string (required, e.g. 'v3.81.0')
    """
    if not efo_version:
        raise ValueError('efo_version is required and cannot be None')

    logger.info(f'Initializing OnToma with cache directory: {ontoma_cache_dir}')

    # Check if we have a cached ontology
    cache_info_file = Path(ontoma_cache_dir) / 'cache_info.txt'

    if cache_info_file.exists():
        cached_version = Path(cache_info_file).read_text().strip()
        if cached_version == efo_version:
            logger.info(f'Found existing EFO {efo_version} cache, reusing...')
        else:
            logger.info(
                f'Cache version mismatch (cached: {cached_version}, requested: {efo_version}), will re-cache...'
            )
    else:
        logger.info(f'No existing cache found, will create new cache for EFO {efo_version}')

    # Initialize OnToma (it will use existing cache if available)
    start_time = time.time()
    ontoma_instance = OnToma(cache_dir=ontoma_cache_dir, efo_release=efo_version)
    initialization_time = time.time() - start_time

    logger.info(f'OnToma initialization completed in {initialization_time:.1f}s')

    # Save cache info for future runs
    Path(cache_info_file).write_text(efo_version)

    return ontoma_instance


def _ontoma_udf(row, ontoma_instance):
    """Try to map first by disease name (because that branch of OnToma is more stable), then by disease ID."""
    try:
        disease_name = None
        if row['diseaseFromSource']:
            disease_name = ' '.join(row['diseaseFromSource'].replace('obsolete', '').split())
        disease_id = row['diseaseFromSourceId'].replace('_', ':') if row['diseaseFromSourceId'] else None

        mappings = []
        if disease_name:
            mappings = _simple_retry(ontoma_instance.find_term, query=disease_name, code=False)
        if not mappings and disease_id and ':' in disease_id:
            mappings = _simple_retry(ontoma_instance.find_term, query=disease_id, code=True)

        return [m.id_ot_schema for m in mappings]
    except Exception as e:
        logger.warning(f'Mapping failed for row {row.name if hasattr(row, "name") else "unknown"}: {e}')
        return []


def add_efo_mapping(evidence_strings, spark_instance, ontoma_cache_dir=None, efo_version=None, cores=1):
    """Given evidence strings with diseaseFromSource and diseaseFromSourceId fields, try to populate EFO mapping.

    field diseaseFromSourceMappedId. In case there are multiple matches, the evidence strings will be exploded
    accordingly.

    Currently, both source columns (diseaseFromSource and diseaseFromSourceId) need to be present in the original
    schema, although they do not have to be populated for all rows.

    Args:
        evidence_strings: Spark DataFrame with evidence data
        spark_instance: Spark session instance
        ontoma_cache_dir: Directory for OnToma cache (will use default if None)
        efo_version: EFO version to use (required, e.g. 'v3.81.0')
        cores: Number of cores to use (1 = sequential, >1 = parallel)
    """
    if not efo_version:
        raise ValueError("efo_version is required. Please specify an explicit EFO version (e.g. 'v3.81.0').")
    logger.info('Collect all distinct (disease name, disease ID) pairs.')
    disease_info_to_map = evidence_strings.select('diseaseFromSource', 'diseaseFromSourceId').distinct().toPandas()
    logger.info(f'Found {len(disease_info_to_map)} unique disease entries to map.')

    # Initialize OnToma with persistent caching
    ontoma_cache_dir = _get_cache_directory(ontoma_cache_dir, efo_version)
    logger.info(f'Using OnToma cache directory: {ontoma_cache_dir}')
    try:
        ontoma_instance = _initialize_ontoma_with_cache(ontoma_cache_dir, efo_version)
    except Exception as e:
        logger.error(f'Failed to initialize OnToma: {e}')
        raise

    # Process based on cores configuration
    if cores > 1:
        # Parallel processing
        logger.info(f'Starting parallel EFO mapping with {cores} cores')
        try:
            from pandarallel import pandarallel

            pandarallel.initialize(nb_workers=cores, progress_bar=True, verbose=1, use_memory_fs=False)

            disease_info_to_map['diseaseFromSourceMappedId'] = disease_info_to_map.parallel_apply(
                lambda row: _ontoma_udf(row, ontoma_instance), axis=1
            )

        except Exception as e:
            logger.error(f'Parallel processing failed: {e}')
            raise
    else:
        # Sequential processing
        logger.info(f'Using sequential processing with cached OnToma. EFO version: {efo_version}')
        mapped_ids = []

        for idx, row in disease_info_to_map.iterrows():
            if idx % 50 == 0:  # Progress logging every 50 rows
                logger.info(f'Processing disease mapping {idx}/{len(disease_info_to_map)}')

            try:
                mapped_id = _ontoma_udf(row, ontoma_instance)
                mapped_ids.append(mapped_id)
            except Exception as e:
                logger.warning(f'Error processing row {idx}: {e}')
                mapped_ids.append([])  # Empty mapping on error

        disease_info_to_map['diseaseFromSourceMappedId'] = mapped_ids

    disease_info_to_map = (
        disease_info_to_map.explode('diseaseFromSourceMappedId')
        # Cast all null values to python None to avoid errors in Spark's DF
        .fillna(nan)
        .replace([nan], [None])
    )

    schema = StructType([
        StructField('diseaseFromSource_right', StringType(), True),
        StructField('diseaseFromSourceId_right', StringType(), True),
        StructField('diseaseFromSourceMappedId', StringType(), True),
    ])
    disease_info_df = spark_instance.createDataFrame(disease_info_to_map, schema=schema).withColumn(
        'diseaseFromSourceMappedId',
        when(col('diseaseFromSourceMappedId') != 'nan', col('diseaseFromSourceMappedId')),
    )

    # WARNING: Spark's join operator is not null safe by default and most of the times,
    # `diseaseFromSourceId` will be null. `eqNullSafe` is a special null safe equality
    # operator that is used to join the two dataframes.
    join_cond = (evidence_strings.diseaseFromSource.eqNullSafe(disease_info_df.diseaseFromSource_right)) & (
        evidence_strings.diseaseFromSourceId.eqNullSafe(disease_info_df.diseaseFromSourceId_right)
    )
    return evidence_strings.join(disease_info_df, on=join_cond, how='left').drop(
        'diseaseFromSource_right', 'diseaseFromSourceId_right'
    )
