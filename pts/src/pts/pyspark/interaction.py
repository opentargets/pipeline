"""Interaction dataset generation.

Ported from platform-etl-backend Interaction step. Computes protein-protein
and RNA interactions from IntAct and STRING databases, producing aggregated
interaction records and per-evidence records.

Scala sources ported:
    - Interaction.scala (main assembly)
    - stringProtein/StringProtein.scala (STRING protein transformation)

Schema design
-------------
A synthetic ``interactionId`` column (64-bit integer xxhash64 over the nine
uniqueness key columns) is added to both output datasets:

* ``interactions``         - one row per pair; carries all dimension/identity
                             columns plus ``interactionId``, count, and scoring.
* ``interactions_evidence``- one row per evidence entry; carries only
                             ``interactionId`` (foreign key into interactions)
                             plus the evidence-specific fields.  The identity
                             columns are *not* duplicated here.
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.types import BooleanType, IntegerType, LongType

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import rename_columns_to_camel_case

# ---------------------------------------------------------------------------
# Column name mappings for A/B swap (intact/reactome/signor are bidirectional)
# ---------------------------------------------------------------------------

_SWAP_MAP: dict[str, str] = {
    'targetA': 'targetB',
    'intA': 'intB',
    'intA_source': 'intB_source',
    'speciesA': 'speciesB',
    'intABiologicalRole': 'intBBiologicalRole',
    'targetB': 'targetA',
    'intB': 'intA',
    'intB_source': 'intA_source',
    'speciesB': 'speciesA',
    'intBBiologicalRole': 'intABiologicalRole',
}

_BIDIRECTIONAL_SOURCES = ('reactome', 'intact', 'signor')

# Evidence channel definitions for STRING data
_STRING_EVIDENCE_CHANNELS = [
    ('coexpression', 'MI:2231'),
    ('cooccurence', 'MI:2231'),
    ('neighborhood', 'MI:0057'),
    ('fusion', 'MI:0036'),
    ('homology', 'MI:2163'),
    ('experimental', 'MI:0591'),
    ('database', ''),
    ('textmining', 'MI:0110'),
]

# Columns that uniquely identify an interaction pair. SINGLE SOURCE OF TRUTH for
# BOTH the aggregation groupBy (_generate_interactions_agg) and the interactionId
# hash (_interaction_id_expr), so the two can never drift. speciesA/speciesB are
# included because the aggregation groups by them; omitting them would let two
# aggregate rows share one interactionId when species is not functionally
# determined by the rest of the key.
_INTERACTION_KEY_COLS: list[str] = [
    'sourceDatabase',
    'targetA',
    'intA',
    'intABiologicalRole',
    'targetB',
    'intB',
    'intBBiologicalRole',
    'speciesA',
    'speciesB',
]

# Dimension/identity columns present in _select_interaction_fields output that
# are dropped from the evidence dataset (replaced by interactionId).
_DIMENSION_COLS: list[str] = [
    'sourceDatabase',
    'targetA',
    'intA',
    'intA_source',
    'intABiologicalRole',
    'targetB',
    'intB',
    'intB_source',
    'intBBiologicalRole',
    'speciesA',
    'speciesB',
]


def _get_code(col_expr: Column) -> Column:
    """Extract the identifier prefix before the first '_' or '-' character.

    Native Spark SQL replacement for the former Python UDF; avoids JVM/Python
    serialisation overhead and enables join optimisations.
    E.g. 'URS123-2_992' → 'URS123'.
    """
    return f.split(f.trim(col_expr), r'[_\-]')[0]


def _interaction_id_expr() -> Column:
    """Return a 64-bit integer (xxhash64) surrogate key over the interaction uniqueness key columns.

    Null values are coalesced to empty strings so that nullable columns
    (targetA, targetB) still produce a stable, deterministic identifier.
    The `.cast('string')` renders speciesA/speciesB structs deterministically
    and is a no-op for flat string columns.
    """
    return f.xxhash64(
        *[f.coalesce(f.col(c).cast('string'), f.lit('')) for c in _INTERACTION_KEY_COLS],
    )


# ---------------------------------------------------------------------------
# Mapping helpers (RNACentral / Human mapping)
# ---------------------------------------------------------------------------


def _transform_rnacentral(df: DataFrame) -> DataFrame:
    """Transform the RNACentral file to (gene_id, mapped_id) pairs.

    Maps column ``_c0`` to ``mapped_id`` and ``_c5`` to ``gene_id``.

    Args:
        df: Raw RNACentral TSV DataFrame with columns _c0 … _c5.

    Returns:
        DataFrame with columns gene_id and mapped_id.

    Examples:
        >>> from pyspark.sql import Row
        >>> from pyspark.sql.types import StringType, StructField, StructType
        >>> schema = StructType([
        ...     StructField('_c0', StringType()),
        ...     StructField('_c1', StringType()),
        ...     StructField('_c2', StringType()),
        ...     StructField('_c3', StringType()),
        ...     StructField('_c4', StringType()),
        ...     StructField('_c5', StringType()),
        ... ])
        >>> df = spark.createDataFrame(
        ...     [Row(_c0='URS001', _c1='9606', _c2='x', _c3='y', _c4='z', _c5='ENSG001')],
        ...     schema,
        ... )
        >>> result = _transform_rnacentral(df)
        >>> row = result.collect()[0]
        >>> row.mapped_id
        'URS001'
        >>> row.gene_id
        'ENSG001'
    """
    return df.withColumnRenamed('_c0', 'mapped_id').withColumnRenamed('_c5', 'gene_id').select('gene_id', 'mapped_id')


def _transform_human_mapping(df: DataFrame) -> DataFrame:
    """Transform the Human Mapping file to (id, mapping_list) pairs.

    Filters to rows where ``_c1 == 'Ensembl'``, groups by ``_c2``, and
    collects the ``_c0`` values into a list.

    Args:
        df: Raw human-mapping TSV DataFrame with columns _c0, _c1, _c2.

    Returns:
        DataFrame with columns id (Ensembl gene id) and mapping_list (array of
        mapped ids).

    Examples:
        >>> from pyspark.sql import Row
        >>> from pyspark.sql.types import StringType, StructField, StructType
        >>> schema = StructType([
        ...     StructField('_c0', StringType()),
        ...     StructField('_c1', StringType()),
        ...     StructField('_c2', StringType()),
        ... ])
        >>> df = spark.createDataFrame(
        ...     [
        ...         Row(_c0='P12345', _c1='Ensembl', _c2='ENSG001'),
        ...         Row(_c0='BRCA1',  _c1='Gene_Name', _c2='ENSG001'),
        ...     ],
        ...     schema,
        ... )
        >>> result = _transform_human_mapping(df)
        >>> row = result.collect()[0]
        >>> row.id
        'ENSG001'
        >>> sorted(row.mapping_list)
        ['P12345']
    """
    return (
        df
        .filter(f.col('_c1') == 'Ensembl')
        .groupBy('_c2')
        .agg(f.collect_list('_c0').alias('mapping_list'))
        .withColumnRenamed('_c2', 'id')
        .withColumn('mapping_list', f.coalesce(f.col('mapping_list'), f.array()))
        .select('id', 'mapping_list')
    )


def _transform_gene_ids(df: DataFrame, human_mapping: DataFrame) -> DataFrame:
    """Extract gene_name → gene_id links from Human Mapping.

    Uses Gene_Name entries to find rows whose mapped_id matches entries in
    ``df`` (which already contains gene_id, mapped_id pairs). Returns the
    combined (gene_id, mapped_id) rows for names that did not directly resolve
    via Ensembl.

    Args:
        df: DataFrame with columns gene_id and mapped_id (from Ensembl mapping).
        human_mapping: Raw human-mapping TSV DataFrame with _c0, _c1, _c2.

    Returns:
        DataFrame with columns gene_id and mapped_id.
    """
    genes = (
        human_mapping
        .filter(f.col('_c1') == 'Gene_Name')
        .groupBy('_c2')
        .agg(f.collect_list('_c0').alias('mapping_list'))
    )

    gene_ids = genes.withColumn('mapped_id', f.explode(f.col('mapping_list'))).drop('mapping_list')

    combination_info = gene_ids.join(df, 'mapped_id', 'left')
    mapped = combination_info.filter(f.col('gene_id').isNotNull()).drop('mapped_id').distinct()
    mapped_not = combination_info.filter(f.col('gene_id').isNull()).drop('gene_id')
    return mapped_not.join(mapped, '_c2').select('gene_id', 'mapped_id').distinct()


def _generate_mapping(
    target_df: DataFrame,
    rnacentral_df: DataFrame,
    human_mapping_df: DataFrame,
) -> DataFrame:
    """Generate the full gene_id ↔ mapped_id lookup table.

    Combines protein IDs, HGNC IDs, Ensembl cross-references from
    Human Mapping, RNACentral mappings, and gene-name derived mappings.

    Args:
        target_df: Target DataFrame with columns id, proteinIds, dbXRefs.
        rnacentral_df: Raw RNACentral TSV DataFrame.
        human_mapping_df: Raw Human Mapping TSV DataFrame.

    Returns:
        Distinct DataFrame with columns gene_id and mapped_id.
    """
    targets_proteins = target_df.withColumn('proteins', f.coalesce(f.col('proteinIds.id'), f.array())).select(
        'id', 'proteins'
    )

    target_hgnc = (
        target_df
        .select(
            f.col('id'),
            f.filter(f.col('dbXRefs'), lambda c: c.getField('source') == 'HGNC').alias('h'),
        )
        .withColumn('mapped_id', f.explode(f.col('h.id')))
        .select(
            f.col('id').alias('gene_id'),
            f.concat(f.lit('HGNC:'), f.col('mapped_id')).alias('mapped_id'),
        )
    )

    human_mapping_result = _transform_human_mapping(human_mapping_df)
    rna_mapping = _transform_rnacentral(rnacentral_df)

    mapping_human = (
        targets_proteins
        .join(human_mapping_result, 'id', 'left')
        .withColumn(
            'mapped_id_list',
            f.when(f.col('mapping_list').isNull(), f.col('proteins')).otherwise(
                f.array_union(f.col('proteins'), f.col('mapping_list'))
            ),
        )
        .select('id', 'mapped_id_list')
        .distinct()
        .withColumnRenamed('id', 'gene_id')
    )

    mapping_explode = mapping_human.withColumn('mapped_id', f.explode(f.col('mapped_id_list'))).drop('mapped_id_list')

    map_gene_ids = _transform_gene_ids(mapping_explode, human_mapping_df)

    mapping = mapping_explode.union(rna_mapping).union(target_hgnc).union(map_gene_ids)

    return mapping.distinct()


# ---------------------------------------------------------------------------
# STRING protein transformation
# ---------------------------------------------------------------------------


def _transform_string_proteins(df: DataFrame, score_threshold: int, string_version: str = '12') -> DataFrame:
    """Transform the STRING protein interaction file into the common schema.

    Filters interactions below ``score_threshold``, constructs interactorA /
    interactorB / interaction / source_info nested columns in the same schema
    used by IntAct data.

    Args:
        df: Raw STRING CSV DataFrame with columns protein1, protein2,
            combined_score, and evidence channel columns.
        score_threshold: Minimum combined_score (inclusive) to keep.
        string_version: STRING database version string for source_info.

    Returns:
        DataFrame with nested columns interactorA, interactorB, interaction,
        source_info.
    """
    logger.info('Transforming STRING proteins with score_threshold=%d', score_threshold)

    filtered = df.withColumn('interaction_score', f.ltrim(f.col('combined_score')).cast(IntegerType())).filter(
        f.col('interaction_score') >= score_threshold
    )

    # Build per-channel evidence structs
    for channel_name, mi_id in _STRING_EVIDENCE_CHANNELS:
        filtered = filtered.withColumn(
            'e_' + channel_name,
            f.struct(
                f.lit(channel_name).alias('interaction_detection_method_short_name'),
                f.lit(mi_id).alias('interaction_detection_method_mi_identifier'),
                f.col(channel_name).cast(LongType()).alias('evidence_score'),
                f.lit(None).cast('string').alias('interaction_identifier'),
                f.lit(None).cast('string').alias('pubmed_id'),
            ),
        )

    return (
        filtered
        .filter(f.col('protein1').contains('9606.'))
        .filter(f.col('protein2').contains('9606.'))
        .withColumn('id_source_p1', f.regexp_replace(f.col('protein1'), '9606\\.', ''))
        .withColumn('id_source_p2', f.regexp_replace(f.col('protein2'), '9606\\.', ''))
        .withColumn('biological_role', f.lit('unspecified role'))
        .withColumn('id_source', f.lit('ensembl_protein'))
        .withColumn(
            'organism',
            f.struct(
                f.lit('human').alias('mnemonic'),
                f.lit('Homo sapiens').alias('scientific_name'),
                f.lit(9606).cast('bigint').alias('taxon_id'),
            ),
        )
        .withColumn(
            'interactorA',
            f.struct(
                f.col('id_source'),
                f.col('biological_role'),
                f.col('id_source_p1').alias('id'),
                f.col('organism'),
            ),
        )
        .withColumn(
            'interactorB',
            f.struct(
                f.col('id_source'),
                f.col('biological_role'),
                f.col('id_source_p2').alias('id'),
                f.col('organism'),
            ),
        )
        .withColumn(
            'source_info',
            f.struct(
                f.lit(string_version).alias('database_version'),
                f.lit('string').alias('source_database'),
            ),
        )
        .withColumn('causal_interaction', f.lit(False).cast(BooleanType()))
        .drop(
            'protein1',
            'protein2',
            'id_source_p1',
            'id_source_p2',
            'biological_role',
            'id_source',
        )
        .withColumn(
            'all_evidence',
            f.array(
                f.col('e_textmining'),
                f.col('e_database'),
                f.col('e_experimental'),
                f.col('e_fusion'),
                f.col('e_neighborhood'),
                f.col('e_cooccurence'),
                f.col('e_coexpression'),
                f.col('e_homology'),
            ),
        )
        .withColumn(
            'interaction',
            f.struct(
                f.col('interaction_score'),
                f.col('causal_interaction'),
                f.col('all_evidence').alias('evidence'),
            ),
        )
        .drop(
            'combined_score',
            'textmining',
            'database',
            'experimental',
            'fusion',
            'neighborhood',
            'cooccurence',
            'coexpression',
            'homology',
            'e_textmining',
            'e_database',
            'e_experimental',
            'e_fusion',
            'e_neighborhood',
            'e_cooccurence',
            'e_coexpression',
            'e_homology',
            'all_evidence',
            'interaction_score',
            'causal_interaction',
            'organism',
        )
    )


# ---------------------------------------------------------------------------
# Core interaction computation
# ---------------------------------------------------------------------------


def _generate_interactions(df: DataFrame, mapping_info: DataFrame) -> DataFrame:
    """Map raw interaction records to (targetA, targetB) via the lookup table.

    Handles the self-interaction case where interactorB is null by falling
    back to interactorA values. For bidirectional sources (intact, reactome,
    signor) the swapped (B→A) direction is also included via union.

    Args:
        df: Raw interaction DataFrame with interactorA, interactorB, interaction,
            source_info nested columns.
        mapping_info: DataFrame with columns gene_id and mapped_id.

    Returns:
        DataFrame of interaction evidence records (one row per evidence entry,
        after exploding the evidences array).  ``sourceDatabase`` is retained
        as a flat column alongside ``interactionResources`` so that downstream
        steps can compute ``interactionId`` without re-extracting it from the
        struct.
    """
    interactions = (
        df
        .withColumn(
            'intB',
            f.when(f.col('interactorB.id').isNull(), f.col('interactorA.id')).otherwise(f.col('interactorB.id')),
        )
        .withColumn(
            'intB_source',
            f.when(f.col('interactorB.id_source').isNull(), f.col('interactorA.id_source')).otherwise(
                f.col('interactorB.id_source')
            ),
        )
        .withColumn(
            'speciesB',
            f.when(f.col('interactorB.organism').isNull(), f.col('interactorA.organism')).otherwise(
                f.col('interactorB.organism')
            ),
        )
        .withColumn(
            'intBBiologicalRole',
            f.when(
                f.col('interactorB.biological_role').isNull(),
                f.col('interactorA.biological_role'),
            ).otherwise(f.col('interactorB.biological_role')),
        )
        .withColumn(
            'interactionScore',
            f.when(
                f.col('interaction.interaction_score') > 1,
                f.col('interaction.interaction_score') / 1000,
            ).otherwise(f.col('interaction.interaction_score')),
        )
        .selectExpr(
            'interactorA.id as intA',
            'interactorA.id_source as intA_source',
            'interactorA.organism as speciesA',
            'interactorA.biological_role as intABiologicalRole',
            'intB',
            'intB_source',
            'speciesB',
            'intBBiologicalRole',
            'source_info.source_database as sourceDatabase',
            'source_info as interactionResources',
            'interaction.evidence as evidencesList',
            'interactionScore',
        )
        .withColumn(
            'speciesA',
            f.struct(
                f.col('speciesA.mnemonic'),
                f.col('speciesA.scientific_name').alias('scientificName'),
                f.col('speciesA.taxon_id').alias('taxonId'),
            ),
        )
        .withColumn(
            'speciesB',
            f.struct(
                f.col('speciesB.mnemonic'),
                f.col('speciesB.scientific_name').alias('scientificName'),
                f.col('speciesB.taxon_id').alias('taxonId'),
            ),
        )
    )

    # Broadcast mapping_info to convert both gene-mapping joins from sort-merge
    # (shuffle on the large interaction table) to broadcast hash joins.
    mapping_bc = f.broadcast(mapping_info)

    interaction_map_left = (
        interactions
        .join(
            mapping_bc,
            _get_code(f.col('intA')) == f.col('mapped_id'),
            'left',
        )
        .withColumn(
            'targetA',
            f.when(f.col('gene_id').isNull(), f.lit(None)).otherwise(f.col('gene_id')),
        )
        .drop('gene_id', 'mapped_id')
    )

    interaction_mapped = (
        interaction_map_left
        .join(
            mapping_bc.alias('mapping'),
            _get_code(f.col('intB')) == f.col('mapping.mapped_id'),
            'left',
        )
        .withColumn(
            'targetB',
            f.when(f.col('gene_id').isNull(), f.lit(None)).otherwise(f.col('gene_id')),
        )
        .drop('gene_id', 'mapping.mapped_id')
    )

    # Swap A/B for bidirectional sources and union.
    # Exclude rows where intA == intB — swapping a self-interaction produces
    # an identical row, which would create duplicates (fixes opentargets/issues#3853).
    reverse_interactions = (
        interaction_mapped
        .filter(f.col('sourceDatabase').isin(*_BIDIRECTIONAL_SOURCES))
        .filter(f.col('intA') != f.col('intB'))
        .select([f.col(c).alias(_SWAP_MAP.get(c, c)) for c in interaction_mapped.columns])
    )

    full_interactions = interaction_mapped.unionByName(reverse_interactions)

    # sourceDatabase is retained (not dropped) so that _interaction_id_expr()
    # can reference it directly in downstream steps without re-extracting from
    # the interactionResources struct.
    return full_interactions.withColumn('evidences', f.explode(f.col('evidencesList'))).drop('evidencesList')


def _select_interaction_fields(df: DataFrame) -> DataFrame:
    """Select and flatten fields for the common interaction-evidence representation.

    The output is used as shared input for both ``_generate_interactions_agg``
    (which needs all dimension columns) and ``_select_evidence_fields`` (which
    replaces them with ``interactionId``).

    Args:
        df: Interaction evidence DataFrame from ``_generate_interactions``.

    Returns:
        DataFrame with flattened dimension and evidence fields.
    """
    return df.selectExpr(
        'sourceDatabase',
        'targetA',
        'intA',
        'intA_source',
        'speciesA',
        'targetB',
        'intB',
        'intB_source',
        'speciesB',
        'interactionResources',
        'interactionScore',
        'evidences.*',
        'intABiologicalRole',
        'intBBiologicalRole',
    )


def _select_evidence_fields(df: DataFrame) -> DataFrame:
    """Produce the evidence-only projection, replacing dimension columns with interactionId.

    The interaction identity columns listed in ``_DIMENSION_COLS`` are dropped;
    they are encoded in ``interactionId`` so consumers can join back to the
    interactions dataset when needed.

    Args:
        df: Output of ``_select_interaction_fields``.

    Returns:
        DataFrame with interactionId, interactionResources, interactionScore,
        and all flattened evidence sub-fields.
    """
    return df.withColumn('interactionId', _interaction_id_expr()).drop(*_DIMENSION_COLS)


def _generate_interactions_agg(df: DataFrame) -> DataFrame:
    """Aggregate interaction evidence rows into per-pair summary records.

    Groups by source database, targetA/B, intA/B, biological roles and species,
    and produces a count of evidence rows and the first interaction score.
    Adds ``interactionId`` post-aggregation as the primary key for the dataset.

    Args:
        df: Interaction evidence DataFrame (from ``_select_interaction_fields``).

    Returns:
        DataFrame with aggregated interaction records, sourceDatabase column,
        and interactionId.
    """
    return (
        df
        # _INTERACTION_KEY_COLS uses the flat ``sourceDatabase`` from
        # _select_interaction_fields, so no rename of the nested struct field is needed.
        .groupBy(*_INTERACTION_KEY_COLS)
        .agg(
            f.count('*').alias('count'),
            f.first(f.col('interactionScore')).alias('scoring'),
        )
        .withColumn('interactionId', _interaction_id_expr())
    )


def _transform_ensembl_protein(df: DataFrame) -> DataFrame:
    """Extract gene_id, protein_id pairs from the Ensembl GTF file.

    Filters to CDS feature rows and extracts ENSG/ENSP identifiers from
    column _c8 via regex.

    Args:
        df: Raw GTF TSV DataFrame (comment lines already excluded by reader).

    Returns:
        DataFrame with columns gene_id and protein_id.
    """
    return (
        df
        .filter(f.col('_c2') == 'CDS')
        .withColumn('gene_id', f.regexp_extract(f.col('_c8'), r'ENSG\w{11}', 0))
        .withColumn('protein_id', f.regexp_extract(f.col('_c8'), r'ENSP\w{11}', 0))
        .select('gene_id', 'protein_id')
    )


def _get_unmatched(intact_df: DataFrame, string_df: DataFrame) -> DataFrame:
    """Collect unmatched interactorB IDs (human, no targetB resolved).

    Args:
        intact_df: IntAct interaction evidence DataFrame.
        string_df: STRING interaction evidence DataFrame.

    Returns:
        Distinct DataFrame with column intB.
    """
    intact_missing = intact_df.filter(f.col('targetB').isNull() & (f.col('speciesB.taxonId') == 9606)).select('intB')

    string_missing = string_df.filter(f.col('targetB').isNull() & (f.col('speciesB.taxonId') == 9606)).select('intB')

    return intact_missing.unionByName(string_missing).select('intB').distinct()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def interaction(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Compute interaction datasets from IntAct and STRING.

    Reads target, RNACentral, Human Mapping, Ensembl protein, IntAct, and
    STRING inputs. Produces aggregated interaction records, per-evidence
    records, and a list of unmatched interactor IDs.

    Output schema
    ~~~~~~~~~~~~~
    ``interactions``          — one row per (source, targetA, targetB) pair with
                                ``interactionId``, dimension columns, count, scoring.
    ``interactions_evidence`` — one row per evidence entry with ``interactionId``
                                (FK) and evidence-specific fields only; dimension
                                columns are not repeated.

    Args:
        source: Input paths keyed by 'targets', 'rnacentral', 'humanmapping',
            'ensproteins', 'intact', 'strings'.
        destination: Output paths keyed by 'interactions',
            'interactions_evidence', 'interactions_unmatched'.
        settings: Step settings; may contain 'scorethreshold' (int, default 0)
            and 'string_version' (str, default '12').
        properties: Spark/GCS properties forwarded to Session.
    """
    spark: SparkSession = Session(app_name='interaction', properties=properties).spark

    score_threshold: int = int(settings.get('scorethreshold', 0))
    string_version: str = str(settings.get('string_version', '12'))
    partition_count = settings.get('partition_count') or {}
    # Auto-scales with the cluster (cores across workers). Used to spread the
    # non-splittable STRING gzip and to parallelise the cached IntAct frame.
    cache_parts = spark.sparkContext.defaultParallelism

    logger.info('Loading target data from %s', source['targets'])
    target_df = spark.read.parquet(source['targets'])

    logger.info('Loading RNACentral data from %s', source['rnacentral'])
    rnacentral_df = spark.read.option('sep', '\t').option('header', 'false').csv(source['rnacentral'])

    logger.info('Loading Human Mapping data from %s', source['humanmapping'])
    human_mapping_df = spark.read.option('sep', '\t').option('header', 'false').csv(source['humanmapping'])

    logger.info('Loading Ensembl protein data from %s', source['ensproteins'])
    ensproteins_raw = (
        spark.read.option('sep', '\t').option('header', 'false').option('comment', '#').csv(source['ensproteins'])
    )
    ensproteins_df = _transform_ensembl_protein(ensproteins_raw)

    logger.info('Loading IntAct data from %s', source['intact'])
    intact_raw = spark.read.json(source['intact'])

    logger.info('Loading STRING data from %s', source['strings'])
    strings_raw = spark.read.option('sep', ' ').option('header', 'true').csv(source['strings']).repartition(cache_parts)

    # Build mapping lookup
    logger.info('Generating ID mapping table')
    mapping_df = _generate_mapping(target_df, rnacentral_df, human_mapping_df)

    # STRING interactions
    logger.info('Transforming STRING proteins (score_threshold=%d)', score_threshold)
    string_proteins = _transform_string_proteins(strings_raw, score_threshold, string_version)
    string_mapping = ensproteins_df.withColumnRenamed('protein_id', 'mapped_id').distinct()

    # Cache the full interaction evidence DataFrames so the three downstream write
    # actions (aggregated, evidence, unmatched) reuse the same materialised result
    # instead of re-executing the expensive gene-mapping joins each time. STRING is
    # already repartitioned at read (its gzip is non-splittable); IntAct is
    # repartitioned here so its cached frame is read in parallel by all three actions.
    logger.info('Transforming STRING interactions')
    string_interactions_df = (
        _generate_interactions(string_proteins, string_mapping)
        .filter(f.col('evidences.evidence_score') > 0)
        .cache()
    )

    logger.info('Transforming IntAct interactions')
    intact_interactions_df = _generate_interactions(intact_raw, mapping_df).repartition(cache_parts).cache()

    # Filter: remove null targetA (keep for unmatched output)
    intact_valid = intact_interactions_df.filter(f.col('targetA').isNotNull())
    string_valid = string_interactions_df.filter(f.col('targetA').isNotNull())

    # Select and flatten fields once; cache so both the aggregation and the
    # evidence projection share a single materialised scan of the join output.
    logger.info('Selecting interaction fields')
    intact_fields = _select_interaction_fields(intact_valid).cache()
    string_fields = _select_interaction_fields(string_valid).cache()

    # Aggregated interactions
    logger.info('Aggregating interaction pairs')
    intact_agg = _generate_interactions_agg(intact_fields)
    string_agg = _generate_interactions_agg(string_fields)
    interactions_parts = partition_count.get('interactions')
    aggregated_raw = rename_columns_to_camel_case(intact_agg.unionByName(string_agg))
    aggregated = aggregated_raw.repartition(interactions_parts) if interactions_parts else aggregated_raw

    # Evidences — dimension columns replaced by interactionId
    logger.info('Generating interaction evidences')
    intact_evidences = _select_evidence_fields(intact_fields)
    string_evidences = _select_evidence_fields(string_fields).withColumn(
        'evidence_score', f.col('evidence_score') / 1000
    )
    evidence_parts = partition_count.get('interactions_evidence')
    evidences_raw = rename_columns_to_camel_case(
        string_evidences.unionByName(intact_evidences, allowMissingColumns=True)
    )
    evidences = evidences_raw.repartition(evidence_parts) if evidence_parts else evidences_raw

    # Unmatched — uses the pre-filter cached DataFrames
    logger.info('Collecting unmatched interactors')
    unmatched = _get_unmatched(intact_interactions_df, string_interactions_df)

    try:
        logger.info('Writing interactions to %s', destination['interactions'])
        aggregated.write.mode('overwrite').parquet(destination['interactions'])

        logger.info('Writing interactions_evidence to %s', destination['interactions_evidence'])
        evidences.write.mode('overwrite').parquet(destination['interactions_evidence'])

        logger.info('Writing interactions_unmatched to %s', destination['interactions_unmatched'])
        unmatched.write.mode('overwrite').parquet(destination['interactions_unmatched'])
    finally:
        intact_fields.unpersist()
        string_fields.unpersist()
        intact_interactions_df.unpersist()
        string_interactions_df.unpersist()
