"""Interaction dataset generation.

Ported from platform-etl-backend Interaction step. Computes protein-protein
and RNA interactions from IntAct and STRING databases, producing aggregated
interaction records and per-evidence records.

Scala sources ported:
    - Interaction.scala (main assembly)
    - stringProtein/StringProtein.scala (STRING protein transformation)
"""

from __future__ import annotations

import re
from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
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

# UDF: truncate a string at the first '_' or '-' character.
# E.g. 'URS123-2_992' → 'URS123'.
_GET_CODE = f.udf(lambda s: re.split(r'[_\-]', s.strip())[0] if s else s)

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
        after exploding the evidences array).
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

    interaction_map_left = (
        interactions
        .join(
            mapping_info,
            _GET_CODE(f.col('intA')) == f.col('mapped_id'),
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
            mapping_info.alias('mapping'),
            _GET_CODE(f.col('intB')) == f.col('mapping.mapped_id'),
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

    return full_interactions.withColumn('evidences', f.explode(f.col('evidencesList'))).drop(
        'evidencesList', 'sourceDatabase'
    )


def _select_fields(df: DataFrame) -> DataFrame:
    """Select and flatten fields for the interaction_evidences index.

    Args:
        df: Interaction evidence DataFrame.

    Returns:
        DataFrame with flattened evidence fields.
    """
    return df.selectExpr(
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


def _generate_interactions_agg(df: DataFrame) -> DataFrame:
    """Aggregate interaction evidence rows into per-pair summary records.

    Groups by source database, targetA/B, intA/B, biological roles and species,
    and produces a count of evidence rows and the first interaction score.

    Args:
        df: Interaction evidence DataFrame (from _select_fields).

    Returns:
        DataFrame with aggregated interaction records and sourceDatabase column.
    """
    return (
        df
        .groupBy(
            'interactionResources.source_database',
            'targetA',
            'intA',
            'intABiologicalRole',
            'targetB',
            'intB',
            'intBBiologicalRole',
            'speciesA',
            'speciesB',
        )
        .agg(
            f.count('*').alias('count'),
            f.first(f.col('interactionScore')).alias('scoring'),
        )
        .withColumnRenamed('source_database', 'sourceDatabase')
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
    strings_raw = spark.read.option('sep', ' ').option('header', 'true').csv(source['strings'])

    # Build mapping lookup
    logger.info('Generating ID mapping table')
    mapping_df = _generate_mapping(target_df, rnacentral_df, human_mapping_df)

    # STRING interactions
    logger.info('Transforming STRING proteins (score_threshold=%d)', score_threshold)
    string_proteins = _transform_string_proteins(strings_raw, score_threshold, string_version)
    string_mapping = ensproteins_df.withColumnRenamed('protein_id', 'mapped_id').distinct()
    string_interactions_df = _generate_interactions(string_proteins, string_mapping).filter(
        f.col('evidences.evidence_score') > 0
    )

    # IntAct interactions
    logger.info('Transforming IntAct interactions')
    intact_interactions_df = _generate_interactions(intact_raw, mapping_df)

    # Filter: remove null targetA (keep for unmatched output)
    intact_valid = intact_interactions_df.filter(f.col('targetA').isNotNull())
    string_valid = string_interactions_df.filter(f.col('targetA').isNotNull())

    # Aggregated interactions
    logger.info('Aggregating interaction pairs')
    intact_agg = _generate_interactions_agg(_select_fields(intact_valid))
    string_agg = _generate_interactions_agg(_select_fields(string_valid))
    aggregated = rename_columns_to_camel_case(intact_agg.unionByName(string_agg)).coalesce(200)

    # Evidences
    logger.info('Generating interaction evidences')
    intact_evidences = _select_fields(intact_valid)
    string_evidences = _select_fields(string_valid).withColumn('evidence_score', f.col('evidence_score') / 1000)

    # Union evidences (string first, then intact — match Scala unionDataframeDifferentSchema order)
    all_columns = list(dict.fromkeys(string_evidences.columns + intact_evidences.columns))
    for col_name in all_columns:
        if col_name not in string_evidences.columns:
            string_evidences = string_evidences.withColumn(col_name, f.lit(None))
        if col_name not in intact_evidences.columns:
            intact_evidences = intact_evidences.withColumn(col_name, f.lit(None))

    evidences_raw = string_evidences.select(all_columns).unionByName(intact_evidences.select(all_columns))
    evidences = rename_columns_to_camel_case(evidences_raw).repartition(200)

    # Unmatched
    logger.info('Collecting unmatched interactors')
    unmatched = _get_unmatched(intact_interactions_df, string_interactions_df)

    logger.info('Writing interactions to %s', destination['interactions'])
    aggregated.write.mode('overwrite').parquet(destination['interactions'])

    logger.info('Writing interactions_evidence to %s', destination['interactions_evidence'])
    evidences.write.mode('overwrite').parquet(destination['interactions_evidence'])

    logger.info('Writing interactions_unmatched to %s', destination['interactions_unmatched'])
    unmatched.write.mode('overwrite').parquet(destination['interactions_unmatched'])
