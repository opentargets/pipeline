"""Target view dataset generation.

Recomposes the pre-refactor monolithic target schema by collating output/target
with the standalone datasets it was split into: homology, target_tractability,
target_safety_event, chemical_probes, and transcript. Intended as a
schema-compatibility view for consumers that still expect the old nested
shape (Platform, API, croissant), not a replacement for output/target within
the pipeline.

Two legacy fields are not reproduced: ``tep`` (dropped) and
``alternativeGenes`` (no source data anywhere in the current pipeline).
``transcripts[].isUniprotReviewed`` is always false and
``transcripts[].alphafoldId``/``uniprotIsoformId`` are always null, since the
new transcript dataset carries no equivalent data.
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import maybe_coalesce


def target_view(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Build the target_view dataset and write it out.

    Args:
        source: Mapping of logical input names to paths. Expected keys:
            ``target``, ``homology``, ``target_tractability``,
            ``target_safety_event``, ``chemical_probes``, ``transcript``
            (all output/ parquet datasets).
        destination: Output path for the target_view parquet dataset.
        settings: Step settings; supports ``partition_count`` (int).
        properties: Spark properties passed to :class:`Session`.
    """
    spark = Session(app_name='target_view', properties=properties).spark

    logger.info('Reading target_view inputs')
    target_df = spark.read.parquet(source['target'])
    homology_df = spark.read.parquet(source['homology'])
    tractability_df = spark.read.parquet(source['target_tractability'])
    safety_event_df = spark.read.parquet(source['target_safety_event'])
    chemical_probes_df = spark.read.parquet(source['chemical_probes'])
    transcript_df = spark.read.parquet(source['transcript'])

    logger.info('Aggregating homologues')
    homologues = _build_homologues(homology_df)

    logger.info('Aggregating tractability')
    tractability = _build_tractability(tractability_df)

    logger.info('Aggregating safety liabilities')
    safety_liabilities = _build_safety_liabilities(safety_event_df)

    logger.info('Aggregating chemical probes')
    chemical_probes = _build_chemical_probes(chemical_probes_df)

    logger.info('Aggregating transcripts')
    transcripts = _build_transcripts(transcript_df)

    logger.info('Assembling target_view')
    result = (
        target_df
        .join(homologues, 'id', 'left_outer')
        .join(tractability, 'id', 'left_outer')
        .join(safety_liabilities, 'id', 'left_outer')
        .join(chemical_probes, 'id', 'left_outer')
        .join(transcripts, 'id', 'left_outer')
    )

    partition_count = (settings or {}).get('partition_count')
    logger.info(f'Writing target_view to {destination}')
    maybe_coalesce(result, partition_count).write.mode('overwrite').parquet(destination)


def _build_homologues(df: DataFrame) -> DataFrame:
    """Group homology rows by target into the old ``homologues`` array.

    Args:
        df: output/homology, one row per (targetId, homologue) pair.

    Returns:
        DataFrame with [id, homologues[{speciesId, speciesName, homologyType,
        targetGeneId, isHighConfidence, targetGeneSymbol,
        queryPercentageIdentity, targetPercentageIdentity, priority}]].
    """
    return (
        df
        .select(
            'targetId',
            f.struct(
                'speciesId',
                'speciesName',
                'homologyType',
                'targetGeneId',
                'isHighConfidence',
                'targetGeneSymbol',
                'queryPercentageIdentity',
                'targetPercentageIdentity',
                'priority',
            ).alias('homologue'),
        )
        .groupBy('targetId')
        .agg(f.collect_list('homologue').alias('homologues'))
        .withColumnRenamed('targetId', 'id')
    )


def _build_tractability(df: DataFrame) -> DataFrame:
    """Group tractability rows by target into the old ``tractability`` array.

    Args:
        df: output/target_tractability, one row per (targetId, modality, id) assessment.

    Returns:
        DataFrame with [id, tractability[{modality, id, value}]].
    """
    return (
        df
        .select('targetId', f.struct('modality', 'id', 'value').alias('assessment'))
        .groupBy('targetId')
        .agg(f.collect_list('assessment').alias('tractability'))
        .withColumnRenamed('targetId', 'id')
    )


def _build_safety_liabilities(df: DataFrame) -> DataFrame:
    """Group safety event rows by target into the old ``safetyLiabilities`` array.

    Args:
        df: output/target_safety_event, one row per (targetId, event) liability.

    Returns:
        DataFrame with [id, safetyLiabilities[{event, eventId, effects,
        biosamples, datasource, literature, url, studies}]].
    """
    return (
        df
        .select(
            'targetId',
            f.struct(
                'event',
                'eventId',
                'effects',
                'biosamples',
                'datasource',
                'literature',
                'url',
                'studies',
            ).alias('liability'),
        )
        .groupBy('targetId')
        .agg(f.collect_list('liability').alias('safetyLiabilities'))
        .withColumnRenamed('targetId', 'id')
    )


def _build_chemical_probes(df: DataFrame) -> DataFrame:
    """Group chemical probe rows by target into the old ``chemicalProbes`` array.

    Args:
        df: output/chemical_probes, one row per (targetId, probe).

    Returns:
        DataFrame with [id, chemicalProbes[{targetFromSourceId, id,
        drugFromSourceId, drugId, mechanismOfAction, origin, control,
        isHighQuality, probesDrugsScore, probeMinerScore, scoreInCells,
        scoreInOrganisms, urls}]].
    """
    return (
        df
        .select(
            'targetId',
            f.struct(
                'targetFromSourceId',
                'id',
                'drugFromSourceId',
                'drugId',
                'mechanismOfAction',
                'origin',
                'control',
                'isHighQuality',
                'probesDrugsScore',
                'probeMinerScore',
                'scoreInCells',
                'scoreInOrganisms',
                'urls',
            ).alias('probe'),
        )
        .groupBy('targetId')
        .agg(f.collect_list('probe').alias('chemicalProbes'))
        .withColumnRenamed('targetId', 'id')
    )


def _build_transcripts(df: DataFrame) -> DataFrame:
    """Group transcript rows by target into the old transcript-related fields.

    Produces transcriptIds, transcripts, canonicalTranscript, and canonicalExons.

    Args:
        df: output/transcript, one row per (targetId, transcript).

    Returns:
        DataFrame with [id, transcriptIds[String], transcripts[{transcriptId,
        biotype, uniprotId, isUniprotReviewed, translationId, alphafoldId,
        uniprotIsoformId, isEnsemblCanonical}], canonicalTranscript{id,
        chromosome, start, end, strand}, canonicalExons[String]].
    """
    per_transcript = (
        df
        .withColumn('uniprotId', f.element_at(f.col('uniprotIds'), 1))
        .select(
            'targetId',
            'transcriptId',
            f.struct(
                'transcriptId',
                'biotype',
                'uniprotId',
                f.lit(False).alias('isUniprotReviewed'),
                f.col('proteinId').alias('translationId'),
                f.lit(None).cast('string').alias('alphafoldId'),
                f.lit(None).cast('string').alias('uniprotIsoformId'),
                'isEnsemblCanonical',
            ).alias('transcript'),
        )
    )

    agg = (
        per_transcript
        .groupBy('targetId')
        .agg(
            f.collect_list('transcriptId').alias('transcriptIds'),
            f.collect_list('transcript').alias('transcripts'),
        )
    )

    canonical = df.filter(f.col('isEnsemblCanonical'))

    canonical_transcript = canonical.select(
        'targetId',
        f.struct(
            f.col('transcriptId').alias('id'),
            'chromosome',
            'start',
            'end',
            'strand',
        ).alias('canonicalTranscript'),
    )

    canonical_exons = (
        canonical
        .select('targetId', f.explode('exons').alias('exon'))
        .select('targetId', f.col('exon.start').alias('start'), f.col('exon.end').alias('end'))
        .groupBy('targetId')
        .agg(f.sort_array(f.collect_list(f.struct('start', 'end'))).alias('sorted_exons'))
        .select(
            'targetId',
            f.flatten(
                f.transform(
                    f.col('sorted_exons'),
                    lambda e: f.array(e['start'].cast('string'), e['end'].cast('string')),
                )
            ).alias('canonicalExons'),
        )
    )

    return (
        agg
        .join(canonical_transcript, 'targetId', 'left_outer')
        .join(canonical_exons, 'targetId', 'left_outer')
        .withColumnRenamed('targetId', 'id')
    )
