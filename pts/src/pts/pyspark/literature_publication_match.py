"""Collapsed literature_publication + literature_match step.

Reads EPMC publications, builds the publication dataset in memory (no
intermediate publication parquet is written), extracts and maps matches, then
writes only the valid and failed match datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from literature.dataset.publication import Publication
from literature.datasource.epmc.publication import EPMCPublication
from literature.datasource.epmc.publication_id_lut import PublicationIdLUT
from loguru import logger
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import maybe_coalesce, maybe_repartition

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _epmc_read_path(epmc_path: str, kind: str, date_prefix: str | None) -> str:
    """Build the EPMC jsonl glob path for a publication kind.

    EPMC publications are laid out as ``{epmc_path}/{kind}/{day}/*.jsonl``, where
    ``{day}`` is a ``YYYY_MM_DD`` folder and the jsonl files sit directly inside
    it. An optional ``date_prefix`` selects a subset of day folders (e.g. a
    single month); a falsy prefix selects every day folder.

    Note: Hadoop/Spark globs are not recursive — ``*`` matches one path
    component and does not cross ``/`` — so each level is matched explicitly.

    Args:
        epmc_path: Base EPMC path, e.g. ``gs://otar025-epmc/ml02``.
        kind: Publication kind, ``abstract`` or ``fulltext``.
        date_prefix: Optional day-folder prefix, e.g. ``2026_03``. Falsy values
            (None, empty string) select all dates.

    Returns:
        A glob path string suitable for ``spark.read.json``.
    """
    base = epmc_path.rstrip('/')
    day_glob = f'{date_prefix}*' if date_prefix else '*'
    return f'{base}/{kind}/{day_glob}/*.jsonl'


def _read_publications(
    session: Session,
    epmc_path: str,
    pub_id_lut: DataFrame,
    date_prefix: str | None,
    repartition: int | None,
) -> Publication:
    """Read EPMC publications into a ``Publication`` dataset, in memory.

    Replicates ``EPMCPublication.from_source`` but with a parameterisable
    date-folder glob and an optional post-read repartition. No intermediate
    publication parquet is written. The library's underscore-prefixed building
    blocks are reused intentionally — the only part that must be reimplemented
    is the read, because the library hardcodes its glob.

    The final publication dataframe is persisted: the downstream pipeline
    triggers several independent actions (write match_valid, write
    match_failed) that all need this dataframe materialised, and without
    persist Spark re-scans the EPMC source jsonl files for each lineage.
    Caller is responsible for ``df.unpersist()``.

    Args:
        session: PTS Spark session wrapper.
        epmc_path: Base EPMC path, e.g. ``gs://otar025-epmc/ml02``.
        pub_id_lut: Parsed publication id lookup table DataFrame.
        date_prefix: Optional day-folder prefix, e.g. ``2026_03``.
        repartition: Optional partition count applied right after the read.

    Returns:
        A ``Publication`` dataset built entirely in memory.
    """
    spark = session.spark
    schema = EPMCPublication.defined_schema

    def _read_kind(kind: str) -> DataFrame:
        df = (
            spark.read
            .schema(schema)
            .json(_epmc_read_path(epmc_path, kind, date_prefix))
            .withColumn('kind', f.lit(kind))
            .withColumn('traceSource', f.input_file_name())
        )
        return maybe_repartition(df, repartition)

    fulltexts = _read_kind('fulltext')
    processed_fulltexts = EPMCPublication._annotate_fulltexts_with_pmid(fulltexts, pub_id_lut)

    abstracts = _read_kind('abstract')
    all_publications = EPMCPublication._merge_abstracts_with_fulltexts(abstracts, processed_fulltexts)

    most_recent = EPMCPublication._get_most_recent_publications(all_publications)
    return Publication(
        _df=most_recent.repartition(f.col('pmid')).persist(),
        _schema=Publication.get_schema(),
    )


def literature_publication_match(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Collapsed literature_publication + literature_match step.

    Reads EPMC publications (optionally restricted to a single month via
    ``settings.date_prefix``), builds the publication dataset in memory, extracts
    and maps matches, disambiguates them, and writes only the valid and failed
    match datasets. The publication parquet is never written.

    Args:
        source: ``pub_id_lut``, ``epmc_publication``, ``ontoma_disease_target_drug_label_lut``.
        destination: ``match_valid``, ``match_failed``.
        settings: optional ``date_prefix`` (str), ``repartition`` (int) for the
            EPMC read, and ``match_valid_coalesce`` / ``match_failed_coalesce``
            (int) for the output partition count of each write.
        properties: Spark properties forwarded to the session.
    """
    spark = Session(app_name='literature', properties=properties)

    date_prefix = settings.get('date_prefix')
    repartition = settings.get('repartition')
    match_valid_coalesce = settings.get('match_valid_coalesce')
    match_failed_coalesce = settings.get('match_failed_coalesce')

    logger.info(f'load publication id lut from: {source["pub_id_lut"]}')
    pub_id_lut = PublicationIdLUT.from_csv(spark, source['pub_id_lut']).persist()

    logger.info(
        f'read EPMC publications from: {source["epmc_publication"]} '
        f'(date_prefix={date_prefix}, repartition={repartition})'
    )
    publication = _read_publications(
        spark,
        source['epmc_publication'],
        pub_id_lut,
        date_prefix,
        repartition,
    )

    logger.info('extract matches and map labels')
    match_mapped = publication.extract_matches().map_labels(
        session=spark,
        label_lut_path=source['ontoma_disease_target_drug_label_lut'],
        label_col_name='label',
        type_col_name='type',
    )
    # consumed by match_disambiguated and the isMapped==False filter
    match_mapped.df.persist()

    logger.info('disambiguate')
    match_disambiguated = match_mapped.disambiguate(
        trusted_sources=[
            'name',
            'ot_curation',
            'eva_clinvar',
            'clinvar_xrefs',
            'approved_name',
            'approved_symbol',
            'trade_name_component',
        ]
    )
    # consumed by the isValid==True and isValid==False filters
    match_disambiguated.df.persist()

    match_valid = match_disambiguated.df.filter(f.col('isValid'))

    # rows that fail mapping are emitted via the isMapped==False branch, so
    # guard the disambiguation branch with isMapped==True to keep the union disjoint
    match_failed = match_mapped.df.filter(~f.col('isMapped')).unionByName(
        match_disambiguated.df.filter(f.col('isMapped')).filter(~f.col('isValid')),
        allowMissingColumns=True,
    )

    logger.info(f'write valid matches to {destination["match_valid"]} (coalesce={match_valid_coalesce})')
    maybe_coalesce(match_valid, match_valid_coalesce).write.mode('overwrite').parquet(destination['match_valid'])

    logger.info(f'write failed matches to {destination["match_failed"]} (coalesce={match_failed_coalesce})')
    maybe_coalesce(match_failed, match_failed_coalesce).write.mode('overwrite').parquet(destination['match_failed'])

    match_mapped.df.unpersist()
    match_disambiguated.df.unpersist()
    publication.df.unpersist()
    pub_id_lut.unpersist()
