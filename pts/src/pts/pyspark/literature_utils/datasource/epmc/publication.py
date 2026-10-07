"""Publications from Europe PMC Data Source."""

from __future__ import annotations

from loguru import logger
from pathlib import Path
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from pts.pyspark.literature_utils.common.schemas import parse_spark_schema
from pts.pyspark.common.session import Session
from pts.pyspark.literature_utils.dataset.publication import Publication

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class EPMCPublication:
    """Class to process publications from Europe PMC."""

    # schema specifying desired subset of columns
    defined_schema = parse_spark_schema("publication.json")

    @classmethod
    def _read_in_with_schema(
        cls: type[EPMCPublication],
        session: Session,
        epmc_path: str,
        publication_kind: str
    ) -> DataFrame:
        """Read in publications from the specified filepath using the specified schema.

        Args:
            session (Session): Spark Session object.
            epmc_path (str): Path to EPMC publications.
            publication_kind (str): Kind of publication.

        Returns:
            DataFrame: DataFrame with EPMC publications.
        """
        publication_path = str(Path(epmc_path) / publication_kind / "**/*.jsonl")
        return (
            session.spark.read.schema(cls.defined_schema)
            .json(publication_path)
            .withColumn("kind", f.lit(publication_kind))
            .withColumn("traceSource", f.input_file_name())
        )

    @staticmethod
    def _annotate_fulltexts_with_pmid(fulltext: DataFrame, lut: DataFrame) -> DataFrame:
        """Add pmid information to fulltexts that do not already have pmids using a publication id lookup table.
        
        Args:
            fulltext (DataFrame): DataFrame with fulltext publications.
            lut (DataFrame): DataFrame with the publication id lookup table.
        
        Returns:
            DataFrame: DataFrame of fulltext publications with pmid information added.
        """
        return (
            fulltext.alias("fulltext")
            .join(
                lut.alias("lut"),
                on = [f.col("fulltext.pmcid") == f.col("lut.pmcid_lut")],
                how="inner"
            )
            .withColumn("pmid", f.coalesce(f.col("pmid"), f.col("pmid_lut")))
            .filter(f.col("pmid") == f.col("pmid_lut"))
            .drop("pmid_lut", "pmcid_lut")
        )

    @staticmethod
    def _merge_abstracts_with_fulltexts(abstract: DataFrame, fulltext: DataFrame) -> DataFrame:
        """Merge abstracts with fulltexts, excluding abstracts for which the fulltext version exists.
        
        Args:
            abstract (DataFrame): DataFrame with abstracts of publications.
            fulltext (DataFrame): DataFrame with fulltext publications.
        
        Returns:
            DataFrame: DataFrame containing complete set of non-repeating publications consisting of both abstracts and fulltexts.
        """
        # exclude abstracts with fulltexts
        abstracts_without_fulltexts = (
            abstract
            .join(
                fulltext,
                on="pmid",
                how="left_anti"
            )
        )

        # combine abstracts with fulltexts
        return (
            abstracts_without_fulltexts
            .unionByName(
                fulltext,
                allowMissingColumns=True
            )
        )
    
    @staticmethod
    def _get_most_recent_publications(df: DataFrame) -> DataFrame:
        """Deduplicate publications by taking the publication with the most recent timestamp.

        Args:
            df (DataFrame): DataFrame with publications for deduplication.
        
        Returns:
            DataFrame: DataFrame with deduplicated set of publications.
        """
        # add timestamp column
        timestamped = (
            df
            .withColumn("int_timestamp", f.unix_timestamp(f.col("timestamp")))
        )

        # get most recent version of each publication
        most_recent_publications = (
            timestamped
            .groupBy(f.col("pmid"), f.col("pmcid"))
            .agg(f.max(f.col("int_timestamp")).alias("max_timestamp"))
        )

        # get other relevant fields for each most recent publication
        return (
            most_recent_publications
            .select(
                f.col("pmid").alias("mrp_pmid"),
                f.col("pmcid").alias("mrp_pmcid"),
                f.col("max_timestamp")
            )
            .join(
                timestamped,
                on=[
                    f.col("mrp_pmid").eqNullSafe(timestamped.pmid) &
                    f.col("mrp_pmcid").eqNullSafe(timestamped.pmcid) &
                    f.col("max_timestamp").eqNullSafe(timestamped.int_timestamp)
                ],
                how="left"
            )
            .drop("mrp_pmcid", "mrp_pmid", "max_timestamp", "int_timestamp", "timestamp", "kind")
        )
    
    @classmethod
    def from_source(
        cls: type[EPMCPublication],
        session: Session,
        epmc_path: str,
        lut: DataFrame
    ) -> Publication:
        """Read publications from the specified filepath.

        Publications are partitioned by pmid.

        Args:
            session (Session): Spark Session object.
            epmc_path (str): Path to EPMC publications.
            lut (DataFrame): DataFrame with the publication id lookup table.

        Returns:
            Publication: Publication dataset with EPMC publications.
        """
        logger.info(f'load fulltexts from {epmc_path}')
        fulltexts = cls._read_in_with_schema(session, epmc_path, "fulltext")
        
        logger.info('annotate fulltexts with pmid')
        processed_fulltexts = cls._annotate_fulltexts_with_pmid(fulltexts, lut)

        logger.info(f'load abstracts from {epmc_path}')
        abstracts = cls._read_in_with_schema(session, epmc_path, "abstract")

        logger.info('merge abstracts with fulltexts')
        all_publications = cls._merge_abstracts_with_fulltexts(abstracts, processed_fulltexts)
        
        logger.info('get most recent publications')
        return Publication(
            _df=(
                cls._get_most_recent_publications(all_publications)
                .repartition(f.col("pmid"))
            ),
            _schema=Publication.get_schema()
        )
    