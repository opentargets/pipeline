"""Match dataset."""

from __future__ import annotations

from dataclasses import dataclass
from ontoma import OnToma
from loguru import logger
from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from pts.pyspark.literature_utils.common.schemas import parse_spark_schema
from pts.pyspark.literature_utils.dataset.dataset import Dataset
from pts.pyspark.literature_utils.dataset.match_mapped import MatchMapped
from pts.pyspark.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql.types import StructType


@dataclass
class Match(Dataset):
    """Match dataset.

    This dataset describes matches extracted from a Publication dataset.
    """

    @classmethod
    def get_schema(cls: type[Match]) -> StructType:
        """Provides the schema for the Match dataset.

        Returns:
            StructType: Schema for the Match dataset.
        """
        return parse_spark_schema("match.json")
    
    def map_labels(
        self: Match,
        session: Session,
        label_lut_path: str,
        label_col_name: str,
        type_col_name: str
    ) -> MatchMapped:
        """Maps labels using the provided label lookup table.

        If there are multiple mappings, the results are exploded accordingly.

        Args:
            session (Session): Spark Session object.
            label_lut_path (str): Path to the label lookup table.
            label_col_name (str): Name of the column containing the label.
            type_col_name (str): Name of the column containing the label type.

        Returns:
            MatchMapped: Dataset with mapped labels.
        """
        logger.info(f'load label lookup table from {label_lut_path}')
        label_lut = OnToma(spark=session.spark, cache_dir=label_lut_path)

        logger.info('map labels')
        mapped_matches = label_lut.map_entities(
            df=self.df,
            result_col_name="entityIds",
            entity_col_name=label_col_name,
            entity_kind="label",
            type_col_name=type_col_name,
            include_normalised_entities=True,
            include_entity_source=True
        )

        logger.info('explode results')
        return MatchMapped(
            _df=(
                mapped_matches
                .withColumn("mappedId", f.explode_outer(f.array_distinct("entityIds.entityId")))
                .withColumn("isMapped", f.col("mappedId").isNotNull())
            ),
            _schema=MatchMapped.get_schema()
        )
