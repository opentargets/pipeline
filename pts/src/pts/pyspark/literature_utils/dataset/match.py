"""Match dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyspark.sql.functions as f
from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import parse_spark_schema
from pts.pyspark.literature_utils.dataset.dataset import Dataset
from pts.pyspark.literature_utils.dataset.match_mapped import MatchMapped
from pts.pyspark.literature_utils.label_mapping import MAPPED_LABELS_DDL, map_label_batches

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
        return parse_spark_schema('literature_match.json')

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
        logger.info(f'map labels with the lookup table at {label_lut_path}')
        mapped_labels = (
            self.df
            .select(f.col(label_col_name).alias('label'), f.col(type_col_name).alias('type'))
            .distinct()
            .mapInArrow(map_label_batches(label_lut_path), MAPPED_LABELS_DDL)
            .alias('mapped')
        )
        mapped_matches = (
            self.df
            .alias('match')
            .join(
                mapped_labels,
                on=(
                    f.col(f'match.{label_col_name}').eqNullSafe(f.col('mapped.label'))
                    & f.col(f'match.{type_col_name}').eqNullSafe(f.col('mapped.type'))
                ),
                how='inner',
            )
            .select('match.*', 'mapped.entityLabelNormalised', 'mapped.entityIds')
            .distinct()
        )

        logger.info('explode results')
        return MatchMapped(
            _df=(
                mapped_matches
                .withColumn('mappedId', f.explode_outer(f.array_distinct('entityIds.entityId')))
                .withColumn('isMapped', f.col('mappedId').isNotNull())
            ),
            _schema=MatchMapped.get_schema()
        )
