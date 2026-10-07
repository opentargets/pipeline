"""Cooccurrence dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pts.pyspark.literature_utils.common.schemas import parse_spark_schema
from pts.pyspark.literature_utils.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql.types import StructType


@dataclass
class Cooccurrence(Dataset):
    """Cooccurrence dataset.

    This dataset describes cooccurrences obtained by combining matches.
    """

    @classmethod
    def get_schema(cls: type[Cooccurrence]) -> StructType:
        """Provides the schema for the Cooccurrence dataset.

        Returns:
            StructType: Schema for the Cooccurrence dataset.
        """
        return parse_spark_schema("cooccurrence.json")
