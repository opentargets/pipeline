"""Definition of Evidence class."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Self

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel

from pts.pyspark.associations_utils.association import Association
from pts.pyspark.associations_utils.dataset import Dataset


@dataclass
class Evidence(Dataset):
    """Methods applied on evidence data."""

    # This is the default set of columns we group evidence by, used for association calculation:
    GROUPBY_COLUMNS: list[str] = field(default_factory=lambda: ['targetId', 'diseaseId', 'datasourceId', 'datatypeId'])

    # These are the columns we are working on:
    MANDATORY_COLUMNS: ClassVar[list[str]] = [
        'targetId',
        'diseaseId',
        'score',
        'datasourceId',
        'datatypeId',
        'year',
    ]

    @classmethod
    def from_raw_evidence(cls: type[Self], raw_evidence_input: DataFrame) -> Evidence:
        """Convert raw OT evidence to Evidence object.

        Args:
            raw_evidence_input (DataFrame): spark dataframe with the raw evidence.

        Returns:
            Evidence: formatted evidence data with relevant columns.
        """
        # Validation:
        raw_input_columns = [col for col in cls.MANDATORY_COLUMNS if col != 'year']
        cls.validate_columns(raw_evidence_input, [*raw_input_columns, 'evidenceDate'])

        # Next year for evidence without evidence date:
        next_year = datetime.now().year + 1

        # Covert and return evidence:
        return Evidence(
            raw_evidence_input.filter(f.col('score') > 0).select(
                *raw_input_columns, cls._evidence_date_to_year(f.col('evidenceDate'), next_year).alias('year')
            )
        )

    def expand_disease(self: Evidence, disease_index: DataFrame) -> Evidence:
        """Calculate indirect evidence based on the provided disease ontology.

        Args:
            disease_index (DataFrame): Open Targets disease index.

        Returns:
            Evidence: evidence exploded to indirect evidence.
        """
        # Exploding disease ancestors:
        processed_disease = disease_index.select(
            f.col('id').alias('diseaseId'),
            f.explode(f.array_union(f.array(f.col('id')), f.col('ancestors'))).alias('specificDiseaseId'),
        )

        # Exploding evidence for all ancestors:
        return Evidence(
            self.df
            .join(f.broadcast(processed_disease), on='diseaseId', how='inner')
            .drop('diseaseId')
            .withColumnRenamed('specificDiseaseId', 'diseaseId')
            .persist(StorageLevel.DISK_ONLY)
        )

    def aggregate_evidence_by_datasource(self: Evidence, persist: bool = False) -> Association:
        """Compute association scores from the evidence scores.

        Args:
            persist (bool): flag if the resulting dataset should be persisted or not. By default: no.

        Returns:
            Association: Calculated association dataset
        """
        # Partitioning: all evidence accumulated for each disease-target-datasource-year triplet until the given year
        window_for_collapsing_scores = (
            Window.partitionBy(*self.GROUPBY_COLUMNS).orderBy('year').rangeBetween(Window.unboundedPreceding, 0)
        )

        association_df = (
            self.df
            .groupby(*self.GROUPBY_COLUMNS, 'year')
            # Collect all evidence scores for a year:
            .agg(f.collect_list('score').alias('yearlyEvidenceScores'))
            # Pool all evidence scores together:
            .withColumn(
                'retrospectiveEvidenceScores',
                f.flatten(f.collect_list('yearlyEvidenceScores').over(window_for_collapsing_scores)),
            )
            .withColumn(
                'associationScore',
                self._get_harmonic_sum(f.col('retrospectiveEvidenceScores')) / self.MAX_HARMONIC_SUM,
            )
            .withColumn('yearlyEvidenceCount', f.size('yearlyEvidenceScores'))
            .withColumnRenamed('datasourceId', 'aggregationValue')
            .withColumn('aggregationType', f.lit('datasourceId'))
            .drop('yearlyEvidenceScores', 'retrospectiveEvidenceScores', 'datatypeId')
        )

        if persist:
            return Association(association_df.persist(StorageLevel.MEMORY_AND_DISK))
        else:
            return Association(association_df)

    @staticmethod
    def _evidence_date_to_year(evidence_date: Column, next_year: int) -> Column:
        """Convert evidence date to year when it is not available, use the last year +1.

        Args:
            evidence_date (Column): spark column with the evidence date
            next_year (int): next year that will be assigned to evidence with no valid evidence date

        Returns:
            Column: year, which should never be null.

        Examples:
            >>> (spark.createDataFrame([('1999-12-31',), ('2021-02-11',), (None,),
            ... ("Batman-Robin",), ("2021",)], ['date'])
            ... .select('date', Evidence._evidence_date_to_year(f.col('date'), 12).alias('year'))
            ... .show())
            +------------+----+
            |        date|year|
            +------------+----+
            |  1999-12-31|1999|
            |  2021-02-11|2021|
            |        NULL|  12|
            |Batman-Robin|  12|
            |        2021|2021|
            +------------+----+
            <BLANKLINE>
        """
        maybe_year = f.when(evidence_date.isNotNull(), f.trim(f.regexp_extract(evidence_date, r'^(\d{4})(?:\D|$)', 1)))  # ty:ignore[missing-argument]

        return f.when(maybe_year != '', maybe_year.cast(t.IntegerType())).otherwise(next_year)  # noqa: PLC1901
