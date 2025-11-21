"""Collection of functionalities required for evidence post-process and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window

from pts.pyspark.common.cast_to_schema import harmonise_to_schema
from pts.pyspark.common.utils import parse_spark_schema, required_columns, update_quality_flag


class EvidenceFlags(StrEnum):
    INVALID_DISEASE = 'No valid disease'
    INVALID_TARGET = 'No valid target'
    DUPLICATED = 'Duplicated'
    NO_VALID_SCORE = 'No valid score'
    INVALID_BIOTYPE = 'Invalid biotype'


@dataclass
class Evidence:
    # Incoming evidence:
    _df: DataFrame

    # Hardcoded column name to capture direction on target:
    DIRECTION_ON_TARGET_COLUMN_NAME: str = 'directionOnTarget'

    # Hardcoded values for evidence processing:
    QC_COLUMN: str = 'qualityControls'

    # Default lenght of variant identifier to start hashing:
    VARIANT_HASH_LENGHT: int = 300

    # Columns we consider for dating evidence:
    EVIDENCE_DATE_COLUMNS: list[str] = field(
        default_factory=lambda: ['publicationDate', 'curationDate', 'studyStartDate']
    )

    def __post_init__(self: Evidence) -> None:
        """Initial process of the evidence."""
        # Initialise QC column if not given:
        if self.QC_COLUMN not in self.df.columns:
            self._df = self._df.withColumn(self.QC_COLUMN, f.lit([]).cast(t.ArrayType(t.StringType())))

        evidence_schema = parse_spark_schema('evidence.json')
        self._df = harmonise_to_schema(self._df, evidence_schema)

    @required_columns(['targetFromSourceId'])
    def validate_target(self: Evidence, target_lut: DataFrame, invalid_biotypes: list[str] | None = None) -> Evidence:
        """Validation of targets.

        Args:
            target_lut (DataFrame): processed target look-up-table.
            invalid_biotypes (list[str] | None): if provided, evidence will be flagged with target of invalid biotype

        Returns:
            Evidence: evidence with mapped targets, and flaggeed evidence without target.
        """
        # If invalid biotypes are not provided, create empty array:
        if not invalid_biotypes:
            invalid_biotypes = []

        return Evidence(
            self.df
            # Resolve target identifiers:
            .join(
                f.broadcast(target_lut).select('targetId', 'biotype', 'targetFromSourceId'),
                on='targetFromSourceId',
                how='leftouter',
            )
            # Flag evidence without mapped targets:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(f.col(self.QC_COLUMN), f.col('targetId').isNull(), EvidenceFlags.INVALID_TARGET),
            )
            # Flag evidence without target of invalid biotype:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN), f.col('biotype').isin(invalid_biotypes), EvidenceFlags.INVALID_BIOTYPE
                ),
            )
            .drop('biotype')
        )

    @required_columns(['diseaseFromSourceMappedId'])
    def validate_diseases(self: Evidence, disease_lut: DataFrame) -> Evidence:
        """Validation of diseases.

        Args:
            disease_lut (DataFrame): processed disease look-up-table.

        Returns:
            Evidence: evidence with mapped targets, and flaggeed evidence without target.
        """
        return Evidence(
            self.df
            # Resolve target identifiers:
            .join(disease_lut, on='diseaseFromSourceMappedId', how='leftouter')
            # Flag evidence without mapped targets:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(f.col(self.QC_COLUMN), f.col('diseaseId').isNull(), EvidenceFlags.INVALID_DISEASE),
            )
        )

    @required_columns(['id'])
    def validate_uniqueness(self: Evidence) -> Evidence:
        """Validate uniqueness of evidence.

        Grouping evidence by evidence identifiers. If more than one evidence share the same ID all but one
        will be flagged.

        Returns:
            Evidence: where non-unique evidence is flagged.
        """
        return Evidence(
            self.df.withColumn('evidence_unique_rank', f.rank().over(Window.partitionBy('id').orderBy(f.rand())))
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN), f.col('evidence_unique_rank') != 1, EvidenceFlags.DUPLICATED
                ),
            )
            .drop('evidence_unique_rank')
        )

    @required_columns(['datasourceId'])
    def validate_datasource(self: Evidence, datasource_id: str) -> Evidence:
        """Validate evidence for datasource identifier.

        Only the specified datasourceId will be kept.

        Args:
            datasource_id (str): identifier of the datasource we want to work with.

        Returns:
            Evidence: with only one datasource
        """
        return Evidence(self.df.filter(f.col('datasourceId') == datasource_id))

    def assign_evidence_identifier(self: Evidence, unique_fields: list[str]) -> Evidence:
        """Adding unique identifier to each evidence based on source specific fields.

        Args:
            unique_fields (list[str]): list of column names that define evidence uniqueness.

        Returns:
            Evidence: with new column (id)
        """
        # Maybe not all columns are actually in the dataframe:
        valid_columns = [
            f.coalesce(f.col(col).cast(t.StringType()), f.lit('null'))
            for col in unique_fields
            if col in self.df.columns
        ]

        # Generate a hash based on the concatenated fields:
        hash_expression = f.sha1(f.concat(*valid_columns))

        return Evidence(self.df.withColumn('id', hash_expression))

    def hash_long_variant_identifiers(self: Evidence) -> Evidence:
        """Hash long variant identifier.

        If `variantId` column is present in the dataset, long ids need to be hashed.
        """
        if 'variantId' not in self.df.columns:
            return self

        return Evidence(
            self.df.withColumn('variantId', self._hash_long_variant_ids(f.col('variantId'), self.VARIANT_HASH_LENGHT))
        )

    def assign_direction_on_target(
        self: Evidence,
        direction_on_target_expression: str | None,
        mechanism_of_action_lut: DataFrame | None,
        target_lut: DataFrame | None,
    ) -> Evidence:
        """Assign direction of target for evidence.

        Args:
            direction_on_target_expression (str | None): spark sql expression to compute direction
            mechanism_of_action_lut (DataFrame | None): drug mechanism of action look up table
            target_lut (DataFrame | None): target look up table

        Returns:
            Evidence: when expression is provided a new column is assigned.

        Raises:
            ValueError: if "drugId" column is missing if moa dataset is used.
        """
        unused_columns = ['moaType', 'TSorOncogene']

        # If no expression is provided return evidence unchanged:
        if direction_on_target_expression is None:
            return self

        evidence_df = self.df

        # Mechanism of action requires the presence of drugId column:
        if mechanism_of_action_lut:
            if 'drugId' not in evidence_df.columns:
                raise ValueError(
                    'To use mechanism of action to annotate effect on target, "drugId" column must be in evidence.'
                )

            # Update evidence data frame, apply expression, and drop unused column:
            evidence_df = evidence_df.join(mechanism_of_action_lut, on=['targetId', 'drugId'], how='left')

        if target_lut:
            evidence_df = evidence_df.join(
                target_lut.select('targetId', 'TSorOncogene').distinct(), on='targetId', how='left'
            )

        return Evidence(
            evidence_df.withColumn(self.DIRECTION_ON_TARGET_COLUMN_NAME, f.expr(direction_on_target_expression)).drop(
                *unused_columns
            )
        )

    def calculate_evidence_score(self: Evidence, score_expression: str) -> Evidence:
        """Using the provided score expression, assign evidence score to evidence.

        Args:
            score_expression (str): valid spark expression as string.

        Returns:
            Evidence: with assigned score
        """
        return Evidence(
            self.df
            # Calculate evidence score and make sure the type is consistent across different datasets:
            .withColumn('score', f.expr(score_expression).cast(t.DoubleType()))
            # Flag evidence invalid scores (missing, zero or negative, more than 1)
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN),
                    f.when(
                        f.col('score').isNull() | (f.col('score') < 0) | (f.col('score') > 1), f.lit(True)
                    ).otherwise(f.lit(False)),
                    EvidenceFlags.NO_VALID_SCORE,
                ),
            )
        )

    @required_columns(['id'])
    def resolve_publication_date(self: Evidence, publication_date_lut: DataFrame) -> Evidence:
        """Date evidence based on literature references.

        ID is a mandatory column to run the logic, however the presence of literature column
        is not. If literature column is missing, self is returned.

        Args:
            publication_date_lut (DataFrarme): look up table with publication identifiers and date

        Returns:
            Evidence: with a new column `publicationDate` if literature is presents.
        """
        # Return self if publication column is not present in the dataset:
        if 'literature' not in self.df.columns:
            return self

        #  Extract publication IDs/evidence ID:
        evidence_with_pub_ids = (
            self.df.select(f.col('id'), f.explode(f.col('literature')).alias('publicationId'))
            .withColumn('publicationId', f.upper(f.trim(f.col('publicationId'))))
            .distinct()
        )

        # Join evidence with publication mapping:
        dated_evidence = (
            evidence_with_pub_ids.join(publication_date_lut, on='publicationId', how='inner')
            # For each evidence identifier find the earliest publication date:
            .withColumn('rank', f.row_number().over(Window.partitionBy('id').orderBy(f.col('publicationDate').asc())))
            .filter(f.col('rank') == 1)
            .select('id', 'publicationDate')
            .distinct()
        )

        # Broadcast for efficiency and join back to main evidence
        dated_evidence_lut = f.broadcast(dated_evidence.orderBy(f.col('id').asc()))

        return Evidence(self.df.join(dated_evidence_lut, on='id', how='left_outer'))

    def resolve_evidence_date(self: Evidence) -> Evidence:
        """Generate evidenceDate for each evidence based on a number of columns.

        Evidence date column has to be presented even if none of the eligible columns are present in the dataset.

        Returns:
            Evidence: with new column `evidenceDate`.
        """
        # Get which columns do we have:
        dating_columns = [f.col(col) for col in self.EVIDENCE_DATE_COLUMNS if col in self.df.columns]

        # If no columns are there, we just return:
        if len(dating_columns) == 0:
            dating_columns = [f.lit(None).cast(t.StringType())]

        # Assign date:
        return Evidence(self.df.withColumn('evidenceDate', f.array_min(f.array(dating_columns))))

    def resolve_direction_of_effect(self: Evidence, mechanism_of_action: DataFrame) -> Evidence:
        raise NotImplementedError

    @staticmethod
    def _hash_long_variant_ids(variant_id: Column, threshold: int = 300) -> Column:
        """Generate hash for long variant identifiers.

        Args:
            variant_id (Column): variant identifier column.
            threshold (int): lenght of the variant identifier above which we hash.

        Returns:
            Column: variant id column where long variants are hashed.
        """
        # Extract chromosome and position from variantId
        chr_col = f.regexp_extract(variant_id, r'^([0-9XYMT]{1,2})_([0-9]+)_([ACGTN]+)_([ACGTN]+)$', 1)
        pos_col = f.regexp_extract(variant_id, r'^([0-9XYMT]{1,2})_([0-9]+)_([ACGTN]+)_([ACGTN]+)$', 2)

        # Apply transformation logic
        return (
            f.when(chr_col.isNull() | pos_col.isNull(), f.concat(f.lit('OTVAR_'), f.md5(variant_id).cast('string')))
            .when(
                f.length(variant_id) > threshold,
                f.concat_ws('_', f.lit('OTVAR'), chr_col, pos_col, f.md5(variant_id).cast('string')),
            )
            .otherwise(variant_id)
        )

    @property
    def df(self: Evidence) -> DataFrame:
        return self._df

    @df.setter
    def df(self: Evidence, new_df: DataFrame) -> Evidence:
        return Evidence(new_df)

    def get_invalid_evidence(self: Evidence) -> DataFrame:
        """Return invalid evidence.

        Returns:
            DataFrame: evidence with non empty QC column.
        """
        return self.df.filter(f.size(self.QC_COLUMN) != 0)

    def get_valid_evidence(self: Evidence) -> DataFrame:
        """Return valid evidence.

        Returns:
            DataFrame: evidence with empty QC column.
        """
        return self.df.filter(f.size(self.QC_COLUMN) == 0)

    def assign_direction_on_trait(self: Evidence, direction_expression: str | None) -> Evidence:
        """Assigning direction on trait.

        Based on a datasource specific expression, the effect on the disease is assessed.

        Args:
            direction_expression (str | None): spark sql expression on how a given source is processed.

        Returns:
            Evidence: 'directionOnTrait' column if directions are provided.
        """
        # If no expresssion is provided, evidence returned unchanged:
        if direction_expression is None:
            return self

        return Evidence(self.df.withColumn('directionOnTrait', f.expr(direction_expression)))
