"""Collection of functions to generate look up tables for validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session


@dataclass
class LookUpTables:
    session: Session
    settings: dict[str, Any]
    disease_lut: DataFrame | None = None
    target_lut: DataFrame | None = None
    publication_lut: DataFrame | None = None

    def __post_init__(self: LookUpTables) -> None:
        # Process disease:
        if disease_lut_path := self.settings.get('disease_path'):
            self.disease_lut = self._prepare_disease_lut(self.session.load_data(disease_lut_path, 'parquet'))

        # Process target:
        if target_lut_path := self.settings.get('target_path'):
            self.target_lut = self._prepare_target_lut(self.session.load_data(target_lut_path, 'parquet'))

        # Process publication:
        if publication_lut_path := self.settings.get('publication_date_lut'):
            self.publication_lut = self._prepare_publication_lut(self.session.load_data(publication_lut_path, 'json'))

    @staticmethod
    def _prepare_disease_lut(df: DataFrame) -> DataFrame:
        """Prepare disease index for disease validation.

        Args:
            df (DataFrame):raw, unprocessed disease index.

        Returns:
            DataFrame: dataframe with column `diseaseId` and `diseaseFromSourceMappedId`.
        """
        return (
            df
            .select(
                f.col('id').alias('diseaseId'),
                f.explode(
                    f.concat(
                        f.array(f.col('id')),
                        f.coalesce(
                            f.col('obsoleteTerms'),
                            f.lit([]).cast(t.ArrayType(t.StringType())),
                        ),
                    )
                ).alias('diseaseFromSourceMappedId'),
            )
            .orderBy(f.col('diseaseFromSourceMappedId').asc())  # ty:ignore[missing-argument]
            .repartition(f.col('diseaseFromSourceMappedId'))
        )

    @staticmethod
    def _prepare_target_lut(df: DataFrame) -> DataFrame:
        """Prepare target index for target validation.

        Args:
            df (DataFrame): raw, unprocessed target index.

        Returns:
            DataFrame: dataframe with column `targetId` and `targetFromSourceId`.
        """
        return (
            df
            .select(
                f.col('id').alias('targetId'),
                'biotype',
                f.array_distinct(
                    f.flatten(
                        f.array(
                            f.array(f.col('id')),
                            f.coalesce(
                                f.transform(f.col('proteinIds'), lambda protein: protein.id),
                                f.array().cast(t.ArrayType(t.StringType())),
                            ),
                            f.array(f.col('approvedSymbol')),
                        )
                    )
                ).alias('targetFromSourceIds'),
                LookUpTables._get_cancer_gene_assessment(f.col('hallmarks.attributes')).alias('TSorOncogene'),
            )
            .select(
                'targetId',
                'biotype',
                'TSorOncogene',
                f.explode('targetFromSourceIds').alias('targetFromSourceId'),
            )
            .distinct()
            .orderBy(f.col('targetFromSourceId').asc())  # ty:ignore[missing-argument]
            .repartition(f.col('targetFromSourceId'))
        )

    @staticmethod
    def _prepare_publication_lut(df: DataFrame) -> DataFrame:
        """Perform modification on the raw publication lookup table.

        Args:
            df (DataFrame): raw literature dataset

        Returns:
            DataFrame: processed lookup table.
        """
        return (
            # Filtering for selected literature sources:
            df.filter(f.col('source').isin('MED', 'PPR', 'AGR'))
            # Extracing relevant coclumns:
            .select(
                f.col('firstPublicationDate').alias('publicationDate'),
                f.explode(f.expr('filter(array(pmid, id, pmcid), x -> x is not null)')).alias('publicationId'),
            )
        )

    @staticmethod
    def _get_cancer_gene_assessment(hallmark_attributes: Column) -> Column:
        """Extract gene role in cancerous development based on cancer hallmark annotation.

        Cancer hallmark annotation provides a list of assessments of the gene. If they suggest
        oncogenic role we flag as `oncogene`, if they suggest tumor supressor function it is flagged as
        `tsg`. In case a gene has both `oncogene` and `tsg` annotation, it is flagged as `bivalent`.

        Args:
            hallmark_attributes (Column): hallmark column from the target index

        Returns:
            Column: with oncogenic/tumor supressor asessment.
        """
        # Get a cleaned version of the hallmark attribute description:
        hallmark_descriptions = f.transform(hallmark_attributes, lambda attribute: f.lower(attribute.description))

        # Flagging 'oncogene' and 'tsg' (tumor suppressor) annotations:
        flagged = f.array_distinct(
            f.transform(
                hallmark_descriptions,
                lambda desc: (
                    f
                    .when(desc.contains('oncogene') & desc.contains('tsg'), f.lit('bivalent'))
                    .when(desc.contains('oncogene'), f.lit('oncogene'))
                    .when(desc.contains('tsg'), f.lit('tsg'))
                ),
            )
        )

        # Resolving direction:
        return (
            f
            .when(f.array_contains(flagged, 'bivalent'), f.lit('bivalent'))
            .when(
                f.array_contains(flagged, 'oncogene') & f.array_contains(flagged, 'tsg'),
                f.lit('bivalent'),
            )
            .when(f.array_contains(flagged, 'oncogene'), f.lit('oncogene'))
            .when(f.array_contains(flagged, 'tsg'), f.lit('tsg'))
        )
