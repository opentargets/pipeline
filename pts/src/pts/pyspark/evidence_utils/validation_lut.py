"""Collection of functions to generate look up tables for validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session

MECHANISM_OF_ACTION_MAP = {
    # Inhibitor list:
    'RNAI INHIBITOR': 'LoF',
    'NEGATIVE MODULATOR': 'LoF',
    'NEGATIVE ALLOSTERIC MODULATOR': 'LoF',
    'ANTAGONIST': 'LoF',
    'ANTISENSE INHIBITOR': 'LoF',
    'BLOCKER': 'LoF',
    'INHIBITOR': 'LoF',
    'DEGRADER': 'LoF',
    'INVERSE AGONIST': 'LoF',
    'ALLOSTERIC ANTAGONIST': 'LoF',
    'DISRUPTING AGENT': 'LoF',
    # Activator list:
    'PARTIAL AGONIST': 'GoF',
    'ACTIVATOR': 'GoF',
    'POSITIVE ALLOSTERIC MODULATOR': 'GoF',
    'POSITIVE MODULATOR': 'GoF',
    'AGONIST': 'GoF',
    'SEQUESTERING AGENT': 'GoF',
    'STABILISER': 'GoF',
}


@dataclass
class LookUpTables:
    session: Session
    settings: dict[str, Any]
    disease_lut: DataFrame | None = None
    target_lut: DataFrame | None = None
    publication_lut: DataFrame | None = None
    mechanism_of_action: DataFrame | None = None

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

        # mechanism of action is optional. Read if available:
        if mechanism_of_action_lut_path := self.settings.get('mechanism_of_action'):
            self.mechanism_of_action_lut = self._prepare_moa_lut(
                self.session.load_data(mechanism_of_action_lut_path, 'parquet')
            )
        else:
            self.mechanism_of_action_lut = None

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

    @staticmethod
    def _prepare_moa_lut(moa_df: DataFrame) -> DataFrame:
        """Generate look up table for chembl evidence with unified MoA assessment.

        Args:
            moa_df (DataFrame): raw mechanism of action dataset.

        Returns:
            DataFrame: drug/target look up -> moa type.
        """
        # Create moa map:
        moa_mapping_expr = f.create_map([f.lit(x) for x in chain(*MECHANISM_OF_ACTION_MAP.items())])

        return (
            moa_df
            # Mapping action types:
            .withColumn(
                'moaType',
                f.coalesce(moa_mapping_expr.getItem(f.col('actionType')), f.lit('noEvaluable')),
            )
            # Exploding drugs:
            .select(
                f.explode_outer('chemblIds').alias('drugId'),
                'moaType',
                'targets',
            )
            # Exploding diseases:
            .select(
                f.explode_outer('targets').alias('targetId'),
                'drugId',
                'moaType',
            )
            # Aggregate action types by drug/target pairs:
            .groupBy('targetId', 'drugId')
            .agg(
                f.collect_set('moaType').alias('moaTypes'),
            )
            .withColumn(
                'moaType',
                f
                .when(
                    f.array_contains('moaTypes', 'GoF') & f.array_contains('moaTypes', 'LoF'),
                    f.lit(None).cast(t.StringType()),
                )
                .when(f.array_contains('moaTypes', 'GoF'), f.lit('GoF'))
                .when(f.array_contains('moaTypes', 'LoF'), f.lit('LoF')),
            )
            .select('targetId', 'drugId', 'moaType')
        )
