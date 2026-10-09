"""Mapped Match dataset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import pyspark.sql.functions as f
from loguru import logger
from pyspark.storagelevel import StorageLevel

from pts.pyspark.common.utils import parse_spark_schema
from pts.pyspark.literature_utils.dataset.cooccurrence import Cooccurrence
from pts.pyspark.literature_utils.dataset.dataset import Dataset

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame
    from pyspark.sql.types import StructType


class IdValidReason(Enum):
    """Reasons for an entityId to be considered valid.

    Attributes:
        ONLY_ID (str): Only entityId for the entityLabel
        ID_FROM_TRUSTED_SOURCE (str): entityId is from a trusted source
        DISAMBIGUATED (str): entityId is successfully disambiguated
    """

    ONLY_ID = 'Only entityId for the entityLabel'
    ID_FROM_TRUSTED_SOURCE = 'entityId is from a trusted source'
    DISAMBIGUATED = 'entityId is successfully disambiguated'


@dataclass
class MatchMapped(Dataset):
    """Mapped Match dataset.

    This dataset describes mapped matches.
    """

    SECTION_TO_SCORE_CONFIG: ClassVar[list[dict[str, Any]]] = [
        {'score': 10, 'section': ['title']},
        {'score': 3,  'section': ['abstract']},
        {'score': 5,  'section': ['results', 'result', 'figure', 'fig', 'table']},
        {'score': 2,  'section': ['discussion', 'discuss', 'conclusion', 'concl']},
        {'score': 1,  'section': ['introduction', 'intro', 'case study', 'case', 'appendix', 'methods', 'other']},
    ]

    # Salt buckets for shuffles keyed by (pmid, mappedId) in the disambig
    # pipeline. The key distribution is Zipfian on full-EPMC mention data:
    # a handful of (pmid, mappedId) pairs account for a disproportionate
    # share of rows. Without salt the hot pairs funnel into a single
    # shuffle partition and create extreme task-duration skew. The same
    # constant is used by `_subset_valid_ids` (two-stage distinct) and
    # `_resolve_ambiguous_mappings` (salt-and-replicate left join); both
    # shuffles share the same skew profile. Sized for the observed skew
    # at full-EPMC scale; shrink if shuffle inflation becomes the new
    # bottleneck.
    DISAMBIG_SALT_BUCKETS = 32

    @classmethod
    def get_schema(cls: type[MatchMapped]) -> StructType:
        """Provides the schema for the MatchMapped dataset.

        Returns:
            StructType: Schema for the MatchMapped dataset.
        """
        return parse_spark_schema('literature_match_mapped.json')

    @staticmethod
    def _update_flag(
        flag_column: Column, flag_condition: Column, flag_text: Enum
    ) -> Column:
        """Update the provided flag column with a new flag if condition is met.

        Args:
            flag_column (Column): Array column with the current list of flags.
            flag_condition (Column): Boolean column indicating which rows should be flagged.
            flag_text (Enum): Text for the new flag.

        Returns:
            Column: Array column with the updated list of flags.
        """
        flag_column = f.when(flag_column.isNull(), f.array()).otherwise(flag_column)
        return f.when(
            flag_condition,
            f.array_union(flag_column, f.array(f.lit(flag_text.value))),
        ).otherwise(flag_column)

    @staticmethod
    def _identify_valid_ids(df: DataFrame, trusted_sources: list[str]) -> DataFrame:
        """Mark ids as valid based on certain criteria.

        Args:
            df (DataFrame): DataFrame containing id and source information.
            trusted_sources (list[str]): List of trusted sources.

        Returns:
            DataFrame: DataFrame where valid ids and their reasons are indicated.
        """
        return (
            df
            .withColumn('validReasons', f.lit(None))
            # mappedId is valid if the id is the only id mapped to the label
            .withColumn(
                'validReasons',
                MatchMapped._update_flag(
                    f.col('validReasons'),
                    f.size(f.array_distinct('entityIds.entityId')) == 1,
                    IdValidReason.ONLY_ID
                )
            )
            # mappedId is valid if the mapping is from a trusted source
            .withColumn(
                'validReasons',
                MatchMapped._update_flag(
                    f.col('validReasons'),
                    f.size(
                        f.array_except(
                            f.col('entityIds.entitySource'),
                            f.array(*[f.lit(x) for x in trusted_sources])
                        )
                    ) == 0,
                    IdValidReason.ID_FROM_TRUSTED_SOURCE
                )
            )
            .withColumn(
                'isValid',
                f.when(f.size('validReasons') > 0, True).otherwise(False)
            )
        )

    @staticmethod
    def _subset_valid_ids(
        df: DataFrame,
        salt_buckets: int = DISAMBIG_SALT_BUCKETS,
    ) -> DataFrame:
        """Subset for entries with valid ids.

        The distinct() over (pmid, mappedId) is Zipfian-skewed in the same way
        as the disambig left join: hot pairs funnel into a single shuffle
        partition and dominate wall-clock. At full-EPMC scale (run-014 stage
        216) the single-task wait on this distinct was ~20 min while p50 was
        ~4 min. The fix is a two-stage distinct: salt the rows, distinct on
        (pmid, mappedId, salt) so hot keys spread across `salt_buckets`
        partitions, drop the salt, then a small final distinct to collapse the
        salt-induced duplicates. The first (salted) distinct carries the heavy,
        balanced shuffle; the second only re-concentrates each key over the
        <= num_distinct * salt_buckets rows that survive, so it is skew-free.

        The persist between the two distincts is load-bearing, not just a
        cache. Without a materialisation barrier Catalyst merges the two
        adjacent distinct aggregates into a single distinct on (pmid, mappedId)
        and prunes the salt column entirely (verified via the optimized plan),
        so the salted shuffle never runs and the skew is unchanged. Persisting
        forces the salted distinct to execute as its own balanced shuffle on
        (pmid, mappedId, salt). DISK_ONLY keeps the lineage recomputable on
        executor loss; a reliable checkpoint would also work but needs a
        configured checkpoint dir, which the session does not set.

        Args:
            df (DataFrame): DataFrame with identified valid ids.
            salt_buckets (int): Number of salt buckets used to spread hot
                (pmid, mappedId) keys across shuffle partitions.

        Returns:
            DataFrame: DataFrame containing only entries with valid ids.
        """
        salted_distinct = (
            df
            .filter(f.col('isValid') == True)  # noqa: E712 (spark Column comparison)
            .select('pmid', 'mappedId')
            .withColumn('_salt', (f.rand(seed=42) * salt_buckets).cast('int'))
            .distinct()
            # barrier: prevents Catalyst from merging the two distincts (which
            # would prune the salt and collapse this back to one skewed shuffle)
            .persist(StorageLevel.DISK_ONLY)
        )
        return (
            salted_distinct
            .drop('_salt')
            .distinct()
            .withColumn('isDisambiguous', f.lit(True))
        )

    @staticmethod
    def _resolve_ambiguous_mappings(
        df: DataFrame,
        valid_id_df: DataFrame,
        salt_buckets: int = DISAMBIG_SALT_BUCKETS,
    ) -> MatchMapped:
        """Resolve ambiguous mappings by using a dataframe of valid ids.

        The left join key (pmid, mappedId) is Zipfian on full-EPMC mention data:
        a handful of (pmid, mappedId) pairs account for a disproportionate share
        of rows, which funnels each hot key onto a single shuffle partition and
        creates extreme task-duration skew. To mitigate this we salt the left
        side with a uniform random bucket, replicate the right side across all
        salt buckets, and join on the salted key. Cardinality and semantics are
        preserved because valid_id_df is already distinct on (pmid, mappedId).

        Args:
            df (DataFrame): DataFrame containing ambiguous mappings.
            valid_id_df (DataFrame): DataFrame containing only valid ids.
            salt_buckets (int): Number of salt buckets used to spread hot
                (pmid, mappedId) keys across shuffle partitions.

        Returns:
            MatchMapped: Dataset with resolved mappings.
        """
        salted_df = df.withColumn(
            '_salt', (f.rand(seed=42) * salt_buckets).cast('int')
        )
        salted_valid_id_df = valid_id_df.withColumn(
            '_salt', f.explode(f.array(*[f.lit(i) for i in range(salt_buckets)]))
        )
        return MatchMapped(
            _df=(
                salted_df
                .join(salted_valid_id_df, on=['pmid', 'mappedId', '_salt'], how='left')
                .drop('_salt')
                .withColumn('isDisambiguous', f.coalesce(f.col('isDisambiguous'), f.lit(False)))
                .withColumn(
                    'validReasons',
                    MatchMapped._update_flag(
                        f.col('validReasons'),
                        (f.col('isDisambiguous') == True) & (f.col('isValid') == False),  # noqa: E712 (spark Column comparison)
                        IdValidReason.DISAMBIGUATED
                    )
                )
                .drop('isDisambiguous')
                .withColumn(
                    'isValid',
                    f.when(f.size('validReasons') > 0, True).otherwise(False)
                )
            ),
            _schema=MatchMapped.get_schema()
        )

    def disambiguate(self: MatchMapped, trusted_sources: list[str]) -> MatchMapped:
        """Identify and annotate mappings that are valid and disambiguous.

        Args:
            trusted_sources (list[str]): List of trusted sources.

        Returns:
            MatchMapped: Dataset with disambiguated mappings.
        """
        # only process successfully mapped matches
        mapped_subset = self.df.filter(f.col('isMapped') == True)  # noqa: E712 (spark Column comparison)

        logger.info('identify valid ids')
        annotated_df = self._identify_valid_ids(mapped_subset, trusted_sources)

        logger.info('subset valid ids')
        valid_id_df = self._subset_valid_ids(annotated_df)

        logger.info('resolve ambiguous mappings')
        return self._resolve_ambiguous_mappings(annotated_df, valid_id_df)

    @staticmethod
    def _section_to_score(section: Column, default_score: float = 1) -> Column:
        """Determine score based on section as specified in the config.

        Args:
            section (Column): Column containing section.
            default_score (float): Score to assign if section is not found in the config.

        Returns:
            Column: Column containing score.
        """
        score = f.lit(default_score)

        # go through config from lowest to highest score
        # if section matches pattern, assign score
        for row in reversed(MatchMapped.SECTION_TO_SCORE_CONFIG):
            pattern = r'\b(' + '|'.join(row['section']) + r')\b'
            score = f.when(section.rlike(pattern), float(row['score'])).otherwise(score)

        return score

    def _generate_pairwise_cooccurrences(self: MatchMapped, type1: str, type2: str) -> Cooccurrence:
        """Generate pairwise cooccurrences given their types.

        Args:
            type1 (str): Type of the left dataset.
            type2 (str): Type of the right dataset.

        Returns:
            Cooccurrence: Cooccurrence dataset.
        """
        logger.info(f'generate cooccurrences between {type1} and {type2}')
        return Cooccurrence(
            _df=(
                self.df
                .filter(f.col('type') == type1)
                .alias('left')
                .join(
                    (
                        self.df
                        .filter(f.col('type') == type2)
                        .select(
                            'pmid', 'text',
                            'label', 'type',
                            'startInSentence', 'endInSentence',
                            'entityLabelNormalised', 'mappedId'
                        )
                        .alias('right')
                    ),
                    on=[
                        (f.col('left.pmid') == f.col('right.pmid')) &
                        (f.col('left.text') == f.col('right.text'))
                    ],
                    how='inner'
                )
                .select(
                    # fields shared between left and right datasets
                    f.col('left.pmid').alias('pmid'),
                    f.col('left.pmcid').alias('pmcid'),
                    f.col('left.pubDate').alias('pubDate'),
                    f.col('left.date').alias('date'),
                    f.col('left.year').alias('year'),
                    f.col('left.month').alias('month'),
                    f.col('left.day').alias('day'),
                    f.col('left.organisms').alias('organisms'),
                    f.col('left.section').alias('section'),
                    f.col('left.text').alias('text'),
                    f.col('left.traceSource').alias('traceSource'),
                    # fields from left dataset
                    f.col('left.label').alias('label1'),
                    f.col('left.type').alias('type1'),
                    f.col('left.startInSentence').alias('start1'),
                    f.col('left.endInSentence').alias('end1'),
                    f.col('left.entityLabelNormalised').alias('entityLabelNormalised1'),
                    f.col('left.mappedId').alias('mappedId1'),
                    # fields from right dataset
                    f.col('right.label').alias('label2'),
                    f.col('right.type').alias('type2'),
                    f.col('right.startInSentence').alias('start2'),
                    f.col('right.endInSentence').alias('end2'),
                    f.col('right.entityLabelNormalised').alias('entityLabelNormalised2'),
                    f.col('right.mappedId').alias('mappedId2'),
                    # fields dependent on left and right values
                    f.concat_ws('-', f.col('left.type'), f.col('right.type')).alias('type'),
                    (f.col('mappedId1').isNotNull() & f.col('mappedId2').isNotNull()).alias('isMapped')
                )
                # assign evidenceScore, given section
                .withColumn('evidenceScore', self._section_to_score(f.col('section')))
            ),
            _schema=Cooccurrence.get_schema()
        )

    def generate_target_disease_cooccurrences(self: MatchMapped) -> Cooccurrence:
        """Generate target disease cooccurrences.

        Returns:
            Cooccurrence: Cooccurrence dataset containing target-disease pairs.
        """
        return self._generate_pairwise_cooccurrences('GP', 'DS')
