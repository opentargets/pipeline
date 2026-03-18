"""Parser to generate evidence for ot_crispr datasets.

This dataset consist of a series of OTAR projects studying various diseases using genome-wide crisp/cas9 knock-outs.
 - The results are expected to arrive in MAGeCK format.
 - The study level metadata is expected to come via filling out a Google spreadseet.
 - These spreadseet downloaded as a tsv and version-ed in the PPP-evidencie-configuration repository.
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Any

from loguru import logger
from pyspark.sql import Column, DataFrame, Row
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session


@dataclass
class DataCheck:
    """A single quality check to apply to the study table.

    Attributes:
        condition (Callable[[], Column]): A lazy factory returning the filter condition column.
        name (str): Human-readable name for the check.
        action (str): Action to take on failing rows; currently only 'drop' is supported.
        report_fn (Callable[[DataFrame], Any]): Function to extract a summary from the failing rows.
        message_fn (Callable[[Any], str]): Function to format the summary into a log message.
    """

    condition: Callable[[], Column]
    name: str
    action: str
    report_fn: Callable[[DataFrame], Any]
    message_fn: Callable[[Any], str]


# This is the list of test that are run on the study table.
STUDY_CHECKS = [
    DataCheck(
        name='StudyId assertion',
        condition=lambda: f.col('studyId').isNull(),
        action='drop',
        report_fn=lambda df: df.count(),
        message_fn=lambda result: f'Rows with missing studyId: {result}',
    ),
    DataCheck(
        name='Disease id presence assertion',
        condition=lambda: f.col('diseases').isNull(),
        action='drop',
        report_fn=lambda df: ', '.join([r['studyId'] for r in df.select('studyId').distinct().collect()]),
        message_fn=lambda result: f'Studies with missing diseaseId: {result}',
    ),
    DataCheck(
        name='DataFile presence assertion',
        condition=lambda: f.col('dataFile').isNull(),
        action='drop',
        report_fn=lambda df: ', '.join([r['studyId'] for r in df.select('studyId').distinct().collect()]),
        message_fn=lambda result: f'Studies with missing dataset: {result}',
    ),
    DataCheck(
        name='Proper project id assertion',
        condition=lambda: f.col('projectId').isNull() | (~f.col('projectId').startswith('OTAR')),
        action='drop',
        report_fn=lambda df: ', '.join([r['studyId'] for r in df.select('studyId').distinct().collect()]),
        message_fn=lambda result: f'Studies with invalid projectId: {result}',
    ),
]


class StudyParser:
    """Process raw CRISPR study table.

    - The versioned, raw study table provided as tsv.
    - The following operations are performed:
        - Read the study table.
        - Drop rows with non-OTAR projects.
        - Split columns with multiple values for disease, filterColumn, and dataFile.
        - Collect replicates for each study.
    """

    def __init__(self, study_table: DataFrame) -> None:
        """Initialise the study parser.

        Args:
            study_table (DataFrame): Table with study metadata
        """
        self.study_table = study_table

    @staticmethod
    def apply_check(study_table: DataFrame, check: DataCheck) -> DataFrame:
        """Apply a single data quality check, logging a warning and dropping failing rows if needed.

        Args:
            study_table (DataFrame): The current study table to check.
            check (DataCheck): The check to apply.

        Returns:
            DataFrame: The study table with failing rows removed if the check action is 'drop'.
        """
        condition = check.condition()
        failing_df = study_table.filter(condition)
        if result := check.report_fn(failing_df):
            message = check.message_fn(result)
            logger.warning(message)

            if check.action == 'drop':
                study_table = study_table.filter(~condition)

        return study_table

    @staticmethod
    def split_column_value(col: Column, separator: str = r'\|') -> Column:
        """Remove whitespace and split column value by the provided separator.

        Args:
            col (Column): A column to be split.
            separator (str): A separator to split the column value.

        Returns:
            Column: A column with split values.
        """
        return f.array_distinct(f.split(f.regexp_replace(col, r'^\s+', ''), separator))

    def process_studies(self) -> DataFrame:
        """Parsing the study table for ot_crispr datasets.

        Args:
            study_table (DataFrame): A DataFrame with study level metadata.

        Returns:
            DataFrame: A DataFrame with parsed study level metadata.
        """
        return (
            reduce(self.apply_check, STUDY_CHECKS, self.study_table)
            # Dropping studies with no OTAR project and the field description row:
            .filter(f.col('projectId').startswith('OTAR'))  # ty:ignore[missing-argument, invalid-argument-type]
            # Selecting relevant columns:
            .select(
                'studyId',
                'projectId',
                'projectDescription',
                'studyOverview',
                'releaseVersion',
                f.col('releaseDate').cast(t.DateType()).alias('releaseDate'),
                # Splitting diseases:
                self.split_column_value(f.col('diseases')).alias('diseaseFromSourceMappedId'),
                'isCellTypeDerived',
                'crisprScreenLibrary',
                'crisprStudyMode',
                'geneticBackground',
                'cellType',
                'cellLineBackground',
                'contrast',
                'dataFileType',
                # Splitting and exploding study if both tail of the distribution are used:
                self.split_column_value(f.col('filterColumn'), ',').alias('filterColumns'),
                # Casting threshold to float:
                f.col('threshold').cast(t.FloatType()).alias('threshold'),
                'dataFile',
                'ControlDataset',
                # Adding replicate identifier when missing:
                f
                .when(f.col('replicateNumber').isNull(), f.lit(1))  # ty:ignore[missing-argument]
                .otherwise(f.col('replicateNumber'))
                .alias('replicateId'),
            )
            # Grouping by study level:
            .groupBy(
                'studyId',
                'projectId',
                'projectDescription',
                'studyOverview',
                'releaseVersion',
                'releaseDate',
                'diseaseFromSourceMappedId',
                'isCellTypeDerived',
                'crisprScreenLibrary',
                'crisprStudyMode',
                'geneticBackground',
                'cellType',
                'cellLineBackground',
                'contrast',
                'filterColumns',
                'threshold',
            )
            # Collecting replicates for each study:
            .agg(f.collect_list(f.struct('dataFile', 'ControlDataset', 'replicateId')).alias('replicates'))
        )


class EvidenceParser:
    """Generate evidence from OTAR CRISPR datasets.

    Based on the provided study-level metadata and the provided raw data files, the evidence is generated.

    The following operations are performed:
        - Process replicates for each study.
        - Combine the results into a single DataFrame.
    """

    def __init__(self, spark: Session, study_table: DataFrame, data_path: str) -> None:
        """Initialise the evidence generator.

        Args:
            spark (Session): An active PySpark session.
            study_table (DataFrame): Parsed study-level metadata from the study table.
            data_path (str): Base path to the directory containing per-project MAGeCK data files.
        """
        self.spark = spark
        self.study_table = study_table
        self.data_path = data_path

    def _read_and_filter_mageck_file(
        self,
        mageck_file: str,
        filter_columns: list[str],
        threshold: float,
    ) -> DataFrame:
        """Read and filter MAGeCK files based on the provided threshold applied on the specified column.

        Args:
            mageck_file (str): path to a MAGeCK file to read.
            filter_columns (list[str]): A filter column name.
            threshold (float): A threshold to filter the data.

        Returns:
            DataFrame: A DataFrame with filtered data.

        Raises:
            ValueError: If the label separator is not recognized.
        """
        logger.info(f'reading MAGeCK file: {mageck_file}')
        raw_data = self.adjust_column_names(self.spark.load_data(mageck_file, format='csv', header=True, sep='\t'))

        # Converting filter columns to float:
        return (
            raw_data
            # Get values for all filter columns:
            .withColumn(
                'filter_value_map',
                f.create_map(
                    *reduce(operator.iadd, ([f.lit(col), f.col(col).cast('double')] for col in filter_columns), [])
                ),
            )
            # Get the minimal value:
            .withColumn('resourceScore', f.array_min(f.array([f.col(col).cast('double') for col in filter_columns])))
            # Dropping non-significant hits:
            .filter(f.col('resourceScore') < threshold)
            # Finish parsing:
            .withColumn(
                'sourceLabel',
                f.expr('filter(map_keys(filter_value_map), x -> filter_value_map[x] = resourceScore)[0]'),
            )
            .select(
                # extract target name:
                f.split(f.col('id'), '_')[0].alias('targetFromSourceId'),
                # Extract log2Fold change value based on where the hit is coming from:
                f
                .when(f.col('sourceLabel').contains('pos'), f.col('pos|lfc'))  # ty:ignore[missing-argument, invalid-argument-type]
                .when(f.col('sourceLabel').contains('neg'), f.col('neg|lfc'))  # ty:ignore[missing-argument, invalid-argument-type]
                .otherwise(None)
                .cast(t.FloatType())
                .alias('log2FoldChangeValue'),
                # Extract which tail of distribution the hit is coming from:
                f
                .when(f.col('sourceLabel').contains('pos'), f.lit('upper tail'))  # ty:ignore[missing-argument, invalid-argument-type]
                .when(f.col('sourceLabel').contains('neg'), f.lit('lower tail'))  # ty:ignore[missing-argument, invalid-argument-type]
                .alias('statisticalTestTail'),
                'resourceScore',
            )
        )

    @staticmethod
    def adjust_column_names(raw_data: DataFrame) -> DataFrame:
        """Adjust column names, as not all MageCK output has the same names.

        Some files have "pos|p-value" others might have "pos.p-value". This method normalises to the first.

        Args:
            raw_data (DataFrame): raw input, as read from the original files.

        Returns:
            DataFrame: where all dots from the columns are replaced with "|"
        """
        # Checking label separator in the third column, which expected to be: neg|p-value or neg.p-value:
        label_separator = '|'
        if '|' in raw_data.columns[3]:
            label_separator = '|'
        elif '.' in raw_data.columns[3]:
            label_separator = '.'
        else:
            raise ValueError(f'Unrecognized label separator in {raw_data.columns[3]}')

        # Updating column names according to the identified label separator:
        return reduce(
            # Rename all columns:
            lambda df, col: df.withColumnRenamed(col, col.replace(label_separator, '|')),
            raw_data.columns,
            raw_data,
        )

    def _process_replicate(
        self,
        data_file: str,
        control_dataset: str | None,
        filter_columns: list[str],
        threshold: float,
    ) -> DataFrame:
        """Process a single replicate: finding hits, exclude controls if provided.

        Args:
            data_file (str): A single file in mageck format.
            control_dataset (str | None): A control dataset in mageck format.
            filter_columns (list[str]): A filter column name.
            threshold (float): A threshold to filter the data.

        Returns:
            DataFrame: A DataFrame with processed data.
        """
        # Extract hits from the data files:
        hits = self._read_and_filter_mageck_file(data_file, filter_columns, threshold)
        # If control dataset is provided, filter out the hits:
        if control_dataset:
            hits = hits.join(
                (
                    self
                    ._read_and_filter_mageck_file(control_dataset, filter_columns, threshold)
                    .select('targetFromSourceId')
                    .distinct()
                ),
                how='left_anti',
                on='targetFromSourceId',
            )
        return hits

    def _process_study_table_row(self, row: Row) -> DataFrame:
        """Process a single row from the study table.

        Args:
            row (Row): A row from the study table.

        Returns:
            DataFrame: A DataFrame with processed data.
        """
        # Process all replicates and collect the results in a list of dataframes:
        replicate_data = [
            self._process_replicate(
                data_file=f'{self.data_path}/{row.projectId}/{replicate.dataFile}',
                control_dataset=f'{self.data_path}/{row.projectId}/{replicate.ControlDataset}'
                if replicate.ControlDataset
                else None,
                filter_columns=row.filterColumns,
                threshold=row.threshold,
            )
            for replicate in row.replicates
        ]

        return (
            # Combine the results into a single DataFrame:
            reduce(lambda df1, df2: df1.unionByName(df2), replicate_data)
            # Aggregating replicate level data:
            .groupBy('targetFromSourceId')
            .agg(
                f.collect_list(f.struct('log2FoldChangeValue', 'resourceScore', 'statisticalTestTail')).alias(
                    'replicates'
                )
            )
            # Add replicate count:
            .withColumn('replicateCount', f.lit(len(replicate_data)))
            # Drop genes, which were not found in all replicates:
            .filter(f.size('replicates') == f.col('replicateCount'))
            # Select the best replicate:
            .select(
                'targetFromSourceId',
                f.col('replicates')[0].log2FoldChangeValue.alias('log2FoldChangeValue'),
                f.col('replicates')[0].resourceScore.alias('resourceScore'),
                f.col('replicates')[0].statisticalTestTail.alias('statisticalTestTail'),
                f.lit(row.studyId).alias('studyId'),
            )
        )

    def parse_experiments(self) -> DataFrame:
        """Extract hits from every study in the study table and combine them into a single DataFrame.

        Returns:
            DataFrame: Combined evidence across all studies, with one row per target per study.
        """
        return reduce(
            lambda df1, df2: df1.unionByName(df2),
            [self._process_study_table_row(study) for study in self.study_table.collect()],
        )


def ot_crispr(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate OT CRISPR evidence and write it to parquet.

    Reads the study metadata table and per-study MAGeCK data files, applies quality checks,
    identifies significant hits across replicates, and joins study-level metadata to produce
    the final evidence dataset.

    Args:
        source (dict[str, str]): Paths to input data. Expected keys:
            - 'study_table': TSV file with study-level metadata.
            - 'ot_crispr_data': Base directory containing per-project MAGeCK result files.
        destination (str): Output path where the evidence parquet will be written.
        settings (dict[str, Any]): Unused; reserved for future configuration.
        properties (dict[str, str]): PySpark session properties passed to the Spark session.
    """
    spark = Session(app_name='ot_crispr', properties=properties)

    logger.info(f'loading data from: {source}')
    study_table_df = spark.load_data(source['study_table'], format='csv', header=True, inferSchema=True, sep='\t')

    logger.info('parse study study table')
    parsed_studies_df = StudyParser(study_table=study_table_df).process_studies()
    logger.info('turn crispr/cas9 knockout screens into disease/target evidence')
    hits_df = EvidenceParser(
        spark=spark, study_table=parsed_studies_df, data_path=source['ot_crispr_data']
    ).parse_experiments()
    evidence_df = (
        hits_df
        # Joining study level metadata:
        .join(parsed_studies_df, on='studyId', how='inner')
        # Selecting relevant columns:
        .select(
            # Project level metadata:
            'projectId',
            'projectDescription',
            'releaseDate',
            'releaseVersion',
            # Study level metadata:
            'studyId',
            'studyOverview',
            'contrast',
            'crisprScreenLibrary',
            'cellType',
            'cellLineBackground',
            f.explode('diseaseFromSourceMappedId').alias('diseaseFromSourceMappedId'),
            # Evidence level data:
            'targetFromSourceId',
            'log2FoldChangeValue',
            'resourceScore',
            'statisticalTestTail',
            # Static fields:
            f.lit('ot_crispr').alias('datasourceId'),
            f.lit('ot_partner').alias('datatypeId'),
        )
    )
    evidence_df.write.parquet(destination, mode='overwrite')
