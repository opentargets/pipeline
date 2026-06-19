from __future__ import annotations

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session


class AggregateExpression:
    """This class is used to take the average of the pseudobulked expression data.

    The output of this class is a dataframe with the average expression of each
     gene (rows) per annotation (columns).
    """

    def __init__(
        self,
        spark: SparkSession,
    ) -> None:
        self.spark = spark
        self.df: DataFrame

    def load_data(self, directory: str, local: bool = False):
        """This function loads the data from the directory and returns a list of DataFrames."""
        # Use spark to read the parquet files in the directory
        if local and not directory.startswith('file://') and not directory.startswith('gs://'):
            directory = f'file://{directory}'

        logger.info(f'Loading data from {directory}')
        df = self.spark.read.parquet(directory)

        self.df = df

    def drop_null_biosample_ids(self):
        """Drop rows where any of the specified celltypeBiosample AND tissueBiosample are null."""
        # First check if the columns exist and add them if they don't, populating with nulls
        if 'tissueBiosampleId' not in self.df.columns:
            self.df = self.df.withColumn('tissueBiosampleId', f.lit(None).cast('string'))
            self.df = self.df.withColumn('tissueBiosampleFromSource', f.lit(None).cast('string'))
            self.df = self.df.withColumn('tissueBiosampleParentId', f.lit(None).cast('string'))
        if 'celltypeBiosampleId' not in self.df.columns:
            self.df = self.df.withColumn('celltypeBiosampleId', f.lit(None).cast('string'))
            self.df = self.df.withColumn('celltypeBiosampleFromSource', f.lit(None).cast('string'))
            self.df = self.df.withColumn('celltypeBiosampleParentId', f.lit(None).cast('string'))

        # Find the rows where both tissue and celltype biosample ID are null or
        # tissue and cell biosample parent ID are null
        nulls = self.df.filter(
            ((self.df['tissueBiosampleId'].isNull()) & (self.df['celltypeBiosampleId'].isNull()))
            | ((self.df['tissueBiosampleParentId'].isNull()) & (self.df['celltypeBiosampleParentId'].isNull()))
        )
        logger.info('The following biosample rows from source have null biosample IDs:')
        nulls.show(truncate=False)
        # Then drop those rows from the dataframe
        self.df = self.df.filter(~((self.df['tissueBiosampleId'].isNull()) & (self.df['celltypeBiosampleId'].isNull())))
        self.df = self.df.filter(
            ~((self.df['tissueBiosampleParentId'].isNull()) & (self.df['celltypeBiosampleParentId'].isNull()))
        )

    def calculate_quartiles(self, local=False):
        """This function calculates the expression quartiles of each gene across all donors."""
        # Define the grouping cols
        groupby_cols = ['targetId', 'datasourceId', 'datatypeId', 'unit']
        if 'targetFromSourceId' in self.df.columns:
            groupby_cols.append('targetFromSourceId')
        if 'tissueBiosampleId' in self.df.columns:
            groupby_cols.append('tissueBiosampleId')
            groupby_cols.append('tissueBiosampleFromSource')
        if 'celltypeBiosampleId' in self.df.columns:
            groupby_cols.append('celltypeBiosampleId')
            groupby_cols.append('celltypeBiosampleFromSource')

        # Partition by grouping keys
        quartile_df = self.df.repartition(*groupby_cols)

        # Define the quantile probabilities
        quartile_probs = [0, 0.25, 0.50, 0.75, 1]

        # Group and compute approximate quantiles, default params
        quartile_df = quartile_df.groupBy(*groupby_cols).agg(
            f.percentile_approx('expression', quartile_probs).alias('q_vals')
        )

        quartile_df = quartile_df.select(
            *groupby_cols,
            f.col('q_vals')[0].alias('min'),
            f.col('q_vals')[1].alias('q1'),
            f.col('q_vals')[2].alias('median'),
            f.col('q_vals')[3].alias('q3'),
            f.col('q_vals')[4].alias('max'),
        )
        self.df = quartile_df

    def calculate_expression_distribution(self, local=False, threshold: float = 0.5):
        """This function groups the dataframe by datasourceId, targetId and datatypeId.

        Then calculates the distribution of expression values for each gene.
        """
        # Group by the relevant columns and count the number of non-zero expressions
        return self.df.groupBy('targetId', 'datasourceId', 'datatypeId', 'unit').agg(
            (f.sum(f.when(f.col('median') > threshold, 1).otherwise(0)) / f.count('*')).alias('distribution_score')
        )

    def load_cellex_data(self, cellex_path, biosample_type='tissue'):
        """Load cellex biosample scores and convert from wide to long format."""
        # Read the cellex data (CSV.gz format)
        cellex_df = (
            self.spark.read.option('header', True).option('inferSchema', True).option('sep', ',').csv(cellex_path)
        )

        # Get the first column (gene IDs) and all other columns (biosample IDs)
        gene_col = cellex_df.columns[0]  # First column is gene ID
        biosample_cols = cellex_df.columns[1:]  # All other columns are biosample IDs

        # Create array of structs for each biosample
        biosample_structs = f.array(*[
            f.struct(f.lit(col_name).alias('biosampleId'), f.col(col_name).alias('specificity_score'))
            for col_name in biosample_cols
        ]).alias('biosample_scores')

        # Convert to long format, enforcing DOUBLE type for specificity_score
        # (inferSchema may infer STRING when CSV cells contain "NA" or similar)
        return (
            cellex_df
            .select(f.col(gene_col).alias('targetId'), f.explode(biosample_structs).alias('x'))
            .select(
                'targetId',
                f.col('x.biosampleId').alias('biosampleId'),
                f.col('x.specificity_score').cast('double').alias('specificity_score'),
            )
            .filter(f.col('specificity_score').isNotNull())  # Remove null scores
        )

    def add_expression_specificity(self, cellex_path, biosample_type='tissue'):
        """Add expression specificity scores to the aggregated expression data."""
        # Load cellex data
        cellex_df = self.load_cellex_data(cellex_path, biosample_type)

        # Determine which biosample column to join on
        if biosample_type == 'tissue':
            biosample_col = 'tissueBiosampleId'
        elif biosample_type == 'celltype':
            biosample_col = 'celltypeBiosampleId'
        elif biosample_type == 'both':
            biosample_col = 'celltypeBiosampleId__tissueBiosampleId'
            # Create the combined column in the main dataframe
            self.df = self.df.withColumn(
                biosample_col, f.concat(f.col('celltypeBiosampleId'), f.lit('__'), f.col('tissueBiosampleId'))
            )
        else:
            raise ValueError("biosample_type must be  'tissue', 'celltype' or 'both'")

        # rename the biosample column to the biosample column in the main dataframe
        cellex_df = cellex_df.withColumnRenamed('biosampleId', biosample_col)

        # Join with the main dataframe
        self.df = self.df.join(cellex_df, on=['targetId', biosample_col], how='left')
        # Drop the celltypeBiosampleId__tissueBiosampleId column
        if biosample_type == 'both':
            self.df = self.df.drop('celltypeBiosampleId__tissueBiosampleId')

    def write_data(self, output_directory, json=False):
        """This function writes the DataFrame to parquet format."""
        if json:
            self.df.write.mode('overwrite').json(f'{output_directory}/json/')
        else:
            # If not JSON, write as parquet
            self.df.write.mode('overwrite').parquet(f'{output_directory}/parquet/')


# Default Spark properties for the aggregate step
_AGGREGATE_DEFAULT_PROPERTIES: dict[str, str] = {
    'spark.driver.memory': '50g',
    'spark.executor.memory': '70g',
    'spark.memory.offHeap.enabled': 'true',
    'spark.memory.offHeap.size': '16g',
    'spark.driver.maxResultSize': '32g',
    'spark.driver.userClassPathFirst': 'true',
    'spark.executor.userClassPathFirst': 'true',
    'spark.sql.shuffle.partitions': '2000',
}


def baseline_expression_aggregate(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression aggregation')

    # Merge step defaults with any caller-supplied overrides
    effective_properties = {**_AGGREGATE_DEFAULT_PROPERTIES, **properties}

    session = Session(app_name='baseline_expression_aggregate', properties=effective_properties)
    spark = session.spark

    # Extract arguments
    directory = source['expression_data']
    tissue_cellex = source.get('tissue_cellex')
    celltype_cellex = source.get('celltype_cellex')
    both_cellex = source.get('tissue_celltype_cellex')

    # Settings
    json_output = settings.get('json', False)
    local = settings.get('local', False)

    eq = AggregateExpression(spark)
    eq.load_data(directory, local=local)

    logger.info('Calculating quartiles...')
    eq.calculate_quartiles(local=local)

    logger.info('Calculating expression distribution...')
    distribution_df = eq.calculate_expression_distribution()
    # Add the distribution score to the main dataframe
    eq.df = eq.df.join(distribution_df, on=['targetId', 'datasourceId', 'datatypeId', 'unit'], how='left')

    # Add expression specificity scores if cellex file is provided
    if tissue_cellex:
        logger.info('Adding tissue-specific expression specificity scores...')
        eq.add_expression_specificity(tissue_cellex, biosample_type='tissue')
    elif celltype_cellex:
        logger.info('Adding celltype-specific expression specificity scores...')
        eq.add_expression_specificity(celltype_cellex, biosample_type='celltype')
    elif both_cellex:
        logger.info('Adding combined tissue+celltype expression specificity scores...')
        eq.add_expression_specificity(both_cellex, biosample_type='both')

    logger.info('Packing data for output...')

    eq.write_data(destination, json=json_output)
    logger.info(f'Data written to {destination}')
