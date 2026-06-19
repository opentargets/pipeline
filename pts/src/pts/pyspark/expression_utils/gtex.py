"""Ingests GTEx V10 data and generates the unaggregated baseline expression data.

Requires Apache Spark (pyspark) for distributed processing.

Expected input formats:
- GTEx source data: gzipped GCT file (.gct.gz) containing sample-by-gene TPM counts.
- Sample metadata: TSV file with columns SAMPID, SMTSD, SMUBRID.
- Subject metadata: TSV file with columns SUBJID, AGE, SEX.
"""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, concat_ws, expr, lit, regexp_replace, split, when


class GtexBaselineExpression:
    """Collection of steps to generate unaggregated baseline expression data."""

    def __init__(
        self,
        spark: SparkSession,
        gtex_source_data_path: str,
        output_directory_path: str,
        sample_metadata_path: str,
        subject_metadata_path: str,
        json: bool = False,
        local: bool = True,
        matrix: bool = False,
        no_efo: bool = False,
    ):
        self.spark = spark
        self.gtex_source_data_path = gtex_source_data_path
        self.output_directory_path = output_directory_path
        self.sample_metadata_path = sample_metadata_path
        self.subject_metadata_path = subject_metadata_path
        self.json = json
        self.local = local
        self.matrix = matrix
        self.no_efo = no_efo
        self.df: DataFrame
        self.df_matrix: DataFrame
        self.sample_ids: list[Any] = []

    def _read_and_prepare_base_df(self):
        """Read the .gct.gz file and perform initial cleaning."""
        rdd = self.spark.sparkContext.textFile(self.gtex_source_data_path)

        # Skip the first 2 lines (version + dimensions) in partition 0 only.
        # This is O(1) per partition — no global shuffle like zipWithIndex.
        def _skip_first_n(index, iterator, n=2):
            if index == 0:
                for _ in range(n):
                    next(iterator, None)
            return iterator

        data_rdd = rdd.mapPartitionsWithIndex(_skip_first_n)

        # Now the first line is the real header (Name\tDescription\tGTEX-...)
        df = self.spark.read.option('sep', '\t').option('header', 'true').csv(data_rdd)

        # Clean up columns
        return (
            df
            .drop('Description')
            .filter(~col('Name').endswith('_PAR_Y'))
            .withColumn('Name', split(col('Name'), r'\.').getItem(0))
        )

    def _read_gtex_data_as_matrix(self, df):
        """Convert cleaned dataframe to wide format matrix."""
        data_cols = [c for c in df.columns if c != 'Name']
        self.df_matrix = df.withColumnRenamed('Name', 'targetId')
        self.sample_ids = data_cols  # Keep track of sample column names

    def _read_gtex_data_as_long(self, df):
        """Convert cleaned dataframe to long format for parquet/json output.

        Uses SQL ``stack()`` generator instead of ``array(struct(…)) + explode``
        to avoid building a massive Catalyst expression tree on the driver,
        which is the main bottleneck for wide DataFrames.
        """
        data_cols = [c for c in df.columns if c != 'Name']
        n = len(data_cols)

        # Build a stack() expression:  stack(N, 'col1', `col1`, 'col2', `col2`, ...)
        # This is the Spark-native unpivot and is orders of magnitude faster than
        # constructing a Python-side array of N structs.
        stack_args = ', '.join(f"'{c}', CAST(`{c}` AS DOUBLE)" for c in data_cols)
        stack_expr = f'stack({n}, {stack_args}) AS (OrigSample, TPM)'

        df_long = df.select('Name', expr(stack_expr))

        # split OrigSample into donorId + sampleId (split on second “-”)
        parts = split(col('OrigSample'), '-', 3)
        df_long = df_long.withColumn('donorId', concat_ws('-', parts.getItem(0), parts.getItem(1))).withColumn(
            'sampleId', parts.getItem(2)
        )

        # Read sample metadata and rename columns
        meta_sample = (
            self.spark.read
            .option('sep', '\t')
            .option('header', 'true')
            .csv(self.sample_metadata_path)
            .select(
                col('SAMPID').alias('OrigSample'),
                col('SMTSD').alias('Tissue'),
                col('SMUBRID').alias('TissueOntologyID'),
            )
        )
        # Read subject metadata and rename columns, also map sex from 1/2 to M/F
        meta_subject = (
            self.spark.read
            .option('sep', '\t')
            .option('header', 'true')
            .csv(self.subject_metadata_path)
            .select(col('SUBJID').alias('donorId'), col('AGE').alias('Age'), col('SEX').alias('Sex'))
            .withColumn('Sex', when(col('Sex') == '1', lit('M')).when(col('Sex') == '2', lit('F')).otherwise(lit('U')))
        )

        # Broadcast the small metadata tables for efficient map-side joins
        from pyspark.sql.functions import broadcast

        df_long = (
            df_long
            .join(broadcast(meta_sample), on='OrigSample', how='left')
            .join(broadcast(meta_subject), on='donorId', how='left')
            .select('Name', 'donorId', 'sampleId', 'TPM', 'Tissue', 'TissueOntologyID', 'Age', 'Sex')
            # Replace : with _ in TissueOntologyID
            .withColumn(
                'TissueOntologyID',
                when(col('TissueOntologyID').isNotNull(), regexp_replace(col('TissueOntologyID'), ':', '_')).otherwise(
                    lit(None)
                ),
            )
        )

        if self.no_efo:
            df_long = df_long.filter(~col('TissueOntologyID').startswith('EFO'))

        # # Renamed most columns to match the desired output format
        self.df = (
            df_long
            .withColumnRenamed('Name', 'targetId')
            .withColumnRenamed('Tissue', 'tissueBiosampleFromSource')
            .withColumnRenamed('TissueOntologyID', 'tissueBiosampleId')
            .withColumn('unit', lit('TPM'))  # Add unit column with constant value "TPM"
            .withColumn('datasourceId', lit('gtex'))  # Add datasourceId column with constant value "gtex"
            .withColumn('datatypeId', lit('bulk rna-seq'))  # Add datatypeId column with constant value "bulk rna-seq"
            .withColumnRenamed('TPM', 'expression')
            .withColumnRenamed('Sex', 'sex')
            .withColumnRenamed('Age', 'age')
        ).drop('sampleId')

    def read_gtex_data(self):
        """Read GTEx data and prepare in requested format(s)."""
        # Always read the base data
        df = self._read_and_prepare_base_df()

        # Cache if used by both matrix and long paths to avoid re-reading
        if self.matrix:
            df = df.cache()
            self._read_gtex_data_as_matrix(df)

        # Convert to long format for parquet/json output
        self._read_gtex_data_as_long(df)

        # Unpersist if we cached
        if self.matrix:
            df.unpersist()

    def pack_data_for_output(self, local: bool = False, json: bool = False, matrix: bool = False):
        """Use spark to write the DataFrame to parquet format."""
        if local:
            output_path = f'file://{self.output_directory_path}/'
        else:
            output_path = f'{self.output_directory_path}/'

        if matrix:
            # Save in matrix form with separate metadata
            self.save_as_matrix(output_path)
        elif json:
            output_path = f'{output_path}/json'
            # If JSON output is requested, convert DataFrame to JSON format
            self.df.write.mode('overwrite').json(output_path)
            logger.info(f'Data written to {output_path} in JSON format')
        else:
            output_path = f'{output_path}/parquet'
            # Repartition so the write is parallelised across tasks
            self.df.repartition(200).write.mode('overwrite').parquet(output_path)
            logger.info(f'Data written to {output_path}')

    def save_as_matrix(self, output_path: str):
        """Save expression data in matrix form with separate metadata file."""
        # Matrix is already in wide format from read_gtex_data
        # Just need to prepare metadata from the sample IDs

        def save_single_tsv(df, output_path):
            """Writes a dataframe to a single TSV file at the specific output_path."""
            # Define a temp directory
            temp_path = output_path + '_temp_write'

            # Write data to temp directory
            (df.coalesce(1).write.mode('overwrite').option('header', 'true').option('sep', '\t').csv(temp_path))

            # Rename the file
            spark = SparkSession.getActiveSession()
            if spark is None:
                raise RuntimeError('no active Spark session found')
            sc = spark.sparkContext
            jvm = sc._jvm
            jsc = sc._jsc
            if jvm is None or jsc is None:
                raise RuntimeError('no jvm found in spark context')
            fs = jvm.org.apache.hadoop.fs.FileSystem.get(jsc.hadoopConfiguration())
            path = jvm.org.apache.hadoop.fs.Path

            # Find the part-00000 file in the temp directory
            temp_path_obj = path(temp_path)
            part_file_path = fs.globStatus(path(temp_path + '/part-*'))[0].getPath()

            # Rename the part file to the final destination
            final_path_obj = path(output_path)
            fs.rename(part_file_path, final_path_obj)

            # Delete the temp directory
            fs.delete(temp_path_obj, True)

        # Read sample metadata
        meta_sample = (
            self.spark.read
            .option('sep', '\t')
            .option('header', 'true')
            .csv(self.sample_metadata_path)
            .select(
                col('SAMPID').alias('OrigSample'),
                col('SMTSD').alias('tissueBiosampleFromSource'),
                col('SMUBRID').alias('TissueOntologyID'),
            )
        )

        if self.no_efo:
            # Identify samples to drop (those starting with EFO)
            efo_rows = meta_sample.filter(col('TissueOntologyID').startswith('EFO')).select('OrigSample').collect()
            efo_samples = [row.OrigSample for row in efo_rows]
            # Filter sample_ids
            self.sample_ids = [s for s in self.sample_ids if s not in efo_samples]
            # Filter df_matrix columns
            cols_to_keep = ['targetId', *self.sample_ids]
            self.df_matrix = self.df_matrix.select(*cols_to_keep)
            # Filter meta_sample to exclude EFOs
            meta_sample = meta_sample.filter(~col('TissueOntologyID').startswith('EFO'))

        # Read subject metadata and map sex from 1/2 to M/F
        meta_subject = (
            self.spark.read
            .option('sep', '\t')
            .option('header', 'true')
            .csv(self.subject_metadata_path)
            .select(col('SUBJID').alias('donorId'), col('AGE').alias('age'), col('SEX').alias('sex'))
            .withColumn('sex', when(col('sex') == '1', lit('M')).when(col('sex') == '2', lit('F')).otherwise(lit('U')))
        )

        # Create metadata for each sample column
        # Split OrigSample into donorId + sampleId
        sample_metadata_rows = []
        for sample_id in self.sample_ids:
            parts = sample_id.split('-', 2)
            donor_id = f'{parts[0]}-{parts[1]}'
            sample_metadata_rows.append((sample_id, donor_id))

        # Create dataframe from sample IDs
        df_samples = self.spark.createDataFrame(sample_metadata_rows, ['OrigSample', 'donorId'])

        # Join with metadata
        df_metadata = (
            df_samples
            .join(meta_sample, on='OrigSample', how='left')
            .join(meta_subject, on='donorId', how='left')
            .withColumn(
                'tissueBiosampleId',
                when(
                    col('TissueOntologyID').isNotNull(), concat_ws('_', split(col('TissueOntologyID'), ':'))
                ).otherwise(lit(None)),
            )
            .withColumn('unit', lit('TPM'))
            .withColumn('datasourceId', lit('gtex'))
            .withColumn('datatypeId', lit('bulk rna-seq'))
            .select(
                'OrigSample',
                'donorId',
                'tissueBiosampleFromSource',
                'tissueBiosampleId',
                'age',
                'sex',
                'unit',
                'datasourceId',
                'datatypeId',
            )
        )

        logger.info('Saving matrix and metadata as TSV files using Spark...')

        # Save matrix as TSV (Spark will handle the write)
        matrix_output = f'{output_path}matrix/expression.tsv'
        metadata_output = f'{output_path}matrix/metadata.tsv'

        # Clean up paths for local mode
        if output_path.startswith('file://'):
            matrix_output = matrix_output.replace('file://', '')
            metadata_output = metadata_output.replace('file://', '')

        # Sort by targetId for consistent output
        df_matrix_sorted = self.df_matrix.orderBy('targetId')

        # Sort metadata by OrigSample for consistent output
        df_metadata_sorted = df_metadata.orderBy('OrigSample')

        # Write the matrix and metadata using the helper function
        logger.info(f'Writing matrix to {matrix_output}...')
        save_single_tsv(df_matrix_sorted, matrix_output)
        logger.info(f'Matrix data written to {matrix_output}')

        logger.info(f'Writing metadata to {metadata_output}...')
        save_single_tsv(df_metadata_sorted, metadata_output)
        logger.info(f'Metadata written to {metadata_output}')

        logger.info(f'Matrix shape: {self.df_matrix.count()} genes x {len(self.sample_ids)} samples')

    def run(self):
        logger.info('Reading GTEx data...')
        self.read_gtex_data()
        logger.info('Packing data for output...')
        output_format = 'JSON' if self.json else 'parquet'
        logger.info(f'Running in {output_format} mode - saving as {output_format}.')
        # Always save parquet/json output
        self.pack_data_for_output(local=self.local, json=self.json)
        # Also save matrix format if requested
        if self.matrix:
            logger.info('Matrix mode enabled - also saving wide format matrix and metadata.')
            self.pack_data_for_output(local=self.local, matrix=self.matrix)
