"""Ingests Database of Immune Cell Expression (DICE) data and generates the unaggregated baseline expression data.

Requires Apache Spark (pyspark) for distributed processing.

Expected input:
- A directory of CSVs, one per cell type, named like "<celltype>_TPM.csv".
  Columns: Feature_name, Transcript_Length(bp), Additional_annotations, 0,1,2,3,...
- A tab-delimited ontology mapping with columns: celltype, full_name, ontology_id
"""

import os

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import array, col, explode, lit, regexp_replace, split, struct, trim
from pyspark.sql.functions import sum as spark_sum


class DiceBaselineExpression:
    """Steps to generate unaggregated baseline expression from DICE."""

    def __init__(
        self,
        spark: SparkSession,
        dice_directory: str,
        mapping_path: str,
        output_directory_path: str,
        json: bool = False,
        local: bool = True,
    ):
        self.spark = spark
        self.dice_directory = dice_directory
        self.mapping_path = mapping_path
        self.output_directory_path = output_directory_path
        self.json = json
        self.local = local
        self.df: DataFrame | None = None

    def _read_and_prepare_single(self, path: str) -> DataFrame:
        """Load one DICE CSV and prepare it.

        Clean columns, prepend celltype to numeric sample headers,
        and collapse duplicated Feature_name rows after stripping version suffixes.
        """
        file = os.path.basename(path)
        logger.info(f'Reading {file}...')
        df = self.spark.read.csv(path, header=True, inferSchema=True)
        logger.info('• file loaded')

        # Drop unused columns if present
        drop_cols = [c for c in ['Transcript_Length(bp)', 'Additional_annotations'] if c in df.columns]
        if drop_cols:
            df = df.drop(*drop_cols)

        # Normalize join key
        if 'Feature_name' not in df.columns:
            raise ValueError(f"{file} missing required column 'Feature_name'")
        df = (
            df.withColumn('Feature_name', trim(col('Feature_name')))
            # strip numeric version suffix only (e.g., ENSG... .1 -> ENSG...)
            .withColumn('Feature_name', regexp_replace(col('Feature_name'), r'\.\d+$', ''))
        )

        # Prepend celltype to purely numeric sample column names (replicates)
        cell_type = file.replace('_TPM.csv', '').replace('.csv', '')
        new_cols = []
        for c in df.columns:
            if c == 'Feature_name':
                new_cols.append(c)
            else:
                new_cols.append(f'{cell_type}-{c}' if c.isdigit() else c)
        df = df.toDF(*new_cols)

        # Sum duplicate genes within this file so Feature_name is unique
        value_cols = [c for c in df.columns if c != 'Feature_name']
        agg_exprs = [spark_sum(col(c).cast('double')).alias(c) for c in value_cols]
        return df.groupBy('Feature_name').agg(*agg_exprs)

    def _read_and_bind_dice_csvs(self, directory: str) -> DataFrame:
        """Outer-join all files by Feature_name (column-bind)."""
        files = os.listdir(directory)
        combined_df: DataFrame | None = None
        for i, file in enumerate(files, 1):
            if file.endswith('TPM.csv'):
                path = os.path.join(directory, file)
                df = self._read_and_prepare_single(path)
                logger.info('• binding into combined df...')
                combined_df = df if combined_df is None else combined_df.join(df, on='Feature_name', how='outer')
                if i % 5 == 0 or i == len(files):
                    logger.info(f'  {i}/{len(files)} files bound')

        if combined_df is None:
            raise RuntimeError('Failed to create combined dataframe.')
        logger.info(f'Combined dataframe has {combined_df.count():,} rows and {len(combined_df.columns)} columns.')
        return combined_df

    def _wide_to_long(self, wide_df: DataFrame) -> DataFrame:
        """Explode sample columns into rows with donorId + expression."""
        data_cols = [c for c in wide_df.columns if c != 'Feature_name']
        if not data_cols:
            raise ValueError('No sample columns found after binding.')
        samp_structs = array(*[
            struct(lit(c).alias('donorId'), col(c).cast('double').alias('expression')) for c in data_cols
        ]).alias('samp_tpm')

        return wide_df.select('Feature_name', explode(samp_structs).alias('x')).select(
            col('Feature_name'), col('x.donorId').alias('donorId'), col('x.expression').alias('expression')
        )

    def _join_celltype_mapping(self, df_long: DataFrame, mapping_path: str) -> DataFrame:
        """Add celltype ontology info based on the donorId prefix before the first '-'."""
        # Extract celltype from donorId (prefix before first '-')
        df_long = df_long.withColumn('celltype', split(col('donorId'), '-').getItem(0))

        # Read mapping
        mapping = self.spark.read.option('header', True).option('sep', '\t').csv(mapping_path)
        required = {'celltype', 'full_name', 'ontology_id'}
        missing = required - set(mapping.columns)
        if missing:
            raise ValueError(f'Mapping file must contain columns {sorted(required)}. Missing: {sorted(missing)}')

        # Join & rename to target schema keys
        return (
            df_long
            .join(mapping, on='celltype', how='left')
            .withColumnRenamed('full_name', 'celltypeBiosampleFromSource')
            .withColumnRenamed('ontology_id', 'celltypeBiosampleId')
        )

    def _write_output(self, df: DataFrame, local: bool, as_json: bool):
        base = f'file://{self.output_directory_path}' if local else self.output_directory_path
        if as_json:
            out_path = f'{base}/json/'
            df.write.mode('overwrite').json(out_path)
            logger.info(f'Data written to {out_path} in JSON format')
        else:
            out_path = f'{base}/parquet/'
            df.write.mode('overwrite').parquet(out_path)
            logger.info(f'Data written to {out_path}')

        # ---------- Public API ----------

    def run(self):
        logger.info('Reading and unifying DICE matrices...')
        wide_df = self._read_and_bind_dice_csvs(self.dice_directory)

        logger.info('Converting wide → long...')
        long_df = self._wide_to_long(wide_df)

        logger.info('Joining ontology mapping...')
        dice_df = self._join_celltype_mapping(long_df, self.mapping_path)

        logger.info('Finalizing schema...')
        self.df = (
            dice_df
            .withColumnRenamed('Feature_name', 'targetId')
            .withColumn('unit', lit('TPM'))
            .withColumn('datasourceId', lit('DICE'))
            .withColumn('datatypeId', lit('bulk rna-seq'))
            .drop('celltype')  # already mapped to *_Biosample* columns
        )

        logger.info('Packing data for output...')
        self._write_output(self.df, local=self.local, as_json=self.json)

        # Optional: basic stats
        logger.info(f'Rows: {self.df.count():,} | Columns: {len(self.df.columns)}')
