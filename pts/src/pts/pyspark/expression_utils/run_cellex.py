import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql.functions import col, concat_ws


def _import_cellex():
    try:
        import cellex
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'cellex is an optional dependency for specificity workflows. '
            'Install PTS with the cellex extra to use this feature.'
        ) from error
    return cellex


class CellexAnalysis:
    def __init__(
        self,
        spark=None,
        mode='parquet',
        expression_matrix_path=None,
        metadata_path=None,
        output_path=None,
        biosample=None,
        sample_id=None,
        log_transform=True,
        do_anova=True,
        input_path=None,
    ):
        self.spark = spark
        self.mode = mode
        self.output_path = output_path
        self.biosample = biosample
        self.sample_id = sample_id
        self.log_transform = log_transform
        self.do_anova = do_anova
        self.input_path = input_path

    def run_cellex_analysis(self, expression_matrix, metadata):
        """Run CELLEX analysis on expression matrix and metadata."""
        cellex = _import_cellex()
        logger.info('Processing CELLEX analysis')

        # Create ESObject (Expression Specificity Object)
        eso = cellex.ESObject(
            data=expression_matrix, annotation=metadata, verbose=True, normalize=False, anova=self.do_anova
        )

        # Compute specificity using different metrics
        # Available metrics: 'esmu' (default), 'nsmu', 'essi', 'nssi'
        eso.compute(verbose=True)

        # Save results
        if self.output_path:
            output_path = f'{self.output_path}/cellex'
            logger.info(f'Saving results to: {output_path}')
            eso.save_as_csv(keys=['all'], verbose=True, path=output_path)
        else:
            logger.info('No output path specified, skipping save.')

    def load_from_matrices(self):
        """Load expression matrix and metadata from files."""
        logger.info(f'Loading expression matrix from: {self.input_path}/matrix/')
        expression_matrix = pd.read_csv(f'{self.input_path}/matrix/expression.tsv', sep='\t', index_col=0)

        logger.info(f'Expression matrix shape: {expression_matrix.shape}')

        # Load metadata
        logger.info(f'Loading metadata from: {self.input_path}/matrix/metadata.tsv')
        metadata = pd.read_csv(f'{self.input_path}/matrix/metadata.tsv', sep='\t')

        logger.info(f'Metadata shape before processing: {metadata.shape}')

        # Process metadata to create sampleId and appropriate biosample column
        if self.biosample in ['celltype', 'tissue']:
            biosample_col = f'{self.biosample}BiosampleId'
            # If sampleId exists and is not in metadata columns, create sampleId by
            # concatenating donorId and biosampleFromSource
            if self.sample_id not in metadata.columns:
                metadata['sampleId'] = metadata['donorId'] + '__' + metadata[biosample_col]
            else:
                metadata['sampleId'] = metadata[self.sample_id]
            # Keep only sampleId and the biosample column
            metadata = metadata[['sampleId', biosample_col]].drop_duplicates()
            metadata = metadata.set_index('sampleId')
        elif self.biosample == 'tissuecelltype':
            # If sampleId not in metadata columns, create sampleId by
            # concatenating donorId, celltypeBiosampleId and tissueBiosampleId
            if self.sample_id not in metadata.columns:
                metadata['sampleId'] = (
                    metadata['donorId'] + '__' + metadata['celltypeBiosampleId'] + '__' + metadata['tissueBiosampleId']
                )
            else:
                metadata['sampleId'] = metadata[self.sample_id]
            # Create combined biosample column
            metadata['celltypeBiosampleId__tissueBiosampleId'] = (
                metadata['celltypeBiosampleId'] + '__' + metadata['tissueBiosampleId']
            )
            # Keep only sampleId and the combined biosample column
            metadata = metadata[['sampleId', 'celltypeBiosampleId__tissueBiosampleId']].drop_duplicates()
            metadata = metadata.set_index('sampleId')
        else:
            raise ValueError(f'Invalid biosample: {self.biosample}. Must be "celltype", "tissue", or "tissuecelltype".')

        logger.info(f'Metadata shape after processing: {metadata.shape}')
        logger.info(metadata.head())
        logger.info(expression_matrix.head())

        # Apply log transformation if requested
        if self.log_transform:
            logger.info('Applying log1p transformation...')
            expression_matrix = np.log1p(expression_matrix)

        return expression_matrix, metadata

    def process_from_parquet(self):
        """Process data from parquet format (original workflow)."""
        logger.info(f'Processing {self.biosample} from parquet format')

        # Read the parquet data
        if self.input_path:
            logger.info(f'Reading parquet from provided input path: {self.input_path}/parquet')
            df = self.spark.read.parquet(f'{self.input_path}/parquet')
        else:
            raise ValueError('input_path is required.')

        # Set biosample column name dynamically
        if self.biosample not in ('both', 'tissuecelltype'):
            biosample_col = f'{self.biosample}BiosampleId'
            # Concatenate biosampleId and donorId to create a unique sample identifier
            df = df.withColumn('sampleId', concat_ws('__', col('donorId'), col(biosample_col)))
            # Create metadata in Spark
            metadata_spark = df.select('sampleId', biosample_col).distinct()
        else:
            df = df.withColumn(
                'sampleId', concat_ws('__', col('donorId'), col('celltypeBiosampleId'), col('tissueBiosampleId'))
            )
            metadata_spark = (
                df
                .withColumn(
                    'celltypeBiosampleId__tissueBiosampleId',
                    concat_ws('__', col('celltypeBiosampleId'), col('tissueBiosampleId')),
                )
                .select('sampleId', 'celltypeBiosampleId__tissueBiosampleId')
                .distinct()
            )

        # Do the pivot in Spark (much faster for large datasets)
        expression_matrix_spark = df.groupBy('targetId').pivot('sampleId').agg({'expression': 'first'})

        # Now convert to pandas (only after the heavy lifting is done in Spark)
        expression_matrix = expression_matrix_spark.toPandas().set_index('targetId')
        metadata = metadata_spark.toPandas().set_index('sampleId')

        # Log-transform the expression data (adding a small constant to avoid log(0))
        expression_matrix = np.log1p(expression_matrix)

        return expression_matrix, metadata

    def run(self):
        # Process based on mode
        if self.mode == 'matrix':
            # Load from matrix files
            expression_matrix, metadata = self.load_from_matrices()
        else:
            # Load from parquet files
            expression_matrix, metadata = self.process_from_parquet()

        # Run CELLEX analysis
        self.run_cellex_analysis(expression_matrix, metadata)
