"""Ingests PRIDE proteomic data and generates the unaggregated baseline expression data.

Requires Apache Spark (pyspark) for distributed processing.

Expected input formats:
- PRIDE source data: TSV file containing sample-by-protein PPB counts.
- Sample metadata: JSON file with experimental designs.
- Tissue to ontology mapping: TSV file with columns PROPERTY VALUE, ONTOLOGY TERM(S)
"""

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array,
    coalesce,
    col,
    concat,
    element_at,
    explode,
    lit,
    regexp_replace,
    split,
    struct,
    when,
)


class PrideBaselineExpression:
    """Collection of steps to generate unaggregated baseline expression data."""

    def __init__(
        self,
        spark: SparkSession,
        pride_source_data_dir: str,
        pride_codes: list,
        output_directory_path: str,
        tissue_ontology_mapping_path: str,
        json: bool = False,
        local: bool = True,
    ):
        self.spark = spark
        self.pride_source_data_dir = pride_source_data_dir
        self.pride_codes = pride_codes
        self.output_directory_path = output_directory_path
        self.tissue_ontology_mapping_path = tissue_ontology_mapping_path
        self.json = json
        self.local = local
        self.df = None

    def read_pride_data(self, pride_code: str):
        pride_source_data_path = f'{self.pride_source_data_dir}/{pride_code}/{pride_code}_OpenTargets_ppb.txt'
        pride_sdrf_path = f'{self.pride_source_data_dir}/{pride_code}/{pride_code}_OpenTargets_sdrf.json'
        # Read the PRIDE matrix file
        pride_matrix = self.spark.read.option('sep', '\t').option('header', 'true').csv(pride_source_data_path)

        # Extract and normalize the UniProt IDs from PRIDE source data.
        orig_cols = pride_matrix.columns

        pride_matrix = (
            pride_matrix
            .withColumn('proteinId', element_at(split(col('Protein IDs'), r'\|'), 2))
            .withColumn('proteinId', regexp_replace('proteinId', r'-\d+$', ''))  # optional, removes isoform suffix
            # if the proteinId column is NULL use the original Protein IDs
            .withColumn('proteinId', coalesce(col('proteinId'), col('Protein IDs')))
            .select('proteinId', *orig_cols)
        )

        # Wide→long: explode an array of structs (one per sample column)
        data_cols = [c for c in pride_matrix.columns if c not in ['proteinId', 'Gene Symbol', 'Protein IDs']]

        # build an array<struct<Sample:string,PPB:double>>
        samp_structs = array(*[
            struct(lit(c).alias('Sample'), col(c).cast('double').alias('PPB')) for c in data_cols
        ]).alias('samp_ppb')

        pride_long = pride_matrix.select(
            'proteinId', 'Gene Symbol', 'Protein IDs', explode(samp_structs).alias('x')
        ).select(
            col('proteinId'), col('Gene Symbol'), col('Protein IDs'), col('x.Sample').alias('assayId'), col('x.PPB')
        )

        # Read the PRIDE SDRF file
        pride_sdrf = (
            self.spark.read
            .option('multiline', True)  # important for JSON
            .json(pride_sdrf_path)
            .withColumn('ex', explode('experimentalDesigns'))
            .select('experimentId', col('ex.*'))
            # In the sex column, replace not available' with NULL
            .withColumn('sex', when(col('sex') == 'not available', lit(None)).otherwise(col('sex')))
            .withColumn('age', when(col('age') == 'not available', lit(None)).otherwise(col('age')))
            # Deduplicate technical replicates: the expression matrix already has
            # them collapsed, so keep only one SDRF row per assayId.
            .dropDuplicates(['assayId'])
        )

        # Read the PRIDE ontology mapping
        pride_ontology_mapping = (
            self.spark.read.option('header', 'true').option('sep', '\t').csv(self.tissue_ontology_mapping_path)
        )

        # Drop the Count column
        pride_ontology_mapping = pride_ontology_mapping.drop('Count')

        # Get all columns except BiosampleId to unpivot
        tissue_columns = [c for c in pride_ontology_mapping.columns if c != 'BiosampleId']

        # Create an array of structs for unpivoting (wide to long transformation)
        tissue_structs = array(*[struct(lit(c).alias('source'), col(c).alias('tissue')) for c in tissue_columns])

        # Unpivot: explode the array and filter out null tissue values
        pride_ontology_mapping = (
            pride_ontology_mapping
            .select('BiosampleId', explode(tissue_structs).alias('tissue_struct'))
            .select(col('BiosampleId'), col('tissue_struct.tissue').alias('tissue'))
            .filter(col('tissue').isNotNull())
            .filter(col('tissue') != lit(''))
            .distinct()  # Remove any duplicates that may arise
        )

        # Join the long DataFrame with the SDRF metadata and ontology mapping
        pride_long = (
            pride_long
            .join(pride_sdrf, on='assayId', how='left')
            .withColumn('donorId', concat(col('experimentId'), lit('-'), col('individual')))  # Add a column for donor
            .join(pride_ontology_mapping, on='tissue', how='left')
            .select('proteinId', 'donorId', 'PPB', 'BiosampleId', 'age', 'sex', 'tissue')
        )

        # Drop rows where tissueOntologyTerm is null
        pride_long = pride_long.filter(col('BiosampleId').isNotNull())

        pride_long = (
            pride_long
            # Keep targetId populated for aggregation; resolve UniProt->Ensembl at merge step.
            .withColumn('targetId', col('proteinId'))
            .withColumnRenamed('PPB', 'expression')
            .withColumnRenamed('tissue', 'tissueBiosampleFromSource')
            .withColumnRenamed('BiosampleId', 'tissueBiosampleId')
            .withColumnRenamed('proteinId', 'targetFromSourceId')
            .withColumn('unit', lit('PPB (iBAQ)'))  # Add unit column with constant value "PPB (iBAQ)"
            # Add datatypeId column with constant value "mass-spectrometry proteomics"
            .withColumn('datatypeId', lit('mass-spectrometry proteomics'))
            .withColumn('datasourceId', lit('PRIDE'))  # Add datasourceId column with constant value "PRIDE"
        )

        # Bind the pride long dataframe to the existing self.df DataFrame
        if self.df is not None:
            self.df = self.df.unionByName(pride_long, allowMissingColumns=True)
        else:
            self.df = pride_long

    def pack_data_for_output(self):
        """Use spark to write the DataFrame to parquet format."""
        if self.local:
            output_path = f'file://{self.output_directory_path}'
        else:
            output_path = f'{self.output_directory_path}'
        if self.json:
            output_path = f'{output_path}/json/'
            # If JSON output is requested, convert DataFrame to JSON format
            self.df.write.mode('overwrite').json(output_path)
            logger.info(f'Data written to {output_path} in JSON format')
        else:
            output_path = f'{output_path}/parquet/'
            # If parquet output is requested, convert DataFrame to parquet format
            self.df.write.mode('overwrite').parquet(output_path)
            logger.info(f'Data written to {output_path} in Parquet format')

    def run(self):
        logger.info('Reading PRIDE data...')
        for pride_code in self.pride_codes:
            logger.info(f'Processing PRIDE code: {pride_code}')
            self.read_pride_data(pride_code)
        logger.info('Packing data for output...')
        self.pack_data_for_output()
