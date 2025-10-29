"""Evidence parser for ClinGen's Gene Validity Curations."""

import pyspark.sql.functions as f
from loguru import logger

from pts.pyspark.common.session import Session
from pts.utils.ontology import add_efo_mapping


def clingen(
    source: dict[str, str],
    destination: str,
    settings: dict[str, str],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='clingen', properties=properties)
    efo_version = settings.get('efo_version')
    cores = int(settings.get('ontology_cores', 1))

    logger.info(f'load data from {source}')
    # Load CSV without header since we need to skip metadata rows
    raw_df = spark.load_data(source['evidence'], format='csv', header=False, inferSchema=True)

    # CSV structure:
    # Row 0-2: metadata rows
    # Row 3: separator (+++)
    # Row 4: column headers (GENE SYMBOL, etc.)
    # Row 5: separator (+++)
    # Row 6+: actual data
    clingen_df = (
        raw_df.withColumn('idx', f.monotonically_increasing_id())
        .filter(f.col('idx') > 5)  # Skip metadata, headers, and separators - start from actual data
        .drop('idx')
        # Rename columns to match the header row we saw in the file
        .toDF(
            'GENE SYMBOL',
            'GENE ID (HGNC)',
            'DISEASE LABEL',
            'DISEASE ID (MONDO)',
            'MOI',
            'SOP',
            'CLASSIFICATION',
            'ONLINE REPORT',
            'CLASSIFICATION DATE',
            'GCEP',
        )
    )

    evidence_df = clingen_df.select(
        f.lit('clingen').alias('datasourceId'),
        f.lit('genetic_literature').alias('datatypeId'),
        f.trim(f.col('GENE SYMBOL')).alias('targetFromSourceId'),
        f.col('DISEASE LABEL').alias('diseaseFromSource'),
        f.col('DISEASE ID (MONDO)').alias('diseaseFromSourceId'),
        f.array(f.col('MOI')).alias('allelicRequirements'),
        f.array(f.struct(f.col('ONLINE REPORT').alias('url'))).alias('urls'),
        f.col('CLASSIFICATION').alias('confidence'),
        f.date_format(f.col('CLASSIFICATION DATE'), 'yyyy-MM-dd').alias('releaseDate'),
        f.col('GCEP').alias('studyId'),
    )

    logger.info('map clingen disease labels')
    mapped_evidence_df = add_efo_mapping(
        evidence_strings=evidence_df, spark_instance=spark.spark, efo_version=efo_version, cores=cores
    )

    logger.info(f'write clingen evidence strings to {destination}')
    mapped_evidence_df.write.parquet(destination, mode='overwrite')
