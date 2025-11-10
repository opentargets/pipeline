"""Disease to target evidence parser for Project Score v2."""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import GenerateDiseaseCellLines

# This parser is specific for the second release of Project Score, so the publication identifier is hardcoded:
PMID = '38215750'


def generate_project_score_evidence(project_score_cell_lines: DataFrame, project_score_hits: DataFrame) -> DataFrame:
    """Generate evidence strings for Project Score v2.

    Args:
        project_score_cell_lines (DataFrame): DataFrame containing cell line annotations
        project_score_hits (DataFrame): DataFrame containing evidence data

    Returns:
        DataFrame: DataFrame containing evidence strings
    """
    return (
        project_score_hits.select(
            f.col('targetSymbol').alias('targetFromSource'),
            f.col('diseaseName').alias('diseaseFromSource'),
            f.col('diseaseId').alias('diseaseFromSourceMappedId'),
            f.col('PRIORITY').cast('float').alias('resourceScore'),
            f.col('targetId').alias('targetFromSourceId'),
            f.array(f.lit(PMID)).alias('literature'),
            f.lit('crispr').alias('datasourceId'),
            f.lit('affected_pathway').alias('datatypeId'),
            f.lower(f.col('cancerType')).alias('cancerType'),
        )
        .join(project_score_cell_lines, on='cancerType', how='left')
        .filter(f.col('cancerType') != 'pancancer')
        .drop('cancerType')
    )


def get_disease_cell_lines(
    cell_passport_file: DataFrame, cell_line_to_uberon_mapping: DataFrame, cell_line_data: DataFrame
) -> DataFrame:
    """Based on the cell passport data and cell line data, generate a dataframe with disease cell lines.

    Args:
        cell_passport_file (DataFrame): Dataframe containing Sanger Cell Model Passport data.
        cell_line_to_uberon_mapping (DataFrame): Dataframe containing tissue label to UBERON mapping
        cell_line_data (DataFrame): Cell line data from ProjectScore

    Returns:
        DataFrame: Prepared disease cell lines objects indexed as cancerType.
    """
    passport_disease_cell_lines = GenerateDiseaseCellLines(
        cell_passport_file, cell_line_to_uberon_mapping
    ).generate_disease_cell_lines()

    # Joining disease cell-lines dataframe for Project Score:
    disease_cell_lines = (
        cell_line_data.select(
            f.lower(f.col('CANCER_TYPE')).alias('cancerType'),
            f.col('CMP_ID').alias('id'),
        )
        .join(passport_disease_cell_lines, on='id', how='right')
        .groupBy('cancerType')
        .agg(f.collect_set('diseaseCellLine').alias('diseaseCellLines'))
    )

    # Are there any cancer types that are not in the cell line file?
    missing_cancer_types = disease_cell_lines.filter(f.col('diseaseCellLines').isNull())
    if missing_cancer_types.count() > 0:
        logger.warning(f'the following cancer types are not in the cell line file: {missing_cancer_types.collect()}')
    else:
        logger.info('all cancer types are in the cell line file.')

    return disease_cell_lines


def project_score(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Main function for parsing Project Score v2 data."""
    spark = Session(app_name='chemical_probes', properties=properties)

    logger.debug(f'loading data from: {source}')
    evidence_table = spark.load_data(path=source['gene_scores'], format='csv', header=True, sep='\t')
    cell_line_data = spark.load_data(path=source['cell_types'], format='csv', header=True, sep='\t')
    cell_passport_file = spark.load_data(
        path=source['cell_passport'], format='csv', header=True, sep=',', quote='"', multiline=True
    )
    cell_line_to_uberon_mapping = spark.load_data(path=source['cell_line_mapping'], format='csv', header=True, sep=',')

    logger.debug('generating a dataframe with disease cell lines')
    disease_cell_lines = get_disease_cell_lines(cell_passport_file, cell_line_to_uberon_mapping, cell_line_data)
    logger.debug('generating evidence from project score')
    project_score_evidence = generate_project_score_evidence(disease_cell_lines, evidence_table)

    logger.debug(f'writing output data to: {destination}')
    project_score_evidence.write.parquet(destination, mode='overwrite')
