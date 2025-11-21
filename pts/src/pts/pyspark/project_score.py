"""Disease to target evidence parser for Project Score v2."""

from typing import Any

from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session

# This parser is specific for the second release of Project Score, so the publication identifier is hardcoded:
PMID = '38215750'


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


class GenerateDiseaseCellLines:
    """Generate "diseaseCellLines" object from a cell passport file.

    !!!
    There's one important bit here: I have noticed that we frequenty get cell line names
    with missing dashes. Therefore the cell line names are cleaned up by removing dashes.
    It has to be done when joining with other datasets.
    !!!

    Args:
        cell_passport_file: Path to the cell passport file.
    """

    def __init__(
        self,
        cell_passport_data: DataFrame,
        cell_line_to_uberon_mapping: DataFrame,
    ) -> None:
        self.cell_passport_data = cell_passport_data
        self.tissue_to_uberon_map = cell_line_to_uberon_mapping

    def generate_disease_cell_lines(self) -> DataFrame:
        """Reading and procesing cell line data from the cell passport file.

        The schema of the returned dataframe is:

        root
        |-- name: string (nullable = true)
        |-- id: string (nullable = true)
        |-- biomarkerList: array (nullable = true)
        |    |-- element: struct (containsNull = true)
        |    |    |-- name: string (nullable = true)
        |    |    |-- description: string (nullable = true)
        |-- diseaseCellLine: struct (nullable = false)
        |    |-- tissue: string (nullable = true)
        |    |-- name: string (nullable = true)
        |    |-- id: string (nullable = true)
        |    |-- tissueId: string (nullable = true)

        Note:
            * Microsatellite stability is the only inferred biomarker.
            * The cell line name has the dashes removed.
            * Id is the cell line identifier from Sanger
            * Tissue id is the UBERON identifier for the tissue, based on manual curation.
        """
        cell_df = (
            self.cell_passport_data.select(
                f.col('model_name').alias('name'),
                f.col('model_id').alias('id'),
                f.lower(f.col('tissue')).alias('tissueFromSource'),
                f.array(self.parse_msi_status(f.col('msi_status'))).alias('biomarkerList'),
            )
            .join(self.tissue_to_uberon_map, on='tissueFromSource', how='left')
            .persist()
        )
        return cell_df.select(
            f.regexp_replace(f.col('name'), '-', '').alias('name'),
            'id',
            'biomarkerList',
            f.struct(f.col('tissueName').alias('tissue'), 'name', 'id', 'tissueId').alias('diseaseCellLine'),
        )

    @staticmethod
    def parse_msi_status(status: Column) -> Column:
        """Based on the content of the MSI status, we generate the corresponding biomarker object."""
        return f.when(
            status == 'MSI',
            f.struct(
                f.lit('MSI').alias('name'),
                f.lit('Microsatellite instable').alias('description'),
            ),
        ).when(
            status == 'MSS',
            f.struct(
                f.lit('MSS').alias('name'),
                f.lit('Microsatellite stable').alias('description'),
            ),
        )
