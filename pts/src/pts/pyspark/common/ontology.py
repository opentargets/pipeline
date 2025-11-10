"""Map diseases using OnToma."""

from loguru import logger
from ontoma import OnToma
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.functions import array, array_union, coalesce, col, explode_outer, lit


def add_efo_mapping(
    spark: SparkSession,
    evidence_df: DataFrame,
    disease_label_lut_path: str,
    label_col_name: str = 'diseaseFromSource',
    disease_id_lut_path: str | None = None,
    id_col_name: str | None = 'diseaseFromSourceId'
) -> DataFrame:
    """Adds a column containing EFO mappings to the provided dataframe.

    Given a dataframe containing evidence with columns containing disease label and id information,
    this function returns the dataframe with the additional column diseaseFromSourceMappedId containing
    the corresponding EFO mappings.
    If there are multiple mappings for an evidence, the results are exploded accordingly.

    Args:
        spark (SparkSession): Spark session.
        evidence_df (DataFrame): Dataframe containing evidence.
        disease_label_lut_path (str): Path to the disease label lookup table.
        label_col_name (str): Name of the column containing the disease label.
        disease_id_lut_path (str | None): Path to the disease id lookup table.
        id_col_name (str | None): Name of the column containing the disease id.

    Returns:
        DataFrame: Dataframe with an additional column diseaseFromSourceMappedId containing the EFO mappings.
    """
    logger.info(f'load disease label lookup table from {disease_label_lut_path}.')
    disease_label_lut = OnToma(spark=spark, cache_dir=disease_label_lut_path)
    label_mapped_df = disease_label_lut.map_entities(
        evidence_df, 'labelMappedResult', label_col_name, 'label', type_col=lit('DS')
    )

    if id_col_name and disease_id_lut_path:
        logger.info(f'load disease id lookup table from {disease_id_lut_path}.')
        disease_id_lut = OnToma(spark=spark, cache_dir=disease_id_lut_path)
        label_id_mapped_df = disease_id_lut.map_entities(
            label_mapped_df, 'idMappedResult', id_col_name, 'id', type_col=lit('DS')
        )

        result_df = (
            label_id_mapped_df
            .withColumn(
                'diseaseFromSourceMappedId',
                _combine_result_columns(col('labelMappedResult'), col('idMappedResult'))
            )
            .drop('labelMappedResult', 'idMappedResult')
        )
    else:
        result_df = (
            label_mapped_df
            .withColumn('diseaseFromSourceMappedId', explode_outer('labelMappedResult'))
            .drop('labelMappedResult')
        )

    return result_df


def _combine_result_columns(label_col: Column, id_col: Column) -> Column:
    """Combine the results from two result columns into a single column.

    Args:
        label_col (Column): Column containing the disease label.
        id_col (Column): Column containing the disease id.

    Returns:
        Column: Column combining the results from mapping labels and IDs.
    """
    return (
        explode_outer(
            array_union(
                coalesce(label_col, array()),
                coalesce(id_col, array())
            )
        )
    )
