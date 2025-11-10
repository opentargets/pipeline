"""Parser to process chemical probes data and generate evidence."""

from typing import Any

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common import Session

# Chemical probe data sources
PROBES_SETS = [
    'Bromodomains chemical toolbox',
    'Chemical Probes for Understudied Kinases',
    'Chemical Probes.org',
    'Gray Laboratory Probes',
    'High-quality chemical probes',
    'MLP Probes',
    'Nature Chemical Biology Probes',
    'Open Science Probes',
    'opnMe Portal',
    'Probe Miner (suitable probes)',
    'Protein methyltransferases chemical toolbox',
    'SGC Probes',
    'Tool Compound Set',
    'Concise Guide to Pharmacology 2019/20',
    'Kinase Chemogenomic Set (KCGS)',
    'Kinase Inhibitors (best-in-class)',
    'Novartis Chemogenetic Library (NIBR MoA Box)',
    'Nuisance compounds in cellular assays',
]


def chemical_probes(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Process chemical probes data and generate evidence."""
    # Starting spark session
    session = Session(app_name='chemical_probes', properties=properties)

    # Extract input dataset locations from config
    probes_excel = source['probes_excel']
    drugs_csv = source['drugs_csv']

    # Process chemical probes data from Excel and CSV files
    probes_data = process_probes_data(session.spark, probes_excel)
    probes_targets_data = process_probes_targets_data(session.spark, probes_excel)
    probes_sets_data = process_probes_sets_data(session.spark, probes_excel)
    targets_xref_data = process_targets_xrefs(session.spark, probes_excel)
    drugs_xref_data = process_drugs_xrefs(session.spark, drugs_csv)

    # Generate evidence from chemical probes
    evidence = generate_chemical_probes_evidence(
        session.spark, probes_data, probes_targets_data, probes_sets_data, targets_xref_data, drugs_xref_data
    )

    # Write the final evidence dataset
    evidence.write.mode('overwrite').parquet(destination)


def collapse_cols_data_in_array(df: DataFrame, source_cols: list[str], destination_col: str) -> DataFrame:
    """Collapses the data in a single column when the information is one-hot encoded.

    Args:
        df (DataFrame): Dataframe containing the data for the different probes
        source_cols (list[str]): List of columns containing the data to be collapsed
        destination_col (str): Name of the column where the array will be stored

    Returns:
        DataFrame: Dataframe with a new column containing the sources that have data for a specific probe
    """
    # Escape the name of the columns in case they contain spaces
    source_cols = [f'`{e}`' for e in source_cols if ' ' in e]
    return df.withColumn(
        destination_col,
        f.array([f.when(df[c] == 1, c.replace(r'`', '')) for c in source_cols]),
    ).withColumn(destination_col, f.array_except(f.col(destination_col), f.array(f.lit(None))))


def clean_origin_col() -> Column:
    """Removes the substring ' probe' from the origin column.

    This states if the probe has been reported from an experimental or computational approach.
    """
    return f.array_distinct(f.expr("transform(origin, x -> trim(regexp_replace(x, ' probe', '')))"))


def extract_hq_flag() -> Column:
    """Returns a flag indicating if the probe is high-quality or not."""
    return f.when(
        f.array_contains(f.col('datasourceIds'), 'High-quality chemical probes'),
        f.lit(True),
    ).otherwise(f.lit(False))


def convert_stringified_array_to_array(col_name: str) -> Column:
    """Converts a column of stringified arrays to an array column.

    Args:
        col_name: Name of the column that contains the stringified array
    """
    return f.split(f.translate(col_name, "[]'", ''), ', ').cast(t.ArrayType(t.StringType()))


def replace_dash(col_name: str) -> Column:
    """Converts to null those values that only contain `-`."""
    return f.when(f.col(col_name).cast(t.StringType()) != '-', f.col(col_name))


def process_scores(col_name: str) -> Column:
    """Helper function to refactor the score processing logic."""
    return f.expr(f'try_cast({col_name} as double)')


def process_probes_data(spark: SparkSession, probes_excel: str) -> DataFrame:
    """Metadata about the compound and the scores given by the different sources."""
    return (
        spark.createDataFrame(
            pd.read_excel(
                probes_excel,
                sheet_name='PROBES',
                header=0,
                index_col=0,
            )
            # Probes that do not have an associated target are marked as nulls
            .query('target.notnull()')
            .reset_index()
            .drop('control_smiles', axis=1)
        )
        # Collect list of datasources for each probe
        .transform(lambda df: collapse_cols_data_in_array(df, PROBES_SETS, 'datasourceIds'))
        # Collecting the list of detection methods of the probe
        .transform(
            lambda df: collapse_cols_data_in_array(
                df,
                ['experimental probe', 'calculated probe'],
                'origin',
            )
        )
        .select(
            'pdid',
            f.col('compound_name').alias('id'),
            clean_origin_col().alias('origin'),
            # Flag the high-quality probes and remove this from the list of datasources
            extract_hq_flag().alias('isHighQuality'),
            f.explode(
                f.array_except(
                    f.col('datasourceIds'),
                    f.array(f.lit('High-quality chemical probes')),
                )
            ).alias('datasourceId'),
            replace_dash('control_name').alias('control'),
        )
    )


def process_probes_targets_data(spark: SparkSession, probes_excel: str) -> DataFrame:
    """Collection of targets associated with the probes and their scores."""
    return (
        spark.createDataFrame(
            pd.read_excel(probes_excel, sheet_name='PROBES TARGETS', header=0, index_col=0)
            # Probes that do not have an associated target are marked with "-"
            .query("gene_name != '-'")
            .reset_index()
            .drop('control_smiles', axis=1)
        )
        .filter(f.col('organism') == 'Homo sapiens')
        .withColumn(
            'mechanismOfAction',
            f.when(
                f.col('action') != '-',
                f.split(f.col('action'), ';'),
            ),
        )
        .select(
            'pdid',
            f.col('target').alias('targetFromSource'),
            'mechanismOfAction',
            process_scores('`P&D probe-likeness score`').alias('probesDrugsScore'),
            process_scores('`Probe Miner Score`').alias('probeMinerScore'),
            process_scores('`Cells score (Chemical Probes.org)`').alias('scoreInCells'),
            process_scores('`Organisms score (Chemical Probes.org)`').alias('scoreInOrganisms'),
        )
    )


def process_probes_sets_data(spark: SparkSession, probes_excel: str) -> DataFrame:
    """Metadata about the different sources of probes."""
    return (
        spark.createDataFrame(pd.read_excel(probes_excel, sheet_name='COMPOUNDSETS', header=0, index_col=0))
        .selectExpr('COMPOUNDSET as datasourceId', 'SOURCE_URL as url')
        .filter(f.col('url').startswith('http'))
    )


def process_targets_xrefs(spark: SparkSession, probes_excel: str) -> DataFrame:
    """Look-up table between the gene symbols and the UniProt IDs."""
    return spark.createDataFrame(
        pd.read_excel(probes_excel, sheet_name='TARGETS', header=0, index_col=0).reset_index()
    ).selectExpr('target as targetFromSource', 'uniprot as targetFromSourceId')


def process_drugs_xrefs(spark: SparkSession, drugs_csv: str) -> DataFrame:
    """Look-up table between the probes IDs in P&Ds and ChEMBL."""
    return (
        spark.read.csv(drugs_csv, header=True)
        .selectExpr('pdid', 'ChEMBL as drugId')
        .filter(f.col('drugId').isNotNull())
    )


def generate_chemical_probes_evidence(
    spark: SparkSession,
    probes_data: DataFrame,
    probes_targets_data: DataFrame,
    probes_sets_data: DataFrame,
    targets_xref_data: DataFrame,
    drugs_xref_data: DataFrame,
) -> DataFrame:
    """Generate evidence from chemical probes data.

    Args:
        spark (SparkSession): Spark session.
        probes_data (DataFrame): Probes data.
        probes_targets_data (DataFrame): Probes targets data.
        probes_sets_data (DataFrame): Probes sets data.
        targets_xref_data (DataFrame): Targets cross-reference data.
        drugs_xref_data (DataFrame): Drugs cross-reference data.

    Returns:
        DataFrame: Generated evidence.
    """
    grouping_cols = [
        'targetFromSourceId',
        'id',
        'drugId',
        'mechanismOfAction',
        'origin',
        'control',
        'isHighQuality',
        'probesDrugsScore',
        'probeMinerScore',
        'scoreInCells',
        'scoreInOrganisms',
    ]

    return (
        probes_targets_data.join(probes_data, on='pdid', how='left')
        .join(targets_xref_data, on='targetFromSource', how='left')
        .join(probes_sets_data, on='datasourceId', how='left')
        .join(drugs_xref_data, on='pdid', how='left')
        .groupBy(grouping_cols)
        .agg(f.collect_set(f.struct(f.col('datasourceId').alias('niceName'), f.col('url').alias('url'))).alias('urls'))
    )
