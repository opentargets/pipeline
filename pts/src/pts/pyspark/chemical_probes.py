"""Chemical probes dataset generation.

Processes the Open Targets chemical probes Excel/CSV sources, resolves gene
symbols to ENSG IDs via output/target, and produces a flat per-probe index
validated against the target universe.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common import Session
from pts.pyspark.common.utils import maybe_coalesce

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
    """Process chemical probes data, resolve target IDs, and write the output dataset.

    Parses the chemical probes Excel/CSV sources, resolves gene symbols/UniProt
    accessions to ENSG IDs via output/target, drops probes that cannot be mapped,
    and writes a flat per-probe parquet to the destination.

    Args:
        source: Mapping with keys ``probes_excel``, ``drugs_csv``,
            ``chembl_molecule``, and ``target`` (output/target parquet for
            ENSG ID resolution and validation).
        destination: Output path for the chemical probes parquet dataset.
        settings: Step settings; supports ``partition_count`` (int, default 2).
        properties: Spark properties passed to :class:`Session`.
    """
    session = Session(app_name='chemical_probes', properties=properties)

    chembl_molecule_df = session.load_data(source['chembl_molecule'])
    target_df = session.spark.read.parquet(source['target'])

    probes_data = process_probes_data(session.spark, source['probes_excel'])
    probes_targets_data = process_probes_targets_data(session.spark, source['probes_excel'])
    probes_sets_data = process_probes_sets_data(session.spark, source['probes_excel'])
    targets_xref_data = process_targets_xrefs(session.spark, source['probes_excel'])
    drugs_xref_data = process_drugs_xrefs(session.spark, source['drugs_csv'], chembl_molecule_df)

    evidence = generate_chemical_probes_evidence(
        session.spark, probes_data, probes_targets_data, probes_sets_data, targets_xref_data, drugs_xref_data
    )

    ensg_lookup = _build_ensg_lookup(target_df)
    result = _resolve_targets(evidence, ensg_lookup)

    partition_count = (settings or {}).get('partition_count', 2)
    maybe_coalesce(result, partition_count).write.mode('overwrite').parquet(destination)


def _build_ensg_lookup(target_df: DataFrame) -> DataFrame:
    """Build a symbol/proteinId → ENSG ID lookup from the target dataset.

    Args:
        target_df: Target parquet from ``output/target``.

    Returns:
        DataFrame with columns ``ensgId`` and ``name``
        (``array<str>`` of protein accessions and approved symbol).
    """
    return target_df.select(
        f.col('id').alias('ensgId'),
        f.flatten(f.array(
            f.col('proteinIds.id'),
            f.array(f.col('approvedSymbol')),
        )).alias('name'),
    )


def _resolve_targets(evidence: DataFrame, ensg_lookup: DataFrame) -> DataFrame:
    """Resolve targetFromSourceId to ENSG IDs.

    Rows with no matching ENSG are kept with a null ``targetId``: the
    drug/probe fields on this dataset (``drugId``, ``drugFromSourceId``) are
    consumed independently of target resolution by ``drug_molecule``, so
    dropping unresolvable rows here would silently shrink that dataset too.
    Consumers that join on ``targetId`` (e.g. ``target.py``) naturally
    exclude null-targetId rows since they match no real target id.

    Args:
        evidence: Chemical probes evidence DataFrame from
            :func:`generate_chemical_probes_evidence`.
        ensg_lookup: ENSG lookup from :func:`_build_ensg_lookup`.

    Returns:
        DataFrame with ``targetId`` column added (nullable) and
        ``targetFromSourceId`` retained.
    """
    return (
        evidence
        .join(ensg_lookup, f.array_contains(f.col('name'), f.col('targetFromSourceId')), 'left_outer')
        .drop('name')
        .withColumnRenamed('ensgId', 'targetId')
    )


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
        spark
        .createDataFrame(
            pd
            .read_excel(
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
        spark
        .createDataFrame(
            pd
            .read_excel(probes_excel, sheet_name='PROBES TARGETS', header=0, index_col=0)
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
        spark
        .createDataFrame(pd.read_excel(probes_excel, sheet_name='COMPOUNDSETS', header=0, index_col=0))
        .selectExpr('COMPOUNDSET as datasourceId', 'SOURCE_URL as url')
        .filter(f.col('url').startswith('http'))
    )


def process_targets_xrefs(spark: SparkSession, probes_excel: str) -> DataFrame:
    """Look-up table between the gene symbols and the UniProt IDs."""
    return spark.createDataFrame(
        pd.read_excel(probes_excel, sheet_name='TARGETS', header=0, index_col=0).reset_index()
    ).selectExpr('target as targetFromSource', 'uniprot as targetFromSourceId')


def process_drugs_xrefs(spark: SparkSession, drugs_csv: str, chembl_molecule: DataFrame) -> DataFrame:
    """Look-up table between the probes IDs in P&Ds and ChEMBL.

    Only includes drugIds that exist in the ChEMBL molecule dataset.

    Args:
        spark: Spark session.
        drugs_csv: Path to the drugs CSV file.
        chembl_molecule: ChEMBL molecule DataFrame for validating drugIds.

    Returns:
        DataFrame with pdid and drugId columns, filtered to valid ChEMBL IDs.
    """
    # Get valid ChEMBL molecule IDs
    valid_ids = chembl_molecule.select(f.col('id').alias('drugId'))

    return (
        spark.read
        .csv(drugs_csv, header=True)
        .selectExpr('pdid', 'ChEMBL as drugId')
        .filter(f.col('drugId').isNotNull())
        .join(valid_ids, on='drugId', how='inner')
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
        'pdid',
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
        probes_targets_data
        .join(probes_data, on='pdid', how='left')
        .join(targets_xref_data, on='targetFromSource', how='left')
        .join(probes_sets_data, on='datasourceId', how='left')
        .join(drugs_xref_data, on='pdid', how='left')
        .groupBy(grouping_cols)
        .agg(f.collect_set(f.struct(f.col('datasourceId').alias('niceName'), f.col('url').alias('url'))).alias('urls'))
        .withColumnRenamed('pdid', 'drugFromSourceId')
    )
