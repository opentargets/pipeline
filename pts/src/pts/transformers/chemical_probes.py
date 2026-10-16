"""Chemical probes dataset generation.

Processes the Open Targets chemical probes Excel/CSV sources, resolves gene
symbols to ENSG IDs via output/target, and produces a flat per-probe index
validated against the target universe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.target import ensg_lookup

# Chemical probe data sources: columns of the PROBES sheet.
PROBES_SETS = [
    'A Collection of Useful Nuisance Compounds (CONS) for Interrogation of Bioassay Integrity',
    'Bromodomains chemical toolbox',
    'Chemical Probes for Understudied Kinases',
    'Chemical Probes.org',
    'Gray Laboratory Probes',
    'High-quality chemical probes',
    'MLP Probes',
    'Nature Chemical Biology Probes',
    'Open Science Probes',
    'opnMe Portal',
    'Protein methyltransferases chemical toolbox',
    'SGC Probes',
    'Concise Guide to Pharmacology 2025/26',
    'Kinase Chemogenomic Set (KCGS)',
    'Kinase Inhibitors (best-in-class)',
    'Novartis Chemogenetic Library (NIBR MoA Box)',
    'Nuisance compounds in cellular assays',
]

PROBES_COLUMNS = ['pdid', 'compound_name', 'control_name', 'experimental probe', 'calculated probe', *PROBES_SETS]

PROBES_TARGETS_COLUMNS = [
    'pdid',
    'organism',
    'target',
    'action',
    'P&D probe-likeness score',
    'Cells score (Chemical Probes.org)',
    'Organisms score (Chemical Probes.org)',
]

MISSING_VALUE_TOKENS = ('-', 'nan')

HIGH_QUALITY = 'High-quality chemical probes'

# Characters spark's string-to-number cast trims from both ends.
_CAST_TRIM = ''.join(chr(c) for c in range(0x21))

_GROUPING_COLS = [
    'targetFromSourceId',
    'drugFromSource',
    'pdid',
    'drugId',
    'mechanismOfAction',
    'origin',
    'control',
    'isHighQuality',
    'probesDrugsScore',
    'scoreInCells',
    'scoreInOrganisms',
]


def chemical_probes(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Process chemical probes data, resolve target IDs, and write the output dataset.

    Parses the chemical probes Excel/CSV sources, resolves gene symbols/UniProt
    accessions to ENSG IDs via output/target, drops probes that cannot be mapped,
    and writes a flat per-probe parquet to the destination.

    Args:
        source: mapping with keys `probes_excel`, `drugs_csv`, `chembl_molecule` and `target`
            (output/target, for ENSG ID resolution and validation).
        destination: output path for the chemical probes dataset.
        settings: unused.
        config: unused.
    """
    chembl_molecule = scan_dataset(source['chembl_molecule']).select('id').collect()
    lookup = ensg_lookup(scan_dataset(source['target']).select('id', 'approvedSymbol', 'proteinIds')).collect()

    evidence = generate_chemical_probes_evidence(
        process_probes_data(source['probes_excel']),
        process_probes_targets_data(source['probes_excel']),
        process_probes_sets_data(source['probes_excel']),
        process_targets_xrefs(source['probes_excel']),
        process_drugs_xrefs(source['drugs_csv'], chembl_molecule),
    )
    result = resolve_targets(evidence, lookup)
    write_dataset(result.sort(result.columns), destination)


def resolve_targets(evidence: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    """Resolve targetFromSourceId to ENSG IDs, dropping rows that match no target.

    Args:
        evidence: chemical probes evidence from `generate_chemical_probes_evidence`.
        lookup: ENSG lookup from `ensg_lookup`.

    Returns:
        The evidence with a `targetId` column added, one row per matching target.
    """
    return evidence.join(lookup, left_on='targetFromSourceId', right_on='name', how='inner').rename({
        'ensgId': 'targetId'
    })


def cell_to_string(value: Any) -> str | None:
    """Render a spreadsheet cell as a string: blanks as null, booleans as '1'/'0'."""
    if pd.isna(value):
        return None
    if isinstance(value, bool | np.bool_):
        return '1' if value else '0'
    return str(value)


def pandas_to_strings(pdf: pd.DataFrame, columns: list[str] | None = None) -> pl.DataFrame:
    """Convert a pandas frame to polars with every cell as a string and blanks as null.

    Args:
        pdf: pandas frame, e.g. a sheet from `pd.read_excel`.
        columns: columns to keep. All columns when omitted.

    Returns:
        DataFrame with one string column per kept column.
    """
    if columns is not None:
        pdf = pdf[columns]
    values = {c: [cell_to_string(v) for v in pdf[c].astype(object)] for c in pdf.columns}
    return pl.DataFrame(values, schema=dict.fromkeys(values, pl.String))


def collapse_cols_data_in_array(df: pl.DataFrame, source_cols: list[str], destination_col: str) -> pl.DataFrame:
    """Collapses the data in a single column when the information is one-hot encoded.

    Only columns whose name contains a space are read.

    Args:
        df: data for the different probes.
        source_cols: columns containing the data to be collapsed.
        destination_col: name of the column where the list will be stored.

    Returns:
        The frame with a new column listing the sources that have data for each probe.
    """
    flags = [pl.when(process_scores(c) == 1).then(pl.lit(c)) for c in source_cols if ' ' in c]
    collected = pl.concat_list(flags).list.drop_nulls().list.unique(maintain_order=True)
    return df.with_columns(collected.alias(destination_col))


def clean_origin_col() -> pl.Expr:
    """Removes the substring ' probe' from the origin column."""
    return (
        pl
        .col('origin')
        .list.eval(pl.element().str.replace_all(' probe', '', literal=True).str.strip_chars(' '))
        .list.unique(maintain_order=True)
    )


def extract_hq_flag() -> pl.Expr:
    """Returns a flag indicating if the probe is high-quality or not."""
    return pl.col('datasourceIds').list.contains(HIGH_QUALITY).fill_null(False)


def convert_stringified_array_to_array(col_name: str) -> pl.Expr:
    """Converts a column of stringified arrays to a list column.

    Args:
        col_name: name of the column that contains the stringified array.
    """
    return pl.col(col_name).str.replace_all(r"[\[\]']", '').str.split(', ')


def replace_missing_placeholder(col: pl.Expr) -> pl.Expr:
    """Converts to null values that only encode "missing" as a placeholder token."""
    value = col.cast(pl.String).str.strip_chars(' ').str.to_lowercase()
    return pl.when(~value.is_in(list(MISSING_VALUE_TOKENS))).then(col)


def process_scores(col_name: str) -> pl.Expr:
    """Cast a text column to double as spark's `try_cast` does, giving null on bad text."""
    value = pl.col(col_name).str.strip_chars(_CAST_TRIM)
    number = value.str.replace(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[dDfF]$', '$1')
    return (
        pl
        .when(~value.str.contains(r'^[+-][nN][aA][nN]$') | value.str.contains(r'^[+-]NaN$'))
        .then(number.cast(pl.Float64, strict=False))
        .alias(col_name)
    )


def _read_sheet(probes_excel: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(StorageHandle(probes_excel).open('rb'), sheet_name=sheet_name, header=0, index_col=0)


def process_probes_data(probes_excel: str) -> pl.DataFrame:
    """Metadata about the compound and the scores given by the different sources."""
    # Probes that do not have an associated target are marked as nulls
    probes = _read_sheet(probes_excel, 'PROBES').query('target.notnull()').reset_index()
    return (
        pandas_to_strings(probes, PROBES_COLUMNS)
        .pipe(collapse_cols_data_in_array, PROBES_SETS, 'datasourceIds')
        .pipe(collapse_cols_data_in_array, ['experimental probe', 'calculated probe'], 'origin')
        .select(
            'pdid',
            pl.col('compound_name').alias('drugFromSource'),
            clean_origin_col().alias('origin'),
            extract_hq_flag().alias('isHighQuality'),
            pl.col('datasourceIds').list.set_difference([HIGH_QUALITY]).alias('datasourceId'),
            replace_missing_placeholder(pl.col('control_name')).alias('control'),
        )
        .filter(pl.col('datasourceId').list.len() > 0)
        .explode('datasourceId')
    )


def process_probes_targets_data(probes_excel: str) -> pl.DataFrame:
    """Collection of targets associated with the probes and their scores."""
    probes_targets = (
        _read_sheet(probes_excel, 'PROBES TARGETS')
        # Probes without an associated target are marked with "-" or left blank
        .query("gene_name.notnull() and gene_name != '-'")
        .reset_index()
    )
    return (
        pandas_to_strings(probes_targets, PROBES_TARGETS_COLUMNS)
        .filter(pl.col('organism') == 'Homo sapiens')
        .select(
            'pdid',
            pl.col('target').alias('targetFromSource'),
            replace_missing_placeholder(pl.col('action')).str.split(';').alias('mechanismOfAction'),
            process_scores('P&D probe-likeness score').alias('probesDrugsScore'),
            process_scores('Cells score (Chemical Probes.org)').alias('scoreInCells'),
            process_scores('Organisms score (Chemical Probes.org)').alias('scoreInOrganisms'),
        )
    )


def process_probes_sets_data(probes_excel: str) -> pl.DataFrame:
    """Metadata about the different sources of probes."""
    compound_sets = _read_sheet(probes_excel, 'COMPOUNDSETS')
    return (
        pandas_to_strings(compound_sets, ['COMPOUNDSET', 'SOURCE_URL'])
        .select(pl.col('COMPOUNDSET').alias('datasourceId'), pl.col('SOURCE_URL').alias('url'))
        .filter(pl.col('url').str.starts_with('http'))
    )


def process_targets_xrefs(probes_excel: str) -> pl.DataFrame:
    """Look-up table between the gene symbols and the UniProt IDs."""
    targets = _read_sheet(probes_excel, 'TARGETS').reset_index()
    return pandas_to_strings(targets, ['target', 'uniprot']).select(
        pl.col('target').alias('targetFromSource'), pl.col('uniprot').alias('targetFromSourceId')
    )


def process_drugs_xrefs(drugs_csv: str, chembl_molecule: pl.DataFrame) -> pl.DataFrame:
    """Look-up table between the probes IDs in P&Ds and ChEMBL.

    Only includes drugIds that exist in the ChEMBL molecule dataset.

    Args:
        drugs_csv: path to the drugs CSV file.
        chembl_molecule: ChEMBL molecules, for validating drugIds.

    Returns:
        DataFrame with pdid and drugId columns, filtered to valid ChEMBL IDs.
    """
    valid_ids = chembl_molecule.select(pl.col('id').alias('drugId'))
    return (
        scan_dataset(drugs_csv, format='csv', infer_schema=False)
        .select('pdid', pl.col('ChEMBL').alias('drugId'))
        .filter(pl.col('drugId').is_not_null())
        .collect()
        .join(valid_ids, on='drugId', how='inner', maintain_order='left')
    )


def generate_chemical_probes_evidence(
    probes_data: pl.DataFrame,
    probes_targets_data: pl.DataFrame,
    probes_sets_data: pl.DataFrame,
    targets_xref_data: pl.DataFrame,
    drugs_xref_data: pl.DataFrame,
) -> pl.DataFrame:
    """Generate evidence from chemical probes data.

    Args:
        probes_data: probes data.
        probes_targets_data: probes targets data.
        probes_sets_data: probes sets data.
        targets_xref_data: targets cross-reference data.
        drugs_xref_data: drugs cross-reference data.

    Returns:
        One row per distinct grouping, with its datasource `urls` sorted.
    """
    return (
        probes_targets_data
        .join(probes_data, on='pdid', how='left')
        .join(targets_xref_data, on='targetFromSource', how='left')
        .join(probes_sets_data, on='datasourceId', how='left')
        .join(drugs_xref_data, on='pdid', how='left')
        .group_by(_GROUPING_COLS)
        .agg(pl.struct(pl.col('datasourceId').alias('niceName'), 'url').unique().sort().alias('urls'))
        .rename({'pdid': 'drugFromSourceId'})
    )
