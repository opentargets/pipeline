"""Project Score v2 (crispr) evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import divide
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

#: The Project Score v2 publication.
PMID = '38215750'


def linear_rescale(
    value: pl.Expr, in_range_min: float, in_range_max: float, out_range_min: float, out_range_max: float
) -> pl.Expr:
    """Rescale `value` linearly between two ranges, clamped to the output range.

    ```
    CASE
      WHEN value IS NULL OR isnan(value) THEN NULL
      ELSE LEAST(out_max, GREATEST(out_min,
        (out_max - out_min) * (CAST(value AS DOUBLE) - in_min) / (in_max - in_min) + out_min
      ))
    END
    ```
    """
    value = value.cast(pl.Float64)
    rescaled = (
        divide((out_range_max - out_range_min) * (value - in_range_min), in_range_max - in_range_min) + out_range_min
    )
    return pl.when(value.is_null() | value.is_nan()).then(None).otherwise(rescaled.clip(out_range_min, out_range_max))


SCORE = linear_rescale(pl.col('resourceScore'), 41.5, 100.0, 0.415, 1.0)

CRISPR = EvidenceDatasource(
    datasource_id='crispr',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId'],
    score=SCORE,
)


def parse_msi_status(status: pl.Expr) -> pl.Expr:
    """The microsatellite stability biomarker for an `MSI` or `MSS` status, else null."""
    return (
        pl
        .when(status == 'MSI')
        .then(pl.struct(name=pl.lit('MSI'), description=pl.lit('Microsatellite instable')))
        .when(status == 'MSS')
        .then(pl.struct(name=pl.lit('MSS'), description=pl.lit('Microsatellite stable')))
        .otherwise(None)
    )


def passport_cell_lines(cell_passport: pl.DataFrame, cell_line_mapping: pl.DataFrame) -> pl.DataFrame:
    """Cell lines from the Cell Model Passports, with their tissue mapped to UBERON.

    The top-level `name` has its dashes removed; the `diseaseCellLine` struct keeps the original.
    """
    return (
        cell_passport
        .select(
            pl.col('model_name').alias('name'),
            pl.col('model_id').alias('id'),
            pl.col('tissue').str.to_lowercase().alias('tissueFromSource'),
            pl.concat_list(parse_msi_status(pl.col('msi_status'))).alias('biomarkerList'),
        )
        .join(cell_line_mapping, on='tissueFromSource', how='left', maintain_order='left')
        .select(
            pl.col('name').str.replace_all('-', '', literal=True),
            'id',
            'biomarkerList',
            pl.struct(pl.col('tissueName').alias('tissue'), 'name', 'id', 'tissueId').alias('diseaseCellLine'),
        )
    )


def disease_cell_lines(
    cell_passport: pl.DataFrame, cell_line_mapping: pl.DataFrame, cell_types: pl.DataFrame
) -> pl.DataFrame:
    """The sorted, distinct `diseaseCellLine`s of each lower-cased Project Score cancer type."""
    cancer_types = cell_types.select(
        pl.col('CANCER_TYPE').str.to_lowercase().alias('cancerType'),
        pl.col('CMP_ID').alias('id'),
    )
    return (
        passport_cell_lines(cell_passport, cell_line_mapping)
        .join(cancer_types, on='id', how='left', maintain_order='left')
        .group_by('cancerType')
        .agg(pl.col('diseaseCellLine').unique().sort().alias('diseaseCellLines'))
    )


def _read(path: str, separator: str) -> pl.DataFrame:
    return scan_dataset(path, format='tsv' if separator == '\t' else 'csv', infer_schema=False).collect()


def generate(gene_scores: str, cell_types: str, cell_passport: str, cell_line_mapping: str) -> pl.LazyFrame:
    """Build Project Score evidence from its prioritised hits and cell line annotations.

    Args:
        gene_scores: location of the mapped Project Score hits tsv.
        cell_types: location of the Project Score cell line tsv.
        cell_passport: location of the Cell Model Passports csv.
        cell_line_mapping: location of the tissue to UBERON mapping csv.

    Returns:
        Raw Project Score evidence, one row per hit outside the pan-cancer analysis.
    """
    cell_lines = disease_cell_lines(
        _read(cell_passport, ','),
        _read(cell_line_mapping, ','),
        _read(cell_types, '\t'),
    )
    return (
        _read(gene_scores, '\t')
        .select(
            pl.col('cancerType').str.to_lowercase(),
            pl.col('targetSymbol').alias('targetFromSource'),
            pl.col('diseaseName').alias('diseaseFromSource'),
            pl.col('diseaseId').alias('diseaseFromSourceMappedId'),
            pl.col('PRIORITY').str.strip_chars().cast(pl.Float32, strict=False).alias('resourceScore'),
            pl.col('targetId').alias('targetFromSourceId'),
            pl.concat_list(pl.lit(PMID)).alias('literature'),
            pl.lit('crispr').alias('datasourceId'),
            pl.lit('affected_pathway').alias('datatypeId'),
        )
        .filter(pl.col('cancerType') != 'pancancer')
        .join(cell_lines, on='cancerType', how='left', maintain_order='left')
        .drop('cancerType')
        .lazy()
    )


def project_score(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Project Score evidence.

    Args:
        source: `gene_scores`, `cell_types`, `cell_passport`, `cell_line_mapping`, `disease`, `target`
            and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        generate(source['gene_scores'], source['cell_types'], source['cell_passport'], source['cell_line_mapping']),
        CRISPR,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
