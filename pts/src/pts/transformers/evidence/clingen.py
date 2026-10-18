"""ClinGen gene-disease validity evidence.

Polars port of the `pyspark clingen` parser together with its post-processing step.
Diseases are mapped with the OnToma lookup tables through `add_efo_mapping`, by label and by
MONDO id.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import confidence_score
from pts.transformers.evidence.utils.ontology import add_efo_mapping
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_strings import spark_trim

#: The file's columns, in order. Its own header row sits among the rows skipped below.
COLUMNS = [
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
]

#: Three metadata rows, then the header between two `+++` separator rows.
SKIPPED_ROWS = 6

CONFIDENCE_SCORES: dict[str, float] = {
    'No Known Disease Relationship': 0.01,
    'Refuted': 0.01,
    'Disputed': 0.01,
    'Limited': 0.01,
    'Moderate': 0.5,
    'Strong': 1.0,
    'Definitive': 1.0,
}

CLINGEN = EvidenceDatasource(
    datasource_id='clingen',
    # order feeds the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'diseaseFromSource',
        'studyId',
        'allelicRequirements',
    ],
    score=confidence_score(CONFIDENCE_SCORES),
)


def _release_date(timestamp: pl.Expr) -> pl.Expr:
    """`date_format(timestamp, 'yyyy-MM-dd')` in a UTC spark session.

    The file mixes UTC timestamps (`2024-03-14T16:00:00.000Z`) with zone-less ones
    (`2018-06-07T00:00:00`), which spark reads as session-local, so in UTC both give their own
    date. Matches the release's `releaseDate` on all 3,678 rows of the 26.09 file. Anything else
    does not parse, and is null as in spark.
    """
    utc = timestamp.str.strptime(pl.Datetime('us', 'UTC'), '%Y-%m-%dT%H:%M:%S%.fZ', strict=False)
    local = timestamp.str.strptime(pl.Datetime('us'), '%Y-%m-%dT%H:%M:%S%.f', strict=False)
    return pl.coalesce(utc, local.dt.replace_time_zone('UTC')).dt.strftime('%Y-%m-%d')


def parse_clingen(path: str) -> pl.LazyFrame:
    """Raw ClinGen evidence from the gene-disease validity CSV, before disease mapping."""
    raw = scan_dataset(
        path,
        format='csv',
        has_header=False,
        skip_rows=SKIPPED_ROWS,
        new_columns=COLUMNS,
        infer_schema=False,
    )
    return raw.select(
        datasourceId=pl.lit('clingen'),
        datatypeId=pl.lit('genetic_literature'),
        targetFromSourceId=spark_trim(pl.col('GENE SYMBOL')),
        diseaseFromSource=pl.col('DISEASE LABEL'),
        diseaseFromSourceId=pl.col('DISEASE ID (MONDO)'),
        allelicRequirements=pl.concat_list('MOI'),
        urls=pl.concat_list(pl.struct(url=pl.col('ONLINE REPORT'))),
        confidence=pl.col('CLASSIFICATION'),
        releaseDate=_release_date(pl.col('CLASSIFICATION DATE')),
        studyId=pl.col('GCEP'),
    )


def clingen(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process ClinGen evidence.

    Args:
        source: `evidence`, `ontoma_disease_label_lut`, `ontoma_disease_id_lut`, `disease` and
            `target` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    raw = add_efo_mapping(
        parse_clingen(source['evidence']),
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    ).collect()
    run_evidence(raw.lazy(), CLINGEN, disease=source['disease'], target=source['target'], destination=destination)
