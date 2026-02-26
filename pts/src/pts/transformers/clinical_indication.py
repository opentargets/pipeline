"""Clinical indication dataset generation."""

from pathlib import Path
from typing import Any

import polars as pl
from clinical_mining.dataset import ClinicalIndication
from loguru import logger


def clinical_indication(source: Path, destination: dict[str, Path], settings: dict[str, Any]) -> None:
    """Generate clinical indication dataset from clinical report data.

    Args:
        source: Path to clinical report data (output/clinical_report)
        destination: Dictionary with destination paths:
            - output: Path to write clinical indication data
            - excluded: Path to write excluded clinical reports
        settings: Dictionary with settings:
            - invalid_clinical_report_qc: List of QC reasons to exclude
    """
    logger.info(f'Source path: {source}')
    reports = pl.read_parquet(source)

    # Filter out clinical reports that fail QC
    invalid_qc_reasons = settings.get('invalid_clinical_report_qc', [])
    if invalid_qc_reasons:
        has_invalid_qc = pl.col('qualityControls').list.set_intersection(invalid_qc_reasons).list.len() > 0
        excluded = reports.filter(has_invalid_qc)
        reports = reports.filter(~has_invalid_qc)
    else:
        excluded = reports.filter(pl.lit(False))

    logger.info(f'Writing excluded clinical reports to {destination["excluded"]}')
    excluded.write_parquet(destination['excluded'], mkdir=True)

    indications = (
        ClinicalIndication.from_report(reports)
        .df.filter(pl.col('mappingStatus') == 'FULLY_MAPPED')
        .drop('drugName', 'diseaseName', 'mappingStatus')
    )

    logger.info(f'Destination path: {destination["output"]}')
    indications.write_parquet(destination['output'], mkdir=True)
