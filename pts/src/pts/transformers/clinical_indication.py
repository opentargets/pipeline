"""Clinical indication dataset generation."""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.clinical_mira import ClinicalIndication
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.quality_flags import valid_rows


def clinical_indication(
    source: Path,
    destination: dict[str, Path],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate clinical indication dataset from clinical report data.

    Args:
        source: Path to clinical report data (output/clinical_report)
        destination: Dictionary with destination paths:
            - output: Path to write clinical indication data
            - excluded: Path to write excluded clinical reports
        settings: Dictionary with settings:
            - invalid_clinical_report_qc: List of QC reasons to exclude
        config: Config object (not used in this transformer)
    """
    logger.info(f'Source path: {source}')
    reports = scan_dataset(str(source)).collect()

    # Filter out clinical reports that fail QC
    reports, excluded = valid_rows(reports, settings.get('invalid_clinical_report_qc', []))

    logger.info(f'Writing excluded clinical reports to {destination["excluded"]}')
    write_dataset(excluded, str(destination['excluded']))

    indications = (
        ClinicalIndication
        .from_report(reports)
        .df.filter(pl.col('mappingStatus') == 'FULLY_MAPPED')
        .drop('drugName', 'diseaseName', 'mappingStatus')
    )

    logger.info(f'Destination path: {destination["output"]}')
    write_dataset(indications, str(destination['output']))
