"""Curated fixes to the diseases of ChEMBL indication reports.

The file comes from the curation repo. Each row names a ChEMBL indication report by its published id (the
DailyMed set id, EMA product, ATC code, or the hash for INN, FDA and USAN references) and one disease
ChEMBL grounded it to, and either removes that disease from the report or replaces its id.
"""

from io import BytesIO

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.clinical_mira.datasets import ClinicalReport
from pts.transformers.clinical_mira.schemas import ClinicalProvider

REMOVED = 'unmapped'
"""A `correctedDiseaseId` that removes the disease from the report."""

CORRECTION_COLUMNS = ['reportId', 'drugFromSource', 'diseaseFromSource', 'diseaseId', 'correctedDiseaseId', 'reason']


def read_chembl_curation(corrections: str) -> pl.DataFrame:
    """Read and validate the ChEMBL indication disease corrections.

    Args:
        corrections: TSV with `CORRECTION_COLUMNS`.

    Returns:
        The corrections, every column a string, with `reportId` lowercased.

    Raises:
        ValueError: on a missing column, an empty key or correction, or a report and disease repeated.
    """
    content, _ = StorageHandle(corrections).read()
    df = pl.read_csv(BytesIO(content), separator='\t', infer_schema=False, quote_char=None)
    missing = [c for c in CORRECTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'chembl indication corrections {corrections} misses columns {missing}')
    df = df.select(CORRECTION_COLUMNS).with_columns(pl.col('reportId').str.to_lowercase())
    empty = df.filter(pl.any_horizontal(pl.col('reportId', 'diseaseId', 'correctedDiseaseId').is_null()))
    if empty.height:
        raise ValueError(
            f'chembl indication corrections {corrections} need reportId, diseaseId and correctedDiseaseId: '
            f'{empty["reportId"].to_list()}'
        )
    duplicated = df.filter(pl.struct('reportId', 'diseaseId').is_duplicated())
    if duplicated.height:
        keys = sorted(set(duplicated.select(pl.concat_str('reportId', 'diseaseId', separator=' | ')).to_series()))
        raise ValueError(f'chembl indication corrections {corrections} has duplicate keys: {keys}')
    return df


def apply_chembl_curation(
    reports: ClinicalReport,
    corrections: pl.DataFrame,
    disease_ids: pl.Series,
) -> ClinicalReport:
    """Remove or re-ground the curated diseases of ChEMBL indication reports.

    A removed disease is dropped from the report's `diseases`; a report left without any disease is
    dropped. A replaced disease keeps its `diseaseFromSource` and takes the curated id. Reports of other
    providers, and diseases of a report that no row names, are untouched.

    Args:
        reports: ChEMBL indication reports, with the disease ids ChEMBL gives.
        corrections: the corrections from :func:`read_chembl_curation`.
        disease_ids: the disease ids a correction may name, as `known_disease_ids` gives them.

    Returns:
        The curated reports.

    Raises:
        ValueError: if a correction names a disease id missing from `disease_ids`.
    """
    named = set(corrections['correctedDiseaseId'].to_list()) - {REMOVED}
    if unknown := sorted(named - set(disease_ids.to_list())):
        raise ValueError(f'chembl indication corrections name disease ids missing from the disease index: {unknown}')

    df = reports.df.with_row_index('__row')
    is_chembl = pl.col('provider') == ClinicalProvider.CHEMBL.value
    links = (
        df
        .filter(is_chembl)
        .select('__row', report=pl.col('id').str.to_lowercase(), disease=pl.col('diseases'))
        .explode('disease')
        .with_columns(diseaseId=pl.col('disease').struct.field('diseaseId'))
    )
    matched = corrections.join(links, left_on=['reportId', 'diseaseId'], right_on=['report', 'diseaseId'], how='semi')
    unmatched = corrections.height - matched.height
    logger.info(f'chembl indication corrections: matched {matched.height}, unmatched {unmatched}')

    curated = (
        links
        .join(
            corrections.select('reportId', 'diseaseId', 'correctedDiseaseId'),
            left_on=['report', 'diseaseId'],
            right_on=['reportId', 'diseaseId'],
            how='left',
        )
        .filter(pl.col('correctedDiseaseId').fill_null('') != REMOVED)
        .with_columns(
            disease=pl
            .when(pl.col('correctedDiseaseId').is_not_null())
            .then(pl.col('disease').struct.with_fields(diseaseId=pl.col('correctedDiseaseId')))
            .otherwise(pl.col('disease'))
        )
        .group_by('__row')
        .agg(pl.col('disease').drop_nulls().unique().alias('__curated'))
    )
    touched = matched.join(links, left_on=['reportId', 'diseaseId'], right_on=['report', 'diseaseId'])['__row']
    out = (
        df
        .join(curated, on='__row', how='left', maintain_order='left')
        .with_columns(
            diseases=pl.when(pl.col('__row').is_in(touched.implode())).then(pl.col('__curated')).otherwise('diseases')
        )
    )
    emptied = out.filter(pl.col('__row').is_in(touched.implode()) & (pl.col('diseases').list.len().fill_null(0) == 0))
    logger.info(f'chembl indication corrections: {emptied.height} reports left without a disease are dropped')
    return ClinicalReport(
        df=out.filter(~pl.col('__row').is_in(emptied['__row'].implode())).drop('__row', '__curated')
    )
