"""Curated fixes to the diseases of indication reports from one provider: ChEMBL or EMA.

Each file comes from the curation repo. A row names a report by its published id and acts on its diseases:

- `diseaseId` with `correctedDiseaseId` = `unmapped`: removes that disease from the report;
- `diseaseId` with a disease id in `correctedDiseaseId`: replaces it;
- an empty `diseaseId` with a disease id in `correctedDiseaseId`: adds that disease, labelled `diseaseFromSource`.
"""

from io import BytesIO

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.clinical_mira.datasets import ClinicalReport

REMOVED = 'unmapped'
"""A `correctedDiseaseId` that removes the disease from the report."""

CORRECTION_COLUMNS = ['reportId', 'drugFromSource', 'diseaseFromSource', 'diseaseId', 'correctedDiseaseId', 'reason']

_DISEASE = pl.Struct({'diseaseFromSource': pl.String, 'diseaseId': pl.String})


def read_indication_corrections(path: str) -> pl.DataFrame:
    """Read and validate a file of indication disease corrections.

    Args:
        path: TSV with `CORRECTION_COLUMNS`.

    Returns:
        The corrections, every column a string, with `reportId` lowercased.

    Raises:
        ValueError: on a missing column, a row without `reportId` or `correctedDiseaseId`, an addition that
            removes, or a report and disease named twice.
    """
    content, _ = StorageHandle(path).read()
    df = pl.read_csv(BytesIO(content), separator='\t', infer_schema=False, quote_char=None)
    missing = [c for c in CORRECTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'indication corrections {path} misses columns {missing}')
    df = df.select(CORRECTION_COLUMNS).with_columns(pl.col('reportId').str.to_lowercase())
    empty = df.filter(pl.any_horizontal(pl.col('reportId', 'correctedDiseaseId').is_null()))
    if empty.height:
        raise ValueError(
            f'indication corrections {path} need reportId and correctedDiseaseId: {empty["reportId"].to_list()}'
        )
    bad_additions = df.filter(pl.col('diseaseId').is_null() & (pl.col('correctedDiseaseId') == REMOVED))
    if bad_additions.height:
        raise ValueError(
            f'indication corrections {path} remove without a diseaseId: {bad_additions["reportId"].to_list()}'
        )
    key = pl.coalesce('diseaseId', 'correctedDiseaseId')
    duplicated = df.filter(pl.struct(pl.col('reportId'), key).is_duplicated())
    if duplicated.height:
        keys = sorted(set(duplicated.select(pl.concat_str(pl.col('reportId'), key, separator=' | ')).to_series()))
        raise ValueError(f'indication corrections {path} has duplicate keys: {keys}')
    return df


def apply_indication_corrections(
    reports: ClinicalReport,
    corrections: pl.DataFrame,
    disease_ids: pl.Series,
    provider: str,
) -> ClinicalReport:
    """Remove, re-ground or add curated diseases of one provider's indication reports.

    A replaced disease keeps its `diseaseFromSource`. An added disease is skipped when the report already
    has it. A report left without any disease is dropped. Reports of other providers, and diseases of a
    report that no row names, are untouched.

    Args:
        reports: the reports, with disease ids.
        corrections: the corrections from :func:`read_indication_corrections`.
        disease_ids: the disease ids a correction may name, as `known_disease_ids` gives them.
        provider: the provider whose reports the corrections target.

    Returns:
        The curated reports.

    Raises:
        ValueError: if a correction names a disease id missing from `disease_ids`.
    """
    named = set(corrections['correctedDiseaseId'].to_list()) - {REMOVED}
    if unknown := sorted(named - set(disease_ids.to_list())):
        raise ValueError(
            f'{provider} indication corrections name disease ids missing from the disease index: {unknown}'
        )

    df = reports.df.with_row_index('__row')
    targets = df.filter(pl.col('provider') == provider).select('__row', report=pl.col('id').str.to_lowercase())
    links = (
        df
        .join(targets, on='__row')
        .select('__row', 'report', disease=pl.col('diseases'))
        .explode('disease', empty_as_null=True)
        .with_columns(diseaseId=pl.col('disease').struct.field('diseaseId'))
    )
    edits = corrections.filter(pl.col('diseaseId').is_not_null())
    additions = corrections.filter(pl.col('diseaseId').is_null())
    matched_edits = edits.join(links, left_on=['reportId', 'diseaseId'], right_on=['report', 'diseaseId'], how='semi')
    matched_additions = additions.join(targets, left_on='reportId', right_on='report', how='semi')
    matched = matched_edits.height + matched_additions.height
    logger.info(f'{provider} indication corrections: matched {matched}, unmatched {corrections.height - matched}')

    kept = (
        links
        .join(
            edits.select('reportId', 'diseaseId', 'correctedDiseaseId'),
            left_on=['report', 'diseaseId'],
            right_on=['reportId', 'diseaseId'],
            how='left',
        )
        .filter(pl.col('disease').is_not_null(), pl.col('correctedDiseaseId').fill_null('') != REMOVED)
        .rechunk()
        .select(
            '__row',
            disease=pl.struct(
                pl.col('disease').struct.field('diseaseFromSource').alias('diseaseFromSource'),
                pl.coalesce('correctedDiseaseId', pl.col('disease').struct.field('diseaseId')).alias('diseaseId'),
            ),
        )
    )
    added = matched_additions.join(targets, left_on='reportId', right_on='report').select(
        '__row',
        disease=pl.struct(
            pl.col('diseaseFromSource').str.to_lowercase().alias('diseaseFromSource'),
            pl.col('correctedDiseaseId').alias('diseaseId'),
        ).cast(_DISEASE),
    )
    touched = pl.concat([
        matched_edits.join(links, left_on=['reportId', 'diseaseId'], right_on=['report', 'diseaseId']).select('__row'),
        added.select('__row'),
    ])['__row']
    curated = (
        pl
        .concat([kept.with_columns(pl.col('disease').cast(_DISEASE)), added])
        .drop_nulls('disease')
        .with_columns(__disease_id=pl.col('disease').struct.field('diseaseId'))
        .unique(['__row', '__disease_id'], keep='first', maintain_order=True)
        .group_by('__row')
        .agg(pl.col('disease').alias('__curated'))
    )
    is_touched = pl.col('__row').is_in(touched.implode())
    out = (
        df
        .join(curated, on='__row', how='left', maintain_order='left')
        .rechunk()
        .with_columns(diseases=pl.when(is_touched).then(pl.col('__curated')).otherwise('diseases'))
    )
    emptied = out.filter(is_touched & (pl.col('diseases').list.len().fill_null(0) == 0))
    logger.info(f'{provider} indication corrections: {emptied.height} reports left without a disease are dropped')
    return ClinicalReport(df=out.filter(~pl.col('__row').is_in(emptied['__row'].implode())).drop('__row', '__curated'))
