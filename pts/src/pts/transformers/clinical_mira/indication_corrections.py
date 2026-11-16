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
    disease_index: pl.DataFrame,
    provider: str,
) -> ClinicalReport:
    """Remove, re-ground or add curated diseases of one provider's indication reports.

    A replaced disease keeps its `diseaseFromSource`. An added disease is skipped when the report already
    has it. A report left without any disease is dropped. Reports of other providers, and diseases of a
    report that no row names, are untouched. A report's obsolete disease id is matched as the current id
    that lists it in `obsoleteTerms`, as `validate_disease` later resolves it.

    Args:
        reports: the reports, with disease ids.
        corrections: the corrections from :func:`read_indication_corrections`.
        disease_index: the disease index's `id` and `obsoleteTerms`; a correction may name a current or an
            obsolete id.
        provider: the provider whose reports the corrections target.

    Returns:
        The curated reports.

    Raises:
        ValueError: if a correction names a disease id missing from the disease index.
    """
    aliases = (
        disease_index
        .select(currentId='id', obsoleteId='obsoleteTerms')
        .explode('obsoleteId', empty_as_null=True)
        .drop_nulls('obsoleteId')
        .unique('obsoleteId', keep='first', maintain_order=True)
    )
    current = disease_index['id'].implode()
    named = set(corrections['correctedDiseaseId'].to_list()) - {REMOVED}
    if unknown := sorted(named - set(disease_index['id'].to_list()) - set(aliases['obsoleteId'].to_list())):
        raise ValueError(
            f'{provider} indication corrections name disease ids missing from the disease index: {unknown}'
        )

    df = reports.df.with_row_index('__row')
    diseases_type = df.schema['diseases']
    disease_type = diseases_type.inner if isinstance(diseases_type, pl.List) else None
    if not isinstance(disease_type, pl.Struct):
        raise TypeError(f'reports have diseases of type {diseases_type}, expected a list of structs')
    field_order = [field.name for field in disease_type.fields]

    def disease(fields: dict[str, pl.Expr]) -> pl.Expr:
        return pl.struct(*[fields[name].alias(name) for name in field_order]).cast(disease_type)

    targets = df.filter(pl.col('provider') == provider).select('__row', report=pl.col('id').str.to_lowercase())
    links = (
        df
        .join(targets, on='__row')
        .select('__row', 'report', disease=pl.col('diseases'))
        .explode('disease', empty_as_null=True)
        .with_columns(diseaseId=pl.col('disease').struct.field('diseaseId'))
        .join(aliases, left_on='diseaseId', right_on='obsoleteId', how='left', maintain_order='left')
        .with_columns(
            matchId=pl
            .when(pl.col('diseaseId').is_in(current))
            .then(pl.col('diseaseId'))
            .otherwise(pl.coalesce('currentId', 'diseaseId'))
        )
    )
    edits = corrections.filter(pl.col('diseaseId').is_not_null())
    additions = corrections.filter(pl.col('diseaseId').is_null())
    matched_edits = edits.join(links, left_on=['reportId', 'diseaseId'], right_on=['report', 'matchId'], how='semi')
    matched_additions = additions.join(targets, left_on='reportId', right_on='report', how='semi')
    matched = matched_edits.height + matched_additions.height
    logger.info(f'{provider} indication corrections: matched {matched}, unmatched {corrections.height - matched}')

    kept = (
        links
        .join(
            edits.select('reportId', 'diseaseId', 'correctedDiseaseId'),
            left_on=['report', 'matchId'],
            right_on=['reportId', 'diseaseId'],
            how='left',
        )
        .filter(pl.col('disease').is_not_null(), pl.col('correctedDiseaseId').fill_null('') != REMOVED)
        .rechunk()
        .select(
            '__row',
            disease=disease({
                'diseaseFromSource': pl.col('disease').struct.field('diseaseFromSource'),
                'diseaseId': pl.coalesce('correctedDiseaseId', pl.col('disease').struct.field('diseaseId')),
            }),
        )
    )
    added = matched_additions.join(targets, left_on='reportId', right_on='report').select(
        '__row',
        disease=disease({
            'diseaseFromSource': pl.col('diseaseFromSource').str.to_lowercase(),
            'diseaseId': pl.col('correctedDiseaseId'),
        }),
    )
    touched = pl.concat([
        matched_edits.join(links, left_on=['reportId', 'diseaseId'], right_on=['report', 'matchId']).select('__row'),
        added.select('__row'),
    ])['__row']
    curated = (
        pl
        .concat([kept, added])
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
