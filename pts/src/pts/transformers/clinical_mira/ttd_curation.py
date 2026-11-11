"""Curated fixes to TTD reports: stage downgrades and drug or disease link corrections.

Both files come from the curation repo and are keyed on the TTD drug id and the indication text
as TTD writes it, the key the TTD report id is built from.
"""

from io import BytesIO

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.clinical_mira.datasets import ClinicalReport
from pts.transformers.clinical_mira.providers.ttd import ttd_report_id
from pts.transformers.clinical_mira.schemas import ClinicalProvider

CURATED_UNKNOWN_PHASE = 'approved (curated: stage unknown)'
"""The `phaseFromSource` of a downgraded TTD report; it maps to the UNKNOWN stage."""

UNMAPPED = 'unmapped'
"""A correction cell that removes the id from that side of the report."""

DOWNGRADE_COLUMNS = ['ttdDrugId', 'drugFromSource', 'diseaseFromSource', 'reason']
CORRECTION_COLUMNS = ['ttdDrugId', 'drugFromSource', 'diseaseFromSource', 'drugId', 'diseaseId', 'reason']


def _read_curation_file(path: str, columns: list[str], what: str) -> pl.DataFrame:
    content, _ = StorageHandle(path).read()
    df = pl.read_csv(BytesIO(content), separator='\t', infer_schema=False, quote_char=None)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f'ttd {what} {path} misses columns {missing}')
    df = df.select(columns).with_columns(reportId=ttd_report_id(pl.col('ttdDrugId'), pl.col('diseaseFromSource')))
    duplicated = df.filter(pl.col('reportId').is_duplicated())
    if duplicated.height:
        keys = sorted(
            set(duplicated.select(pl.concat_str('ttdDrugId', 'diseaseFromSource', separator=' | ')).to_series())
        )
        raise ValueError(f'ttd {what} {path} has duplicate keys: {keys}')
    return df


def read_ttd_curation(downgrades: str, corrections: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read and validate the TTD stage downgrades and link corrections.

    Args:
        downgrades: TSV with `DOWNGRADE_COLUMNS`.
        corrections: TSV with `CORRECTION_COLUMNS`; an empty `drugId` or `diseaseId` leaves that side as grounded.

    Returns:
        The two tables, every column a string, each with the `reportId` of the TTD report it targets.

    Raises:
        ValueError: on a missing column, a key repeated in one file, or a correction with neither id.
    """
    down = _read_curation_file(downgrades, DOWNGRADE_COLUMNS, 'stage downgrades')
    fix = _read_curation_file(corrections, CORRECTION_COLUMNS, 'link corrections')
    empty = fix.filter(pl.col('drugId').is_null() & pl.col('diseaseId').is_null())
    if empty.height:
        raise ValueError(f'ttd link corrections with neither drugId nor diseaseId: {empty["ttdDrugId"].to_list()}')
    return down, fix


def _require_known(ids: pl.Series, known: pl.Series, what: str) -> None:
    curated = set(ids.drop_nulls().to_list()) - {UNMAPPED}
    unknown = sorted(curated - set(known.to_list()))
    if unknown:
        raise ValueError(f'ttd link corrections name {what} ids missing from the {what} index: {unknown}')


def _corrected_side(side: str, label: str, id_field: str, curated: str, is_target: pl.Expr) -> pl.Expr:
    first_label = pl.col(side).list.eval(pl.element().struct.field(label)).list.drop_nulls().list.first()
    curated_id = pl.when(pl.col(curated) == UNMAPPED).then(pl.lit(None, dtype=pl.String)).otherwise(pl.col(curated))
    replaced = pl.concat_list(pl.struct(first_label.alias(label), curated_id.alias(id_field)))
    return pl.when(is_target & pl.col(curated).is_not_null()).then(replaced).otherwise(pl.col(side)).alias(side)


def apply_ttd_curation(
    reports: ClinicalReport,
    downgrades: pl.DataFrame,
    corrections: pl.DataFrame,
    drug_ids: pl.Series,
    disease_ids: pl.Series,
) -> ClinicalReport:
    """Downgrade and correct the curated TTD reports.

    A downgraded report gets `CURATED_UNKNOWN_PHASE` as `phaseFromSource`, so its stage becomes UNKNOWN.
    A correction replaces the whole `drugs` or `diseases` list with one entry: the first label of that
    side and the curated id, or a null id for `UNMAPPED`. Reports of other providers are untouched.

    Args:
        reports: the reports with mapped drugs and diseases.
        downgrades: the stage downgrades from :func:`read_ttd_curation`.
        corrections: the link corrections from :func:`read_ttd_curation`.
        drug_ids: the ids of the drug index.
        disease_ids: the ids of the disease index.

    Returns:
        The curated reports, with `clinicalStage` recomputed.

    Raises:
        ValueError: if a correction names an id missing from the drug or disease index.
    """
    _require_known(corrections['drugId'], drug_ids, 'drug')
    _require_known(corrections['diseaseId'], disease_ids, 'disease')

    df = reports.df
    is_ttd = pl.col('provider') == ClinicalProvider.TTD.value
    ttd_ids = set(df.filter(is_ttd)['id'].to_list())
    for what, curated in (('stage downgrades', downgrades), ('link corrections', corrections)):
        matched = sum(report_id in ttd_ids for report_id in curated['reportId'].to_list())
        logger.info(f'ttd curation {what}: matched {matched}, unmatched {curated.height - matched}')

    fixes = corrections.select(
        pl.col('reportId').alias('id'),
        pl.col('drugId').alias('__curated_drug'),
        pl.col('diseaseId').alias('__curated_disease'),
    )
    curated = (
        df
        .with_columns(
            phaseFromSource=pl
            .when(is_ttd & pl.col('id').is_in(downgrades['reportId'].to_list()))
            .then(pl.lit(CURATED_UNKNOWN_PHASE))
            .otherwise(pl.col('phaseFromSource'))
        )
        .join(fixes, on='id', how='left', maintain_order='left')
        .with_columns(
            _corrected_side('drugs', 'drugFromSource', 'drugId', '__curated_drug', is_ttd),
            _corrected_side('diseases', 'diseaseFromSource', 'diseaseId', '__curated_disease', is_ttd),
        )
        .drop('__curated_drug', '__curated_disease')
    )
    return ClinicalReport(df=curated)
