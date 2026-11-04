"""Pharmacovigilance signal detection from FAERS adverse event reports."""

import zipfile

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.pharmacovigilance import faers_schema

_ROW_COLUMNS = ['safetyreportid', 'chembl_id', 'event']
_EMPTY_LIST = pl.lit([], dtype=pl.List(pl.String()))


def normalise_event(expr: pl.Expr) -> pl.Expr:
    """Lowercase, turn ``^`` into ``'`` and strip surrounding spaces only."""
    return expr.str.to_lowercase().str.replace_all('^', "'", literal=True).str.strip_chars(' ')


def drug_names(drug_molecule: pl.DataFrame) -> pl.DataFrame:
    """Build distinct (chembl_id, drug_name) pairs from names, trade names and synonyms."""

    def labels(column: str) -> pl.Expr:
        return pl.col(column).list.eval(pl.element().struct.field('label')).fill_null(_EMPTY_LIST)

    return (
        drug_molecule
        .select(
            pl.col('id').alias('chembl_id'),
            pl.concat_list(
                labels('tradeNames'),
                pl.concat_list(pl.col('name')),
                labels('synonyms'),
            ).alias('drug_name'),
        )
        .explode('drug_name')
        .filter(pl.col('drug_name').is_not_null())
        .with_columns(pl.col('drug_name').str.to_lowercase())
        .unique(maintain_order=True)
    )


def read_blacklist(text: str) -> pl.DataFrame:
    """Normalise the blacklisted event terms, one per line."""
    return pl.DataFrame({'event': text.splitlines()}, schema={'event': pl.String()}).select(
        normalise_event(pl.col('event'))
    )


def report_rows(reports: pl.DataFrame, drugs: pl.DataFrame, blacklist: pl.DataFrame) -> pl.DataFrame:
    """Reduce FAERS reports to distinct (safetyreportid, chembl_id, event) rows."""
    kept = (
        reports
        .filter(
            pl.col('safetyreportid').is_not_null()
            & pl.col('seriousnessdeath').fill_null('0').eq('0')
            & pl.col('primarysource').struct.field('qualification').is_in(['1', '2', '3'])
        )
        .select(
            'safetyreportid',
            pl.col('patient').struct.field('reaction').alias('reaction'),
            pl.col('patient').struct.field('drug').alias('drug'),
        )
        .with_row_index('_report')
    )

    events = (
        kept
        .select('_report', 'safetyreportid', 'reaction')
        .explode('reaction')
        .rechunk()
        .select(
            '_report',
            'safetyreportid',
            normalise_event(pl.col('reaction').struct.field('reactionmeddrapt')).alias('event'),
        )
        .filter(pl.col('event').is_not_null())
        .join(blacklist, on='event', how='anti')
        .unique()
    )

    linked = (
        kept
        .select('_report', 'drug')
        .explode('drug')
        .rechunk()
        .filter(pl.col('drug').struct.field('drugcharacterization').eq('1'))
        .select(
            '_report',
            pl.concat_list(
                pl.col('drug').struct.field('openfda').struct.field('brand_name').fill_null(_EMPTY_LIST),
                pl.concat_list(pl.col('drug').struct.field('medicinalproduct')),
                pl.col('drug').struct.field('openfda').struct.field('generic_name').fill_null(_EMPTY_LIST),
                pl.col('drug').struct.field('openfda').struct.field('substance_name').fill_null(_EMPTY_LIST),
            ).alias('drug_name'),
        )
        .explode('drug_name')
        .with_columns(pl.col('drug_name').str.to_lowercase())
        .filter(pl.col('drug_name').is_not_null() & (pl.col('drug_name').str.len_chars() > 0))
        .join(drugs, on='drug_name', how='inner')
        .select('_report', 'chembl_id')
        .unique()
    )

    return events.join(linked, on='_report', how='inner').select(_ROW_COLUMNS).unique()


def read_archive(path: str, drugs: pl.DataFrame, blacklist: pl.DataFrame) -> pl.DataFrame:
    """Reduce one FAERS zip archive to distinct report rows."""
    with StorageHandle(path).open('rb') as handle, zipfile.ZipFile(handle) as archive:
        with archive.open(archive.namelist()[0]) as entry:
            content = entry.read()
    parsed = pl.read_json(content, schema=faers_schema)
    del content
    reports = parsed.explode('results').unnest('results')
    return report_rows(reports, drugs, blacklist)
