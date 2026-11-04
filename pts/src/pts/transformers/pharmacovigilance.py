"""Pharmacovigilance signal detection from FAERS adverse event reports."""

import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.pharmacovigilance import faers_schema, pharmacovigilance_schema
from pts.transformers.utils.dataset import check_destinations_absent, scan_dataset, write_dataset

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
    return (
        pl.DataFrame({'event': text.splitlines()}, schema={'event': pl.String()})
        .select(normalise_event(pl.col('event')))
        .filter(pl.col('event') != '')
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
    reports = pl.read_json(content, schema=faers_schema).explode('results').unnest('results')
    del content
    return report_rows(reports, drugs, blacklist)


def likelihood_ratio() -> pl.Expr:
    """Log-likelihood ratio from the contingency columns ``A``, ``B``, ``C`` and ``D``."""
    a, b, c, d = (pl.col(name).cast(pl.Float64) for name in 'ABCD')
    return (
        a * (a.log() - (a + b).log())
        + c * (c.log() - (c + d).log())
        - (a + c) * ((a + c).log() - (a + b + c + d).log())
    )


def contingency_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Count distinct reports per drug, event and pair, and score each pair with its LLR."""
    total = rows['safetyreportid'].n_unique()
    return (
        rows
        .select('safetyreportid', 'chembl_id', 'event')
        .unique()
        .with_columns(
            pl.col('safetyreportid').n_unique().over('chembl_id').alias('drug_reports'),
            pl.col('safetyreportid').n_unique().over('event').alias('event_reports'),
            pl.col('safetyreportid').n_unique().over('chembl_id', 'event').alias('A'),
        )
        .select('chembl_id', 'event', 'A', 'drug_reports', 'event_reports')
        .unique()
        .with_columns(pl.col('A', 'drug_reports', 'event_reports').cast(pl.Int64))
        .with_columns(
            (pl.col('event_reports') - pl.col('A')).alias('B'),
            (pl.col('drug_reports') - pl.col('A')).alias('C'),
            (pl.lit(total, dtype=pl.Int64) - pl.col('drug_reports') - pl.col('event_reports') + pl.col('A')).alias('D'),
        )
        .with_columns(likelihood_ratio().alias('llr'))
        .filter(pl.col('llr').is_not_null() & pl.col('llr').is_not_nan())
        .select('chembl_id', 'event', 'A', 'B', 'C', 'D', 'drug_reports', 'event_reports', 'llr')
    )


def critical_value(permutations: int, n_j: int, n_i: np.ndarray, total: int, percentile: float) -> float:
    """Monte Carlo percentile of the maximum LLR over simulated events for one drug."""
    if n_j is None or total is None or n_i is None or len(n_i) == 0 or n_j <= 0 or total <= 0:
        return 0.0

    rng = np.random.default_rng(0)
    total_n = float(total)
    z = float(n_j)
    yi = np.array(n_i, dtype=np.float64)

    p = yi / total_n
    p_sum = p.sum()
    if p_sum <= 0:
        return 0.0
    p /= p_sum

    xi = rng.multinomial(int(n_j), p, size=permutations).astype(np.float64)

    t1_valid = (xi > 0) & (yi > 0)
    t2_valid = ((z - xi) > 0) & ((total_n - yi) > 0)
    term1 = np.where(t1_valid, xi * (np.log(np.where(t1_valid, xi, 1.0)) - np.log(yi)), 0.0)
    term2 = np.where(t2_valid, (z - xi) * (np.log(np.where(t2_valid, z - xi, 1.0)) - np.log(total_n - yi)), 0.0)

    max_llrs = (term1 + term2).max(axis=1)
    max_llrs += -z * np.log(z) + z * np.log(total_n)
    max_llrs = np.where(np.isfinite(max_llrs), max_llrs, 0.0)

    index = min(int(percentile * permutations), permutations - 1)
    return float(np.sort(max_llrs)[index])


def critical_values(table: pl.DataFrame, permutations: int, percentile: float) -> pl.DataFrame:
    """Compute one critical value per drug from the surviving rows of the contingency table."""
    drugs = (
        table
        .group_by('chembl_id')
        .agg(
            pl.col('drug_reports').first(),
            (pl.col('A') + pl.col('B') + pl.col('C') + pl.col('D')).first().alias('total'),
            pl.col('event_reports').sort().alias('n_i'),
        )
        .sort('chembl_id')
    )
    jobs = [
        (permutations, int(row['drug_reports']), np.array(row['n_i'], dtype=np.float64), int(row['total']), percentile)
        for row in drugs.iter_rows(named=True)
    ]
    with ThreadPoolExecutor() as pool:
        values = list(pool.map(lambda job: critical_value(*job), jobs))
    return pl.DataFrame(
        {'chembl_id': drugs['chembl_id'], 'critval': values}, schema={'chembl_id': pl.String(), 'critval': pl.Float64()}
    )


def significant_events(table: pl.DataFrame, critical: pl.DataFrame) -> pl.DataFrame:
    """Keep pairs whose LLR exceeds a positive critical value."""
    return (
        table
        .join(critical, on='chembl_id', how='inner')
        .filter((pl.col('llr') > pl.col('critval')) & (pl.col('critval') > 0))
        .select(
            'chembl_id',
            'event',
            pl.col('A').alias('count'),
            'llr',
            'critval',
            pl.lit('').alias('meddraCode'),
        )
        .cast(pharmacovigilance_schema)
        .sort('chembl_id', 'event')
    )



def read_archives(archives: list[str], drugs: pl.DataFrame, blacklist: pl.DataFrame, workers: int) -> pl.DataFrame:
    """Reduce every archive concurrently and combine them into distinct report rows."""
    frames: list[pl.DataFrame | None] = [None] * len(archives)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_archive, path, drugs, blacklist): i for i, path in enumerate(archives)}
        done = 0
        try:
            for future in as_completed(futures):
                index = futures[future]
                frame = future.result()
                frames[index] = frame
                done += 1
                logger.info(f'read archive {done}/{len(archives)}: {archives[index]} ({frame.height} rows)')
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return pl.concat([f for f in frames if f is not None]).unique()


def pharmacovigilance(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Detect drug-event safety signals in FAERS reports with a Monte Carlo likelihood ratio test.

    Args:
        source: ``drug_molecule`` dataset, ``fda_events`` directory of FAERS zip archives, and the
            ``blacklisted_events`` text file.
        destination: output dataset directory.
        settings: ``montecarlo_permutations``, ``montecarlo_percentile`` and ``archive_workers``.
        config: Otter config.

    Raises:
        ValueError: if no archive is found or the destination already exists.
    """
    permutations = int(settings['montecarlo_permutations'])
    percentile = float(settings['montecarlo_percentile'])
    workers = int(settings['archive_workers'])

    check_destinations_absent([destination])
    archives = sorted(StorageHandle(source['fda_events']).glob('**/*.zip'))
    if not archives:
        raise ValueError(f'no FAERS archives found in {source["fda_events"]}')

    molecules = scan_dataset(source['drug_molecule']).select('id', 'name', 'tradeNames', 'synonyms').collect()
    drugs = drug_names(molecules)
    blacklist = read_blacklist(StorageHandle(source['blacklisted_events']).read_text()[0])
    logger.info(f'{len(archives)} FAERS archives, {drugs.height} drug names, {blacklist.height} blacklisted events')

    rows = read_archives(archives, drugs, blacklist, workers)
    if rows.height == 0:
        raise ValueError(f'no report rows found in {len(archives)} FAERS archives')
    logger.info(f'{rows.height} distinct report rows from {rows["safetyreportid"].n_unique()} reports')

    table = contingency_table(rows)
    critical = critical_values(table, permutations, percentile)
    result = significant_events(table, critical)
    logger.info(f'{result.height} significant drug-event pairs from {table.height} scored')
    write_dataset(result, destination)
