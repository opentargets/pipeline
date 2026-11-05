"""Select the publications of a pool's trials and fetch their abstracts from Europe PMC."""

from __future__ import annotations

import html
import io
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError

from pts.transformers.clinical_mira.providers.aact import read_aact_tables

if TYPE_CHECKING:
    from collections.abc import Callable

    from otter.config.model import Config

CHUNK_SIZE = 40
ATTEMPTS = 3
PAUSE_SECONDS = 5
TIMEOUT_SECONDS = 60
USER_AGENT = 'opentargets-pts'
MAX_CONSECUTIVE_FAILED_CHUNKS = 3

ABSTRACT_SCHEMA = pl.Schema({'pmid': pl.String, 'title': pl.String, 'abstract': pl.String})
STORE_SCHEMA = pl.Schema({
    'pmid': pl.String,
    'title': pl.String,
    'abstract': pl.String,
    'fetched_at': pl.Datetime(time_unit='us', time_zone='UTC'),
})


def trial_papers(archive: str, scratch_root: str, reference_types: list[str]) -> pl.DataFrame:
    """Return the ``(nct_id, pmid)`` links of the given reference types whose PMID is 1 to 10 digits."""
    references = read_aact_tables(archive, ['study_references'], scratch_root)['study_references']
    return (
        references
        .filter(pl.col('reference_type').is_in(reference_types) & pl.col('pmid').str.contains(r'^\d{1,10}$'))
        .select('nct_id', 'pmid')
        .unique(maintain_order=True)
    )


def select_papers(trials: pl.DataFrame, papers: pl.DataFrame, per_pool: int) -> pl.DataFrame:
    """Return the newest ``per_pool`` publications of each drug's trials, with the smallest trial linked to each."""
    return (
        trials
        .select('nct_id', 'drug_name')
        .join(papers.select('nct_id', 'pmid'), on='nct_id', how='inner', maintain_order='left')
        .group_by('drug_name', 'pmid')
        .agg(pl.col('nct_id').min())
        .sort(pl.col('pmid').cast(pl.Int64), descending=True)
        .group_by('drug_name', maintain_order=True)
        .head(per_pool)
        .select('drug_name', 'pmid', 'nct_id')
        .sort('drug_name', pl.col('pmid').cast(pl.Int64), descending=[False, True])
    )


def paper_id(pmid: str) -> str:
    """Return the id a publication is cited by."""
    return f'pmid{pmid}'


def _clean(text: str | None) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html.unescape(text or ''))).strip()


def paper_text(pmid: str, title: str | None, abstract: str, abstract_chars: int) -> str:
    """Render a publication the way a trial is rendered in a pool's text."""
    return (
        f'Trial ID: {paper_id(pmid)}\nOfficial Title: {_clean(title)}\nDescription: {_clean(abstract)[:abstract_chars]}'
    )


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        return json.load(response)


def _fetch_chunk(chunk: list[str], url: str, get: Callable[[str], dict[str, Any]]) -> list[dict[str, Any]] | None:
    query = 'SRC:MED AND (' + ' OR '.join(f'EXT_ID:{p}' for p in chunk) + ')'
    params = urllib.parse.urlencode({'query': query, 'resultType': 'core', 'format': 'json', 'pageSize': 100})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return get(f'{url}?{params}').get('resultList', {}).get('result', [])
        except Exception as exc:
            logger.warning(f'Europe PMC request for {len(chunk)} PMIDs failed (attempt {attempt}/{ATTEMPTS}): {exc}')
            if attempt < ATTEMPTS:
                time.sleep(PAUSE_SECONDS)
    return None


def fetch_abstracts(
    pmids: list[str],
    url: str,
    get: Callable[[str], dict[str, Any]] | None = None,
) -> pl.DataFrame:
    """Fetch the title and abstract of each PMID that Europe PMC returns with an abstract.

    A chunk of PMIDs that fails every attempt is logged and left out of the result. After
    :py:data:`MAX_CONSECUTIVE_FAILED_CHUNKS` such chunks in a row the remaining PMIDs are not
    requested, and what was fetched is returned.

    Args:
        pmids: The PubMed ids to look up.
        url: The Europe PMC search endpoint.
        get: Returns the parsed JSON of a URL; defaults to an HTTP fetch.

    Returns:
        pl.DataFrame: Columns ``pmid``, ``title`` and ``abstract``.
    """
    fetch = get or _get_json
    rows: list[dict[str, str | None]] = []
    failed_in_a_row = 0
    for start in range(0, len(pmids), CHUNK_SIZE):
        if failed_in_a_row >= MAX_CONSECUTIVE_FAILED_CHUNKS:
            logger.error(
                f'Europe PMC failed {failed_in_a_row} chunks in a row: '
                f'skipping the remaining {len(pmids) - start} of {len(pmids)} PMIDs'
            )
            break
        results = _fetch_chunk(pmids[start : start + CHUNK_SIZE], url, fetch)
        if results is None:
            failed_in_a_row += 1
            logger.error(f'no abstracts for {len(pmids[start : start + CHUNK_SIZE])} PMIDs starting at {pmids[start]}')
            continue
        failed_in_a_row = 0
        rows.extend(
            {'pmid': r.get('pmid'), 'title': r.get('title'), 'abstract': r.get('abstractText')}
            for r in results
            if (r.get('abstractText') or '').strip() and r.get('pmid')
        )
    return pl.DataFrame(rows, schema=ABSTRACT_SCHEMA).unique('pmid', keep='first', maintain_order=True)


def read_abstracts(store_uri: str, config: Config) -> pl.DataFrame:
    """Read the cached abstracts, the newest fetch of each PMID winning."""
    try:
        files = sorted(StorageHandle(f'{store_uri}/moa/papers', config=config).glob('*.parquet'))
    except NotFoundError:
        files = []
    frames = []
    for file in files:
        data, _ = StorageHandle(file, config=config).read()
        frames.append(pl.read_parquet(io.BytesIO(data)).select(STORE_SCHEMA.names()).cast(STORE_SCHEMA))
    if not frames:
        return pl.DataFrame(schema=STORE_SCHEMA)
    return pl.concat(frames).sort('fetched_at', maintain_order=True).unique('pmid', keep='last', maintain_order=True)


def write_abstracts(df: pl.DataFrame, store_uri: str, run_id: str, config: Config) -> None:
    """Cache the fetched abstracts as the file of one run."""
    stamped = df.select(ABSTRACT_SCHEMA.names()).with_columns(
        fetched_at=pl.lit(datetime.now(UTC), dtype=STORE_SCHEMA['fetched_at'])
    )
    buffer = io.BytesIO()
    stamped.write_parquet(buffer, compression='zstd')
    StorageHandle(f'{store_uri}/moa/papers/{run_id}.parquet', config=config).write(buffer.getvalue())
