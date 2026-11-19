"""Load the release's modelled outputs into a staging dataset, then swap them into prod."""

from collections.abc import Callable, Iterable
from typing import Any

from google.cloud import bigquery
from loguru import logger
from ot_datamodels.discovery import discover_datasets
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import OtterError


class BigQueryError(OtterError):
    """A BigQuery load, check or swap failed."""


#: One-row table holding the release version, swapped with the outputs.
RELEASE_TABLE = 'ot_release'
RELEASE_SCHEMA = [bigquery.SchemaField('release', 'STRING')]


def published_tables(release_uri: str) -> list[str]:
    """Modelled datasets with at least one `*.parquet` in `<release_uri>/output/<name>/`."""
    found = StorageHandle(f'{release_uri.rstrip("/")}/output').glob('*/*.parquet')
    folders = {path.rstrip('/').rsplit('/', 2)[-2] for path in found}
    models = set(discover_datasets())
    for name in sorted(models - folders):
        logger.warning(f'{name} has a model but no output, skipped')
    for name in sorted(folders - models):
        logger.warning(f'output/{name} has no model, not published')
    return sorted(models & folders)


class BigQueryRelease:
    """Staging and prod datasets of one project; `client` is a `bigquery.Client` or a stand-in."""

    def __init__(self, client: Any, project: str, dataset: str, staging: str, location: str) -> None:
        self._client = client
        self._project = project
        self._dataset = dataset
        self._staging = staging
        self._location = location

    def publish(self, tables: list[str], release_uri: str, release: str) -> None:
        """Load and check staging, then swap. Prod is untouched unless every table staged."""
        if not tables:
            raise BigQueryError('no tables to publish')
        if RELEASE_TABLE in tables:
            raise BigQueryError(f'{RELEASE_TABLE} is reserved for the release version')
        swapped = [*tables, RELEASE_TABLE]
        self.load_staging(tables, release_uri, release)
        self.check_staging(swapped)
        self.swap(swapped)

    def load_staging(self, tables: Iterable[str], release_uri: str, release: str) -> None:
        """Load each table's parquet, and the `ot_release` row, into staging, replacing them."""
        staging = bigquery.Dataset(f'{self._project}.{self._staging}')
        staging.location = self._location
        self._client.create_dataset(staging, exists_ok=True)
        config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        release_uri = release_uri.rstrip('/')
        self._await_all(
            'load',
            {
                t: lambda t=t: self._client.load_table_from_uri(
                    f'{release_uri}/output/{t}/*.parquet',
                    self._id(self._staging, t),
                    job_config=config,
                    location=self._location,
                )
                for t in tables
            }
            | {RELEASE_TABLE: lambda: self._load_release(release)},
        )

    def _load_release(self, release: str) -> Any:
        config = bigquery.LoadJobConfig(
            schema=RELEASE_SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        return self._client.load_table_from_json(
            [{'release': release}],
            self._id(self._staging, RELEASE_TABLE),
            job_config=config,
            location=self._location,
        )

    def check_staging(self, tables: Iterable[str]) -> None:
        """Raise if any staged table is empty."""
        empty = [t for t in tables if not self._client.get_table(self._id(self._staging, t)).num_rows]
        if empty:
            raise BigQueryError(f'empty staging tables: {", ".join(empty)}')

    def swap(self, tables: list[str]) -> None:
        """Copy every staged table over prod, then drop prod tables not in `tables`."""
        config = bigquery.CopyJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        self._await_all(
            'copy',
            {
                t: lambda t=t: self._client.copy_table(
                    self._id(self._staging, t),
                    self._id(self._dataset, t),
                    job_config=config,
                    location=self._location,
                )
                for t in tables
            },
        )
        keep = set(tables)
        for table in self._client.list_tables(f'{self._project}.{self._dataset}'):
            if table.table_type == 'TABLE' and table.table_id not in keep:
                logger.info(f'dropping {self._dataset}.{table.table_id}, no longer published')
                self._client.delete_table(self._id(self._dataset, table.table_id), not_found_ok=True)

    def _await_all(self, kind: str, start: dict[str, Callable[[], Any]]) -> None:
        """Start every job, then wait on each; raise naming every table that failed."""
        jobs, errors = {}, {}
        for table, begin in start.items():
            try:
                jobs[table] = begin()
            except Exception as e:
                errors[table] = e
        for table, job in jobs.items():
            try:
                job.result()
                logger.debug(f'{kind} {table} done')
            except Exception as e:
                errors[table] = e
        if errors:
            detail = '; '.join(f'{t}: {e}' for t, e in sorted(errors.items()))
            raise BigQueryError(f'{kind} failed for {", ".join(sorted(errors))} ({detail})')

    def _id(self, dataset: str, table: str) -> str:
        return f'{self._project}.{dataset}.{table}'
