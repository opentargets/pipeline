# Clickhouse load task
from pathlib import Path

from clickhouse_connect.driver.binding import quote_identifier
from clickhouse_connect.driver.exceptions import ClickHouseError
from clickhouse_connect.driver.tools import insert_file
from loguru import logger
from ot_datamodels.base import DatasetModel
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

from pos.clickhouse_ddl import input_dir, landing_ddl, landing_table, served_sql, served_views, uniqueness_sql
from pos.services.clickhouse import connect

# synchronous, so an insert returns once its data is written
INSERT_SETTINGS = {'async_insert': 0}


class ClickhouseLoadError(OtterError):
    """Base class for exceptions in this module."""


class ClickhouseLoadSpec(Spec):
    """Configuration fields for the Load Clickhouse task."""

    dataset: str
    """A served view."""
    clickhouse_host: str = 'localhost'
    clickhouse_username: str = 'default'
    clickhouse_password: str = ''
    clickhouse_database: str = 'ot'
    clickhouse_port: int = 8123
    data_dir_parent: str
    product: str


def check_unique(client, model: type[DatasetModel]) -> None:
    """Fail when the view's RocksDB key repeats in its landing table.

    Raises:
        ClickhouseLoadError: if it does.
    """
    try:
        client.command(uniqueness_sql(model))
    except ClickHouseError as e:
        raise ClickhouseLoadError(f'loading {model.dataset_name} failed: {e}') from e


def build_served(client, model: type[DatasetModel]) -> None:
    """The view's table and dictionaries from its filled landing table, which is then dropped."""
    serving = model.clickhouse
    assert serving is not None
    if serving.engine == 'rocksdb':
        check_unique(client, model)
    for sql in served_sql(model):
        logger.info(sql)
        client.command(sql)
    if serving.engine == 'mergetree':
        _log_count(client, model.dataset_name)


def _log_count(client, table: str) -> None:
    count = client.command(f'SELECT count() FROM {quote_identifier(table)}')  # noqa: S608 quoted identifier
    logger.info(f'{table}: {count} rows')


class ClickhouseLoad(Task):
    """Load a served view: its landing table from `view/<dataset_name>`, then its table and dictionaries."""

    def __init__(self, spec: ClickhouseLoadSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: ClickhouseLoadSpec
        try:
            self._model = served_views()[self.spec.dataset]
        except (KeyError, ValueError) as e:
            raise ClickhouseLoadError(f'unable to load config for {self.spec.dataset}: {e}') from e
        if self.spec.product not in self._model.products:
            raise ClickhouseLoadError(f'{self.spec.dataset}: is not served for {self.spec.product}')
        self._table_name = landing_table(self._model)
        self._file: Path | None = None  # the file being inserted

    @report
    def run(self) -> Task:
        files = self._files(self._get_parquet_path(), self.spec.dataset)
        client = self._connect()
        create = landing_ddl(self._model)
        client.command(f'DROP TABLE IF EXISTS {quote_identifier(self._table_name)} SYNC')
        logger.info(create)
        client.command(create)
        self._load_named(client, self.spec.dataset, self._table_name, files, create)
        build_served(client, self._model)
        return self

    def _connect(self):
        return connect(
            host=self.spec.clickhouse_host,
            port=self.spec.clickhouse_port,
            username=self.spec.clickhouse_username,
            password=self.spec.clickhouse_password,
            database=self.spec.clickhouse_database,
        )

    @staticmethod
    def _files(path: Path, name: str) -> list[Path]:
        files = sorted(path.glob('*.parquet'))
        if not files:
            raise ClickhouseLoadError(f'no parquet files for dataset {name} in {path}/*.parquet')
        return files

    def _load_named(self, client, name: str, table: str, files: list[Path], create: str) -> None:
        """`_load`; an error the retry does not clear names the dataset, the file and the generated table."""
        try:
            self._load(client, table, files)
        except ClickHouseError as e:
            message = f'loading {name} into {table} failed on {self._file}: {e}; its table: {create}'
            raise ClickhouseLoadError(message) from e

    def _load(self, client, table: str, files: list[Path]) -> None:
        """Insert the files; on an error truncate the table and insert them all once more."""
        try:
            self._insert(client, table, files)
        except ClickHouseError as e:
            logger.warning(f'loading {table} failed ({e}); truncating it and loading every file again')
            client.command(f'TRUNCATE TABLE {quote_identifier(table)}')
            self._insert(client, table, files)
        _log_count(client, table)

    def _insert(self, client, table: str, files: list[Path]) -> None:
        for file in files:
            self._file = file
            logger.debug(f'Inserting file {file} into Clickhouse table {table}')
            insert_file(client, table, str(file), fmt='Parquet', settings=INSERT_SETTINGS)

    def _get_parquet_path(self) -> Path:
        return Path(f'{self.context.config.work_path}/{self.spec.data_dir_parent}/{input_dir(self._model)}')
