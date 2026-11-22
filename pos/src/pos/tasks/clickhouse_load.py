# Clickhouse load task
from pathlib import Path

from clickhouse_connect.driver.binding import quote_identifier
from clickhouse_connect.driver.exceptions import ClickHouseError
from clickhouse_connect.driver.tools import insert_file
from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

from pos.clickhouse_ddl import columns, ddl_for, member_table, members
from pos.services.clickhouse import connect
from pos.sql import split_statements
from pos.utils import get_config

# synchronous, so an insert returns once its data is written
INSERT_SETTINGS = {'async_insert': 0}


class ClickhouseLoadError(OtterError):
    """Base class for exceptions in this module."""


class ClickhouseLoadSpec(Spec):
    """Configuration fields for the Load Clickhouse task."""

    dataset: str
    clickhouse_host: str = 'localhost'
    clickhouse_username: str = 'default'
    clickhouse_password: str = ''
    clickhouse_database: str = 'ot'
    clickhouse_port: int = 8123
    data_dir_parent: str
    dataset_config_path: str = 'config/datasets.yaml'
    product: str = 'platform'


class ClickhouseLoad(Task):
    def __init__(self, spec: ClickhouseLoadSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: ClickhouseLoadSpec
        try:
            self._entry = get_config(self.spec.dataset_config_path).clickhouse[self.spec.dataset]
            self._table_name = self._entry['table']
            self._derived = bool(self._entry.get('derived', False))
            self._optional = bool(self._entry.get('optional', False))
            self._input_dir = self._entry.get('input_dir')
            self._pre_load_sql = self._entry.get('preload_script')
            self._post_load_sql = self._entry.get('postload_script')
            self._models = self._entry.get('models')
            self._optional_models = set(self._entry.get('optional_models', []))
        except AttributeError:
            raise ClickhouseLoadError(f'unable to load config for {self.spec.dataset}')

    @report
    def run(self) -> Task:
        if self._models is not None:
            member_files = self._member_files()
            client = self._connect()
            self._load_members(client, member_files)
        else:
            files = [] if self._derived else self._files(self._get_parquet_path(), self._optional, self.spec.dataset)
            client = self._connect()
            ddl = ddl_for(self._entry, self.spec.product)
            if ddl:
                self._create(client, ddl[0][1])
            else:
                self._execute_script(client, self._pre_load_sql)
            if not self._derived:
                self._load(client, self._table_name, files)
        if self._post_load_sql is not None:
            self._execute_script(client, self._post_load_sql)
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
    def _files(path: Path, optional: bool, name: str) -> list[Path]:
        files = sorted(path.glob('*.parquet'))
        if not files:
            where = f'{path}/*.parquet'
            if not optional:
                raise ClickhouseLoadError(f'no parquet files for dataset {name} in {where}')
            logger.warning(f'no parquet files for optional dataset {name} in {where}, loading none')
        return files

    def _member_files(self) -> dict[str, list[Path]]:
        """Each member's files for the product, before any table is created; optional members with none left out."""
        found = {}
        for model in members(self._models, self.spec.product):
            name = model.dataset_name
            files = self._files(self._get_parquet_path() / name, name in self._optional_models, name)
            if files:
                found[name] = files
        return found

    def _load_members(self, client, member_files: dict[str, list[Path]]) -> None:
        """Each member into its own table, then all of them into the union table; the member tables are dropped."""
        ddl = dict(ddl_for(self._entry, self.spec.product))
        loaded = [m for m in members(self._models, self.spec.product) if m.dataset_name in member_files]
        for model in loaded:
            self._create(client, ddl[member_table(model)])
            self._load(client, member_table(model), member_files[model.dataset_name])
        self._create(client, ddl[self._table_name])
        inserts = []
        for model in loaded:
            cols = ', '.join(quote_identifier(name) for name, _ in columns(model))
            source = quote_identifier(member_table(model))
            target = quote_identifier(self._table_name)
            inserts.append(f'INSERT INTO {target} ({cols}) SELECT {cols} FROM {source}')  # noqa: S608 quoted identifiers
        try:
            self._run_all(client, inserts)
        except ClickHouseError as e:
            logger.warning(f'loading {self._table_name} failed ({e}); truncating it and inserting every member again')
            client.command(f'TRUNCATE TABLE {quote_identifier(self._table_name)}')
            self._run_all(client, inserts)
        self._log_count(client, self._table_name)
        for model in loaded:
            client.command(f'DROP TABLE {quote_identifier(member_table(model))}')

    @staticmethod
    def _run_all(client, statements: list[str]) -> None:
        for sql in statements:
            client.command(sql)

    @staticmethod
    def _create(client, sql: str) -> None:
        logger.info(sql)
        client.command(sql)

    def _load(self, client, table: str, files: list[Path]) -> None:
        """Insert the files; on an error truncate the table and insert them all once more."""
        try:
            self._insert(client, table, files)
        except ClickHouseError as e:
            logger.warning(f'loading {table} failed ({e}); truncating it and loading every file again')
            client.command(f'TRUNCATE TABLE {quote_identifier(table)}')
            self._insert(client, table, files)
        self._log_count(client, table)

    @staticmethod
    def _log_count(client, table: str) -> None:
        count = client.command(f'SELECT count() FROM {quote_identifier(table)}')  # noqa: S608 quoted identifier
        logger.info(f'{table}: {count} rows')

    @staticmethod
    def _insert(client, table: str, files: list[Path]) -> None:
        for file in files:
            logger.debug(f'Inserting file {file} into Clickhouse table {table}')
            insert_file(client, table, str(file), fmt='Parquet', settings=INSERT_SETTINGS)

    def _get_parquet_path(self) -> Path:
        return Path(f'{self.context.config.work_path}/{self.spec.data_dir_parent}/{self._input_dir}')

    def _execute_script(self, client, script: str) -> None:
        for sql in split_statements(Path(script).read_text()):
            client.query(sql)
