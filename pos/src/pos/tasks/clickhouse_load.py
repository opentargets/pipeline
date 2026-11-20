# Clickhouse load task
from pathlib import Path

from clickhouse_connect.driver.tools import insert_file
from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

from pos.services.clickhouse import connect
from pos.sql import split_statements
from pos.utils import get_config


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


class ClickhouseLoad(Task):
    def __init__(self, spec: ClickhouseLoadSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: ClickhouseLoadSpec
        try:
            self._config = get_config(self.spec.dataset_config_path).clickhouse
            self._table_name = self._config[self.spec.dataset]['table']
            self._derived = bool(self._config[self.spec.dataset].get('derived', False))
            self._input_dir = self._config[self.spec.dataset].get('input_dir')
            self._pre_load_sql = self._config[self.spec.dataset].get('preload_script')
            self._post_load_sql = self._config[self.spec.dataset].get('postload_script')
            self._glob_pattern = self._config[self.spec.dataset].get('glob_pattern', '*.parquet')
        except AttributeError:
            raise ClickhouseLoadError(f'unable to load config for {self.spec.dataset}')

    @report
    def run(self) -> Task:
        files = [] if self._derived else list(self._get_parquet_path().glob(self._glob_pattern))
        if not self._derived and not files:
            raise ClickhouseLoadError(
                f'no parquet files for dataset {self.spec.dataset} in {self._get_parquet_path()}/{self._glob_pattern}'
            )
        client = connect(
            host=self.spec.clickhouse_host,
            port=self.spec.clickhouse_port,
            username=self.spec.clickhouse_username,
            password=self.spec.clickhouse_password,
            database=self.spec.clickhouse_database,
        )
        self._execute_script(client, self._pre_load_sql)
        for file in files:
            logger.debug(f'Inserting file {file} into Clickhouse table {self._table_name}')
            insert_file(client, self._table_name, str(file), fmt='Parquet')
        if self._post_load_sql is not None:
            self._execute_script(client, self._post_load_sql)
        return self

    def _get_parquet_path(self) -> Path:
        return Path(f'{self.context.config.work_path}/{self.spec.data_dir_parent}/{self._input_dir}')

    def _execute_script(self, client, script: str) -> None:
        for sql in split_statements(Path(script).read_text()):
            client.query(sql)
