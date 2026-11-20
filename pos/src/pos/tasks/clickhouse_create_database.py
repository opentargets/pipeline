# Clickhouse create database task
from clickhouse_connect.driver.exceptions import DatabaseError
from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

from pos.services.clickhouse import connect, create_database, tables
from pos.utils import WIPE_DISKS


class ClickhouseCreateDatabaseError(OtterError):
    """Base class for exceptions in this module."""


class ClickhouseCreateDatabaseSpec(Spec):
    """Configuration fields for the Clickhouse create database task."""

    host: str = 'localhost'
    username: str = 'default'
    password: str = ''
    port: int = 8123
    clickhouse_database: str = 'ot'
    exist_ok: bool = False


class ClickhouseCreateDatabase(Task):
    def __init__(self, spec: ClickhouseCreateDatabaseSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: ClickhouseCreateDatabaseSpec

    @report
    def run(self) -> Task:
        logger.debug(f'creating Clickhouse database {self.spec.clickhouse_database}')
        client = connect(
            host=self.spec.host,
            port=self.spec.port,
            username=self.spec.username,
            password=self.spec.password,
        )
        try:
            create_database(client, self.spec.clickhouse_database, exists_ok=self.spec.exist_ok)
            existing = tables(client, self.spec.clickhouse_database)
        except DatabaseError as db_err:
            raise ClickhouseCreateDatabaseError(f'Create Clickhouse database error : {db_err}') from db_err
        if existing:
            shown = ', '.join(existing[:10]) + (', ...' if len(existing) > 10 else '')
            raise ClickhouseCreateDatabaseError(
                f'database {self.spec.clickhouse_database} already has {len(existing)} tables ({shown}). {WIPE_DISKS}'
            )
        return self
