# OpenSearch create index task

from pathlib import Path

from loguru import logger
from opensearchpy import RequestError
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

from pos.services.opensearch import connect
from pos.utils import get_config

ALREADY_EXISTS = 'resource_already_exists_exception'


class OpenSearchCreateIndexError(OtterError):
    """Base class for exceptions in this module."""


class OpenSearchCreateIndexSpec(Spec):
    """Configuration fields for the create index OpenSearch task."""

    dataset: str
    prefix: str
    es_host: str = 'localhost'
    es_port: int = 9200
    dataset_config_path: str = 'config/datasets.yaml'


class OpenSearchCreateIndex(Task):
    def __init__(self, spec: OpenSearchCreateIndexSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: OpenSearchCreateIndexSpec
        try:
            self._config = get_config(self.spec.dataset_config_path).opensearch
            self._index_name = self._get_index_name()
            self._mappings = Path(self._config[self.spec.dataset]['mappings'])
        except AttributeError:
            raise OpenSearchCreateIndexError(f'unable to load config for {self.spec.dataset}')

    @report
    def run(self) -> Task:
        opensearch = connect(self.spec.es_host, self.spec.es_port)
        try:
            opensearch.indices.create(index=self._index_name, body=self._mappings.read_text())
        except RequestError as e:
            if e.error != ALREADY_EXISTS:
                raise
            logger.info(f'index {self._index_name} exists')
            return self
        logger.info(f'index {self._index_name} created')
        return self

    def _get_index_name(self) -> str:
        return f'{self.spec.prefix}_{self._config[self.spec.dataset]["index"]}'
