# OpenSearch prepare task: an empty cluster and its settings, before any index is created

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

from pos.services.opensearch import QUERY_INSIGHTS_OFF, connect, disable_query_insights
from pos.utils import WIPE_DISKS, get_config


class OpenSearchPrepareError(OtterError):
    """Base class for exceptions in this module."""


class OpenSearchPrepareSpec(Spec):
    """Configuration fields for the OpenSearch prepare task."""

    prefix: str
    es_host: str = 'localhost'
    es_port: int = 9200
    dataset_config_path: str = 'config/datasets.yaml'


class OpenSearchPrepare(Task):
    def __init__(self, spec: OpenSearchPrepareSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: OpenSearchPrepareSpec
        datasets = get_config(self.spec.dataset_config_path).opensearch
        self._indices = sorted({f'{self.spec.prefix}_{d["index"]}' for d in datasets.values()})

    @report
    def run(self) -> Task:
        opensearch = connect(self.spec.es_host, self.spec.es_port)
        existing = [i for i in self._indices if opensearch.indices.exists(index=i)]
        if existing:
            raise OpenSearchPrepareError(f'indices already exist: {", ".join(existing)}. {WIPE_DISKS}')
        disable_query_insights(opensearch)
        logger.info(f'query insights off: {", ".join(QUERY_INSIGHTS_OFF)}')
        return self
