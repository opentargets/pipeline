# OpenSearch finish task: no query-insights indices left on the loaded cluster

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

from pos.services.opensearch import connect, top_queries_indices


class OpenSearchFinishError(OtterError):
    """Base class for exceptions in this module."""


class OpenSearchFinishSpec(Spec):
    """Configuration fields for the OpenSearch finish task."""

    es_host: str = 'localhost'
    es_port: int = 9200


class OpenSearchFinish(Task):
    def __init__(self, spec: OpenSearchFinishSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: OpenSearchFinishSpec

    @report
    def run(self) -> Task:
        opensearch = connect(self.spec.es_host, self.spec.es_port)
        stray = top_queries_indices(opensearch)
        if stray:
            logger.warning(f'deleting query-insights indices: {", ".join(stray)}')
            opensearch.indices.delete(index=','.join(stray), expand_wildcards='all')
        left = top_queries_indices(opensearch)
        if left:
            raise OpenSearchFinishError(f'query-insights indices remain: {", ".join(left)}')
        logger.info('no query-insights indices')
        return self
