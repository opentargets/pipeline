# OpenSearch prepare task: cluster settings before any index is created

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

from pos.services.opensearch import QUERY_INSIGHTS_OFF, connect, disable_query_insights


class OpenSearchPrepareSpec(Spec):
    """Configuration fields for the OpenSearch prepare task."""

    es_host: str = 'localhost'
    es_port: int = 9200


class OpenSearchPrepare(Task):
    def __init__(self, spec: OpenSearchPrepareSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: OpenSearchPrepareSpec

    @report
    def run(self) -> Task:
        opensearch = connect(self.spec.es_host, self.spec.es_port)
        disable_query_insights(opensearch)
        logger.info(f'query insights off: {", ".join(QUERY_INSIGHTS_OFF)}')
        return self
