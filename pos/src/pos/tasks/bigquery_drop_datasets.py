# Drop scratch BigQuery datasets

from google.cloud import bigquery
from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

SCRATCH_PROJECT = 'open-targets-eu-dev'
SCRATCH_SUFFIXES = ('_scratch', '_scratch_staging')


class BigqueryDropDatasetsSpec(Spec):
    """Configuration fields for the BigQuery drop datasets task.

    project_id: project holding the datasets; only `open-targets-eu-dev`
    dataset_ids: datasets to delete with their tables; each must end in `_scratch` or `_scratch_staging`
    """

    project_id: str
    dataset_ids: list[str]


class BigqueryDropDatasets(Task):
    def __init__(self, spec: BigqueryDropDatasetsSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: BigqueryDropDatasetsSpec

    @report
    def run(self) -> Task:
        project = self.spec.project_id
        if project != SCRATCH_PROJECT:
            raise OtterError(f'refusing to drop datasets in {project}: only {SCRATCH_PROJECT}')
        if others := [d for d in self.spec.dataset_ids if not d.endswith(SCRATCH_SUFFIXES)]:
            raise OtterError(f'refusing to drop {", ".join(others)}: not a {" or ".join(SCRATCH_SUFFIXES)} dataset')
        client = bigquery.Client(project=project)
        for dataset in self.spec.dataset_ids:
            client.delete_dataset(f'{project}.{dataset}', delete_contents=True, not_found_ok=True)
            logger.info(f'dropped {project}.{dataset}')
        return self
