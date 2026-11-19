# Load every modelled output into BigQuery staging, check it, then swap it into prod

from google.cloud import bigquery
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

from pos.gcp.bigquery import BigQueryRelease, published_tables


class BigqueryLoadSpec(Spec):
    """Configuration fields for the BigQuery task.

    release_uri: the run whose `output/` is published
    project_id: project holding both datasets
    dataset_id: prod dataset, never deleted
    staging_dataset_id: dataset loaded and checked before the swap
    location: BigQuery location of both datasets
    """

    release_uri: str
    project_id: str
    dataset_id: str
    staging_dataset_id: str
    location: str


class BigqueryLoad(Task):
    def __init__(self, spec: BigqueryLoadSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: BigqueryLoadSpec

    @report
    def run(self) -> Task:
        client = bigquery.Client(project=self.spec.project_id, location=self.spec.location)
        release = BigQueryRelease(
            client,
            project=self.spec.project_id,
            dataset=self.spec.dataset_id,
            staging=self.spec.staging_dataset_id,
            location=self.spec.location,
        )
        release.publish(published_tables(self.spec.release_uri), self.spec.release_uri)
        return self
