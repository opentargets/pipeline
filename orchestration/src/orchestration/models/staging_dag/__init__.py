"""Schema of the staging DAG configs.

The config of a staging DAG is the file `dags/<dag_id>/config.yaml`, read with
`orchestration.dags.config.staging_pipeline.StagingDagConfig` in three stages:

1. `env` and `environments` are validated once, on the raw config, against
   `StagingEnvironments`. `env` is the name of the selected environment.
2. The whole config is rendered as text with the values of the selected environment:
   `{{input.<name>}}` (an input root), `{{output}}` (the run folder, which contains the
   `config/`, `bin/` and `output/` folders) and `{{<key>}}` for every `scratchpad` key,
   such as a tool version.
   The rendered text contains no placeholder.
3. The rest of the rendered config is validated against `StagingDagConfigModel`: the
   infrastructure first, then every step against the model of its kind.

The rules of the environments:

* the output of a run is a folder of the data bucket named after a timestamp,
  `<data bucket>/<env>/<timestamp>`, set in the config for each run of the DAG, as the
  `run_name` of the unified pipeline,
* each test input root is the prod root (e.g. a reference), or
  `<data bucket>/fixtures/<ts>/<prod root without gs://>`, a dated static slice of the
  prod input in the data bucket.

The prod environment is required. The test environment and the setup are optional.

A step is executed on Dataproc, on Google Batch or on a VM. The rendered DAG config of a run
is recorded in the `config/` folder of the run. The optional `setup` list is the files copied
into the `bin/` folder of the run before the steps, such as a cluster init script or the
main python file of a Dataproc job.
"""

from orchestration.models.staging_dag.dag import StagingDagConfigModel
from orchestration.models.staging_dag.environment import EnvironmentName, StagingEnvironment, StagingEnvironments
from orchestration.models.staging_dag.infrastructure import StagingClusterConfig, StagingInfrastructure
from orchestration.models.staging_dag.setup import SetupCopy
from orchestration.models.staging_dag.step import (
    BatchStep,
    DataprocStep,
    StagingStep,
    StepDefinition,
    StepKind,
    VmStep,
)

__all__ = [
    'BatchStep',
    'DataprocStep',
    'EnvironmentName',
    'SetupCopy',
    'StagingClusterConfig',
    'StagingDagConfigModel',
    'StagingEnvironment',
    'StagingEnvironments',
    'StagingInfrastructure',
    'StagingStep',
    'StepDefinition',
    'StepKind',
    'VmStep',
]
