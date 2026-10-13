"""Schema of the staging DAG configs.

Each staging DAG lives in `dags/<dag_id>/` with a `dag.py` and a `config.yaml`. The
config is read with `AppConfig`, rendered with the scratchpad of its selected
environment, and validated against `StagingDagConfigModel`. Reading it is the job of
`orchestration.dags.config.staging.StagingDagConfig`.

A step is either a pps step, run on one of the DAG's Dataproc clusters, or a Batch
step, run on Google Batch from its own job spec.
"""

from __future__ import annotations

from datetime import timedelta
from graphlib import CycleError, TopologicalSorter
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from orchestration.models.batch.operator import BatchIndexOperatorSpec, BatchJobOperatorSpec
from orchestration.operators.dataproc import CustomClusterConfig

EnvironmentName = Literal['prod', 'test']


class StagingEnvironment(BaseModel):
    """Values that differ between the environments a DAG can run in."""

    model_config = ConfigDict(extra='forbid')

    name: EnvironmentName
    scratchpad: dict[str, str] = {}
    """Values rendering the `{{placeholders}}` of the DAG config, and overriding the
    scratchpad of `pps/config.yaml`."""


class StagingEnvironments(BaseModel):
    """The environments of a DAG config, and the one it runs in.

    Validated before the rest of the config, which can only be validated once it is
    rendered with the selected environment's scratchpad.
    """

    env: EnvironmentName
    environments: list[StagingEnvironment]

    @model_validator(mode='after')
    def _validate_env(self) -> Self:
        names = [e.name for e in self.environments]
        if len(names) != len(set(names)):
            raise ValueError(f'duplicated environments: {names}')
        if self.env not in names:
            raise ValueError(f'environment {self.env} is not defined in environments {names}')
        return self

    @property
    def scratchpad(self) -> dict[str, str]:
        """The scratchpad of the selected environment."""
        return next(e.scratchpad for e in self.environments if e.name == self.env)


class StagingInfrastructure(BaseModel):
    """The infrastructure the steps of a DAG run on."""

    model_config = ConfigDict(extra='forbid')

    clusters: dict[str, CustomClusterConfig] = {}
    """Dataproc clusters, by name. A cluster is created before its first step and
    deleted after its last."""

    @model_validator(mode='before')
    @classmethod
    def _staging_cluster_defaults(cls, data: Any) -> Any:
        # staging clusters run with the default compute service account and public ips
        for cluster in (data or {}).get('clusters', {}).values():
            if isinstance(cluster, dict):
                cluster.setdefault('service_account', None)
                cluster.setdefault('internal_ip_only', False)
        return data


class StagingStep(BaseModel):
    """A step of a staging DAG, run as one Airflow task (two for Batch steps)."""

    model_config = ConfigDict(extra='forbid')

    cluster: str | None = None
    """Cluster the pps step of the same name runs on. Set for pps steps only."""
    google_batch: BatchJobOperatorSpec | None = None
    """Job spec of a Batch step. Set for Batch steps only."""
    google_batch_index_specs: BatchIndexOperatorSpec | None = None
    """Index the Batch job is fanned out over. Required for Batch steps."""
    depends_on: list[str] = []
    """Steps that must finish before this one starts."""
    execution_timeout: timedelta | None = None
    """Kill the step if it runs longer than this (seconds, or an ISO 8601 duration)."""

    @model_validator(mode='after')
    def _validate_kind(self) -> Self:
        if (self.cluster is None) == (self.google_batch is None):
            raise ValueError('a step needs exactly one of `cluster` (pps step) or `google_batch` (Batch step)')
        if self.google_batch is not None and self.google_batch_index_specs is None:
            raise ValueError('a Batch step needs `google_batch_index_specs`')
        if self.cluster is not None and self.google_batch_index_specs is not None:
            raise ValueError('`google_batch_index_specs` is only valid for Batch steps')
        return self

    @property
    def is_batch(self) -> bool:
        """Whether the step runs on Google Batch rather than as a pps step on Dataproc."""
        return self.google_batch is not None


class StagingDagConfigModel(StagingEnvironments):
    """The `config.yaml` of a staging DAG, once rendered."""

    model_config = ConfigDict(extra='forbid')

    description: str
    """Description of the DAG."""
    staging_bucket: str
    """GCS location for the files a run uploads: step configs, entrypoint and init script."""
    pps_version: str
    """pps release the pps steps run, as in its `pps@v<pps_version>` tag."""
    infrastructure: StagingInfrastructure = StagingInfrastructure()
    steps: dict[str, StagingStep]

    @model_validator(mode='after')
    def _validate_steps(self) -> Self:
        for name, step in self.steps.items():
            if step.cluster is not None and step.cluster not in self.infrastructure.clusters:
                raise ValueError(f'step {name} runs on unknown cluster {step.cluster!r}')
            if unknown := set(step.depends_on) - set(self.steps):
                raise ValueError(f'step {name} depends on unknown steps {sorted(unknown)}')

        try:
            tuple(TopologicalSorter({n: s.depends_on for n, s in self.steps.items()}).static_order())
        except CycleError as e:
            raise ValueError(f'steps form a cycle: {e.args[1]}') from e
        return self
