"""Infrastructure of the staging DAGs: the Dataproc clusters of the steps."""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from orchestration.operators.dataproc import CustomClusterConfig
from orchestration.utils.common import GCP_PROJECT_GENETICS
from orchestration.utils.staging import RUN_RESOURCE_SUFFIX_LENGTH

MAX_CLUSTER_NAME_LENGTH = 51 - RUN_RESOURCE_SUFFIX_LENGTH
"""Maximum length of a cluster name: 51 characters for Dataproc, without the suffix of the run."""


class StagingClusterConfig(CustomClusterConfig):
    """A Dataproc cluster of a staging DAG.

    Unlike in `CustomClusterConfig`, an unknown key is a validation error. The init actions of a cluster are usually files
    copied into the `bin/` folder of the run by `setup`.

    The defaults are those of the Dataproc clusters of the gentropy DAGs: the cluster is in
    the genetics project, the project of the tasks that create, use and delete it, with the
    default service account of the project, and with external IPs, for the downloads of its
    init actions.

    Examples:
        >>> from pydantic import ValidationError
        >>> cluster = StagingClusterConfig.model_validate({
        ...     'num_workers': '4',
        ...     'metadata': {'GENTROPY_REF': 'v4.0.0-dev.1'},
        ...     'init_actions_uris': ['gs://gwas_catalog_top_hits/prod/20261026T153052/bin/install_dependencies_on_cluster.sh'],
        ... })
        >>> cluster.num_workers, cluster.init_actions_uris
        (4, ['gs://gwas_catalog_top_hits/prod/20261026T153052/bin/install_dependencies_on_cluster.sh'])
        >>> cluster.project_id, cluster.service_account, cluster.internal_ip_only
        ('open-targets-genetics-dev', None, False)

        The cluster is in the genetics project:

        >>> try:
        ...     StagingClusterConfig.model_validate({'project_id': 'open-targets-eu-dev'})
        ... except ValidationError as e:
        ...     print(e.errors()[0]['msg'])
        Value error, the project of a staging cluster is open-targets-genetics-dev, the project of its tasks, not 'open-targets-eu-dev'

        >>> try:
        ...     StagingClusterConfig.model_validate({'num_worker': 4})
        ... except ValidationError as e:
        ...     print(e.errors()[0]['msg'])
        Extra inputs are not permitted
    """  # noqa: E501

    model_config = ConfigDict(extra='forbid')

    project_id: str = GCP_PROJECT_GENETICS
    """The project of the cluster, the genetics project."""
    service_account: str | None = None
    """The service account of the cluster VMs, the default service account of the project when unset."""
    internal_ip_only: bool | None = False
    """Whether the cluster VMs have internal IP addresses only."""

    @model_validator(mode='after')
    def _validate_project(self) -> Self:
        """Check that the cluster is in the genetics project, the project of its create, submit and delete tasks."""
        if self.project_id != GCP_PROJECT_GENETICS:
            raise ValueError(
                f'the project of a staging cluster is {GCP_PROJECT_GENETICS}, the project of its tasks, '
                f'not {self.project_id!r}'
            )
        return self


class StagingInfrastructure(BaseModel):
    """The Dataproc clusters of the `dataproc` steps of a DAG.

    Examples:
        >>> infrastructure = StagingInfrastructure.model_validate({
        ...     'clusters': {'otg-tophit-gwascatalog': {'project_id': 'open-targets-genetics-dev', 'num_workers': 4}}
        ... })
        >>> list(infrastructure.clusters)
        ['otg-tophit-gwascatalog']

        A cluster name is a valid Dataproc name once the suffix of the run, `-<hash>`, is added:

        >>> from pydantic import ValidationError
        >>> for name in ['otg_tophit', 'otg-tophit-gwascatalog-harmonisation-quality-control']:
        ...     try:
        ...         StagingInfrastructure.model_validate({'clusters': {name: {}}})
        ...     except ValidationError as e:
        ...         print(e.errors()[0]['msg'])
        Value error, cluster name 'otg_tophit' is not lowercase letters, digits and dashes, starting with a letter
        Value error, cluster name 'otg-tophit-gwascatalog-harmonisation-quality-control' is longer than 45 characters, the maximum with the suffix of the run
    """  # noqa: E501

    model_config = ConfigDict(extra='forbid')

    clusters: dict[str, StagingClusterConfig] = {}
    """Dataproc clusters, by name. A cluster is created before its first step and deleted
    after its last, as `<name>-<hash>`, a name distinct for each DAG run."""

    @model_validator(mode='after')
    def _validate_names(self) -> Self:
        """Check that each cluster name, with the run suffix, is a valid Dataproc cluster name."""
        for name in self.clusters:
            if not re.fullmatch(r'[a-z][a-z0-9-]*', name):
                raise ValueError(
                    f'cluster name {name!r} is not lowercase letters, digits and dashes, starting with a letter'
                )
            if len(name) > MAX_CLUSTER_NAME_LENGTH:
                raise ValueError(
                    f'cluster name {name!r} is longer than {MAX_CLUSTER_NAME_LENGTH} characters, '
                    'the maximum with the suffix of the run'
                )
        return self
