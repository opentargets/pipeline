"""Environments of the staging DAGs: the input and output locations of a run."""

from __future__ import annotations

import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from orchestration.models.staging_dag.placeholders import RESERVED_PLACEHOLDERS, TIMESTAMP_RE, check_rendered

EnvironmentName = Literal['prod', 'test']


class StagingEnvironment(BaseModel):
    """The input and output locations of a DAG run in one environment, and the values of its placeholders.

    `output` is the run folder, which contains the `config/`, `bin/` and `output/` folders.
    The roots and the scratchpad are the values of the placeholders of the config:
    `{{input.<name>}}`, `{{output}}` and `{{<scratchpad key>}}`.

    Examples:
        >>> from pydantic import ValidationError
        >>> test = StagingEnvironment(
        ...     input={'gwas_catalog': 'gs://gwas_catalog_top_hits/fixtures/20261026T153000/gwas_catalog_inputs'},
        ...     output='gs://gwas_catalog_top_hits/test/20261026T153052',
        ...     scratchpad={'gentropy_version': '4.0.0-dev.1'},
        ... )
        >>> test.data_bucket, test.fixtures_root
        ('gs://gwas_catalog_top_hits', 'gs://gwas_catalog_top_hits/fixtures')
        >>> test.config_root
        'gs://gwas_catalog_top_hits/test/20261026T153052/config'
        >>> for name, value in test.template_context.items():
        ...     print(f'{{{{{name}}}}}: {value}')
        {{input.gwas_catalog}}: gs://gwas_catalog_top_hits/fixtures/20261026T153000/gwas_catalog_inputs
        {{output}}: gs://gwas_catalog_top_hits/test/20261026T153052
        {{gentropy_version}}: 4.0.0-dev.1

        The placeholders of the paths are invalid scratchpad keys:

        >>> try:
        ...     StagingEnvironment(input={}, output='', scratchpad={'output': 'x'})
        ... except ValidationError as e:
        ...     print(e.errors()[0]['msg'])
        Value error, scratchpad keys ['output'] are reserved placeholders
    """

    model_config = ConfigDict(extra='forbid')

    input: dict[str, str]
    """Read-only input roots, by name, used as `{{input.<name>}}`."""
    output: str
    """Folder of the run: `<data bucket>/<env>/<timestamp>`, set for each run of the DAG."""
    scratchpad: dict[str, str] = {}
    """Values of the other placeholders of the DAG config, such as a tool version or a cluster size."""

    @model_validator(mode='after')
    def _validate_scratchpad(self) -> Self:
        if clash := sorted(k for k in self.scratchpad if k in RESERVED_PLACEHOLDERS or k.startswith('input.')):
            raise ValueError(f'scratchpad keys {clash} are reserved placeholders')
        return self

    @property
    def data_bucket(self) -> str:
        """The data bucket of the DAG: `output` without `/<env>/<timestamp>`."""
        return self.output.rsplit('/', 2)[0]

    @property
    def fixtures_root(self) -> str:
        """The folder of the static test inputs in the data bucket, as `<ts>/<prod input root without gs://>`."""
        return f'{self.data_bucket}/fixtures'

    @property
    def config_root(self) -> str:
        """The `config/` folder of the run, the location of the record of the rendered DAG config."""
        return f'{self.output}/config'

    @property
    def template_context(self) -> dict[str, str]:
        """The values of the placeholders of the config, by placeholder name."""
        return {
            **{f'input.{name}': root for name, root in self.input.items()},
            'output': self.output,
            **self.scratchpad,
        }


class StagingEnvironments(BaseModel):
    """The environments of a DAG config, the selected one, and the rules of their layout.

    Validated on the raw config, before the rest of the config, which is valid only once
    rendered with the values of the selected environment. The output of every environment
    is a folder of the data bucket named after a timestamp, `<data bucket>/<env>/<timestamp>`,
    set for each run of the DAG, as the `run_name` of the unified pipeline. Each test input
    root is the prod root, or a dated static slice of the prod root in
    `<data bucket>/fixtures/<ts>/<prod root without gs://>`.

    Examples:
        The GWAS Catalog top hits DAG, with the test environment selected. The test input
        `gwas_catalog` is a static slice of the prod input, in a dated folder of the data
        bucket, `fixtures/<ts>/`, under the path of the prod input root. The test input
        `gnomad` is the prod input bucket:

        >>> from pydantic import ValidationError
        >>> raw = {
        ...     'env': 'test',
        ...     'environments': {
        ...         'prod': {
        ...             'input': {'gwas_catalog': 'gs://gwas_catalog_inputs', 'gnomad': 'gs://gnomad_data_2'},
        ...             'output': 'gs://gwas_catalog_top_hits/prod/20261026T153052',
        ...         },
        ...         'test': {
        ...             'input': {
        ...                 'gwas_catalog': 'gs://gwas_catalog_top_hits/fixtures/20261026T153000/gwas_catalog_inputs',
        ...                 'gnomad': 'gs://gnomad_data_2',
        ...             },
        ...             'output': 'gs://gwas_catalog_top_hits/test/20261026T153052',
        ...         },
        ...     },
        ...     'steps': {'pics': '{{output}}/output/credible_set'},  # the rest of the config is ignored
        ... }
        >>> environments = StagingEnvironments.model_validate(raw)
        >>> environments.environment.output
        'gs://gwas_catalog_top_hits/test/20261026T153052'
        >>> prod, test = environments.environments['prod'], environments.environments['test']
        >>> {name: root == prod.input[name] for name, root in test.input.items()}
        {'gwas_catalog': False, 'gnomad': True}

        >>> def check(change):
        ...     changed = {**raw, 'environments': {n: {**e, 'input': {**e['input']}} for n, e in raw['environments'].items()}}
        ...     change(changed['environments'])
        ...     try:
        ...         return StagingEnvironments.model_validate(changed).environment
        ...     except ValidationError as e:
        ...         print(e.errors()[0]['msg'])
        >>> check(lambda e: e['test']['input'].update(gwas_catalog='gs://gwas_catalog_top_hits/test/input/gwas_catalog'))
        Value error, test input root gwas_catalog is 'gs://gwas_catalog_top_hits/test/input/gwas_catalog', but it can only be gs://gwas_catalog_inputs (the prod root, read directly), gs://gwas_catalog_top_hits/fixtures/<ts>/gwas_catalog_inputs (a dated static slice of the prod root)

        The test output is outside the prod folders:

        >>> check(lambda e: e['test'].update(output='gs://gwas_catalog_top_hits/prod/20261026T153052'))
        Value error, test output 'gs://gwas_catalog_top_hits/prod/20261026T153052' is not <data bucket>/test/<timestamp>

        And an output folder is named after a timestamp:

        >>> check(lambda e: e['prod'].update(output='gs://gwas_catalog_top_hits/prod/latest'))
        Value error, prod output 'gs://gwas_catalog_top_hits/prod/latest' is not <data bucket>/prod/<timestamp>
    """  # noqa: E501

    env: EnvironmentName
    """The selected environment."""
    environments: dict[EnvironmentName, StagingEnvironment]

    @property
    def environment(self) -> StagingEnvironment:
        """The selected environment."""
        return self.environments[self.env]

    @model_validator(mode='before')
    @classmethod
    def _check_rendered(cls, data: Any) -> Any:
        """Check that the environments contain no placeholder.

        The environments are the values the config is rendered with, so they are checked on
        the raw config, before the rendering would fill a placeholder in them.
        """
        if isinstance(data, dict):
            check_rendered(data.get('environments'))
        return data

    @model_validator(mode='after')
    def _validate_environments(self) -> Self:
        """Check the layout of the environments, on the raw config."""
        self._check_defined()
        self._check_outputs()
        prod, test = self.environments['prod'], self.environments.get('test')
        if test is not None:
            self._check_test_matches_prod(prod, test)
            self._check_test_inputs(prod, test)
        return self

    def _check_defined(self) -> None:
        """Check that prod and selected environment are defined."""
        if 'prod' not in self.environments:
            raise ValueError('the prod environment is required')
        if self.env not in self.environments:
            raise ValueError(f'environment {self.env} is not defined in environments {sorted(self.environments)}')

    def _check_outputs(self) -> None:
        """Check that the `output` of every environment is a run folder of its data bucket.

        The folder is `<data bucket>/<env>/<timestamp>`, e.g. `prod/20261026T153052`.
        """
        for name, environment in self.environments.items():
            if not re.fullmatch(rf'gs://[^\s{{}}]+/{name}/{TIMESTAMP_RE}', environment.output):
                raise ValueError(f'{name} output {environment.output!r} is not <data bucket>/{name}/<timestamp>')

    @staticmethod
    def _check_test_matches_prod(prod: StagingEnvironment, test: StagingEnvironment) -> None:
        """Check that the test input names, scratchpad keys and data bucket are the prod ones.

        The steps are common to both environments, so every placeholder of a step renders in both.
        """
        if names := sorted(set(prod.input) ^ set(test.input)):
            raise ValueError(f'the prod and test input names differ: {names}')
        if keys := sorted(set(prod.scratchpad) ^ set(test.scratchpad)):
            raise ValueError(f'the prod and test scratchpad keys differ: {keys}')
        if prod.data_bucket != test.data_bucket:
            raise ValueError(f'the prod and test data buckets differ: {prod.data_bucket}, {test.data_bucket}')

    @staticmethod
    def _check_test_inputs(prod: StagingEnvironment, test: StagingEnvironment) -> None:
        """Check that each test input root is an allowed location.

        For an input `gwas_catalog` with the prod root `gs://gwas_catalog_inputs`, in the data
        bucket `gs://gwas_catalog_top_hits`, the test root is one of:

        * the prod input, read directly: `gs://gwas_catalog_inputs`,
        * a dated static slice of the prod root, kept in the data bucket under the prod root
          path: `gs://gwas_catalog_top_hits/fixtures/<ts>/gwas_catalog_inputs`, where `<ts>`
          is a timestamp such as `20261026T153000`.
        """
        for name, root in test.input.items():
            prod_root = prod.input[name]
            slice_root = f'{test.fixtures_root}/<ts>/{prod_root.removeprefix("gs://")}'
            forms = {  # description: (path shown in the error, pattern the root matches)
                'the prod root, read directly': (prod_root, re.escape(prod_root)),
                'a dated static slice of the prod root': (
                    slice_root,
                    re.escape(slice_root).replace(re.escape('<ts>'), TIMESTAMP_RE),
                ),
            }
            if not any(re.fullmatch(pattern, root) for _, pattern in forms.values()):
                raise ValueError(
                    f'test input root {name} is {root!r}, but it can only be '
                    + ', '.join(f'{path} ({description})' for description, (path, _) in forms.items())
                )
