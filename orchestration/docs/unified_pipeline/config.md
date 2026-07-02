# Unified Pipeline pipeline config

This document describes the **Unified Pipeline configuration**

## Unified Pipeline configuration

The Unified Pipeline configuration is defined in multiple YAML files throughout
the repository. Beside the configuration for the pipeline itself, each software
component run by it owns its configuration:

- [`unified_pipeline.yaml`](../../../orchestration/src/orchestration/dags/config/unified_pipeline.yaml) -
    configuration for the Unified Pipeline itself
- [`clusters.yaml`](../../../orchestration/src/orchestration/dags/config/clusters.yaml) -
    configuration for the Spark clusters used by the Unified Pipeline
- [PIS `config.yaml`](../../../pis/config.yaml) - configuration for the Platform Input Stage
    (PIS)
- [PTS `config.yaml`](../../../pts/config.yaml) - configuration for the Platform Transformation
    Stage (PTS)
- [`gentropy.yaml`](../../../orchestration/src/orchestration/dags/config/gentropy.yaml) -
    configuration for the Platform Genetics (Gentropy)
- [`gentropy.overrides.yaml`](../../../orchestration/src/orchestration/dags/config/ppp/gentropy.overrides.yaml) -
    PPP overrides for the Platform Genetics (Gentropy)

Typically the configuration file has to have at least the `steps` key defined that
marks all of the pipeline steps that should be executed within the pipeline run.

```
steps:
    biosample:
        - name: copy cell ontology
        source: cl.json
        destination: input/biosample/cl.json
```

Each step config (_biosample_ in the example above) should hold the definition of
the step parameters required to execute the step.

Each config structure is unique to the tool. Refer to the specific tool documentation
for more details on the configuration structure.

### Template variables

The template variables can be used in the configuration files to define the
dynamic parts of the configuration. Currently one has to register the template
variables in the `src/orchestration/dags/config/unified_pipeline.py` file, which
is then used to render the configuration files.

### Infrastructure configuration

Along with that the configuration also includes infrastructure specific config,
which defines the Spark clusters in `clusters.yaml`.

## Overriding configurations

> NOTE! The functionality describe here can be used in the PPP (Partner Preview
> Platform) unified pipeline runs, as it allows to override specific parts of
> the configuration.

In order to override the default configuration one can define the configuration
files in the `src/orchestration/dags/config` directory. By convention, the
override files should be named as `${pipeline_part}.override.yaml` where
`${pipeline_part}` is of:

- pis
- pts
- gentropy

To override the specific configuration one need to define the same config file
as the original one defined in the `src/orchestration/dags/config` directory,
but with the `.override.yaml` suffix. For example, to override the `pis.yaml`
configuration, one should create a file named
`src/orchestration/dags/config/ppp/pis.override.yaml`.

> IMPORTANT! The override functionality is performed on the **rendered and
> parsed** configs! Any template variables in the override files will be parsed
> with the templates from the original config files. Only the content of the
> `steps` key will be overridden, the rest of the configuration will be dropped
> after rendering.
