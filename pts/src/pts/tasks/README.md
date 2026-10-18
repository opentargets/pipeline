# Experimental pyspark runner task

The `pyspark` task can be used to launch pyspark jobs. They are local jobs for now, but we'll add
dataproc cluster support.

The file [baseline_expression_harmonise.py](../pyspark/baseline_expression_harmonise.py) is an example of a
pyspark job.


## Launching

Just like any other Otter app:

```sh
$ uv run pts -s baseline_expression_dice
```


## Config

It is configured in [config.yaml](../../../config.yaml) like so:

```yaml
---
work_path: ./work
log_level: DEBUG
release_uri: gs://open-targets-pre-data-releases/jferrer/test-pts-pyspark
pool: 8

steps:
  baseline_expression_dice:
    - name: pyspark baseline expression harmonise dice
      pyspark: baseline_expression_harmonise
      source:
        dice_directory: input/baseline_expression/bulkRNAseq/DICE
        mapping_path: input/baseline_expression/bulkRNAseq/DICE/DICE_ontology_mapping.txt
      destination:
        baseline_expression: intermediate/baseline_expression/unaggregated/DICE/celltype
      settings:
        harmoniser: dice
```

You pass either a `str` or a `dict[str, str]` to both `source` and `destination`, and a `pyspark`
which is a method name inside a python file with the same name under `pts.pyspark` package.

You can also pass a `properties` `dict[str, str]` with extra configuration for Spark.


## Path support

The paths in both `source` and `destination` are relative to either `release_uri` or `work_path`,
in that order. That means, if a `release_uri` is defined (in the form `gs://...`), the spark job
will fetch the data from a GCS bucket. Otherwise, data must be local under `work_path`.

This can be used to develop and test locally with data in the host.

> [!NOTE]
>
> GCS Support is still a bit flimsy, and the application must be launched with:
> `$ GCLOUD_PROJECT=open-targets-prod GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json uv run pts -s baseline_expression_dice`
> to specify a credentials file and project. We'll fix that in the future.
