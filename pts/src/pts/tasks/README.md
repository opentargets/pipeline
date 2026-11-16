# Experimental pyspark runner task

The `pyspark` task can be used to launch pyspark jobs. They are local jobs for now, but we'll add
dataproc cluster support.

The file [literature_entity_lut.py](../pyspark/literature_entity_lut.py) is an example of a
pyspark job port of the scala ETL's literature entity lookup table step.


## Launching

Just like any other Otter app:

```sh
$ uv run pts -s literature_entity_lut
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
  literature_entity_lut:
    - name: pyspark literature entity lut
      pyspark: literature_entity_lut
      source:
        matches: intermediate/literature/match
      destination:
        literature_entity_lut: output/literature_entity_lut
      settings:
        partition_count: 35
      properties:
        'spark.sql.shuffle.partitions': '10000'
```

You pass either a `str` or a `dict[str, str]` to both `source` and `destination`, and a `pyspark`
which is a method name inside a python file with the same name under `pts.pyspark` package.

You can also pass a `settings` `dict` for the job, and a `properties` `dict[str, str]` with extra
configuration for Spark.


## Path support

The paths in both `source` and `destination` are relative to either `release_uri` or `work_path`,
in that order. That means, if a `release_uri` is defined (in the form `gs://...`), the spark job
will fetch the data from a GCS bucket. Otherwise, data must be local under `work_path`.

This can be used to develop and test locally with data in the host.

> [!NOTE]
>
> GCS Support is still a bit flimsy, and the application must be launched with:
> `$ GCLOUD_PROJECT=open-targets-prod GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json uv run pts -s literature_entity_lut`
> to specify a credentials file and project. We'll fix that in the future.
