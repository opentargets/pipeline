---
name: staging-dag-run
description: Test-run a staging DAG (a datasource ingestion or fine-mapping DAG built from `<dag_dir>/config.yaml`) on Dataproc from a laptop, watch it and check its output.
---

# Running a staging DAG

A staging DAG is a package under `orchestration/src/orchestration/dags/<dag_id>/` with a
`config.yaml` read by `StagingDagConfig` (`dags/config/staging_pipeline.py`). Every path
it writes is under the run folder, `environments.<env>.output`.

## Before a run

- **Log in twice.** `gcloud auth login` covers `gcloud storage` and `gcloud compute`;
  `gcloud auth application-default login` covers the Airflow Google operators. Tokens
  expire mid-session with `Reauthentication failed. cannot prompt during
  non-interactive execution`: log in again, do not work around it.
- **Never write over data.** Set `output` to a new folder
  `<data bucket>/<env>/<YYYYMMDDTHHMMSS>` (`date -u +%Y%m%dT%H%M%S`). List the bucket
  first, since other runs live there, and check the folder is empty:
  `gcloud storage ls "<run folder>/**"`.
- **Check inputs by listing or reading objects.** `gcloud storage buckets describe` is
  denied on public buckets (e.g. `gs://finngen-public-data-multiome`) even when reading
  works.
- **Check quota** for the clusters in `config.yaml`:
  `gcloud compute regions describe europe-west1 --project open-targets-genetics-dev`
  (`SSD_TOTAL_GB`, `CPUS`). Clusters default to `pd-ssd` disks.

**Done when** both logins work, the run folder is empty, and the quota covers the
clusters.

## Running from a laptop

Run each branch from its own git worktree, with a scratch `AIRFLOW_HOME`:

```sh
export AIRFLOW_HOME=<scratch>/airflow_<dag_id>
export AIRFLOW__CORE__DAGS_FOLDER=<worktree>/orchestration/src/orchestration/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT='{"conn_type": "google_cloud_platform", "extra": {"project": "open-targets-genetics-dev"}}'
cd <worktree>/orchestration
uv run --frozen airflow db migrate
uv run --frozen python -c "from orchestration.dags.<dag_id> import dag; dag.test()"
```

- `python -m orchestration.dags.<dag_id>` fails: the DAG is a package (`is a package
  and cannot be directly executed`).
- Without `AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT`, tasks fail with ``The conn_id
  `google_cloud_default` isn't defined``, after the `bin/` uploads have succeeded.
  Re-running into the same folder is fine.
- `dag.test()` runs tasks one at a time: parallel branches run in series.
- `AirflowException: Error during cancellation handling: task_instance is not set on
  the trigger` is logged for every deferrable task and is harmless. Judge each task by
  `[DAG TEST] end task` and `new_state=`.
- **`dag.test()` stops at the first failed task and never runs `delete_cluster_*`.**
  Delete the cluster yourself:
  `gcloud dataproc clusters delete <name>-<hash> --region europe-west1`.
  `idle_delete_ttl` is only a backstop.

**Done when** every task ended `success`, and both of these list nothing of yours:

```sh
gcloud dataproc clusters list --region europe-west1
gcloud compute instances list --filter="name~staging"
```

## Watching a run

- Poll the job state and the count of `[DAG TEST] end task` lines. `tail -F` on
  several logs floods the output with `==>` headers.
- Job errors: `gcloud dataproc jobs describe <job>
  --format="value(driverOutputResourceUri)"`, then `gcloud storage cat <uri>.000000000`.
- Cluster load: `gcloud dataproc clusters describe <cluster> --format=json`, under
  `metrics.yarnMetrics`. `yarn-containers-pending` above 0 with no
  `yarn-vcores-available` means the cluster is too small.
- A running cluster can grow mid-job:
  `gcloud dataproc clusters update <cluster> --num-workers N` (4 to 16 took about
  7 minutes). Then set the size that worked in `config.yaml`.
- Cluster properties are fixed at creation: a property fix needs a new run.

## gentropy on Dataproc

- A gentropy step that reads block-gzipped TSVs (`Session(use_enhanced_bgzip_codec=True)`)
  needs the cluster property `'spark:spark.jars.packages':
  'org.seqdoop:hadoop-bam:7.10.0'`; the session config adds it too late for
  `spark-submit` (`Compression codec org.seqdoop.hadoop_bam.util.BGZFEnhancedGzipCodec
  not found`). Copy it from `finngen_multiome_ingestion/config.yaml`. Keep the codec
  on: Hadoop's own gzip reader stops after the first BGZF block, silently.
- Size the cluster for the input: about 340 GB of gzipped TSVs needed 16
  `n2d-highmem-16` workers (26 minutes); 4 workers left 62 executors pending.

## Checking the output

- Compare counts with the numbers in the gentropy PR, and count the sources too (rows,
  credible sets, features) to catch silent truncation.
- Credible sets: unique `id`s, every credible set has a study and variants, and the
  posterior probabilities of each credible set sum to between 0.95 and 1.
- Read outputs with `duckdb` on a local copy of a sample, not at full scale on a laptop.

## Fixing metadata without a re-run

For a wrong metadata column (a `pubmedId`, a sample size) in a finished run, patching
can replace a re-run:

1. Leave the run folder as it is. Write to a new, empty run folder.
2. Patch only the broken columns. Check rows, types and the other columns are unchanged.
3. Copy the untouched datasets across and compare name, size and `crc32c`.
4. Record the patch script and a "Patch of <date>" note in the new folder's docs, and
   point consumers (`pis/config.yaml`) at it.

Fix the code in gentropy in its own PR, so the next run needs no patch.
