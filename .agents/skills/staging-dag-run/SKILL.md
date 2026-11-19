---
name: staging-dag-run
description: Test-run a staging DAG (a datasource ingestion or fine-mapping DAG built from `<dag_dir>/config.yaml`) on Dataproc from a laptop, watch it and check its output.
---

# Running a staging DAG

A staging DAG is a package under `orchestration/src/orchestration/dags/<dag_id>/` with a
`config.yaml` read by `StagingDagConfig` (`dags/config/staging_pipeline.py`). Every path
it writes is under the run folder, `environments.<env>.output`.

Staging clusters and VMs live in `open-targets-genetics-dev` (`utils/staging.py`). Pass
`--project open-targets-genetics-dev --region europe-west1` to every `gcloud dataproc`
and `gcloud compute` command below: the gcloud default project may be another one, and
then a leaked cluster lists as nothing.

## Before a run

- **Log in twice.** `gcloud auth login` covers `gcloud storage` and `gcloud compute`;
  `gcloud auth application-default login` covers the Airflow Google operators. If
  tokens expire mid-session (`Reauthentication failed. cannot prompt during
  non-interactive execution`), ask the user to log in again.
- **Write a test to its own folder.** Every staging `config.yaml` ships with `env: prod`,
  and its `output` sits next to the real run folders. Add a test environment (copy
  `prod`) whose `output` is a new, empty folder outside `<data bucket>/prod/`, ending in
  `date -u +%Y%m%dT%H%M%S`, and set `env` to it. Keep this change out of the PR. Check
  the folder is empty: `gcloud storage ls "<run folder>/**"`.
- **Check inputs by listing or reading objects.** `gcloud storage buckets describe` is
  denied on public buckets (e.g. `gs://finngen-public-data-multiome`) even when reading
  works.
- **Check quota** for the clusters in `config.yaml`:
  `gcloud compute regions describe europe-west1 --project open-targets-genetics-dev`
  (`SSD_TOTAL_GB`, `CPUS`). Clusters default to `pd-ssd` disks.

**Done when** both logins work, `env` names the test environment, its run folder is
empty, and the quota covers the clusters.

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

- Without `AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT`, tasks fail with ``The conn_id
  `google_cloud_default` isn't defined``, after the `bin/` uploads have succeeded.
  Re-running into the same folder is fine.
- `dag.test()` runs tasks one at a time: parallel branches run in series.
- `AirflowException: Error during cancellation handling: task_instance is not set on
  the trigger` is logged for every deferrable task and is harmless. Judge each task by
  `[DAG TEST] end task` and `new_state=`.
- **After any failed run, delete the cluster yourself.** `delete_cluster_*` has
  `trigger_rule=ALL_SUCCESS`, so a failed step leaves its cluster up, in a scheduled run
  as well as in `dag.test()`. The cluster name ends in a hash of the run id; take it from
  `gcloud dataproc clusters list`, then `gcloud dataproc clusters delete <name>`.
  `idle_delete_ttl` is only a backstop.

**Done when** every task ended `success`, and both of these list nothing of yours:

```sh
gcloud dataproc clusters list --project open-targets-genetics-dev --region europe-west1
gcloud compute instances list --project open-targets-genetics-dev --filter="name~staging"
```

## Watching a run

- Poll the job state and the count of `[DAG TEST] end task` lines.
- Job errors: `gcloud dataproc jobs describe <job>
  --format="value(driverOutputResourceUri)"`, then `gcloud storage cat <uri>.000000000`.
- Cluster load: `gcloud dataproc clusters describe <cluster> --format=json`, under
  `metrics.yarnMetrics`. `yarn-containers-pending` above 0 with no
  `yarn-vcores-available` means the cluster is too small.
- A running cluster can grow mid-job:
  `gcloud dataproc clusters update <cluster> --num-workers N`. Then set the size that
  worked in `config.yaml`.
- Cluster properties are fixed at creation: a property fix needs a new run.

**Done when** each job has ended and, for a job that failed, the driver output names
the cause.

## gentropy on Dataproc

- A gentropy step that reads block-gzipped TSVs (`Session(use_enhanced_bgzip_codec=True)`)
  needs the cluster property `'spark:spark.jars.packages':
  'org.seqdoop:hadoop-bam:7.10.0'`; the session config adds it too late for
  `spark-submit` (`Compression codec org.seqdoop.hadoop_bam.util.BGZFEnhancedGzipCodec
  not found`). Copy it from `finngen_multiome_ingestion/config.yaml`. Keep the codec
  on: Hadoop's own gzip reader stops after the first BGZF block, silently.
- Size the cluster for the input, and grow it while `yarn-containers-pending` stays
  above 0.

## Checking the output

- Compare counts with the numbers in the gentropy PR, and count the sources too (rows,
  credible sets, features) to catch silent truncation.
- Credible sets: unique `id`s, and every credible set has a study and variants.
- For fine-mapped credible sets (SuSiE, CARMA), the posterior probabilities of each set
  sum to between 0.95 and 1. PICS, top-hit and 99% sets, and sets trimmed by purity
  filters, can fall outside this.
- Read outputs with `duckdb` on a local copy of a sample, not at full scale on a laptop.

**Done when** the counts match the gentropy PR and every check above passes.

## Fixing metadata without a re-run

For a wrong metadata column (a `pubmedId`, a sample size) in a finished run, patching
can replace a re-run:

1. Leave the run folder as it is. Write to a new, empty run folder.
2. Patch only the broken columns. Check rows, types and the other columns are unchanged.
3. Copy the untouched datasets across and compare name, size and `crc32c`.
4. Record the patch script and a "Patch of <date>" note in the new folder's docs, and
   point consumers (`pis/config.yaml`) at it.

Fix the code in gentropy in its own PR, so the next run needs no patch.
