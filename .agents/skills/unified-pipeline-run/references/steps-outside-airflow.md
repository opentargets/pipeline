# Testing steps outside Airflow

To test a chain of pts or gentropy steps on real data without an Airflow VM, build the
same configs and specs the DAG builds, and run them yourself into a test folder.

## Step configs

Build them with the orchestration code, pointed at the test folder:

1. Wrap `PipelineRunConfig.__init__` (`orchestration/src/orchestration/models/run_config.py`):
   call the original, then set `release_uri` to the test folder. Replacing it drops
   `run_name`, `is_ppp` and `release_name`, which `UnifiedPipelineConfig` reads.
2. Call `UnifiedPipelineConfig().step_config(<stage>_<step>)`, e.g. `pts_target` or
   `gentropy_l2g`.

pts then reads its sources from and writes its outputs to the test folder, as in
production. Copy the inputs the steps need into it first.

The clinical steps (`aact_*`, `moa_literature`, `clinical_mining_models`) keep a shared
record store and snapshot outside `release_uri`, under the pts scratchpad variable
`clinical_staging`. Point it at the test folder as well, or the test run's records join
the store that production runs reuse.

**Done when** every `gs://` path in the generated configs starts with the test folder.

## Dataproc specs

- Build the cluster with `CustomClusterConfig(...).create_cluster()`
  (`orchestration/src/orchestration/operators/dataproc.py`).
- In `open-targets-genetics-dev`, pass `service_account=None`. Leaving the field out
  keeps the default, `up-airflow-dev@open-targets-eu-dev`.
- A gentropy job: `GentropyJobBuilder(...).build()`.
- A pts job: `PTSDataprocStep.build_job()` (`models/pts_step.py`). Upload what the DAG
  uploads first: the step config (`config_uri`), the entrypoint
  (`dataproc_script_run_source` to `dataproc_script_run_uri`) and the cluster init
  action (`pts_init_action_source` to `pts_init_action_uri`), as in
  `dags/unified_pipeline.py`.
- Dataproc staging and temp settings take a bucket name, not a folder. Leave the
  default staging bucket.

## VM steps

- COS VMs have no `buildx`, and `pts/Dockerfile` needs BuildKit (`RUN --mount`).
  Download the `docker-buildx` plugin into `$DOCKER_CONFIG/cli-plugins` first.

## Release data

- `gs://open-targets-data-releases` is requester pays: pass `--billing-project` set to a
  project where you can use services. The pipeline bills `open-targets-eu-dev`
  (`requester_pays_project_id`); `open-targets-genetics-dev` also works if you have
  access there.
- Releases are read-only. Copy inputs out of them into the test folder.
