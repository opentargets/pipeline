# Testing steps outside Airflow

To test a chain of pts or gentropy steps on real data without an Airflow VM, build the
same configs and specs the DAG builds, and run them yourself into a test folder.

## Step configs

Build them with the orchestration code, pointed at the test folder:

1. Patch `PipelineRunConfig.__init__` (`orchestration/src/orchestration/models/run_config.py`)
   so `release_uri` is the test folder.
2. Call `UnifiedPipelineConfig().step_config(<step>)`.

pts then reads its sources from and writes its outputs to the test folder, as in
production. Copy the inputs the steps need into it first.

**Done when** every `gs://` path in the generated configs starts with the test folder.

## Dataproc specs

- Build the cluster with `CustomClusterConfig(...).create_cluster()` and the job with
  `GentropyJobBuilder(...).build()` (`orchestration/src/orchestration/operators/dataproc.py`).
- Drop `service_account` when running in `open-targets-genetics-dev`: the default
  service account belongs to `open-targets-eu-dev`.
- Dataproc staging and temp settings take a bucket name, not a folder. Leave the
  default staging bucket.

## VM steps

- COS VMs have no `buildx`, and `pts/Dockerfile` needs BuildKit (`RUN --mount`).
  Download the `docker-buildx` plugin into `$DOCKER_CONFIG/cli-plugins` first.

## Release data

- `gs://open-targets-data-releases` is requester pays: pass
  `--billing-project=open-targets-genetics-dev`.
- Releases are read-only. Copy inputs out of them; never write there.
- In zsh, quote `gs://.../*`, or the shell expands it and the copy fails with
  `no matches found`.
