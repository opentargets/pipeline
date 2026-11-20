## orchestration

Airflow 3 DAGs (`src/orchestration/dags/`), heavily coupled to Google Cloud
primitives (GCS, Dataproc, Google Batch, Compute Engine). `unified_pipeline.py` is
the main DAG that runs the full pipeline; other DAGs handle ingestion/harmonisation
(e.g. `gwas_catalog_*`, `finngen_*`, `eqtl_catalogue_*`).

- DAG configuration lives in `dags/config/` (Pydantic models, plus `ppp/` for the
  partner preview pipeline). In `unified_pipeline.yaml`, a step's `depends_on`
  lists its **direct** dependencies only, leaving edges implied transitively out.
- **A PTS step runs on Dataproc if — and only if — one of its tasks is named
  `pyspark …`** (`models/pts_step.py`, `pts_step_from_config`). Every other step
  runs on a plain GCE VM. The check is on the *task name* in `pts/config.yaml`, not
  the task type, so a single remaining `pyspark` task keeps the whole step on a
  cluster.
- The DAG reads `pis/config.yaml` and `pts/config.yaml` from the Airflow checkout,
  but runs code from the images pinned by `pis_version`/`pts_version` in
  `unified_pipeline.yaml`. A config change that names a new or renamed transformer
  needs a `pts_version` bump to an image containing it.
- Custom `operators/` (dataproc, gcs, gce, batch, diff, and `operators/differs/`)
  wrap the cloud calls; `models/` holds the typed specs (`pts_step`, `batch`, `secret`).
- `assets/dataproc_pts_run.py` is the Dataproc cluster entrypoint, excluded from
  ruff and ty.

Local dev spins Airflow via Docker Compose (`make local-airflow` from this
directory, UI at `localhost:8080`, creds `airflow`/`airflow`). New DAGs under
`dags/` are only picked up on the ~300s filesystem scan.

`release_candidate.py` builds a POS release candidate (disk snapshots and a hand-off
JSON) from a run; `release.py` publishes it (`scratch` for test runs). One-time setup and
operation: `docs/pos_release.md`.

Operating a run (the Airflow VM, triggering, re-running steps, checking output) is
the `unified-pipeline-run` skill.
Test-running a staging DAG (a package with its own `config.yaml`, read by
`dags/config/staging_pipeline.py`) is the `staging-dag-run` skill.
