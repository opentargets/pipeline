# Checking the output

## Predict before you look

Before re-running a step, measure the current output and write down what the change
should do to it. A prediction made beforehand can be wrong; a number read afterwards
just gets explained.

For a fix that should change content but not shape, two invariants catch most mistakes:

```
row count      unchanged
score sum      unchanged
```

## Count rows, not bytes

Byte sizes change with encoding and compression, and a real row loss can hide inside
an expected size change. `drug_mechanism_of_action` once lost 73% of its rows across
three release candidates unnoticed because only sizes were compared.

Row counts from parquet footers are cheap (no data is read):

```python
import polars as pl

def rows(uri: str) -> int:
    return pl.scan_parquet(f'{uri}/**/*.parquet').select(pl.len()).collect().item()

rows('gs://open-targets-pipeline-runs/<run>/output/target')
```

Three comparisons answer different questions:

| Compare with | Answers |
| --- | --- |
| the published release | what users will see change |
| the previous run | what this run alone changed |
| the previous run of the same release cycle | whether this cycle's fixes landed |

## Rows moving is not rows lost

Before reporting a drop, check whether the rows went to `excluded/`:

```
output/study      4,074,174 -> 3,004,247   (-26.3%)
excluded/study      102,696 -> 1,172,741   (+1,070,045)
```

These reconcile exactly, so this was an intended exclusion, not loss. An exact
reconciliation is strong evidence; an approximate one is not.

A large drop can also be a fix: datasets that converge on one row per entity (e.g.
`target`, `target_prioritisation` and `search_target` all reaching the same count)
point to a removed fan-out.

## Reading Airflow honestly

- **Query failures directly** (`state=failed`, read `total_entries`). A listing of
  task instances stops at 100 (`vm-and-config.md`, "Airflow API"), so it can look
  complete and miss failures.
- **A task's `duration` is its previous attempt's** after a clear. Compute elapsed time
  from `start_date`.
- **Clearing resets the DagRun's `start_date`**, so a run that needed intervention
  under-reports its span. `manifest.json` keeps the real step times and survives the VM;
  its steps are keyed `<runner>_<step>` (`pis_variant`, `pts_variant`).
- **The exception text is in a log entry's `error_detail`** key, not `event`.
- **`manifest.json` can miss steps that ran.** On `ds/ppp-2612-1`, four steps that
  succeeded in Airflow (`pis_disease`, `pts_evidence_orphanet`,
  `pts_evidence_uniprot_literature`, `pts_evidence_uniprot_variants`) had no manifest
  entry. Before reporting step counts or times from the manifest, check its step list
  against the `run_` tasks, and say where a time comes from Airflow instead.
- **A failed `run_` can hide a step that succeeded**: when the exit-status log entry is
  lost, the sensor times out (`full-run.md`, failure modes). The manifest, the container
  log and the output settle what happened.

## Step logs

When a step VM finishes, its container output is copied into the `run_<step>` Airflow
task log, which is lost with the VM. It stays in Cloud Logging. VM names are
`up-<step, _ as ->-<run hash>`; find the exact one in the `run_` task log, or with
`gcloud compute instances list --filter='name~^up-'` while the step runs.

Container output:

```sh
gcloud logging read --project open-targets-eu-dev --freshness=2d --limit=500 \
  'resource.type="gce_instance" AND jsonPayload.instance.name="<vm>" AND jsonPayload.container.name="/workload_container"' \
  --format='value(jsonPayload.message)'
```

Startup script output, including the container's exit status:

```sh
gcloud logging read --project open-targets-eu-dev --freshness=2d --limit=500 \
  'resource.type="gce_instance" AND labels.instance_name="<vm>"' \
  --format='value(jsonPayload.message)'
```
