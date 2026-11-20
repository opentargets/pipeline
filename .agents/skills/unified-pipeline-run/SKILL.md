---
name: unified-pipeline-run
description: Run, re-run or check the output of the unified pipeline (the `unified_pipeline` DAG) on Google Cloud.
---

# Running the unified pipeline

## The one rule

**Green is not evidence.** Each of these has happened on a real run:

- a step reported `success` after fetching the wrong input version
- a step reported `success` after skipping itself, with nothing written
- a config change was pulled on the VM and never reached the containers

Check the thing you care about (the input file, the output row count), not the status
of the task that should have produced it. Before believing a negative result ("no
objects", "0 matches"), run a control query that should return something.

## Router

| What you want | Read |
| --- | --- |
| Start the VM, change a version or config, make the change apply | `references/vm-and-config.md` |
| Run the pipeline end to end | `references/full-run.md` (templates in `references/reporting.md`) |
| Re-run some steps inside an existing run | `references/partial-reruns.md` |
| Check the output; compare against another run or release | `references/verification.md` |
| Test pts or gentropy steps on real data without Airflow | `references/steps-outside-airflow.md` |

## Where things live

| What | Where |
| --- | --- |
| `run_name`, `pis_version`, `pts_version`, `gentropy_version`, boot images | `orchestration/src/orchestration/dags/config/unified_pipeline.yaml` |
| input `*_version`s, `ot_curation`, step sources/destinations/settings | `pis/config.yaml`, `pts/config.yaml` |
| Dataproc cluster specs | `orchestration/src/orchestration/dags/config/clusters.yaml` |
| Airflow VM | `up-airflow-dev-<suffix>`, `europe-west1-d`, project `open-targets-eu-dev` |

With `release_candidate: true` in `unified_pipeline.yaml` the run ends by triggering the
`release_candidate` DAG: don't `make clean-vm` until the triggered run has finished.

A config change reaches a live VM only after a pull and a DAG re-parse; see
`references/vm-and-config.md`.

Every run writes to `gs://open-targets-pipeline-runs/<run_name>/`:

```
input/          fetched by pis
intermediate/   scratch between steps
output/         release datasets
excluded/       rows dropped by validation
etc/config/     the config each step actually ran with
etc/bin/        the pts init action uploaded for this run
manifest.json   otter's per-step record
```

`manifest.json` and the bucket survive the VM. Airflow's database and task logs do not.
