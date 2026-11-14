# What the PR, the issue and its comments contain

Templates for the full-run stages in `full-run.md`. The previous run's PR and tracking
issue are the live examples; match their layout.

## Config PR body (stage 1)

- what the run is for, and the run it follows
- a versions table: `main` value, this branch's value, and why each moved (the PRs each
  new rc brings in)
- the digests printed by the image check, and the gentropy tag
- input changes in `pis/config.yaml` / `pts/config.yaml`, or that they are unchanged
- the tracking issue link, and that the run is launched from this branch

## Tracking issue (stage 2)

In `opentargets/issues`, labelled `Pipeline`, titled
`Full pipeline run <run_name> (<release>)`. The body is the run definition:

- what the run is for and which run it follows (link that run's issue)
- a configuration table comparing the previous run with this one: `pis`, `pts`,
  `gentropy`, `ot_curation`, and any input versions that moved
- the configuration PR, and that the image check and the gentropy tag check passed
- **what changed since the previous run**, grouped (genetics, inputs, new or replaced
  steps and datasets, build), each item linking its PR
- where the data goes: `gs://open-targets-pipeline-runs/<run_name>/`
- "Progress is posted here as comments."

## "Run triggered." (stage 4)

- **Airflow run:** the `dag_run_id` and start time (UTC)
- **VM:** its name, and the branch and commit it serves
- **Config check:** the served `run_name` and versions
- **DAG parse:** last parse time, after the last config change; import errors
- **At start:** total task instances, how many are running, 0 failed

## Incident (stage 5)

One comment per incident, posted when found and edited when resolved:

- **what happened:** the step, the task, the error text (from the logs, not the task
  state)
- **cause:** what it is, or that it is not yet known
- **action:** what was done (retry, config fix and its PR, upstream URL change), or
  what is needed from someone
- **outcome:** whether the retry succeeded, and what was checked to confirm it

## "Run complete" (stage 6)

Compare with the previous run where it helps:

- **outcome:** finished time, wall clock, tasks succeeded out of total, how many needed
  retries, and any incidents with links to their comments
- **health:** task counts by state from the full list, manifest steps and their
  results, and no step VMs or clusters left behind
- **longest steps:** the top 10 by run task, with step time and the previous run's time
- **totals:** summed run-task hours, wall clock hours, and their ratio (parallelism)
- **every step's walltime** in a collapsed `<details>` table
- **dataset comparison:** row counts against the previous run and the published
  release, with every large change explained or flagged as open (`verification.md`)

## "VM deleted" (stage 7)

The VM name, that it was deleted, and that the run's outputs, configs and
`manifest.json` remain in the run bucket.
