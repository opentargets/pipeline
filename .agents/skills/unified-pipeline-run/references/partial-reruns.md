# Partial re-runs: clearing steps inside an existing run

Re-running a few steps inside a finished DagRun is the common case: a fix lands and one
dataset needs rebuilding. Almost every mechanism involved can silently do nothing.

## What decides whether a step runs

Every step is an Airflow task group named after the step, starting with
`diff_<step>` → `upload_config_<step>`. What follows depends on the step kind
(`orchestration/src/orchestration/dags/unified_pipeline.py`):

| Kind | After `upload_config_` | Differs |
| --- | --- | --- |
| pis, and pts steps on a VM | `run_` → `delete_vm_` / `end_` | config, manifest artifacts, step result |
| pts steps on Dataproc | `upload_entrypoint_` → `create_cluster_<cluster>` → `run_` → `end_` | config, manifest artifacts, step result |
| gentropy | `cluster_create_<cluster>` → `run_` → `end_` | config, Spark job |

A shared pts cluster is deleted by its own `cluster_delete_<cluster>` task, only once
every step using it has succeeded.

`DiffOperator.choose_branch` (`orchestration/src/orchestration/operators/diff.py`)
decides:

1. **Did a `depends_on` step run in this DagRun?** Each diff task that decides "run"
   records it in the `steps_that_ran` XCom. If an upstream step did, this step runs
   with no config consulted. A VM serving a branch from before opentargets/pipeline#341
   (`d9973f1e`) never finds an upstream run here: check with
   `git merge-base --is-ancestor d9973f1e HEAD` in `/opt/pipeline`.
2. Otherwise each differ in turn; any one returning true runs the step.
3. Otherwise it branches to `end_<step>`: the step is **skipped** and `end_` is green.

`ConfigDiffer` compares the step's current config with `etc/config/<step>.yaml` in the
run bucket. An image change (`pts_version`) is not part of the step config, so bumping
the image alone does not make a step run.

**To force a step, delete its uploaded config** (`ConfigDiffer` runs the step when it
is missing):

```sh
gcloud storage rm gs://open-targets-pipeline-runs/<run>/etc/config/<step>.yaml
```

Do this before clearing any step that no upstream step in the same clear will force.

## How to clear

Use the REST API (setup in `vm-and-config.md`). `airflow tasks clear` cannot target a
manual run in Airflow 3: it selects on `logical_date`, which is null, matches nothing,
and prints nothing.

```
POST /api/v2/dags/unified_pipeline/clearTaskInstances
{
  "dag_run_id": "manual__2026-11-10T20:25:46.920642+00:00",
  "task_ids": ["<step>.diff_<step>", "<step>.upload_config_<step>", "<step>.run_<step>", ...],
  "include_downstream": false,
  "include_upstream": false,
  "only_failed": false,
  "dry_run": true
}
```

- **Always `dry_run: true` first.** Check the states it lists: a `success` you did not
  mean to redo means the selection is wrong.
- **`task_ids` are full task ids** (`<step>.<task>_<step>`), from the table above. A
  group name clears nothing.
- **Keep `include_downstream: false`, and list each step you want re-run.** Downstream
  inclusion drags in everything transitively (an evidence step pulls in
  `pts_association` and everything behind it).
- **`upstream_failed` is a finished state.** Tasks stuck behind a failure need clearing
  too.

## Traps, in the order they bite

### Recovering a failed step: clear from `run_` down

`upload_config_` writes the step config **before** the step runs, so after a failure
the uploaded config already matches. Clearing `diff_` then makes the step skip itself
and report success with no output, and its consumers fail or skip in turn.

Clear from `run_<step>` downward (for Dataproc steps, from the cluster task), leaving
`diff_` and `upload_config_` in `success`. The dry run shows it: no `success` states in
the list. To re-run the differ anyway, delete `etc/config/<step>.yaml` before **every**
attempt; each attempt re-uploads it.

### Polars transformer steps refuse to overwrite

`write_dataset` (`pts/src/pts/transformers/utils/dataset.py`) refuses any destination
that already exists: `pl.PartitionBy` adds part files rather than replacing them, so
writing over a previous run would mix old and new parts. **Delete the output before
re-running a transformer step**, including partial output from a failed attempt:

```sh
gcloud storage rm --recursive gs://open-targets-pipeline-runs/<run>/output/<dataset>/
```

`pyspark` steps write with Spark's overwrite mode and need no delete.

### Re-runs duplicate per-uuid outputs

otter has no resume: a cleared step redoes all its tasks. A destination containing
`${uuid}` gets a fresh name on every attempt, so a re-run writes a second full set
beside the first and the consumer reads both, silently double counting. Today that is
the `literature` pis step (`input/literature/literature_export/${uuid}.json.gz`); find
the current ones with `grep -n 'destination:.*uuid' pis/config.yaml pts/config.yaml`.
Delete those outputs before re-running the step.

### Make a running task terminal before clearing it

Ending an Airflow task does not stop the VM or cluster job it launched. Mark it
`failed` first, check what it left behind, then clear:

```
PATCH /api/v2/dags/unified_pipeline/dagRuns/{run_id}/taskInstances/{task_id}
{"new_state": "failed"}
```

Step VM names are deterministic within a DagRun (`up-<step, _ as ->-<run hash>`), so a
leftover VM makes the next attempt fail with `409 already exists`. Delete it before
retrying:

```sh
gcloud compute instances list --project open-targets-eu-dev --filter='name~^up-'
```

### Config changed on the VM? Wait for the parse

A task cleared before the DAG re-parses runs the old configuration. See
`vm-and-config.md`.

## A worked sequence

Re-running one polars transformer step after an image bump:

```
1. bump pts_version, push, git pull in /opt/pipeline
   (directory mount: no recreate needed for unified_pipeline.yaml)
2. wait for a DAG parse newer than the file's mtime
3. delete output/<dataset>/            # write_dataset refuses to overwrite
4. delete etc/config/<step>.yaml       # force the differ
5. clear the step's tasks with include_downstream: false, dry run first
6. confirm diff_ branched to upload_config_, not end_
7. check the output, not the task status
```

Step 6 is the early warning: `end_<step>` green while `upload_config_<step>` is
`skipped` means the step skipped.

**Done when** `run_<step>` succeeded in this attempt (not skipped), and the output
matches the prediction you wrote before clearing (`verification.md`).
