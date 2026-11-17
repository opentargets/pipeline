# A full run, start to finish

Seven stages, in order. Each ends on a **Done when** line; finish it before starting
the next. Every check in them has failed a run at least once. What each PR, issue and
comment contains is in `reporting.md`.

| # | Stage | Leaves behind |
| --- | --- | --- |
| 1 | Configure on a new branch off `main`, open a draft PR | draft PR in `opentargets/pipeline` |
| 2 | Open the tracking issue | issue in `opentargets/issues` |
| 3 | Spawn the Airflow VM from the branch | VM serving the branch |
| 4 | Trigger the run | "Run triggered." comment |
| 5 | Watch it; handle incidents | one comment per incident |
| 6 | Report the finished run and its statistics | "Run complete" comment |
| 7 | Delete the Airflow VM | "VM deleted" comment |

Read the previous run's tracking issue first. Its "Run complete" comment is the
reference for how long this run should take and which steps are slowest. The newest
one is:

```sh
gh issue list -R opentargets/issues --label Pipeline --state all --limit 1 \
  --search 'Full pipeline run in:title'
```

## 1. Configure on a branch and open a draft PR

```sh
git fetch origin && git switch -c config/full-run-<YYMM>-<N> origin/main
```

Change only the run definition:

- **`run_name`** in `unified_pipeline.yaml`: `<prefix>/<platform|ppp>-<YYMM>-<N>`. It
  decides the output path, and `ppp` switches on the partner preview overrides and
  `ppp_only` steps. Use a new `N` for every run; reusing one collides with the previous
  run's outputs and uploaded configs.
- **`pis_version`, `pts_version`** in `unified_pipeline.yaml`: the newest rcs on
  `main`, moved together, since orchestration changes on `main` can need a newer pis
  or pts. `gentropy_version` only when a newer one is wanted.
- **`ot_curation` and input versions** in `pis/config.yaml`, when they move.

Run the image and gentropy tag checks (`vm-and-config.md`, "Image tags"). Commit
(`orchestration: configure the unified pipeline for <run_name>`), push, and open a
**draft** PR against `main` (`reporting.md`).

The PR stays draft while the run is going. Merging it, or any other PR, is the run
owner's call: ask.

**Done when** both checks pass for the pushed commit and the draft PR is open.

## 2. Open the tracking issue

Open it in `opentargets/issues` (`reporting.md`), then add its link to the PR body.

**Done when** the issue and the PR link to each other.

## 3. Spawn the Airflow VM from the branch

From `orchestration/` on the config branch, with no Airflow VM running:

```sh
gcloud compute instances list --project open-targets-eu-dev --filter='name~^up-airflow-dev'
make cloud-airflow
```

The VM serves the branch it was created from (`vm-and-config.md`). Confirm on the VM:

```sh
cd /opt/pipeline && git rev-parse --abbrev-ref HEAD && git log --oneline -1
grep -E '^(run_name|pis_version|pts_version|gentropy_version):' \
  orchestration/src/orchestration/dags/config/unified_pipeline.yaml
```

**Done when** the VM serves the branch's head commit and the DAG has re-parsed with 0
import errors (`vm-and-config.md`).

## 4. Trigger the run

The DAG ships paused, and a run triggered while paused stays queued. Unpause, then
trigger (API setup in `vm-and-config.md`):

```sh
curl -s -X PATCH "$B/api/v2/dags/unified_pipeline?update_mask=is_paused" \
  -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"is_paused": false}'

curl -s -X POST "$B/api/v2/dags/unified_pipeline/dagRuns" \
  -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"logical_date": null, "conf": {}}'
```

`logical_date: null` makes it a manual run. Keep the returned `dag_run_id`
(`manual__<timestamp>`) as `$RID`; every later API call needs it. Post
"Run triggered." (`reporting.md`).

**Done when** the comment is posted and tasks are running.

## 5. Watch it, and report every incident

Count task instances by state rather than listing them:

```sh
curl -s -H "Authorization: Bearer $T" \
  "$B/api/v2/dags/unified_pipeline/dagRuns/$RID/taskInstances?limit=1&state=failed" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["total_entries"])'
```

Most steps run on their own GCE VM, named `up-<step, _ as ->-<run hash>` (e.g.
`up-pis-variant-…`), which Compute Engine deletes after 8 hours regardless. Which pts
steps run on Dataproc instead is in `orchestration/AGENTS.md`; gentropy steps always
do. No step sets `execution_timeout`: a step running well past its time in the previous
run is stuck, not slow; read its logs (`verification.md`).

**Every incident gets its own issue comment** (`reporting.md`), posted when found and
updated when resolved. An incident is anything that needs action: a failed task, a
stuck step, a retry, a config fix, a wrong-looking output.

Retrying a failed step needs no permission; follow `partial-reruns.md`, where a
careless clear makes a failed step skip itself and go green. Retry the same step at
most **4 times**; then stop, leave it failed, and say so on the issue. A fix that
changes code goes in its own PR, not the config PR, and merging it is the run owner's
call.

Failure modes seen on real runs:

| Symptom | Cause |
| --- | --- |
| every Dataproc cluster fails at init | the init action failing, e.g. a public download (GitHub, PyPI) erroring |
| a step fails with `409 ... already exists` | a VM left over from an earlier attempt (`partial-reruns.md`) |
| a step succeeds but used the old config | cleared before the DAG re-parsed the pulled config (`vm-and-config.md`) |
| a step fails reaching an external service (OpenAI, GitHub, Hugging Face, an upstream data host) | public services rate-limit or fail at fan-out scale; retry, or check whether the upstream URL moved |
| one failed step shows as several failures | `delete_vm_*` and downstream tasks fail or go `upstream_failed` alongside it |

**Done when** the DagRun has ended and every incident comment has an outcome.

## 6. Report the finished run and gather statistics

When the DagRun ends, check it rather than its state: count tasks by every state, and
read `manifest.json` (every step's `result` is `success`).

**Save the task instances before anything else.** They live in the Airflow database
on the VM and go with it. Page through the API and write them to a file:

```python
import json, urllib.request

def task_instances(base, token, run_id):
    out, offset = [], 0
    while True:
        url = f'{base}/api/v2/dags/unified_pipeline/dagRuns/{run_id}/taskInstances?limit=100&offset={offset}'
        req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
        page = json.load(urllib.request.urlopen(req))
        out += page['task_instances']
        offset += 100
        if offset >= page['total_entries']:
            return out
```

From them, per step:

- **run task:** `end_date - start_date` of `run_<step>` (the step's own work)
- **step:** first start to last end across the step's task group (adds VM or cluster
  create and delete)
- `try_number` above 1 marks a retry

Check nothing was left behind:

```sh
gcloud compute instances list --project open-targets-eu-dev --filter='name~^up-' \
  --format='value(name)'
gcloud dataproc clusters list --project open-targets-eu-dev --region europe-west1 \
  --format='value(clusterName)'
```

Post "Run complete" (`reporting.md`), then mark the config PR ready for review.

**Done when** the task instances are saved to a file outside the VM, the "Run
complete" comment is posted, and the PR is ready for review.

## 7. Delete the Airflow VM

Only after the statistics are saved and posted. From `orchestration/` on the config
branch:

```sh
make clean-vm
terraform -chdir=deployment state list
gcloud compute instances list --project open-targets-eu-dev --filter='name~^up-' \
  --format='value(name)'
```

`make clean-vm` runs `terraform destroy`: the Airflow database and task logs are gone
for good. Post "VM deleted" (`reporting.md`).

**Done when** both listing commands print nothing and the comment is posted.
