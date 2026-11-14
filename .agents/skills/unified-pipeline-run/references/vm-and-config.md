# The Airflow VM, and making a config change apply

## Starting and stopping

From `orchestration/`, on the branch you want the VM to serve:

```sh
make cloud-airflow     # deployment/start.sh
```

It applies the terraform in `deployment/`, waits for the VM's `/ready` marker, opens an
SSH tunnel to `localhost:8081` and opens the Airflow UI (`airflow`/`airflow`). Needs
`terraform` and `gcloud`, with application default credentials
(`gcloud auth application-default login`).

**Done when** the tunnel answers:
`curl -s http://localhost:8081/api/v2/monitor/health` reports the scheduler healthy.

- **The branch is set only when the VM is created.** `main.tf` ignores later changes to
  it, so `make cloud-airflow` from another branch leaves an existing VM on its old one.
  To serve another branch, `git checkout` it in `/opt/pipeline`, or destroy and
  recreate the VM.
- **The VM never pulls by itself.** Boot runs `git fetch` and `git checkout <branch>`,
  which does not fast-forward a branch already checked out. After pushing config
  changes, pull on the VM.
- **The tunnel drops** on VM stop/restart and on any container recreate. Re-open it
  with `make tunnel`.
- **Stop keeps state, destroy loses it.** `gcloud compute instances stop` keeps the
  disk, so the Airflow database and run history survive. `make clean-vm` runs
  `terraform destroy`: the Airflow DB and task logs are gone, and the next VM gets a
  new name suffix.

## Airflow API

The API is on `localhost:8081` through the tunnel (`localhost:8080` on the VM). Get a
token from the FAB auth endpoint:

```sh
B=http://localhost:8081
T=$(curl -s -X POST $B/auth/token -H 'Content-Type: application/json' \
      -d '{"username":"airflow","password":"airflow"}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
```

Then pass `-H "Authorization: Bearer $T"`. List endpoints return at most 100 items per
page and silently ignore a larger `limit`: read `total_entries`, or page with `offset`.

## Applying a config change

The repo lives at `/opt/pipeline` (`/opt/orchestration` is a symlink into it). The
Airflow containers mount `orchestration/src`, `pis` and `pts` from it as directories
(`orchestration/compose.yaml`), so a pull reaches them. Push, then on the VM:

```sh
cd /opt/pipeline && git pull
```

A VM created before opentargets/pipeline#350 mounts `pis/config.yaml` and
`pts/config.yaml` as single files, which a pull does not reach: the step runs, succeeds
and fetches the old version. If `sudo docker exec $(sudo docker ps -q --filter
name=scheduler | head -1) ls /opt/pis` lists only `config.yaml`, recreate the Airflow
services once, between runs, from a checkout that includes #350 (`git merge-base
--is-ancestor 35b2900c HEAD`); `--no-deps` keeps postgres and the run history:

```sh
cd /opt/pipeline/orchestration && sudo docker compose up -d --force-recreate --no-deps \
  airflow-scheduler airflow-dag-processor airflow-apiserver airflow-triggerer
```

## Wait for the DAG to re-parse

Step configs and cluster specs are built when the DAG is **parsed**, not when a task
runs. A task cleared before the re-parse runs the old configuration.

```sh
stat -c %Y /opt/pipeline/pis/config.yaml       # the file you changed
curl -s -H "Authorization: Bearer $T" "$B/api/v2/dags/unified_pipeline" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["last_parsed_time"])'
curl -s -H "Authorization: Bearer $T" "$B/api/v2/importErrors" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["total_entries"])'
```

**Done when** `last_parsed_time` is later than the changed file's mtime and there are
0 import errors.

## Image tags

Run the image check before setting or trusting any pinned version:

```sh
cd orchestration && uv run --frozen python -m orchestration.utils.images \
  src/orchestration/dags/config/unified_pipeline.yaml
```

It resolves `pis`, `pts` and `pts-nlp` (a pts version is usable only when both pts
images exist), prints each digest, and exits 1 naming any that is missing.

It does **not** check gentropy. Dataproc installs gentropy from its git tag, which
`clusters.yaml` builds as `v{{gentropy_version}}` (the config value has no leading
`v`). Check the tag exists:

```sh
gh api repos/opentargets/gentropy/git/ref/tags/v<gentropy_version> --jq .object.sha
```

**Done when** the image check exits 0 and the gentropy tag resolves.

## Init actions

The pts Dataproc init action is uploaded from the repo
(`orchestration/src/orchestration/assets/install_dependencies_on_cluster.sh`) into
`<run>/etc/bin/` at the start of every run, so the repo copy is what runs. The gentropy
init action is a shared object in `gs://genetics_etl_python_playground/initialisation/`
(named in `clusters.yaml`), deployed by hand. If gentropy clusters fail at init,
download that object and read it.
