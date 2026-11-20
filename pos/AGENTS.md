## pos

`pos` (Platform Output Support, the Pipeline Output Stage) loads the release datasets
the pipeline wrote into the Platform's backends (OpenSearch, ClickHouse, BigQuery) and
publishes the release (GCS, FTP, AWS, tarballs, Croissant). It is a bare
[Otter](https://opentargets.github.io/otter) application, like `pis` and `pts`:
`src/pos/core.py` builds a `Runner` and registers `pos.tasks`; `config/config.yaml`
declares the steps.

It runs from an image (`pos/Dockerfile`, built from the repo root) and talks to
ClickHouse and OpenSearch over `localhost`. It starts no servers and creates no
infrastructure.

## Running

```sh
pos -s <step> -r <release_uri> -w <work> -p <pool>   # -l <level> optional
cd pos && uv run pos -s <step> -r <release_uri> -w ../work
cd pos && uv run --frozen pytest -rxs
cd pos && uv run --frozen ruff check . && uv run --frozen ty check
```

`release_uri` names the run: its last segment must be `<product>-YYMM-N[-suffix]`
(`gs://.../platform-2612-3`; `parse_release_uri` also takes a local path, for tests,
but otter's `-r` needs a scheme and host). `pos.run` derives product,
release, `database_namespace` and the product's fixed values (buckets, BigQuery
datasets) from it. A task with `products: [...]` runs only for those products.

## Steps

| Step | Does | Needs |
| --- | --- | --- |
| `backend` | syncs `output/`, `view/`, `export/` to the work dir, writes `croissant.json` to `release_uri`, creates and loads the OpenSearch indices and the ClickHouse database | ClickHouse `:8123`, OpenSearch `:9200`, on empty disks: it fails if the database has tables or an index exists |
| `tarballs` | streams `/mnt/clickhouse` and `/mnt/opensearch` through tar, pigz and `rclone rcat` to `<release_uri>/disk_images/` | the two disks, restored from snapshots |
| `bigquery` | loads every modelled output into `platform_staging`, then swaps each table into `platform` (Platform only) | BigQuery access |
| `gcs_sync` | `<release_uri>` to the public release bucket | |
| `aws_sync` | rclone to S3 (Platform only) | `AWS_SHARED_CREDENTIALS_FILE` |
| `ftp_sync` | manual, from a workstation: `uv run pos -s ftp_sync -r <release_uri>`; over SSH to the EBI cluster, where `gcloud` runs (Platform only) | SSH key for `$USER` (unset in a container, so paramiko would log in as root) |
| `publish_manifest` | last, after every other step: copies `<release_uri>/manifest.json` to the release bucket and S3 (S3 Platform only) | `AWS_SHARED_CREDENTIALS_FILE` |
| `scratch_cleanup` | scratch mode only: drops the scratch BigQuery datasets (Platform only; for PPP it does nothing and exits 0) | BigQuery access |

- `backend` writes nothing to the release buckets besides `croissant.json`.
- `bigquery` tables are `ot_datamodels` datasets present under `<release_uri>/output/`;
  `ot_release` is kept. It leaves prod untouched if any staging load fails or is empty.
  Rerun it after a failed swap: staging stays complete.
- `config/datasets.yaml` maps datasets to ClickHouse tables and OpenSearch indices.
  `config/clickhouse/` holds the SQL, schemas and server config (`config.d/`,
  `users.d/`); `config/opensearch/` the index settings.

## Scratch mode

`POS_SCRATCH=1` (unset otherwise; any other value fails) runs a step against scratch
destinations, for test runs that must not touch the public release:

- The run location moves from `release_uri` to
  `gs://open-targets-pos-scratch/runs/<prefix>-<run>`, so the manifest and every
  `release_uri` output go there. Inputs (`data_source`) are still read from
  `release_uri`. The scratchpad key `run_location` holds the effective location in
  both modes.
- `release_gcs_bucket` becomes `gs://open-targets-pos-scratch/<product>`; BigQuery
  goes to `open-targets-eu-dev`, datasets `platform_scratch` and
  `platform_scratch_staging` (`SCRATCH_PRODUCTS` in `src/pos/run.py`).
- There is no FTP or AWS address: `aws_sync`, `ftp_sync` and the S3 copy in
  `publish_manifest` are `scratch: skip`, and `aws_sync` and `ftp_sync` refuse to run.
- `scratch_cleanup` (`scratch: only`) drops the scratch datasets; it refuses any
  project but `open-targets-eu-dev` and any dataset not ending in `_scratch` or
  `_scratch_staging`.
- The log opens with a `SCRATCH MODE` line listing the swapped values.
- A task gated with `scratch: skip|only` runs only outside or only in scratch mode.
- An image without scratch mode ignores the variable: orchestration refuses scratch on
  older images.

## Image contract (phase 4)

- **Command:** `pos -s <step> -r <release_uri> -w /mnt/disks/work -p <pool>`, with
  `-l <level>` optional.
- **Network:** ClickHouse on `localhost:8123` and OpenSearch on `localhost:9200`, so the
  pos container needs host networking or an equivalent.
- **Mounts:**
  - `backend`: the work disk, plus the ClickHouse and OpenSearch data disks mounted
    into their containers. pos itself only talks to them over HTTP.
  - `tarballs`: `/mnt/clickhouse` and `/mnt/opensearch` from disks restored from
    snapshots.
- **ClickHouse config:** the image ships `config.d/` and `users.d/` at
  `/app/config/clickhouse/`. Extract them and write them to the ClickHouse data disk
  root before starting ClickHouse, so `tarballs` contains them.
- **Secrets:** `AWS_SHARED_CREDENTIALS_FILE` for `aws_sync` and `publish_manifest`, as a
  file mount from Secret Manager; the service account reads the secret for the mount.
- **Exit code:** non-zero on failure.
- **Outputs:**
  - `backend`: the data disks, plus `croissant.json` under `release_uri`.
  - `tarballs`: `<release_uri>/disk_images/`.
  - `gcs_sync`: the release bucket.
  - `aws_sync`: S3.
  - `bigquery`: BigQuery.
  - `publish_manifest`: the manifest in the release bucket and S3; in scratch mode, the
    scratch bucket only.
  - `scratch_cleanup`: drops the scratch BigQuery datasets.
  - Every step: the manifest, except a PPP `scratch_cleanup`, a no-op that writes none.
- **Retries:** `backend` cannot be retried in place. A retry needs fresh or wiped disks;
  the rerun guard refuses otherwise.
- **BigQuery datasets:** created only if missing (`get_dataset` first), so the prod account
  needs no project-level dataset create; an existing dataset is never deleted or recreated.
- **Concurrency:** never run two `bigquery` steps at once; they share
  `platform_staging`.
- **Ordering:** `tarballs` must finish before `gcs_sync`; `publish_manifest` last. The
  public tarballs reach the release bucket only through `gcs_sync`.
- **Service account:** read `open-targets-pipeline-runs`; write the release buckets
  (`open-targets-data-releases`, `open-targets-ppp-data-releases`); create datasets and
  write tables in BigQuery `open-targets-prod` (EU). Scratch mode: write
  `gs://open-targets-pos-scratch`; create, write and delete datasets in BigQuery
  `open-targets-eu-dev`.
- **Labels:** `tool=pos`, `step=<step>`.

## Building a release

- `pos@v26.12.0rc3`, the last tag with `pos remote` and Terraform, cannot build 26.12:
  its `backend` tasks `require:` names that exist only as explode children, and otter
  26.9.1 `load_specs` exits with EINVAL on them.
- The `release_candidate` and `release` DAGs (`orchestration/`) build and publish a
  release from a run; setup and operation are in `orchestration/docs/pos_release.md`.
- Fallback by hand, on an `n2d-highmem-96` (256 GB runs ClickHouse out of memory) with
  400 GB pd-ssd disks at `/mnt/clickhouse` and `/mnt/opensearch` and a work disk at
  `/mnt/disks/work`. The exact server command lines are `server_lines` in
  `orchestration/src/orchestration/operators/gce_backend.py`:
  1. Copy `config.d` and `users.d` from `/app/config/clickhouse/` in the image
     (`docker create` + `docker cp`) into `/mnt/clickhouse`; make `data/` on both disks
     and `chown 1000:1000 /mnt/opensearch/data`.
  2. Start the stock servers (`clickhouse/clickhouse-server:26.8.2.7`,
     `opensearchproject/opensearch:3.5.0`) on the host network, data on the disks.
     ClickHouse takes `--ulimit nofile=262144:262144`. For OpenSearch run
     `sysctl -w vm.max_map_count=262144` first, pass `--ulimit memlock=-1:-1
     --ulimit nofile=65536:65536`, and set `OPENSEARCH_JAVA_OPTS='-Xms31g -Xmx31g'`,
     `discovery.type=single-node`, `DISABLE_SECURITY_PLUGIN=true`,
     `bootstrap.memory_lock=true`, `thread_pool.write.queue_size=-1` and the three
     `search.insights.top_queries.{latency,cpu,memory}.enabled=false`.
  3. Run `docker run --rm --network host -v /mnt/disks/work:/mnt/disks/work <pos-image>
     -s backend -r gs://<bucket>/platform-2612-N -w /mnt/disks/work -p 8`, then stop the
     servers before snapshotting the disks.
  4. Run `tarballs` (disks also mounted into the pos container), `bigquery`, `gcs_sync`
     and `aws_sync`; delete the VM and disks.

## Tests

`tests/test_contract.py` checks that every dataset `config/datasets.yaml` loads from
`output/`, `view/` or `export/` is a `destination:` of a step in `pts/config.yaml`. A
dataset pts stops writing fails it. Keep `datasets.yaml` and `pts/config.yaml` in step
in the same PR.

## Croissant

`pos.croissant` (`src/pos/croissant/`, script `ot_croissant`) builds the
[Croissant](https://docs.mlcommons.org/croissant/docs/croissant-spec.html) descriptor
(`croissant.json`) for a release: one distribution per output dataset and one record
set describing its fields. It reads the parquet schemas of the output datasets and
fills in descriptions, keys and foreign keys from curated JSON in
`src/pos/croissant/assets/`. The `ot_croissant` task (`pos/tasks/ot_croissant.py`)
runs in `backend`.

Curation assets:

| File | Describes | Model (`models.py`) |
| --- | --- | --- |
| `assets/distribution.json` | each dataset: `nice_name`, `description`, `tags`, `key` | `DistributionAnnotation` |
| `assets/recordset/<dataset>.json` | each field of a dataset: `description`, `isPrimaryKey`, `foreign_key` | `RecordsetFieldAnnotation` |
| `assets/instance.json` | the Platform instances (public, ppp) | `InstanceAnnotation` |

- Every model is `extra='forbid'`: an unknown key fails validation.
- Field ids are relative to the dataset (`targetId`, not `evidence/targetId`). Struct
  and array-of-struct sub-fields need their own entries, keyed `parent/field`
  (e.g. `trialSponsor/agencyClass`).
- `foreign_key` is a full `dataset/field` id and must exist in the corpus.

Missing curation never fails; each gap is a WARNING in the log and a silent defect in
the descriptor:

- A field with no description ships the literal `PLACEHOLDER for <field> description`.
  A description in the parquet field metadata takes precedence over the curation.
- A curated key component missing from the schema is dropped from the record set key.
  The key still validates, but may no longer be unique; check it against the data
  after an output schema change.
- A dataset missing from `distribution.json` ships "Description of the distribution
  '<id>' is not available."
- A snake_case column is warned about; output columns are camelCase.

So a pts change that adds, renames or removes an output column is a two-package
change: update `assets/recordset/<dataset>.json` in the same PR. A new dataset also
needs a `distribution.json` entry.

```sh
cd pos && uv run ot_croissant --help
cd pos && uv run --frozen pytest tests/croissant -rxs
```

The tests (`tests/croissant/`) check the curation files against the models, unique
ids, the absence of dataset prefixes, that every foreign key points at an existing
field, and that the assets ship in the wheel. They do not check curation against the
real output schemas: to see the warnings for a release, run `ot_croissant` on its
output datasets and read the log.

## Cut-over

- 26.12 is built from the monorepo `pos/`, by the DAGs (`orchestration/docs/pos_release.md`), or the VM fallback above.
- pos is not ruff-formatted tree-wide: never run `ruff format` over it. Only
  `ruff check` and `ty check` are enforced.

## Syncing from `opentargets/pos`

New commits on the standalone repo reach the monorepo by rewriting it into a `pos/`
subdirectory and merging the result. Save this as `pos-rewrite.sh`:

```sh
#!/usr/bin/env bash
# Rewrite opentargets/pos into a pos/ subdirectory for the pipeline monorepo.
# Usage: pos-rewrite.sh <clone-path>   (path must not exist yet)
set -euo pipefail

dest=${1:?usage: pos-rewrite.sh <clone-path>}
keep=(main 2612-schema-updates api-rewrite)

/usr/bin/git clone --quiet https://github.com/opentargets/pos.git "$dest"
for b in "${keep[@]}"; do
  [ "$b" = main ] || /usr/bin/git -C "$dest" branch --quiet "$b" "origin/$b"
done
/usr/bin/git -C "$dest" checkout --quiet main
/usr/bin/git -C "$dest" tag -l | xargs -r /usr/bin/git -C "$dest" tag -d >/dev/null
/usr/bin/git -C "$dest" remote remove origin
/usr/bin/git -C "$dest" for-each-ref --format='%(refname:short)' refs/heads |
  while read -r b; do
    case " ${keep[*]} " in *" $b "*) ;; *) /usr/bin/git -C "$dest" branch -D --quiet "$b" ;; esac
  done

(cd "$dest" && git-filter-repo --force \
  --to-subdirectory-filter pos \
  --message-callback '
import re
return re.sub(rb"(?<![\w/#])#(\d+)(?!\w)", rb"opentargets/pos#\1", message)
')

/usr/bin/git -C "$dest" tag -l | xargs -r /usr/bin/git -C "$dest" tag -d >/dev/null
/usr/bin/git -C "$dest" for-each-ref --format='%(refname)'
```

Then, in a monorepo worktree:

1. `bash pos-rewrite.sh <new-clone-path>` (the path must not exist).
2. `git fetch <new-clone-path> main`, then check `git merge-base HEAD FETCH_HEAD`
   finds the previously merged tip. The rewrite is deterministic, so the old commits
   keep their ids and `git merge FETCH_HEAD` brings in only the new ones. If there is
   no merge base, stop and ask.
3. Resolve conflicts in `pos/` in favour of the monorepo's package conventions
   (`pyproject.toml`, `uv.lock`, `Makefile`, `AGENTS.md`), then relock with the pinned
   uv.
4. David pushes; do not push from a session.

The import used git-filter-repo a40bce548d2c. The same SHAs need the same
filter-repo version and options.
