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
| `backend_clickhouse` | syncs `output/` and `view/` to the work dir, creates and loads the ClickHouse database | ClickHouse `:8123` on an empty disk: it fails if the database has tables |
| `backend_opensearch` | syncs `output/` and `export/` to the work dir, writes `croissant.json` to `release_uri`, creates and loads the OpenSearch indices | OpenSearch `:9200` on an empty disk: it fails if an index exists; a complete `output/` (every dataset another dataset references must be present, or croissant generation fails) |
| `tarballs` | streams `/mnt/clickhouse` and `/mnt/opensearch` through tar, pigz and `rclone rcat` to `<release_uri>/disk_images/` | the two disks, restored from snapshots |
| `bigquery` | loads every modelled output into `platform_staging`, then swaps each table into `platform` (Platform only) | BigQuery access |
| `gcs_sync` | `<release_uri>` to the public release bucket, except `view/` | |
| `aws_sync` | rclone to S3, without `view/` (Platform only) | `AWS_SHARED_CREDENTIALS_FILE` |
| `ftp_sync` | manual, from a workstation: `uv run pos -s ftp_sync -r <release_uri>`; over SSH to the EBI cluster, where `gcloud` runs; leaves out `view/` and `excluded/` (Platform only) | SSH key for `$USER` (unset in a container, so paramiko would log in as root) |
| `publish_manifest` | last, after every other step: copies `<release_uri>/manifest.json` to the release bucket and S3 (S3 Platform only) | `AWS_SHARED_CREDENTIALS_FILE` |
| `scratch_cleanup` | scratch mode only: drops the scratch BigQuery datasets (Platform only; for PPP it does nothing and exits 0) | BigQuery access |

- `backend_opensearch` writes nothing to the release buckets besides `croissant.json`.
- The two backend steps are independent, each needs only its own server: run them on separate VMs or one after the other on one.
- `bigquery` tables are `ot_datamodels` datasets present under `<release_uri>/output/`;
  `ot_release` is kept. It leaves prod untouched if any staging load fails or is empty.
  Rerun it after a failed swap: staging stays complete.
- `config/datasets.yaml` maps datasets to OpenSearch indices; the ClickHouse tables are declared on the
  served view models (ClickHouse landing tables, below).
  `config/clickhouse/` holds the server config (`config.d/`, `users.d/`) and nothing else;
  `config/opensearch/` the index settings.

## ClickHouse landing tables

Every ClickHouse table is built from a landing table with an `ot_datamodels` model
(`discover_models()`: the outputs and the `view/` models in `ot_datamodels.views`):
`clickhouse_load` builds the landing table from the model (`pos.clickhouse_ddl`) at load time,
loads it, builds the served tables and dictionaries from it, then drops it.

**Served views** are declared on their models, and every ClickHouse table is one. A view model with
`clickhouse = Serving(...)` (`ot_datamodels.serving`) is one served table named after its `dataset_name`, built
from the whole view: `Serving` holds only the engine (`rocksdb` or `mergetree`), a MergeTree's `order_by`,
the dictionaries built from the table (`ClickHouseDictionary`, which reads the view's own table) and
`column_types`, the ClickHouse type of a primitive by `spec.walk` path where it differs from the dtype's
(`columns()` fails on a path that is not a primitive of the view). A primitive's landing type is its
`column_types` entry, else the dtype's.
Everything else is derived (`pos.clickhouse_ddl`, `served_views()`): the input folder `view/<dataset_name>`, the
landing table `<dataset_name>_log`, the RocksDB key (the model's one `PrimaryKey`) and the products (the model's).
Every view model has a `Serving` (datamodels `tests/test_serving.py`); an output model with `clickhouse` set
fails `served_views()`. A table cut from another (rows filtered, columns chosen) is a view pts writes, with
`copy_view` and `drop_nulls` (the evidence index views, the clinical splits); pos never filters. Evidence is one
table per source (`evidence_<source>_view` and its index views, `datamodels` `views/evidence_views.py`): nothing
builds a union, and the API chooses tables from the datasource ids.

A load drops and creates the landing table, inserts the parquet files (on an error it truncates and inserts
them all once more), checks a RocksDB key is unique in the landing table, builds the table (`allow_nullable_key`
when an `order_by` column is nullable, `OPTIMIZE FINAL` on a MergeTree) and its dictionaries, then drops the
landing table. A view with no parquet fails its load.

**Load order** (`config/config.yaml`): `clickhouse_load_views load served views` appends one `clickhouse_load
clickhouse load <dataset_name>` per served view whose model lists the run's product, so a PPP-only view
(`association_timeseries_view`, `otar_view`, the `encore`, `ot_crispr` and `ot_crispr_validation` evidence views)
loads for PPP alone (`test_the_served_views_each_product_loads` pins both lists). Its `exclude` leaves out
`colocalisation_by_credible_set_view`, which loads alone after it: its RocksDB bulk insert is the largest
ClickHouse memory peak of the step. Platform loads 83 served views and PPP 91, colocalisation included
(`test_every_clickhouse_load_builds_with_the_product`). Every landing table is dropped after its load
(`test_serving_sql`).

Print the generated DDL:

```sh
cd pos && uv run pos-ddl disease_view
cd pos && uv run pos-ddl evidence_eva_view
cd pos && uv run pos-ddl --all --product ppp
```

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
  - `backend_clickhouse` and `backend_opensearch`: the work disk, plus the step's data disk
    mounted into its server container. pos itself only talks to the server over HTTP.
  - `tarballs`: `/mnt/clickhouse` and `/mnt/opensearch` from disks restored from
    snapshots.
- **ClickHouse config:** the image ships `config.d/` and `users.d/` at
  `/app/config/clickhouse/`. Extract them and write them to the ClickHouse data disk
  root before starting ClickHouse, so `tarballs` contains them.
- **Secrets:** `AWS_SHARED_CREDENTIALS_FILE` for `aws_sync` and `publish_manifest`, as a
  file mount from Secret Manager; the service account reads the secret for the mount.
- **Exit code:** non-zero on failure.
- **Outputs:**
  - `backend_clickhouse`: the ClickHouse data disk.
  - `backend_opensearch`: the OpenSearch data disk, plus `croissant.json` under `release_uri`.
  - `tarballs`: `<release_uri>/disk_images/`.
  - `gcs_sync`: the release bucket.
  - `aws_sync`: S3.
  - `bigquery`: BigQuery.
  - `publish_manifest`: the manifest in the release bucket and S3; in scratch mode, the
    scratch bucket only.
  - `scratch_cleanup`: drops the scratch BigQuery datasets.
  - Every step: the manifest, except a PPP `scratch_cleanup`, a no-op that writes none.
- **Retries:** a backend step cannot be retried in place. A retry needs fresh or wiped disks;
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
     -s backend_clickhouse -r gs://<bucket>/platform-2612-N -w /mnt/disks/work -p 8`, and
     the same with `-s backend_opensearch` (each step needs only its own server; one VM may
     run both), then stop the servers before snapshotting the disks.
  4. Run `tarballs` (disks also mounted into the pos container), `bigquery`, `gcs_sync`
     and `aws_sync`; delete the VM and disks.

  The DAG builds ClickHouse on 400 GB and copies it onto 250 GB before snapshotting; a
  manual build may snapshot the build disk.

## Tests

`tests/test_contract.py` checks that every dataset `config/datasets.yaml` or a served view loads from
`output/`, `view/` or `export/` is a `destination:` of a step in `pts/config.yaml`, and that every
`evidence_*` output pts writes has its `evidence_<source>_view`. A dataset pts stops writing fails it. Keep
`datasets.yaml`, the view models and `pts/config.yaml` in step in the same PR.

`tests/test_serving_sql.py` runs the `backend_clickhouse` landing DDL and served-table SQL, in `requires:` order, against empty tables on a live ClickHouse (`POS_TEST_CLICKHOUSE`, default
`localhost:8123`; CI starts one and sets `POS_REQUIRE_CLICKHOUSE=1`; without a server it
skips). Locally: `docker run -d --rm --name pos-test-clickhouse -p 8123:8123 -e
CLICKHOUSE_SKIP_USER_SETUP=1 clickhouse/clickhouse-server:26.8.2.7`. It also loads polars-written
Enum parquet (top level, list, struct) into generated `Enum8` landing tables.

## Schema audit (phase 5)

The phase-5 audit (`scripts/schema_drift.py`, the change register and `scripts/build_register.py`) is in git
history. `docs/api_follow_ups.csv` (`dataset, field, follow_up`) lists the API changes still open.

## Croissant

`pos.croissant` (`src/pos/croissant/`, script `ot_croissant`) builds the
[Croissant](https://docs.mlcommons.org/croissant/docs/croissant-spec.html) descriptor
(`croissant.json`) for a release: one file set per output dataset folder and one record
set describing its fields. Both come from the dataset's `ot_datamodels` output model
(`discover_datasets()`, `dataset_spec`); no parquet is read, only the folder names
(`<column>=<value>` hive partitions set the file set's `includes`). The `ot_croissant`
task (`pos/tasks/ot_croissant.py`) runs in `backend_opensearch`.

| Croissant | From the model |
| --- | --- |
| file set `name` | `dataset_title` |
| record set `description` | `dataset_description`, then `dataset_tags` as `[Tag, Tag]` |
| record set `key` | the `PrimaryKey` columns |
| field `description` | the field's `description`, then ` [bioregistry:<prefix>]` for a `Bioregistry` value |
| field `references` | the value's `Reference` (on a list, its element's) as `<dataset>/<column>` |
| field `dataType` | the polars dtype (`TYPE_DICT` in `constants.py`; an Enum is Text) |

- A list is `repeated`, a struct has `subField`s, a list of lists is flattened with a WARNING.
- A dataset folder with no output model is left out with a WARNING; views are never described.
- A field with no description fails the build (and pos's tests).
- `assets/instance.json` (model `InstanceAnnotation`) is the only curation left: the Platform
  instances (public, ppp).

So a pts change that adds, renames or removes an output column changes croissant through its
model; there is nothing to edit in pos. A `Reference` to a dataset the release lacks makes the
descriptor fail mlcroissant validation.

```sh
cd pos && uv run ot_croissant --help
cd pos && uv run --frozen pytest tests/croissant -rxs
```

The tests (`tests/croissant/`) build record sets from fixture models (types, shapes, keys,
references, file sets of a flat and a partitioned folder, a folder with no model) and from every
output model (a description on every field, keys and references matching the models, every
reference pointing at an existing field), and validate the result with mlcroissant.

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
