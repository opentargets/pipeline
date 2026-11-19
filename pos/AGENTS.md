## pos

`pos` (Platform Output Support, the Pipeline Output Stage) loads the release datasets
the pipeline wrote into the Platform's backends (OpenSearch, ClickHouse, BigQuery),
snapshots them, and publishes the release (GCS, FTP, AWS, tarballs, Croissant). It is
an [Otter](https://opentargets.github.io/otter) application, like `pis` and `pts`:
`src/pos/core.py` builds a `Runner` and registers `pos.tasks`; each step in
`config/config.yaml` (Platform) or `config/config_ppp.yaml` (PPP) is a list of tasks.

Its code was imported with full history from `opentargets/pos`. It is not built into
an image yet; the root `makefile` covers it for `dev`, `lint`, `test` and `clean`.

## Running

Run from inside `pos/`: config and Terraform paths are relative to it.

```sh
cd pos && uv sync --frozen --all-groups
cd pos && uv run pos -s <step>            # `local` is the default command
cd pos && uv run --frozen pytest -rxs
cd pos && uv run --frozen ruff check . && uv run --frozen ty check
```

Before a run, set the `scratchpad:` values at the top of the config (`release`,
`data_source`, `database_namespace`, the disk and snapshot names). Steps work in
`work_path` (`./work`, git-ignored).

## Commands

All are `uv run pos <command>`; the default is `local`.

| Command | Does | Runs |
| --- | --- | --- |
| `local -s <step>` | any config step | here |
| `remote -s <step>` | any config step on a VM from `deployment/` | Terraform |
| `backend --product` | the `backend` step: disk snapshots, OpenSearch snapshot, ClickHouse backup | Terraform |
| `tarballs --product` | OpenSearch and ClickHouse tarballs from the disk snapshots | Terraform |
| `clean-remote` | destroys the Terraform VM and workspace | here |
| `bigquery --instance dev\|prod` | the `bigquery_{dev,prod}_load_all` step | here |
| `gcs-sync --product` | the `gcs_sync` step | here |
| `aws-sync` | the `aws_sync` step (Platform only) | here |
| `ftp-sync` | the `ftp_sync` step, over SSH to the EBI cluster (Platform only) | here |
| `restore-clickhouse`, `restore-opensearch` | restore a running instance from its backup, over an SSH tunnel | here |

`p2j` converts parquet to NDJSON (`pos.parquet2json`).

Release commands (`bigquery --instance prod`, `*-sync`) ask for confirmation. `backend`
refuses to run when its snapshots or backups already exist.

## Infrastructure

Until phase 3, `pos` runs on a VM that Terraform creates from `deployment/` (state in
`gs://open-targets-ops/terraform/platform-pos-v2`). Its startup script
(`deployment/startup.sh`) clones `https://github.com/opentargets/pos` at
`--pos-branch` into `/opt/pos` and runs `uv sync`. That clone is the standalone repo,
not this monorepo.

`config/clickhouse/` and `config/opensearch/` hold the database Dockerfiles, schemas
and settings; `config/datasets.yaml` maps datasets to tables and indices;
`config/aws_sync.txt` and `config/rclone.conf` drive the rclone sync.
`.credentials/` and `config/clickhouse/config.d/s3.xml` are git-ignored; the VM
fetches `s3.xml` from instance metadata.

## Croissant

`pos.croissant` (`src/pos/croissant/`, script `ot_croissant`) builds the
[Croissant](https://docs.mlcommons.org/croissant/docs/croissant-spec.html) descriptor
(`croissant.json`) for a release: one distribution per output dataset and one record
set describing its fields. It reads the parquet schemas of the output datasets and
fills in descriptions, keys and foreign keys from curated JSON in
`src/pos/croissant/assets/`. The `ot_croissant` task (`pos/tasks/ot_croissant.py`)
runs it when publishing a release.

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

## Interim rules

- 26.12 may still be built from `opentargets/pos`, which stays live until cut-over.
  Change `pos/` here only with a plan for getting the change into the 26.12 run.
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

(cd "$dest" && /opt/homebrew/bin/git-filter-repo --force \
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
   (`pyproject.toml`, `uv.lock`, `makefile`, `AGENTS.md`), then relock with the pinned
   uv.
4. David pushes; do not push from a session.
