# POS release: one-time setup and operation

The `release_candidate` and `release` DAGs build and publish a release from a run's
output. **Nothing here is run by an agent**: David applies it. Each command is marked:

- **scratch-side (apply now):** needed for scratch runs; grants nothing on public destinations.
- **prod (apply only when David approves publishing):** touches `open-targets-prod`, the
  release buckets or S3. Nothing public is written before 26.12 is approved.

```sh
SA=pos-release@open-targets-eu-dev.iam.gserviceaccount.com
```

## Setup

### Service account (scratch-side, apply now)

```sh
gcloud iam service-accounts create pos-release --project open-targets-eu-dev \
  --display-name "POS release VMs"
```

The VMs run as this account; Airflow only creates them, so it needs no instance role
itself, but `up-airflow-dev` must be allowed to attach it:

```sh
gcloud iam service-accounts add-iam-policy-binding $SA --project open-targets-eu-dev \
  --member serviceAccount:up-airflow-dev@open-targets-eu-dev.iam.gserviceaccount.com \
  --role roles/iam.serviceAccountUser
```

Logging and monitoring writer, artifact registry reader (pulls the pos image):

```sh
for role in roles/logging.logWriter roles/monitoring.metricWriter roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding open-targets-eu-dev \
    --member serviceAccount:$SA --role $role --condition=None
done
```

### Scratch bucket (scratch-side, apply now)

Uniform bucket-level access, with a 7-day lifecycle so scratch objects expire:

```sh
cat > /tmp/pos-scratch-lifecycle.json <<'JSON'
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 7}}]}
JSON
gcloud storage buckets create gs://open-targets-pos-scratch --project open-targets-eu-dev \
  --location europe-west1 --uniform-bucket-level-access \
  --lifecycle-file /tmp/pos-scratch-lifecycle.json
gcloud storage buckets add-iam-policy-binding gs://open-targets-pos-scratch \
  --member serviceAccount:$SA --role roles/storage.objectAdmin
```

### Scratch-side data grants (scratch-side, apply now)

Scratch BigQuery datasets are created by `pos` in `open-targets-eu-dev`
(`<product>_scratch`, `<product>_scratch_staging`), so the account needs project-level
BigQuery access there:

```sh
for role in roles/bigquery.dataEditor roles/bigquery.jobUser; do
  gcloud projects add-iam-policy-binding open-targets-eu-dev \
    --member serviceAccount:$SA --role $role --condition=None
done
```

A scratch run only reads the run under `release_uri`; its tarballs and step records go
to the scratch bucket. The bucket keeps its object ACLs; these are IAM grants only
(never enable UBLA):

```sh
gcloud storage buckets add-iam-policy-binding gs://open-targets-pipeline-runs \
  --member serviceAccount:$SA --role roles/storage.objectViewer
```

### AWS secret (prod, apply when publishing is approved)

`pos-aws-credentials` holds an AWS shared-credentials file with a `[default]` profile
for the S3 bucket. Rotation is up to whoever holds the AWS account.

```sh
gcloud secrets create pos-aws-credentials --project open-targets-eu-dev \
  --replication-policy automatic
gcloud secrets versions add pos-aws-credentials --project open-targets-eu-dev \
  --data-file <path-to-aws-credentials-file>
gcloud secrets add-iam-policy-binding pos-aws-credentials --project open-targets-eu-dev \
  --member serviceAccount:$SA --role roles/secretmanager.secretAccessor
```

### Pipeline-runs writes (prod, apply when publishing is approved)

A release writes `disk_images/` and overwrites `manifest.json` under `release_uri`, so
create alone is not enough. (`release_candidate.json` is written by Airflow's own
account, not this one.)

```sh
gcloud storage buckets add-iam-policy-binding gs://open-targets-pipeline-runs \
  --member serviceAccount:$SA --role roles/storage.objectUser
```

### Release buckets and BigQuery (prod, apply when publishing is approved)

Bucket level, not project level:

```sh
for b in open-targets-data-releases open-targets-ppp-data-releases; do
  gcloud storage buckets add-iam-policy-binding gs://$b \
    --member serviceAccount:$SA --role roles/storage.objectAdmin
done
```

BigQuery: the account runs jobs in `open-targets-prod` and edits only datasets
`platform` and `platform_staging`. Create staging once by hand, so the account needs no
project-level `bigquery.datasets.create`:

```sh
gcloud projects add-iam-policy-binding open-targets-prod \
  --member serviceAccount:$SA --role roles/bigquery.jobUser --condition=None
bq --project_id open-targets-prod mk --dataset --location EU open-targets-prod:platform_staging
for ds in platform platform_staging; do
  bq show --format=prettyjson open-targets-prod:$ds > /tmp/$ds.json
  # add {"role": "WRITER", "userByEmail": "<SA>"} to "access" in /tmp/$ds.json, then:
  bq update --source /tmp/$ds.json open-targets-prod:$ds
done
```

`WRITER` is the dataset form of `roles/bigquery.dataEditor`. `bigquery` is skipped for
PPP, which has no BigQuery grant.

## Trim (prod)

In this order; step 2 only after its precondition.

1. **Remove `up-airflow-dev`'s project-wide `roles/storage.admin` on
   `open-targets-prod`, then verify** with a unified-pipeline run (or a staging DAG reading
   `open-targets-data-releases`). If the run fails, re-grant it (rollback):

   ```sh
   gcloud projects remove-iam-policy-binding open-targets-prod \
     --member serviceAccount:up-airflow-dev@open-targets-eu-dev.iam.gserviceaccount.com \
     --role roles/storage.admin

   # rollback
   gcloud projects add-iam-policy-binding open-targets-prod \
     --member serviceAccount:up-airflow-dev@open-targets-eu-dev.iam.gserviceaccount.com \
     --role roles/storage.admin --condition=None
   ```

2. **Trim the default compute account** (`<project-number>-compute@developer.gserviceaccount.com`
   of `open-targets-eu-dev`): its `storage.admin` on `open-targets-prod` and on
   `open-targets-data-releases`, and its `billing.projectManager`. Precondition: after
   cut-over and the old repo's archive; first check what else in eu-dev runs as it
   (`gcloud compute instances list --format='value(name,serviceAccounts[].email)'`).

## Operating

Both DAGs ship paused, and a run triggered while paused stays queued (the Airflow VM's
8 h cap keeps running). Unpause `release_candidate` before a unified-pipeline run with
`release_candidate: true` or before triggering it by hand, and unpause `release` before
triggering it (API setup in the `unified-pipeline-run` skill, `vm-and-config.md`):

```sh
for dag in release_candidate release; do
  curl -s -X PATCH "$B/api/v2/dags/$dag?update_mask=is_paused" \
    -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
    -d '{"is_paused": false}'
done
```

### Release candidate

DAG `release_candidate`, param `release_uri` (e.g.
`gs://open-targets-pipeline-runs/do/platform-2612-3`). It starts a VM with ClickHouse
and OpenSearch on two disks, runs `backend`, stops the servers, snapshots the disks and
deletes the VM and disks. Outputs:

- snapshots `<prefix>-<run>-ch` and `<prefix>-<run>-os` in `open-targets-eu-dev`;
- `<release_uri>/release_candidate.json`, the hand-off: the pos image and version, the
  product, the snapshots and the disk sizes. Read it to see what a candidate is.

The guard refuses to start if either snapshot, either working disk
(`pos-<prefix>-<run>-ch|os`) or the hand-off exists. **To rebuild, delete both snapshots
and the hand-off first.**

**Never clear only the clean-up tasks of an old `release_candidate` run.** They reuse
that run's old task states and XComs and could delete another run's same-named disks.
Clear from `names`/`guard` downstream, or trigger a new run.

The unified pipeline triggers this DAG at the end when `release_candidate: true` in
`unified_pipeline.yaml`. Do not `make clean-vm` until that triggered run has finished.

### Release

DAG `release`, params `release_uri` and `scratch` (default false). It reads the
hand-off, restores the disks from the snapshots and runs the hand-off's own pos image:
`tarballs`, `gcs_sync`, `aws_sync`, `bigquery`, then the manifest last. `aws_sync` is
skipped in scratch mode and for PPP; `bigquery` is skipped for PPP.

`scratch=true` requires the candidate's `pos_version` >= `26.12.0rc6` and writes only to:

- `gs://open-targets-pos-scratch/<product>`;
- `gs://open-targets-pos-scratch/runs/<prefix>-<run>` (manifest, run outputs);
- BigQuery `open-targets-eu-dev` datasets `<product>_scratch` and
  `<product>_scratch_staging`, dropped at the end, even after a failure.

Scratch objects expire after 7 days. Nothing reaches the release buckets, S3 or
`open-targets-prod`. To check a scratch run, compare tarball members with the
snapshots, objects and md5s with the run, BigQuery row counts, and the manifest last.
