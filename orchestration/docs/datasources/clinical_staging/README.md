# Clinical staging

This document was updated on 2026-10-30.

The `clinical_report` step needs data that comes from an LLM or from models:
the trial extraction, the stop reasons, and the drug and disease entities in
AACT trials and in the TTD, EMA and PMDA texts. The release calls no model and
imports no `torch` or `transformers`. Everything is computed ahead of time by
the `clinical_staging` DAG, written to `gs://clinical_staging`, and copied into
the release by `pis/config.yaml`.

AACT (Aggregate Analysis of ClinicalTrials.gov) is published as dated
PostgreSQL archives by CTTI, fetched from
`https://aact.ctti-clinicaltrials.org/static/static_db_copies/monthly/<version>`.
The clinical mining inputs (TTD, EMA, PMDA, the ChEMBL mapping) are curated by
hand under `gs://ot-team/irene/clinical_mining/<clinical_mining_version>/`.

| Pinned in | Names |
| --- | --- |
| `aact_version`, `clinical_mining_version` in `clinical_staging.yaml` | the versions the DAG stages |
| the same names in `pis/config.yaml` | the versions the release copies |
| `*_model_revision`, `*_ner_revision` in `pis/config.yaml` | the models |
| `*_tag` in `pis/config.yaml` | the snapshots the release copies |

## Layout

```{bash}
gs://clinical_staging/store/{stop_reasons,drug_entities,disease_entities,llm_extraction}/  # record stores, shared across versions
gs://clinical_staging/aact/<aact_version>/input/                   # aact.zip and the models, <org>/<name>/<revision>/<file>
gs://clinical_staging/aact/<aact_version>/extraction/              # data.parquet + metadata.json
gs://clinical_staging/aact/<aact_version>/stop_reasons/<tag>/      # data.parquet + metadata.json
gs://clinical_staging/aact/<aact_version>/drug_entities/<tag>/
gs://clinical_staging/<source>/<version>/etc/config/               # the config each step ran with
gs://clinical_staging/clinical_mining/<clinical_mining_version>/input/                # ttd, ema, pmda, chembl mapping, models
gs://clinical_staging/clinical_mining/<clinical_mining_version>/{drug_entities,disease_entities}/<tag>/
```

## The DAG

Each step is named `{stage}_{step}`, where the stage is the application that
runs it, and belongs to a `source` (`aact` or `clinical_mining`). The source
gives the step its release URI, `gs://clinical_staging/<source>/<version>`,
and its version. Each step runs on its own short-lived GCE VM.

1. `pis_aact` — downloads the monthly archive to `input/aact.zip` and the
   stop-reasons, chemical and drugtemist models to `input/models/`.
2. `pts_aact_trial_extraction` (n1-highmem-8) — asks the LLM to read every trial the record
   store does not already hold, and publishes the extraction. A trial that
   fails all 3 attempts is published without, with a warning giving the count;
   a rerun of the version computes nothing.
3. `pts_aact_trial_models` (n2d-standard-16) — runs the stop-reasons model over
   the distinct `why_stopped` texts and the drug NER over the distinct drug
   labels, and publishes `stop_reasons/<tag>` and `drug_entities/<tag>`.
4. `pis_clinical_mining` — copies the TTD, EMA, PMDA and ChEMBL mapping inputs
   and the chemical, drugtemist and disease models.
5. `pts_clinical_mining_models` (n2d-standard-16) — runs the drug and disease
   NER over the TTD, EMA and PMDA texts and publishes `drug_entities/<tag>` and
   `disease_entities/<tag>`. It waits for `pts_aact_trial_models`, because both
   write the shared drug-entity store.

ChEMBL is not staged. The release fails on any drug label with no id that no
snapshot covers, whatever its source.

There is no Airflow dependency between this DAG and `unified_pipeline`. The
release names the versions in `pis/config.yaml`; if it names one that was
never staged, its copy fails on a missing path. The OpenAI key is fetched from
Secret Manager onto the VM and mounted into the container.

## Record stores, snapshots and revision tags

A record store holds one record per text, model and output revision, so a new
version only runs the models over texts it has not seen. Each run publishes
the records for its texts as a snapshot: a `data.parquet` plus a
`metadata.json` written last. The snapshot directory is named by a tag:

```
<hub revision>+<hub revision>-p<digest>
```

The hub revisions come in the order of the kind's models. The digest is the
first 8 hex digits of the sha256 of the kind's parameters and its
`logic_version`, both in `pts/src/pts/clinical_staging/staged_outputs.py`. The tag is the store's
revision and the snapshot directory name, so one tag means one output.

| Snapshot | Hub revisions in the tag |
| --- | --- |
| `stop_reasons` | `stop_reasons_model_revision` |
| `drug_entities` | `biobert_chemical_ner_revision`, `drugtemist_ner_revision` |
| `disease_entities` | `biobert_diseases_ner_revision` |

`pis/config.yaml` pins the full tags as `stop_reasons_tag`, `drug_entities_tag`
and `disease_entities_tag`, and the release copies those paths. A pts test
computes each tag from the pinned hub revisions and fails, printing the new
tag, when a pin is stale. Reading a snapshot staged by code with other
parameters fails before any work, naming both digests.

Snapshots are write-once per (version, tag), and the extraction is write-once
per AACT version.

### Recovery when a publish is refused

A rerun that would change a published snapshot is refused
(`SnapshotConflictError`). Parameter and code changes do not cause it, because
they give a new tag. To recover from a refused extraction, delete the
extraction directory and rerun the step. The stores keep every computed record,
so only texts they have not seen are computed.

Recovering an extraction for an AACT version that was already staged for trial
models changes that version's drug labels. The rerun of `pts_aact_trial_models`
then publishes different data under the same `drug_entities/<tag>/`, because
the tag does not depend on the extraction, and is refused. Delete that
version's `drug_entities/<tag>/` directory as well before rerunning. Releases
that copied that version's extraction or drug snapshot no longer reproduce from
the bucket. The stop reasons and the mining snapshots are unaffected.

### Model bumps

A new model revision gives a new tag, and its snapshot lands beside the old
one, so snapshots need no deletion.

1. Update the hub revision in `pis/config.yaml`.
2. Run the pts test `test_pinned_tags_equal_the_code`. It prints the new tag;
   update the matching `*_tag` pin.
3. Rerun the staging steps of each affected version.

A version that was already staged is the catch: its pis step adds
`input/models/<org>/<name>/<new revision>/` next to the old revision, and the
models step then fails with "staged at 2 revisions". Before rerunning, delete
`input/models/<org>/<name>/<old revision>/` of every bumped model in that
version's directory, rerun its pis step, then its models step. Move the release
last. Until the new snapshot exists, the release copy fails on the missing tag.

### Parameter and code changes

A change to the parameters of a kind, or to the code that computes its output,
also gives a new tag.

1. Change the constants that the kind's `params` reference: `STOP_REASONS_THRESHOLD`
   in `stop_reasons.py`, or `DRUG_NAME_PATTERNS`, `BIOBERT_LABELS`,
   `DRUGTEMIST_LABELS`, `DISEASE_LABELS`, `DISEASE_MIN_LENGTH` and
   `DISEASE_DROP_DIGITS` in `ner.py`. `params` only records them; editing it
   does not change what is computed. When the compute code changes without a
   constant, bump the kind's `logic_version` by one.
2. Update the matching `*_tag` pin in `pis/config.yaml`, with the tag the pin
   test prints.
3. Release pts and bump `pts_version` in `clinical_staging.yaml`. Staging
   computes the tag from the code in the pts image, so restaging with the old
   image reproduces the old tag. Do the same for the release image when moving
   the release.
4. Restage the affected versions. The new snapshot lands beside the old one,
   which stays: snapshots remain write-once.
5. Move the release last.

## Rollout

1. Release pts and pis, and bump `pts_version` and `pis_version` in
   `clinical_staging.yaml`, and the pins in `pis/config.yaml`.
2. First run only, in `pts/config.yaml`: set `import_extraction_cache` on the
   `aact_trial_extraction` step (task `llm_extract`) to
   `gs://aact_data/cache/trial_extraction/<schema digest>`, and `seed_ner_cache`
   on the `aact_trial_models` step (task `trial_models`) to
   `gs://ot-team/irene/clinical_mining/<clinical_mining_version>/input/clinical_report/ner_cache.parquet`.
   The schema digest is the sha256 of the response schema, as the step
   computes it; from `pts/`:

   ```bash
   uv run python -c "import hashlib, json; from pts.transformers.clinical.schemas import ClinicalReportExtractionSchema as M; print(hashlib.sha256(json.dumps(M.model_json_schema(by_alias=True), sort_keys=True).encode()).hexdigest())"
   ```

   Remove both afterwards: once the schema changes, an `import_extraction_cache`
   left set names another schema's namespace, and the step fails. The mining
   disease NER (about 3k texts) is computed in full on the first run. Without
   the import and the seed the first run pays for every trial and every label
   again.
3. Trigger the DAG. It runs `clinical_mining_models` after `aact_trial_models`,
   so the seed covers the mining labels. Check that every snapshot of both
   sources exists.
4. Only then move the unified pipeline onto the new pts and pis.

`gs://aact_data` is kept as history and is no longer read or written, apart
from the one-off import.

## Consumers

`unified_pipeline` copies, through the `clinical_report` step, the archive, the
extraction, the stop-reasons and drug snapshots of the AACT version, the
clinical mining inputs, and the drug and disease snapshots of the clinical
mining version. `pts.chembl_molecule` mines the extraction for drug synonyms
and reads the copy the `clinical_report` step makes.

Only AACT trials with an extraction row are included in the clinical report.

## Changelog

### 2026-10-30

- Snapshot tags end in a digest of the kind's parameters and `logic_version`,
  and pis pins the full tags. A code change gives a new tag, so it no longer
  needs a delete and restage.

### 2026-10-20

- The DAG is now `clinical_staging` and stages both AACT and the clinical
  mining inputs into `gs://clinical_staging`, with revision-tagged snapshots.
  The release reads nothing from `gs://ot-team` or `gs://aact_data` and runs no
  model.
- The LLM record store replaces the extraction cache.

### 2026-10-19

- The extraction code now lives in `pts.transformers.clinical`.

### 2026-08-06

- Initial version. Replaces a one-off OpenAI Batch job run outside the
  pipeline, whose raw output was hand-staged to a personal bucket.
