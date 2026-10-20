# Clinical staging

This document was updated on 2026-10-20.

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
| `*_model_revision`, `*_ner_revision` in `pis/config.yaml` | the models, and so the snapshot tags |

## Layout

```{bash}
gs://clinical_staging/store/{stop_reasons,drug_entities,disease_entities,llm_extraction}/  # record stores, shared across versions
gs://clinical_staging/aact/<aact_version>/input/                   # aact.zip and the models, <org>/<name>/<revision>/<file>
gs://clinical_staging/aact/<aact_version>/extraction/              # data.parquet + metadata.json
gs://clinical_staging/aact/<aact_version>/stop_reasons/<tag>/      # data.parquet + metadata.json
gs://clinical_staging/aact/<aact_version>/drug_entities/<tag>/
gs://clinical_staging/aact/<aact_version>/etc/config/              # the config each step ran with
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
2. `pts_aact_trial_extraction` — asks the LLM to read every trial the record
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
   `disease_entities/<tag>`.

ChEMBL is not staged. The release fails on any drug label with no id that no
snapshot covers, whatever its source.

There is no Airflow dependency between this DAG and `unified_pipeline`. The
release names the versions in `pis/config.yaml`; if it names one that was
never staged, its copy fails on a missing path. The OpenAI key is fetched from
Secret Manager onto the VM and mounted into the container.

## Record stores, snapshots and revision tags

A record store holds one record per text, model and model revision, so a new
version only runs the models over texts it has not seen. Each run publishes
the records for its texts as a snapshot: a `data.parquet` plus a
`metadata.json` written last. The snapshot directory is named by a tag built
from the revisions of the models behind it:

| Snapshot | Tag |
| --- | --- |
| `stop_reasons` | `<stop_reasons_model_revision>` |
| `drug_entities` | `<biobert_chemical_ner_revision>+<drugtemist_ner_revision>` |
| `disease_entities` | `<biobert_diseases_ner_revision>` |

`pis/config.yaml` builds the same paths from the same revisions, so the
release can only read a snapshot made with the models it pins. Snapshots are
write-once per (version, tag), and the extraction is write-once per AACT
version.

### Recovery when a publish is refused

A rerun that would change a published snapshot is refused
(`SnapshotConflictError`). To recover, delete that snapshot directory (or the
extraction directory) and rerun the step. The stores keep every computed
record, so only texts they have not seen are computed.

### Model bumps

A new model revision gives a new tag, so a bump needs no deletion: change the
revision in `pis/config.yaml`, rerun the staging steps of each affected
version, then move the release. Until the new snapshot exists, the release
copy fails on the missing tag.

## Rollout

1. Release pts and pis, and bump `pts_version` and `pis_version` in
   `clinical_staging.yaml`, and the pins in `pis/config.yaml`.
2. First run only: set `import_extraction_cache` on the `aact_trial_extraction`
   step to `gs://aact_data/cache/trial_extraction/<schema digest>`, and
   `seed_ner_cache` on `aact_trial_models` to the release NER cache. Remove
   both afterwards. Without them the first run pays for every trial and every
   label again.
3. Trigger the DAG, and check that every snapshot of both sources exists.
4. Only then move the unified pipeline onto the new pts and pis.

`gs://aact_data` is kept as history and is no longer read or written.

## Consumers

`unified_pipeline` copies, through the `clinical_report` step, the archive, the
extraction, the stop-reasons and drug snapshots of the AACT version, the
clinical mining inputs, and the drug and disease snapshots of the clinical
mining version. The `drug` step copies the extraction, which
`pts.chembl_molecule` mines for drug synonyms.

Only AACT trials with an extraction row are included in the clinical report.

## Changelog

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
