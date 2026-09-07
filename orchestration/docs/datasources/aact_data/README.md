# AACT

This document was updated on 2026-09-04.

AACT (Aggregate Analysis of ClinicalTrials.gov) is published as a monthly
PostgreSQL archive by CTTI, fetched from
`https://aact.ctti-clinicaltrials.org/static/static_db_copies/monthly/<version>`.

The AACT archive version is the single pin used by both the preprocessing and
release DAGs. The preprocessing DAG publishes the exact archive it processed,
and the release copies that archive and its extraction from the same snapshot.

| Pinned in | Names |
| --- | --- |
| `aact_version` in the preprocessing config | the raw CTTI archive downloaded and processed |
| `aact_version` in `unified_pipeline.yaml` | the same archive and `aact_data` snapshot consumed by the release |

The release pin sits with the other release input pins. `pis/config.yaml`
carries the default so the steps still run standalone.

Derived data is stored under `gs://aact_data` with the following structure:

```{bash}
gs://aact_data/<aact_version>/input/           # the raw CTTI archive
gs://aact_data/<aact_version>/prompts/         # the prompt sent for each trial
gs://aact_data/<aact_version>/extraction/      # the LLM extraction
gs://aact_data/<aact_version>/etc/config/      # the config each step ran with
gs://aact_data/cache/trial_extraction/         # the extraction cache, shared across versions
```

The cache sits outside the version directory on purpose. It is keyed on the
trial ID and output schema, so an accepted extraction is reused across AACT
snapshots and does not depend on volatile publication data. The rendered
prompt hash is retained as audit metadata. The model and system instructions
are operational choices and are not part of the key.

## Preprocessing

### aact_trial_extraction dag

The **aact_trial_extraction.py** dag contains the following steps:

1. `pis_aact` — downloads the monthly archive to `input/aact.zip`.
2. `pts_aact_trial_extraction` — builds one prompt per trial and sends the ones
   that are not already in the cache to the OpenAI Responses API. It restores
   and reads the required AACT tables inline through the shared PTS PostgreSQL
   reader.

As in `unified_pipeline`, a step is named `{stage}_{step}`: the stage is the
application that runs it, and the step itself is defined in that application's
own config file, so `pis_aact` runs the `aact` step from
`pis/config.yaml`. The dependency graph is declared in
`config/aact_trial_extraction.yaml`.

Each step runs on its own short-lived GCE VM and the VM is deleted afterwards.
The dag does no diffing: it is already incremental where it matters, because
the extraction only sends the model the trials that are not in its cache.

The output dataset is the trial extraction under
`gs://aact_data/<aact_version>/extraction/`. The raw input archive is retained
under `gs://aact_data/<aact_version>/input/aact.zip`.

The OpenAI key is fetched from Secret Manager onto the VM and mounted into the
container.

## Processing description

The extraction returns, per trial: the drug intent and a confidence for it, the
primary indications and background conditions, and the investigated, comparator
and supportive drugs, each with synonyms and dosages.

Only trials that are not already in the cache are sent to the model. The cache
key hashes the `trial ID` and the `JSON schema` the response is validated
against. Changing AACT text, publications, the model or system instructions
therefore does not re-extract accepted results; changing the schema does.

Only results are cached. A trial the model failed on has no row, so it is
indistinguishable from one never attempted and the next run tries it again.
Nothing tracks failures, retry counts or how long ago something was tried: this
dag runs a few times a year, and re-attempting a few hundred stubborn trials on
each run is cheaper than the bookkeeping needed to remember not to.

Work is sharded, and each shard is written to `cache/trial_extraction/staging/`
as it completes. Rerunning the dag against the same `aact_version` picks those
shards back up rather than paying for the same API calls twice.

### Importing earlier Batch API results

Earlier OpenAI Batch API output can be imported once by setting the optional
`legacy_batch_results` field on `llm_extract` to a directory containing the
`*_output.jsonl` files. Set `legacy_import_only: true` for the first run. The
task parses valid responses, matches them to the current prompts, and seeds the
shared cache using the current trial-and-schema keys without making API calls.
Rate-limited, malformed, and unmatched responses remain cache misses. Inspect
the migration counts, then remove both migration fields and run the DAG
normally; only the remaining misses are sent to the configured model.

## Consumers

`unified_pipeline` copies the raw input and extraction through PIS tasks that
name the shared version literally:

- the `clinical_report` step copies `input/aact.zip` and `extraction/`, which
  `pts.clinical_report` reads inline and
- the `drug` step consumes `extraction/`, which `pts.chembl_molecule` mines for
  drug synonyms and `pts.clinical_report` uses for indications and drug intent

There is no Airflow dependency between this dag and `unified_pipeline`. Moving a
release onto a newer AACT archive means changing the shared `aact_version` pin
and ensuring the preprocessing DAG has published that snapshot first.

If the release asks for an archive nobody processed, its PIS copy fails because
`gs://aact_data/<version>/` does not exist.

Only AACT trials with a successful extraction row are included in the clinical
report. A trial the model answered negatively still produces a row, so a
missing row means the extraction never succeeded and the trial is excluded
rather than being combined with partially populated AACT data.

## Changelog

### 2026-09-04

- Reuse earlier Batch API results by optionally seeding the cache; cache
  identity is the trial ID and output schema, not prompt text, publications,
  the model or system instructions.
- Exclude AACT trials without successful extraction from the clinical report.

### 2026-08-06

- Initial version. Replaces a one-off OpenAI Batch job run outside the pipeline,
  whose raw output was hand-staged to a personal bucket and parsed at release
  time by both `pts.clinical_report` and `pts.chembl_molecule`.
