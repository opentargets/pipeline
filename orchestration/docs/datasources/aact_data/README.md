# AACT

This document was updated on 2026-09-24.

AACT (Aggregate Analysis of ClinicalTrials.gov) is published as dated
PostgreSQL archives by CTTI, fetched from
`https://aact.ctti-clinicaltrials.org/static/static_db_copies/monthly/<version>`.

The same AACT archive version is pinned in both the preprocessing configuration
and the release input configuration. The preprocessing DAG publishes the exact
archive it processed, and the release copies that archive and its extraction
from the same snapshot.

| Pinned in | Names |
| --- | --- |
| `aact_version` in `aact_trial_extraction.yaml` | the raw CTTI archive downloaded and processed |
| `aact_version` in `pis/config.yaml` | the same archive and `aact_data` snapshot consumed by the release |

The release pin sits with the other release input pins. `pis/config.yaml`
carries the default so the steps still run standalone.

Derived data is stored under `gs://aact_data` with the following structure:

```{bash}
gs://aact_data/<aact_version>/input/           # the raw CTTI archive
gs://aact_data/<aact_version>/prompts/         # the prompt generated for this snapshot
gs://aact_data/<aact_version>/extraction/      # the LLM extraction
gs://aact_data/<aact_version>/extraction/trials_needing_extraction.parquet  # cache misses at snapshot start
gs://aact_data/<aact_version>/errors/          # diagnostics for failed LLM calls
gs://aact_data/<aact_version>/analysis/        # cache-wide summary and per-trial flags
gs://aact_data/<aact_version>/evaluation/      # LLM judge summary and findings
gs://aact_data/<aact_version>/etc/config/      # the config each step ran with
gs://aact_data/cache/trial_extraction/<schema-digest>/  # extraction cache, shared across AACT versions
```

The cache sits outside the version directory on purpose. Each response schema
has its own cache directory, named by the SHA-256 digest of its JSON schema.
Within that directory, rows are keyed on the trial ID and schema digest, so an
accepted extraction is reused across AACT snapshots. The rendered prompt hash
is retained as audit metadata. The model and system instructions are
operational choices and are not part of the key.

For metric definitions, denominators, and a practical guide to turning the
analysis and judge outputs into review actions, see [interpreting extraction
quality checks](evaluation.md).

## Preprocessing

### aact_trial_extraction dag

The **aact_trial_extraction.py** dag contains the following steps:

1. `pis_aact` — downloads the monthly archive to `input/aact.zip`.
2. `pts_aact_trial_extraction` — builds one prompt per trial and sends the ones
   that are not already in the cache to the OpenAI Responses API. It restores
   and reads the required AACT tables inline through the shared PTS PostgreSQL
   reader.
3. `pts_aact_trial_extraction_analysis` — reads every accepted extraction in
   the current schema cache and writes `analysis/summary.json`,
   `analysis/flags.parquet` and one row per cached trial in
   `analysis/trials.parquet`. The summary reports, separately for each entity
   field, how often an extracted label does not occur literally in its own
   `evidence_quote`. Its denominator is the number of entities with quotes;
   it also reports the share of trials containing that field with at least one
   such label. Missing quotes are counted separately. The trial table includes
   field-specific flag counts and changes between cached and current prompt
   hashes. Other summary statistics include current-snapshot extraction
   coverage, intent and confidence distributions, and entity-count shapes. A missing
   extraction cannot be counted from the cache alone; the current snapshot's
   prompt file supplies that denominator. Name-in-quote checks are literal,
   case-insensitive substring checks: synonyms and abbreviations can be flagged,
   and a name appearing in a quote does not establish the claimed clinical role.
   Historical Batch API prompts sometimes included literature omitted from
   later regenerated prompt files, so this step does not use those files to
   determine whether a quote was present in the original model input.
4. `pts_aact_trial_extraction_judge` — selects up to 100 successful extractions
   from the trials that needed extraction when this snapshot began. The
   extraction step writes that cohort before updating the shared cache, so a
   retry of the same snapshot uses the same cohort. The judge requires matching
   stored and current prompt hashes. It selects up to 50 trials with systematic
   flags and fills the remaining places with unflagged trials; either group can
   fill unused places if the other is too small. When no new extraction
   qualifies, the step samples earlier extractions without a cached verdict
   for the current prompt, extraction, judge model, and rubric. Karenina uses
   `gpt-6-luna` to compare the extraction with the trial text for investigated
   drugs, primary indications, `drug_intent`,
   and other clinical roles. It writes `evaluation/results.parquet`,
   `evaluation/summary.json`, and `evaluation/findings.jsonl`. Each finding
   names the failed criterion, explains the suspected error, suggests a fix,
   and records whether its source quote occurs in the prompt. The JSONL file
   has one record per failed criterion and is empty when every criterion
   passes. Review these findings before changing the extractor: the judge's
   suggested fixes are hypotheses, not ground truth. The pass rate describes
   the selected sample, not all trials. `selection_pool` identifies whether
   it used `trials_needing_extraction` or `previous_unjudged`. If neither pool has eligible
   records, the step publishes an empty evaluation with `sample_size: 0`.

As in `unified_pipeline`, a step is named `{stage}_{step}`: the stage is the
application that runs it, and the step itself is defined in that application's
own config file, so `pis_aact` runs the `aact` step from
`pis/config.yaml`. The dependency graph is declared in
`config/aact_trial_extraction.yaml`.

The PTS publish workflow also publishes `pts-with-karenina` with the same
version tag as `pts`. The judge step selects it with
`image: pts-with-karenina` in its DAG definition. Other PTS steps continue
to use the regular `pts` image. Before enabling this DAG version, publish a
PTS tag containing these changes and update `pts_version` in
`config/aact_trial_extraction.yaml` to that tag; both image names must exist
under it.

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

Work is sharded, and each shard is written under the current schema cache's
`staging/` directory as it completes. Rerunning the dag against the same
`aact_version` picks those shards back up rather than paying for the same API
calls twice.

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
use the shared version:

- the `clinical_report` step copies `input/aact.zip`, which
  `pts.clinical_report` reads inline, and
- the `drug` step copies `extraction/`, which `pts.chembl_molecule` mines for
  drug synonyms and `pts.clinical_report` uses for indications and drug intent.

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

### 2026-09-23

- Include AACT detailed descriptions in prompts under their intended field name.

### 2026-09-04

- Reuse earlier Batch API results by optionally seeding the cache; cache
  identity is the trial ID and output schema, not prompt text, publications,
  the model or system instructions.
- Exclude AACT trials without successful extraction from the clinical report.

### 2026-08-06

- Initial version. Replaces a one-off OpenAI Batch job run outside the pipeline,
  whose raw output was hand-staged to a personal bucket and parsed at release
  time by both `pts.clinical_report` and `pts.chembl_molecule`.
