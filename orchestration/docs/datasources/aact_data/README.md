# AACT

This document was updated on 2026-08-06.

AACT (Aggregate Analysis of ClinicalTrials.gov) is published as a monthly
PostgreSQL archive by CTTI, fetched from
`https://aact.ctti-clinicaltrials.org/static/static_db_copies/monthly/<version>`.

Two versions are pinned, independently and on purpose. They are different
things that happen to share a value: one names a CTTI archive, the other names
an `aact_data` snapshot built from it.

| Pinned in | Names |
| --- | --- |
| `aact_version` in `config/aact_trial_extraction.yaml` | the raw CTTI archive this dag downloads and extracts |
| `aact_data_version` in `config/unified_pipeline.yaml` | the `aact_data` snapshot a release reads |

The release pin sits with the rest of the release's version pins, next to
`pis_version` and `gentropy_version`, because it is a property of the release
rather than of the tool. `pis/config.yaml` carries defaults for both so the
steps still run standalone, and each dag overrides the one it owns.

This dag can therefore run ahead of any release. Extracting each archive as it
is published keeps the cache warm and makes the next release cheap, and does not
move a release onto it. A release moves only when someone bumps its own pin.

Derived data is stored under `gs://aact_data` with the following structure:

```{bash}
gs://aact_data/<aact_version>/input/           # the raw CTTI archive
gs://aact_data/<aact_version>/tables/          # AACT tables exported as parquet
gs://aact_data/<aact_version>/prompts/         # the prompt sent for each trial
gs://aact_data/<aact_version>/extraction/      # the LLM extraction
gs://aact_data/<aact_version>/etc/config/      # the config each step ran with
gs://aact_data/cache/trial_extraction/         # the extraction cache, shared across versions
```

The cache sits outside the version directory on purpose. It is keyed on a hash
of the prompt, so a trial whose text did not change between two monthly
archives has to be a cache hit — and it cannot be one if the cache is filed
under the version.

## Preprocessing

### aact_trial_extraction dag

The **aact_trial_extraction.py** dag contains the following steps:

1. `pis_aact_tables` — downloads the monthly archive, restores it into a
   throwaway postgres server, and exports the tables as parquet.
2. `pts_aact_trial_extraction` — builds one prompt per trial and sends the ones
   that are not already in the cache to the OpenAI Responses API.

As in `unified_pipeline`, a step is named `{stage}_{step}`: the stage is the
application that runs it, and the step itself is defined in that application's
own config file, so `pis_aact_tables` runs the `aact_tables` step from
`pis/config.yaml`. The dependency graph is declared in
`config/aact_trial_extraction.yaml`.

Each step runs on its own short-lived GCE VM and the VM is deleted afterwards.
The dag does no diffing: it is already incremental where it matters, because
the extraction only sends the model the trials that are not in its cache.

The output datasets are:

- [x] AACT tables under `gs://aact_data/<aact_version>/tables/`
- [x] Trial extraction under `gs://aact_data/<aact_version>/extraction/`

The OpenAI key is fetched from Secret Manager onto the VM and mounted into the
container; it never appears in configuration or in code.

## Processing description

The extraction returns, per trial: the drug intent and a confidence for it, the
primary indications and background conditions, and the investigated, comparator
and supportive drugs, each with synonyms and dosages.

Only trials that are not already in the cache are sent to the model. The cache
key hashes the `rendered prompt`, the `system prompt text`, the `model name` and the
`JSON schema` the response is validated against, so editing any one of them
re-extracts exactly the affected trials and nothing else.

Failures are cached alongside successes and retried after 30 days. Without
that, every run would re-attempt the same permanently failing trials forever,
and the cost of doing so would not show up anywhere.

Work is sharded, and each shard is written to `cache/trial_extraction/staging/`
as it completes. Rerunning the dag against the same `aact_version` picks those
shards back up rather than paying for the same API calls twice.

## Consumers

`unified_pipeline` reads both outputs through PIS `copy_many` tasks that name
the version literally:

- the `clinical_report` step copies `tables/`
- the `drug` step copies `extraction/`, which `pts.chembl_molecule` mines for
  drug synonyms and `pts.clinical_report` uses for indications and drug intent

There is no Airflow dependency between this dag and `unified_pipeline`. Moving a
release onto a newer AACT archive means making sure this dag has extracted it,
then bumping `aact_data_version` in `config/unified_pipeline.yaml` — the same
way every other ingested datasource works here.

If those two pins disagree in the direction that matters — a release asking for
an archive nobody extracted — the release fails on its first `copy_many`,
because `gs://aact_data/<version>/` does not exist. The opposite case, this dag
being ahead of the release, is normal and does nothing.

Note that a release can legitimately contain trials with no extraction row: a
trial the model answered negatively still produces a row, so a missing row means
the extraction never succeeded for it. `pts.clinical_report` warns about the
coverage gap rather than failing — a slightly thinner extraction is not a reason
to stop a release.

## Changelog

### 2026-08-06

- Initial version. Replaces a one-off OpenAI Batch job run outside the pipeline,
  whose raw output was hand-staged to a personal bucket and parsed at release
  time by both `pts.clinical_report` and `pts.chembl_molecule`.
