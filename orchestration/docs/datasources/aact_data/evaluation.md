# Interpreting AACT extraction quality checks

The AACT extraction DAG produces two complementary checks after extraction. The
systematic analysis scans **every accepted row in the current response-schema
cache**. The LLM judge reads the trial text and extraction for a **small,
selected subset**. Use the first to find coverage gaps and candidates for
review; use the second to understand possible clinical role errors. Neither
output alone is a validated accuracy estimate.

For a snapshot `<aact_version>`, start with:

```text
gs://aact_data/<aact_version>/analysis/summary.json
gs://aact_data/<aact_version>/analysis/flags.parquet
gs://aact_data/<aact_version>/analysis/trials.parquet
gs://aact_data/<aact_version>/extraction/trials_needing_extraction.parquet
gs://aact_data/<aact_version>/evaluation/summary.json
gs://aact_data/<aact_version>/evaluation/results.parquet
gs://aact_data/<aact_version>/evaluation/findings.jsonl
```

The extraction cache is shared between AACT versions. The analysis summary
therefore describes the whole current-schema cache, while its
`current_snapshot` section describes the prompts generated for this AACT
version. The extraction step records which trial IDs needed an extraction
before updating the cache. Judge eligibility is narrower again: it requires a
successful extraction in that cohort whose stored prompt hash **exactly
matches** the current prompt hash. When none qualifies, the judge selects
earlier prompt-matched extractions without a cached verdict for the current
judge model and rubric. Keep these populations separate when reporting
percentages.
For imported historical Batch API results, the stored hash can refer to a
later regenerated prompt that omitted literature supplied in the original
Batch request. A hash match is therefore an operational eligibility check,
not proof that the reconstructed text is everything the extractor saw.

## A practical review sequence

1. Check `analysis/summary.json` → `current_snapshot` for missing extractions
   and changed prompt hashes. Open `analysis/trials.parquet` for the affected
   trial IDs. A cached row with a different prompt hash may reflect older trial
   text or a different prompt; inspect it before judging the extraction against
   today's prompt.
2. Check `label_quote_by_field` and `quote_checks` for unusually frequent
   missing quotes or label/quote mismatches. Open `analysis/flags.parquet` to
   see the actual labels and quotes. Group recurring examples by field before
   changing a prompt or parser.
3. Check `evaluation/summary.json` by criterion and source. Open
   `evaluation/findings.jsonl` for the judge's evidence, explanation and
   proposed fix; use `evaluation/results.parquet` for one row per judged trial,
   including passing verdicts and `drug_intent` reasoning.
4. Turn recurring, credible examples into one specific action: correct a
   source/input problem, clarify a field definition or prompt example, fix
   extraction code, or add cases to the next judge run. Re-run the same checks
   after the change. A judge suggestion is a lead to investigate, not an
   automatic edit to the extracted data.

## Systematic analysis: what each metric means

| Summary field | Denominator and interpretation | Investigation it suggests |
| --- | --- | --- |
| `cache_rows`, `unique_trial_ids`, `duplicate_trial_id_rows` | Accepted cache rows, distinct trial IDs, and their difference. | Duplicate IDs merit a cache audit. Do not treat rows as unique trials if duplicates exist. |
| `current_snapshot.prompt_count` | Prompts generated for this AACT snapshot. | Establishes the coverage denominator. |
| `current_snapshot.cached_count`, `missing_extraction_count`, `coverage_percent` | Current prompt IDs with any cache row; without a cache row; and `cached_count / prompt_count`. | For missing IDs, inspect extraction errors/retries. Coverage does **not** mean the cached text matches the current prompt. |
| `current_snapshot.cached_prompt_differs_count` | Current prompt IDs with a cached row whose stored prompt hash differs. | Inspect stale input and decide whether re-extraction is needed. The cache key does not include the prompt hash, so a changed prompt does not automatically trigger extraction. |
| `drug_intent_counts` | Number of cache rows per extracted intent. | Investigate shifts in label mix, especially a large rise in `other`; frequency does not establish correctness. |
| `confidence_counts`, `confidence_by_intent` | Counts in `<0.7`, `0.7-<0.9`, `>=0.9`, or missing bins; count/mean/median/min/max of the model's reported confidence per intent. | Use low or missing confidence to prioritize review. The number is model reported and is **not calibrated accuracy**. |
| `entity_shape_counts`, `entity_shape_percent_of_cache` | Cache rows with zero, one or multiple investigated drugs and primary indications; percentages use `cache_rows`. | Investigate unexpected empty fields or changes in complexity. These are counts of extracted lists, not verified drug–disease relationships. |
| `label_quote_by_field.<field>.non_null_trials`, `trials_with_entities`, `entities` | Cache rows where a field is non-null; rows with at least one entity; total entities in that field. Null and empty lists are distinct here. | Check whether a field is systematically omitted, and choose the right denominator for its quote rates. |
| `label_quote_by_field.<field>.missing_quote` | Entities with no `evidence_quote`. | Review extraction instructions or schema handling for that field. |
| `label_quote_by_field.<field>.name_not_in_quote` | Quoted entities whose extracted `drug` or `name` is not a case-insensitive literal substring of their **own** quote. | Inspect synonyms, abbreviations and unsupported labels. This flag does not prove hallucination. |
| `label_quote_by_field.<field>.percent_of_quoted_entities_with_name_not_in_quote` | `name_not_in_quote / with_quote`; excludes entities without quotes. | Compare fields or runs only alongside their entity counts and missing-quote counts. |
| `label_quote_by_field.<field>.trials_with_name_not_in_quote`, `percent_of_trials_with_entities_affected` | Distinct affected trial IDs; divided by trials with entities in that field. | Estimate how widely a field-specific issue is spread across trials. |
| `quote_checks` | The same quote checks pooled across all five entity fields. `affected_trial_counts` counts distinct IDs; `affected_trial_percent` divides by `cache_rows`. | Use for an overview, then inspect the field-specific table; pooled rates can hide one problematic field. |
| `flag_counts` | Number of **flag rows** by category, not distinct trials. A trial can contribute several rows. | Drill into `flags.parquet`; use `trials.parquet` or distinct IDs when counting affected trials. |

The five quote-checked fields are `investigated_drugs`, `comparator_drugs`,
`supportive_drugs`, `primary_indications`, and `background_conditions`.
`flags.parquet` also marks `no_investigated_drug`, `no_primary_indication`,
`multiple_drug_indication_combinations`, and
`current_prompt_differs_from_cached_prompt`. An empty indication can be valid
when the trial text states no clinical indication, and multiple extracted drugs
and indications do not establish every possible drug–indication pairing.

**Quote checks have a narrow meaning.** The name check is a case-insensitive
substring comparison between an extracted label and its associated quote. It
does not check that the quote occurred in the original model input or that the
clinical role is right. Older Batch API prompts could contain literature not
present in later regenerated prompts. Conversely, a matching word in a quote
does not prove the field assignment is correct.

## LLM judge: reading `evaluation/summary.json`

The judge evaluates four boolean criteria: `investigated_drugs_correct`,
`primary_indications_correct`, `drug_intent_correct`, and
`other_roles_correct` (comparator drugs, supportive drugs, and background
conditions). It uses only the matched current prompt and the cached
extraction. The rubric returns false when the evidence is insufficient or the
role is ambiguous, so a failure can mean **uncertainty** rather than a
demonstrated extraction error.

| Summary field | Meaning | How to use it |
| --- | --- | --- |
| `selection_pool` | `trials_needing_extraction` when eligible new extractions exist; otherwise `previous_unjudged`. | State which cohort the reported pass rates describe. |
| `sample_size`, `eligible_count` | Judged trials; eligible trials in the selected pool with matching current prompt hashes. | Check the scale of the judgment and whether 100 trials were available. |
| `source_counts` | Judged trials from the `flagged` and `random` sources within the selected pool. | Check the actual mix before interpreting totals. |
| `source_stats.<source>.sample_size`, `trait_pass_counts`, `all_correct_count`, `all_correct_percent` | Count per passing criterion and count/share passing all four criteria within that source. | Compare issue types in flagged and unflagged new trials. |
| `trait_pass_counts`, `all_correct_count`, `all_correct_percent` | The same counts over the **selected sample**. `all_correct` requires all four booleans to be true. | Describe this run only; it is not a population accuracy measure. |
| `finding_count`, `findings_by_criterion` | Number of failed-criterion records, split by criterion. | Prioritize common issue types. `finding_count` can exceed the number of affected trials; count distinct IDs in `findings.jsonl` when needed. |
| `findings_with_source_quote_in_prompt` | Findings whose judge-provided source quote is a **case-sensitive exact substring** of the current prompt. | Open findings without a matching quote; a false value may be paraphrasing or combined fragments, not proof that the extraction fabricated evidence. |
| `model`, `rubric_version`, `sample_seed` | Provenance of the judge run. | Keep these with reported results; changes make direct comparisons harder. |

The DAG first selects up to 100 trials from `extraction/trials_needing_extraction.parquet`,
restricted to successful extractions with matching prompts. If none qualify,
it selects up to 100 earlier extractions without a cached verdict for the
current prompt, extraction, judge model, and rubric. It aims for 50 `flagged`
trials (missing quote, name/quote mismatch, no investigated drug, or no
primary indication) and 50 `random` trials from the remaining unflagged
trials. Either source fills vacancies if the other contains too few trials.
Selection is deterministic for a given pool and seed. When fewer than 100
qualify, it judges all that qualify; when neither pool has candidates, it
writes `sample_size: 0` and empty results. The `random` source is random
**within unflagged trials of the selected pool**, not within all trials.
The combined pass rate is therefore not a calibrated accuracy estimate for
all new trials or the full historical cache. A rerun with no new trials can
take the next 100 previously unjudged records as the judge cache fills.

`findings.jsonl` has one record per failed criterion with `id`, `source`, `selection_pool`,
`criterion`, `issue`, `source_quote`, `source_quote_in_prompt`, and
`suggested_fix`. Intent findings also have `extracted_label`,
`recommended_label`, and `reason`. A single finding can describe multiple
errors. `results.parquet` has the complete per-trial verdicts; it also includes
the judge's recommended intent and reasoning even when intent passes. An
empty findings file means no criteria failed **in this sample**.

The judge sees the current prompt, not every source that might have been
available to an older Batch extraction. Treat discrepancies for those trials
as possible input-provenance issues until the original request is checked.

## From signals to action points

| Signal | Inspect | Plausible next action |
| --- | --- | --- |
| Missing extractions or falling coverage | Current prompt IDs absent from the cache; extraction `errors/` | Fix failed calls or input handling, then rerun extraction. |
| Many changed prompt hashes | `trials.parquet` rows with `current_prompt_differs = true`; compare the stored and current input | Decide whether the change merits a new extraction/cache namespace. Do not score an old extraction against changed text as though it were its original input. |
| High missing-quote rate in one field | A few `flags.parquet` rows for that field | Clarify the quote requirement or fix serialization if quotes were lost. |
| High name/quote mismatch rate | Entity, quote, and synonyms from `flags.parquet` | Separate benign synonyms from unsupported labels; add precise prompt examples or normalization only after seeing the pattern. |
| Many `no_primary_indication` or `no_investigated_drug` flags | Trial text and extraction for those IDs | Distinguish valid nulls from omissions. Add cases to the judge sample if the role needs clinical interpretation. |
| Judge flags investigated drugs | `findings.jsonl` quotes and arms in trial text | Clarify experimental intervention versus comparator or previous treatment. |
| Judge flags primary indications or other roles | Target condition, enrolled population, context, and intervention purpose | Refine `primary_indications`/`background_conditions` definitions with both positive and negative examples. |
| Judge flags `drug_intent` | `extracted_label`, `recommended_label`, `reason`, and source text | Review a group of similar transitions before changing the prompt; check whether the judge rubric itself is too strict. |
| Repeated judge findings with unmatched source quotes | Original prompt and any available source text | Check judge evidence quality before treating those suggestions as actionable. |

Track action points as concrete hypotheses with trial IDs and supporting
quotes. For example: “Several trials with safety-only outcomes were assigned
`therapeutic`; inspect these IDs and add an example distinguishing drug use
from study objective.” Avoid turning every free-text `suggested_fix` into an
independent task. Group by clinical mistake, make one change, and check whether
it improves the next run without worsening other fields.

A useful handover note for each run is: snapshot and schema; extraction
coverage; top two field-specific systematic signals with their denominators;
judge sample sizes and per-criterion pass counts by source; two or three
representative trial IDs; the proposed change and who will verify it. State
clearly when an issue is only a judge hypothesis or when its source prompt is
uncertain.

## Comparing runs

Record the AACT version, schema/cache namespace, extraction model and prompt
version, number of eligible trials, sample selection settings, judge model,
and rubric version. Compare analysis **rates with their denominators**, not
raw flag counts when the cache grows. The cache-wide metrics include historical
accepted rows, so a new snapshot's summary is not a clean “new trials only”
cohort. For that question, identify trial IDs added between the old and new
snapshot prompt files, then join those IDs to `trials.parquet`.

For judge runs, use a fixed set of trial IDs when comparing extractor models or
prompts. The new-trial cohort changes between AACT snapshots, and fallback
runs consume previously unjudged records over time. Judge findings are useful
for triage, but validate a small set
of representative examples with a domain expert before declaring an extraction
change better. There is no established pass threshold or calibrated confidence
score in the current pipeline.

## Airflow smoke test on an already extracted snapshot

The `aact_trial_extraction` DAG runs the extraction, systematic analysis, and
judge in that order. It uses the AACT version and PTS image tag pinned in
`config/aact_trial_extraction.yaml`. Check that **both** `pts:<pts_version>` and
`pts-with-karenina:<pts_version>` exist in Artifact Registry before triggering
it. The Airflow dev VM may be running another branch; inspect its checkout
before replacing its code or services.

To use a separate local Airflow UI without disturbing another Compose project,
run from `orchestration/`:

```bash
COMPOSE_PROJECT_NAME=aact-evaluation AIRFLOW_HOST_PORT=8082 make local-airflow
```

The UI is at `http://localhost:8082`. This Airflow runs the same DAG operators:
each PTS step still starts a GCE VM, uses the pinned registry image and reads or
writes GCS. Local Docker images are **not** used by those VMs. The command needs
the Google application credentials described in the orchestration README.

Before triggering, compare the current prompts' `cache_key` values with the
accepted extraction cache. For `2026-09-23`, the check on 2026-09-30 found
237,557 prompts, 237,562 accepted cache rows, **zero cache misses**, and
69,305 rows with a matching current prompt hash. Recheck this after any cache
or schema change; zero misses is what makes this a test of the fallback path
without new extractor calls. A full DAG run still downloads the AACT archive,
rewrites the snapshot outputs and cache pointer, and sends up to 100 selected
trials to the judge. Run it only when those writes are intended.

After the run, check:

1. `extraction/trials_needing_extraction.parquet` has zero rows and the
   extraction step reports zero failed trials.
2. `analysis/summary.json` exists; its `cache_rows` and
   `current_snapshot.prompt_count` agree with the extraction/cache snapshot.
   Inspect `analysis/flags.parquet` for real flagged IDs.
3. `evaluation/summary.json` has `selection_pool: "previous_unjudged"`,
   `sample_size: 100`, the expected `model` and `rubric_version`, and source
   counts totalling 100. `evaluation/results.parquet` has 100 distinct IDs;
   `evaluation/findings.jsonl` contains one record per failed criterion.
4. The judge cache's `latest.json` points to a snapshot containing those 100
   verdict keys. The fallback deliberately excludes judged keys on a later
   run, so rerunning the DAG selects the **next** unjudged trials and incurs
   more judge calls. A second full run is unnecessary to verify that the first
   verdicts were cached.

Stop the isolated Airflow stack from `orchestration/` with:

```bash
COMPOSE_PROJECT_NAME=aact-evaluation AIRFLOW_HOST_PORT=8082 \
  docker compose -f compose.yaml -f compose.local.yaml down
```
