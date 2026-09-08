# Migrating L2G training and prediction from gentropy into PTS

Date: 2026-09-08
Status: design approved, experimental

## Summary

Replace three gentropy steps — `l2g_train_test_split`, `l2g_training` and `l2g_prediction` — with
a single PTS step, `l2g`, built on polars, XGBoost and shap. The step runs on one plain GCE VM.
It uses no Google Batch, no Dataproc, no Hugging Face Hub and no Weights & Biases.

`l2g_feature_matrix` stays in gentropy on Dataproc. `l2g_evidence` stays in gentropy and keeps
reading `output/l2g_prediction`, whose schema and content contract do not change.

```
before:  gentropy_l2g_feature_matrix → gentropy_l2g_train_test_split (Dataproc)
                                     → gentropy_l2g_training         (Dataproc)
                                     → gentropy_l2g_prediction       (Batch, 1000 tasks)
                                     → gentropy_l2g_evidence
after:   gentropy_l2g_feature_matrix → pts_l2g (one GCE VM)
                                     → gentropy_l2g_evidence
```

## Measurements this design rests on

All figures from the `do/platform-2609-1` and `do/platform-2609-2` runs, measured 2026-09-08.

| dataset | rows | parquet |
| --- | --- | --- |
| `intermediate/l2g_feature_matrix` | 61,190,215 | 2.50 GB |
| `intermediate/l2g_train_split` | 77,149 | 0.01 GB |
| `intermediate/l2g_test_split` | 17,431 | 0.01 GB |
| `output/l2g_prediction` | 3,219,816 | 0.50 GB |

The feature matrix carries 31 float features plus `studyLocusId`, `geneId` and `isProteinCoding`.

SHAP throughput, measured with the 26.09-2 model itself (`XGBClassifier`, 300 trees, `max_depth=5`)
over real feature-matrix rows, using `shap.TreeExplainer` with `feature_perturbation="interventional"`
and `model_output="probability"` — the same algorithm gentropy uses:

| background size | rows/s/process | 3.22M rows | wall clock on 32 vCPU |
| --- | --- | --- | --- |
| 100 | 181-317 | 2.8-4.9 process-hours | ~6-10 min |
| 1000 | 32 | 28 process-hours | ~53 min |

The range at background 100 is measurement spread, not uncertainty about the approach. 317
rows/s was measured on an idle machine with xgboost 3.4.1 / shap 0.52.0; 181 rows/s on the same
rows and the same background under concurrent load with xgboost 3.2.0 / shap 0.51.0, which is
what production ships (see the dependency note below). A controlled same-rows comparison put the
library-version difference itself at ~6% -- 181 against 193 rows/s -- so the spread is machine
load, not the versions. Size the step against the slow end.

Production pins xgboost 3.2.0 and shap 0.51.0 rather than the newest releases: uv forks both by
Python version, and requiring the newer ones would push pts off Python 3.11 and off its
`python:3.11-slim` base image. The older pair loads the released `classifier.skops` and
reproduces the published scores at max |delta| = 0, and `shap.maskers.Independent` takes
`max_samples` in 0.51, which is all this design needs from it.

`predict_proba` runs at 5.0M rows/s, so scoring the entire 61.2M-row matrix costs ~12 s. XGBoost's
native `Booster.predict(pred_contribs=True)` does all 3.22M rows in 11.5 s but produces log-odds
margin contributions rather than probability-space SHAP values, so it is not used.

## Findings that shaped the design

### The production SHAP background is 100 rows, not 1000

`L2GPrediction._explain` samples 1000 background rows, but `shap` ≥ 0.46 wraps a `data=` argument in
a `shap.maskers.Independent` masker whose default `max_samples` is 100. The remaining 900 rows are
discarded silently. gentropy pins `shap>=0.50.0`, so every released run has used 100.

Consequence: the cost to match current behaviour is the 2.8-4.9 process-hour row, not the 28 one. The
design makes the background size an explicit setting so it can never again be decided by a library
default.

### The 1000-way Batch fan-out buys nothing

Each of the 1000 Batch tasks runs a full Spark job on an `n1-standard-2`, reads the *entire* 2.5 GB
feature matrix (only the credible-set side is partitioned), and produces roughly 3,200 output rows.
Against a `max_run_duration` of 1 h that is an envelope of up to 2,000 vCPU-hours. The measured work
is under 5 vCPU-hours of SHAP plus ~12 s of scoring. A single VM is the right shape.

### SHAP is already not reproducible

Each Batch task draws its own unseeded `sample(n=1_000)` from the Hub-hosted training data, so the
background — and therefore `shapBaseValue` — differs per output partition. Measured across **all
200 partitions** of `platform-2609-1`: **200 distinct base values spanning 0.028042 to 0.137713, a
4.91× spread** within a single released dataset. (An earlier six-partition sample suggested 1.75×;
the full population is far worse.) Row-for-row SHAP parity with 26.09-2 is unachievable even by
re-running gentropy, so it is not an acceptance criterion. One fixed seeded background for the
whole run replaces this, yielding a single `shapBaseValue` — confirmed on a real prediction run.

### The OTG-curation gold-standard branch is dead code

`LocusToGeneTrainTestSplitStep._parse_gold_standard` has three branches. The curated file at
`gs://otar001-core/l2g/goldStandard/2025-06-25` carries
`studyLocusId, geneId, diseaseIds, variantId, studyId, goldStandardSet`, so schema comparison against
`L2GGoldStandard` yields only `unexpected_columns: ["diseaseIds"]` and the second branch fires: drop
that column and use the file as-is. The OTG branch — which needs `find_overlaps`, the variant index
and the PPI dataset — never runs on this data. Confirmed independently by the absence of
`traitFromSourceMappedId`, which is why `l2g_train_test_split_stats.json` has no
`n_unique_positive_gene_disease_pairs` key.

**It is not ported.** If a future gold standard arrives in OTG-curation format, the step must fail
loudly rather than silently mis-parse.

### The released model's hyperparameters are not in version control

`etc/model/locus_to_gene_model/classifier.skops` from 26.09-2 carries `n_estimators=300`, `gamma=0`
and `max_delta_step=1`. gentropy's `LocusToGeneConfig.hyperparameters` at `v3.3.0-rc.1` contains
none of them, and `n_estimators` appears nowhere in either repository — XGBoost's default yields 100
rounds, not 300. They came from an override outside version control.

The full parameter set actually used:

```
objective=binary:logistic  eval_metric=aucpr   random_state=777
n_estimators=300           max_depth=5         min_child_weight=10
eta=0.05                   subsample=0.8       colsample_bytree=0.8
reg_alpha=1                reg_lambda=1.0      scale_pos_weight=0.8
gamma=0                    max_delta_step=1
```

These move into `settings:` in `pts/config.yaml`, seeded from the model itself.

### The three intermediates have no consumers outside the L2G chain

`intermediate/l2g_feature_matrix`, `l2g_train_split` and `l2g_test_split` are referenced only by the
gentropy L2G steps. Nothing in `pts`, `croissant` or the rest of `orchestration` reads them, and
neither appears in the croissant distribution. They are handoffs, not products.

### Fixed-model parity is achievable exactly

Loading 26.09-2's own `classifier.skops`, applying the imputation and float32 cast in polars, and
scoring gave **max |Δ| = 0.000e+00** against the published `score` column on the 56 overlapping rows
of a two-partition sample.

**That figure was too optimistic, and the acceptance criterion must not be exact equality.** Running
the assembled `l2g_predict` over the same released model and a larger slice — 212,847 rows scored,
38,738 above threshold — matched the published dataset on **every** row, but with **max |Δ| =
1.192e-07** on 184 of them. That value is exactly one float32 ULP.

It is not non-determinism on our side: XGBoost's prediction here is bit-identical across repeat runs
and across `n_jobs` of 1, 2 and the default. It is not the locus-mean imputation either — both
gene-count columns are null-free in this data, so that path never fires. It is almost certainly the
prediction kernel differing by a ULP between the XGBoost version gentropy's image pinned and the
3.2.0 this ships. The 56-row sample simply contained no affected row.

So the gate is `max |Δ| < 1e-6`, not equality, and the row set at threshold must match exactly — it
did, 38,738 of 38,738. A tolerance of zero would fail on a difference far below the resolution of
the 0.05 threshold it feeds.

## Architecture

### Step and task shape

One PTS config step, `l2g`, with two tasks:

```yaml
l2g:
  - name: transform l2g_train
    transformer: l2g_train
    source:
      feature_matrix: intermediate/l2g_feature_matrix
      credible_set: output/credible_set
      gold_standard: input/l2g/gold_standard.json
      predefined_test: input/l2g/predefined_test_for_split
    destination:
      model: etc/model/locus_to_gene_model/classifier.skops
      background: etc/model/locus_to_gene_model/shap_background.parquet
      metrics: etc/model/locus_to_gene_model/metrics.json
      train_split: intermediate/l2g_train_split
      test_split: intermediate/l2g_test_split
      split_stats: intermediate/l2g_train_test_split_stats.json
    settings:
      features_list: &l2g_features [eQtlColocClppMaximum, ...]   # 31 names, order load-bearing
      hyperparameters: {n_estimators: 300, max_depth: 5, ...}
      train_on_full_dataset: true
      shap_background_size: 100
      shap_background_seed: 42

  - name: transform l2g_predict
    requires:
      - transform l2g_train
    transformer: l2g_predict
    source:
      feature_matrix: intermediate/l2g_feature_matrix
      credible_set: output/credible_set
      model: etc/model/locus_to_gene_model/classifier.skops
      background: etc/model/locus_to_gene_model/shap_background.parquet
    destination: output/l2g_prediction
    settings:
      features_list: *l2g_features
      l2g_threshold: 0.05
      explain_predictions: true
```

The feature list is shared between the two tasks by a **YAML anchor and alias**, not by the
scratchpad. Otter's `Scratchpad` is `string.Template` substitution, so a `${...}` sentinel can only
ever produce a string — a list would arrive as `"['eQtlColocClppMaximum', ...]"`. Anchors are
resolved by `yaml.safe_load` before otter sees the document, so a real list arrives.

The list is the 31 names from `LocusToGeneConfig.features_list` in gentropy, copied verbatim **in
order**. The order is load-bearing twice over: it is the column order the model was fitted on, and
it is the order of the `features` array in the output. The anchor is what stops the two tasks
drifting apart.

One config step is one Airflow task on one VM. Because no task is named `pyspark …`, PTS step
routing (`orchestration/src/orchestration/models/pts_step.py`) sends it to a plain GCE VM rather
than a Dataproc cluster. Two tasks rather than one keeps training and prediction independently
testable while still satisfying the single-step requirement; the in-step handoff by path follows the
existing `disease_efo → disease_phenotype` pattern.

The merge means a prediction failure re-runs training. On 94k rows that costs about a minute, which
is the correct trade.

One consequence to hold in mind on retries: `write_dataset` refuses to write into a destination that
already exists, and otter has no resume, so a re-run of `pts_l2g` after a partial failure needs its
destinations cleared first — the same handling every other PTS step already needs.

### Module layout

```
pts/src/pts/transformers/l2g_train.py      def l2g_train(source, destination, settings, config)
pts/src/pts/transformers/l2g_predict.py    def l2g_predict(source, destination, settings, config)
pts/src/pts/transformers/l2g/
    gold_standard.py   parse the curated file, build the annotated gold-standard feature matrix
    split.py           the predefined-test derivation and its statistics
    features.py        feature order, float32 cast, imputation — shared by train and predict
    model.py           fit, evaluate, save, load
    explain.py         background construction and the SHAP process-pool driver
```

New dependencies for `pts`: `xgboost`, `shap`, `scikit-learn`, `skops`.

### Training

1. Read the gold standard as ndjson; drop `diseaseIds`. If the schema is not the curated one, raise.
2. Left-join `credible_set` projected to `studyLocusId, variantId, studyId` onto the feature matrix.
3. Inner-join the gold standard on `(studyId, variantId, geneId)`.
4. Filter `isProteinCoding == 1`; drop `studyId`, `variantId`; deduplicate on the full remaining
   column set with `maintain_order=True`, reproducing spark's `.distinct()`.
5. Impute and select features (see below).
6. Predefined-test derivation:
   - test-positive genes = `predefined_test` rows whose `goldStandardSet` is `1` or `"positive"`,
     distinct on `geneId`;
   - contaminating `studyLocusId`s = annotated rows joining any test-positive gene. The row's own
     label is deliberately not consulted: a locus whose gene label flipped between runs is still in
     the test set and must stay out of training;
   - train = annotated rows anti-joined on those `studyLocusId`s;
   - test = annotated rows inner-joined to `predefined_test` on `(studyLocusId, geneId)`;
   - labels encoded `negative → 0`, `positive → 1`, tolerating already-encoded integers.
7. Fit `XGBClassifier` on train with the configured hyperparameters.
8. Evaluate once on test: `areaUnderROC`, `accuracy`, `weightedPrecision`, `averagePrecision`,
   `weightedRecall`, `f1` — the same six metrics gentropy reports, from scikit-learn.
9. Refit on `train + test` (`train_on_full_dataset: true`), matching what produced the 26.09-2 model.
   Reported metrics come from step 8 and are unaffected.
10. Build the SHAP background: a seeded sample of `train + test` of the configured size. Write it.
11. Save the model as `.skops` at the existing path; write `metrics.json` and the split statistics.

No cross-validation, no hierarchical re-split, no second split for model export. The only split
honoured is the pinned one staged by `pis_l2g_predefined_test_for_split`.

### Prediction

1. Scan credible sets projecting `studyLocusId, studyType`; keep `studyType == "gwas"`; distinct.
2. Scan the feature matrix; filter `isProteinCoding == 1`; semi-join to those study loci.
3. Impute and cast (see below).
4. `predict_proba` in chunks; take the positive-class column as `score`.
5. Filter `score >= l2g_threshold`.
6. SHAP over the surviving rows.
7. Assemble `features` as an array of `{name, value, shapValue}` in model feature order, plus
   `shapBaseValue`.
8. Sort by `(studyLocusId, geneId)`; pin dtypes; `write_dataset`.

The features are carried through from the matrix rather than dropped and re-joined, which is what
gentropy's `add_features` does. There is no join.

### Imputation and casting (shared)

`geneCount500kb` and `proteinGeneCount500kb` are filled with the mean over their `studyLocusId`
partition; every remaining null becomes `0.0`. All features are then cast to `Float32` in the
configured feature order. This mirrors `L2GFeatureMatrix.fill_na` and `select_features`, which
gentropy applies on both the training and the prediction path.

### SHAP

One background array for the entire run, sampled once with a recorded seed and written to
`etc/model/locus_to_gene_model/shap_background.parquet`, so any run can be reproduced.

The masker is constructed explicitly as `shap.maskers.Independent(background, max_samples=N)` with
`N = shap_background_size`, rather than passing a bare array and inheriting whatever `max_samples`
default the installed `shap` happens to carry.

Rows are chunked across a `ProcessPoolExecutor`; each worker constructs its explainer once in the
pool initialiser. `check_additivity=False`, as gentropy does.

Default `shap_background_size: 100` reproduces current behaviour at ~6-10 min on 32 vCPU. Raising it
to 1000 is a one-line change costing ~53 min.

Chunk rows so there are several chunks per worker, not one: a chunk count below the worker count
caps the speedup at the chunk count regardless of how many cores the VM has.

### Sizing

`machine_type: n1-highmem-32` for `pts_l2g` in `unified_pipeline.yaml`. Peak memory is roughly
7.6 GB for the full 61.2M × 31 float32 matrix, so this is headroom rather than a requirement; the
32 vCPUs are what the SHAP pool consumes.

## Artifacts

| path | change |
| --- | --- |
| `output/l2g_prediction` | unchanged schema and content contract |
| `etc/model/locus_to_gene_model/classifier.skops` | unchanged path and format |
| `etc/model/locus_to_gene_model/shap_background.parquet` | new — the run's SHAP background |
| `etc/model/locus_to_gene_model/metrics.json` | new — held-out metrics, feature missingness, hyperparameters, background seed and size |
| `intermediate/l2g_train_split` | kept at its path, audit-only |
| `intermediate/l2g_test_split` | kept at its path, audit-only |
| `intermediate/l2g_train_test_split_stats.json` | kept at its path |

The two splits and the statistics file have no consumers. They are retained so a run stays auditable
and so the diff against 26.09-2 is direct.

Output dtypes are pinned before writing: `studyLocusId` and `geneId` string, `score` double,
`features[].value`, `features[].shapValue` and `shapBaseValue` float32. `gentropy_l2g_evidence` reads
this dataset through `session.load_data(..., schema=...)`, which *imposes* the schema on read rather
than validating it, so a type drift would be coerced silently rather than raised. Pinning is what
prevents that.

## Orchestration changes

Removed:

- `gentropy.yaml`: the `l2g_train_test_split`, `l2g_training` and `l2g_prediction` step blocks,
  including the `google_batch` and `google_batch_index_specs` configuration and the `hfhub-key` /
  `wandb-key` secret mappings.
- `orchestration/src/orchestration/assets/l2g_predict.sh`.
- `unified_pipeline.py`: the `l2g_training_version` scratchpad variable and the
  `gentropy_step_outputs` entries for the three removed steps.
- `unified_pipeline.yaml`: the three `gentropy_l2g_*` dependency entries.

Added:

- `unified_pipeline.yaml`: `pts_l2g` with `machine_type: n1-highmem-32`, depending on
  `gentropy_l2g_feature_matrix`, `pis_l2g` and `pis_l2g_predefined_test_for_split`.
- `gentropy_l2g_evidence` now depends on `pts_l2g`.

`croissant` needs no change: the `l2g_prediction` distribution and recordset describe a schema that
does not move.

## Spark-to-polars divergences designed against

Against the recurring list in `CLAUDE.md`, each of which has produced a real defect before:

- **Ordering.** Spark's `.distinct()` is set-backed and happened to be stable; polars `unique()` is
  not. Every deduplication uses `maintain_order=True` and the output is sorted by
  `(studyLocusId, geneId)` before writing, which also makes the baseline diff positional.
- **Casts.** The feature cast reproduces spark's `CAST(x AS FLOAT)`; it is asserted against the
  baseline rather than assumed.
- **Nulls.** Spark's windowed `mean` ignores nulls and so does polars', but the imputation gets an
  explicit test rather than an assumption, because this is exactly where the two have diverged.

## Verification

1. **Fixed-model prediction parity.** Load 26.09-2's own `classifier.skops`, run the new prediction
   path over 26.09-2's feature matrix, and require:
   - `score` equal to the published column within float tolerance;
   - the row set at `score >= 0.05` equal to the published 3,219,816 rows.

   This isolates the migration from the retraining and is the sharpest test available. Demonstrated
   at max |Δ| = 0 on a two-partition sample during design; that sample was too small to contain an
   affected row. Measured against `do/platform-2609-1` (the reference run used by the harness --
   `2609-2` has no `output/l2g_prediction` to compare against) over two feature-matrix partitions:
   212,847 rows scored, 38,738 above the 0.05 threshold, 38,738 of 38,738 matched the published row
   set exactly, and max |Δ| = 1.192e-07 on the score (184 rows above 1e-9). That value is exactly one
   float32 ULP -- ruled out as this migration's non-determinism by measurement (XGBoost's
   `predict_proba` is bit-identical here across repeat runs and across `n_jobs` of 1, 2 and default,
   and both gene-count columns are null-free in this data, so the locus-mean imputation never fires)
   and attributed instead to the XGBoost prediction kernel differing by one ULP between the version
   gentropy's image pinned and the one this ships. Wall clock not recorded for this sample.

2. **SHAP distributional agreement.** Per-feature SHAP distributions and the mean-|SHAP| feature
   ranking must agree with the baseline. Per-row equality is explicitly not required, and the single
   consistent `shapBaseValue` is recorded as an intended improvement over the current 1.75× spread.

3. **Unit tests** per module on small fixtures: gold-standard parsing and its rejection of the
   unported OTG format, the contamination anti-join, label encoding, imputation, feature ordering,
   and the assembled output schema.

4. **Retrained-model metrics** are written to `metrics.json` for information. They are not an
   acceptance gate, since the retrained model legitimately differs from 26.09-2's.

## Result of the full-scale parity gate

Run on 2026-09-08 against `do/platform-2609-1`, using that release's own `classifier.skops` over
its own feature matrix — all 200 partitions, not a sample. 20,015,785 rows survived the GWAS and
protein-coding filters; scoring and comparison took 632 s, of which 603 s was the GCS scan.

| gate | result | |
| --- | --- | --- |
| Scores match the published column | all 3,219,816 published rows found; `max \|Δ\| = 1.192e-07`, mean `6.655e-11`, **zero rows above 1e-6** | PASS |
| Row set at threshold matches | 3,219,816 against 3,219,816; zero only-ours, zero only-published | PASS |
| Baseline's own base value is not constant | 200 distinct values, 0.028042–0.137713, ratio 4.91× | PASS |

The training half, which this gate cannot reach because it holds the model fixed, is covered
separately: the chain reproduces all fourteen of the release's recorded split statistics exactly,
and a model retrained by this port scores 0.933660 against the released model's 0.933736 on the
same held-out set.

## Out of scope

- `l2g_feature_matrix` stays in gentropy. Its inputs in 2609-2 are 283,303,467 colocalisation rows
  (16.52 GB), 48,808,491 enhancer-to-gene rows, 7,886,762 variants and 4,761,925 credible sets. That
  is a genuine distributed join and its own migration.
- Publishing the model to `opentargets/locus_to_gene_<version>` on the Hugging Face Hub stops. The
  repository is public and may have external consumers; restoring it, if wanted, is a separate
  out-of-band job and not part of the pipeline.
- Weights & Biases logging stops. The metrics it carried are written to `metrics.json` in the
  release instead, versioned with the run that produced them.
- Hyperparameter tuning. The degenerate single-point W&B sweep that exists today tunes nothing; a
  real search would change the model and belongs in its own piece of work.
