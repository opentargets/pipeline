# L2G into PTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace gentropy's `l2g_train_test_split`, `l2g_training` and `l2g_prediction` steps with one PTS step, `l2g`, running polars + XGBoost + shap on a single GCE VM.

**Architecture:** One PTS config step containing two transformer tasks (`transform l2g_train`, then `transform l2g_predict` via `requires:`). Because no task is named `pyspark …`, orchestration routes the step to a plain GCE VM rather than a Dataproc cluster. Pure helper functions live in a `pts/transformers/l2g/` package and are unit-tested directly; the two transformer entry points do IO only. `l2g_feature_matrix` stays in gentropy.

**Tech Stack:** Python 3.11–3.13, polars, XGBoost, shap, scikit-learn, skops, otter, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-l2g-pts-migration-design.md` (mirrored in the vault at `Areas/Work/Projects/Project-PTS/specs/2026-09-08-l2g-pts-migration-design.md`)

## Global Constraints

- **Branch:** `experiment/l2g-in-pts`, already cut from `origin/main` at `17df29da`.
- **Style:** single quotes for inline strings, double for multiline and docstrings; line length 120; Google-convention docstrings. Config is shared in `ruff.toml`; each package does `extend = "../ruff.toml"`.
- **Type checking is `ty` (Astral), not mypy.** `make lint` runs `ruff check` + `ty check`.
- **Every command uses `--frozen`.** The lockfile is authoritative: `cd pts && uv run --frozen pytest …`.
- **Run package commands from inside the package directory.** There is no root `pyproject.toml`.
- **Commit messages for package-specific changes start with the package name** (`pts: …`, `orchestration: …`). No `Co-Authored-By` lines and no Anthropic/Claude references.
- **Do not bump versions by hand** and **do not run `make pr`** — it is broken and pushes a bump to `main` before failing.
- **The feature order is load-bearing.** `FEATURES` is the column order the model was fitted on *and* the order of the `features` array in the output. Never sort it, never derive it from a schema.
- **Output dtypes are pinned before writing:** `studyLocusId` and `geneId` `pl.String`, `score` `pl.Float64`, `features[].value`, `features[].shapValue` and `shapBaseValue` `pl.Float32`.
- **Hyperparameters, verbatim from the 26.09-2 model** (they are in neither repo — see the spec):
  `objective=binary:logistic`, `eval_metric=aucpr`, `random_state=777`, `n_estimators=300`, `max_depth=5`, `min_child_weight=10`, `eta=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=1`, `reg_lambda=1.0`, `scale_pos_weight=0.8`, `gamma=0`, `max_delta_step=1`.
- **`write_dataset` refuses an occupied destination.** Tests must write into a fresh `tmp_path` subdirectory.

## File Structure

| File | Responsibility |
| --- | --- |
| `pts/src/pts/transformers/l2g/__init__.py` | package marker, no logic |
| `pts/src/pts/transformers/l2g/features.py` | the `FEATURES` contract; imputation and float32 cast shared by both tasks |
| `pts/src/pts/transformers/l2g/gold_standard.py` | parse the curated gold standard; build the annotated gold-standard feature matrix |
| `pts/src/pts/transformers/l2g/split.py` | the predefined-test derivation, label encoding, split statistics |
| `pts/src/pts/transformers/l2g/model.py` | matrix building, fit, evaluate, save, load |
| `pts/src/pts/transformers/l2g/explain.py` | SHAP background construction and the process-pool driver |
| `pts/src/pts/transformers/l2g_train.py` | `l2g_train` transformer entry point (IO only) |
| `pts/src/pts/transformers/l2g_predict.py` | `l2g_predict` transformer entry point (IO only) |
| `pts/tests/test_l2g_features.py` … `test_l2g_transformers.py` | unit tests, one module per source module |
| `pts/tests/test_l2g_parity.py` | the gated parity harness against 26.09-2 |
| `pts/config.yaml` | the `l2g` step |
| `pts/pyproject.toml` | `xgboost`, `shap`, `scikit-learn`, `skops` |
| `orchestration/…/dags/config/gentropy.yaml` | three step blocks removed |
| `orchestration/…/dags/config/unified_pipeline.yaml` | `pts_l2g` added, three `gentropy_l2g_*` removed |
| `orchestration/…/dags/unified_pipeline.py` | differ entries and the `l2g_training_version` variable removed |
| `orchestration/…/assets/l2g_predict.sh` | deleted |

---

### Task 1: Dependencies and the feature contract

**Files:**
- Modify: `pts/pyproject.toml`
- Create: `pts/src/pts/transformers/l2g/__init__.py`
- Create: `pts/src/pts/transformers/l2g/features.py`
- Test: `pts/tests/test_l2g_features.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FEATURES: tuple[str, ...]` (31 names, ordered), `FIXED_COLUMNS: tuple[str, ...]`, `LOCUS_MEAN_IMPUTED: tuple[str, ...]`, and `impute_and_cast(frame: pl.LazyFrame, features: Sequence[str], *, keep: Sequence[str] = ()) -> pl.LazyFrame`.

- [ ] **Step 1: Add the dependencies**

In `pts/pyproject.toml`, add to `[project].dependencies`:

```toml
  "xgboost>=3.4.1",
  "shap>=0.52.0",
  "scikit-learn>=1.7.0",
  "skops>=0.11.0",
```

- [ ] **Step 2: Lock and install**

Run from `pts/`: `uv lock` then `uv sync --all-extras --dev`
Expected: `uv.lock` updated, install succeeds.

- [ ] **Step 3: Write the failing test**

Create `pts/tests/test_l2g_features.py`:

```python
"""Tests for the shared L2G feature contract."""

import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES, impute_and_cast


def test_features_has_31_names_in_fitted_order() -> None:
    assert len(FEATURES) == 31
    assert len(set(FEATURES)) == 31
    assert FEATURES[0] == 'eQtlColocClppMaximum'
    assert FEATURES[-1] == 'transPQtlColocH4MaximumNeighbourhood'


def test_impute_and_cast_fills_gene_counts_with_the_locus_mean() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a', 'a', 'a'],
        'geneId': ['g1', 'g2', 'g3'],
        'geneCount500kb': [2.0, 4.0, None],
        'proteinGeneCount500kb': [1.0, 1.0, 1.0],
    })
    out = impute_and_cast(frame, ['geneCount500kb', 'proteinGeneCount500kb']).collect()
    assert out['geneCount500kb'].to_list() == [2.0, 4.0, 3.0]


def test_impute_and_cast_does_not_borrow_the_mean_across_loci() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a', 'b'],
        'geneId': ['g1', 'g2'],
        'geneCount500kb': [8.0, None],
        'proteinGeneCount500kb': [1.0, 1.0],
    })
    out = impute_and_cast(frame, ['geneCount500kb', 'proteinGeneCount500kb']).collect()
    # locus 'b' has no non-null value of its own, so the mean is null and the 0.0 fill applies
    assert out['geneCount500kb'].to_list() == [8.0, 0.0]


def test_impute_and_cast_fills_every_other_feature_with_zero() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a'],
        'geneId': ['g1'],
        'vepMaximum': [None],
    })
    out = impute_and_cast(frame, ['vepMaximum']).collect()
    assert out['vepMaximum'].to_list() == [0.0]


def test_impute_and_cast_returns_features_in_the_requested_order_as_float32() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a'],
        'geneId': ['g1'],
        'vepMean': [1.0],
        'vepMaximum': [2.0],
    })
    out = impute_and_cast(frame, ['vepMaximum', 'vepMean']).collect()
    assert out.columns == ['studyLocusId', 'geneId', 'vepMaximum', 'vepMean']
    assert out.schema['vepMaximum'] == pl.Float32
    assert out.schema['vepMean'] == pl.Float32


def test_impute_and_cast_keeps_extra_columns_after_the_fixed_ones() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
        'vepMaximum': [2.0],
    })
    out = impute_and_cast(frame, ['vepMaximum'], keep=['goldStandardSet']).collect()
    assert out.columns == ['studyLocusId', 'geneId', 'goldStandardSet', 'vepMaximum']


def test_impute_and_cast_raises_when_a_feature_is_absent() -> None:
    frame = pl.LazyFrame({'studyLocusId': ['a'], 'geneId': ['g1']})
    with pytest.raises(ValueError, match='vepMaximum'):
        impute_and_cast(frame, ['vepMaximum']).collect()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_features.py -rxs`
Expected: FAIL, `ModuleNotFoundError: No module named 'pts.transformers.l2g'`

- [ ] **Step 5: Write the implementation**

Create `pts/src/pts/transformers/l2g/__init__.py`:

```python
"""Locus-to-gene training and prediction, ported from gentropy onto polars."""
```

Create `pts/src/pts/transformers/l2g/features.py`:

```python
"""The L2G feature contract, shared by training and prediction.

`FEATURES` is the order the model was fitted on AND the order of the `features` array in
`output/l2g_prediction`. It is a contract, not a preference: sorting it or deriving it from a
schema silently changes what the classifier is handed.
"""

from collections.abc import Sequence

import polars as pl

FEATURES: tuple[str, ...] = (
    'eQtlColocClppMaximum',
    'pQtlColocClppMaximum',
    'sQtlColocClppMaximum',
    'eQtlColocH4Maximum',
    'pQtlColocH4Maximum',
    'sQtlColocH4Maximum',
    'eQtlColocClppMaximumNeighbourhood',
    'pQtlColocClppMaximumNeighbourhood',
    'sQtlColocClppMaximumNeighbourhood',
    'eQtlColocH4MaximumNeighbourhood',
    'pQtlColocH4MaximumNeighbourhood',
    'sQtlColocH4MaximumNeighbourhood',
    'distanceSentinelFootprint',
    'distanceSentinelFootprintNeighbourhood',
    'distanceFootprintMean',
    'distanceFootprintMeanNeighbourhood',
    'distanceTssMean',
    'distanceTssMeanNeighbourhood',
    'distanceSentinelTss',
    'distanceSentinelTssNeighbourhood',
    'vepMaximum',
    'vepMaximumNeighbourhood',
    'vepMean',
    'vepMeanNeighbourhood',
    'e2gMean',
    'e2gMeanNeighbourhood',
    'geneCount500kb',
    'proteinGeneCount500kb',
    'credibleSetConfidence',
    'transPQtlColocH4Maximum',
    'transPQtlColocH4MaximumNeighbourhood',
)
"""The 31 features, in the order `LocusToGeneConfig.features_list` declares them."""

FIXED_COLUMNS: tuple[str, ...] = ('studyLocusId', 'geneId')
"""Identity columns that are never handed to the classifier."""

LOCUS_MEAN_IMPUTED: tuple[str, ...] = ('geneCount500kb', 'proteinGeneCount500kb')
"""Gene attributes imputed with the mean over their study locus before the zero fill."""


def impute_and_cast(
    frame: pl.LazyFrame,
    features: Sequence[str],
    *,
    keep: Sequence[str] = (),
) -> pl.LazyFrame:
    """Reproduce gentropy's `L2GFeatureMatrix.fill_na` followed by `select_features`.

    The two gene-count columns are filled with the mean over their `studyLocusId` partition;
    every remaining null in every feature becomes 0.0. Features are then cast to `Float32` and
    returned in the requested order, which is the order the classifier expects.

    The locus mean is computed in `Float64` and cast back to `Float32`, because spark's
    `avg` returns a double and `select_features` casts the result back to float. Averaging in
    `Float32` throughout would round differently.

    Args:
        frame: feature matrix, one row per (study locus, gene).
        features: feature names, in the order the model was fitted on.
        keep: extra columns to carry through, placed after the fixed columns.

    Returns:
        LazyFrame of `FIXED_COLUMNS + keep + features`, features as `Float32`.

    Raises:
        ValueError: if any requested feature is absent from the frame.
    """
    available = set(frame.collect_schema().names())
    if missing := [name for name in features if name not in available]:
        msg = f'feature matrix is missing {missing}'
        raise ValueError(msg)

    cast = frame.with_columns(pl.col(name).cast(pl.Float32).alias(name) for name in features)

    imputed = cast.with_columns(
        pl.col(name)
        .fill_null(pl.col(name).cast(pl.Float64).mean().over('studyLocusId').cast(pl.Float32))
        .alias(name)
        for name in LOCUS_MEAN_IMPUTED
        if name in available
    )

    return imputed.select(
        *FIXED_COLUMNS,
        *keep,
        *(pl.col(name).fill_null(0.0).alias(name) for name in features),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_features.py -rxs`
Expected: 7 passed

- [ ] **Step 7: Lint**

Run: `cd pts && uv run --frozen ruff check src/pts/transformers/l2g tests/test_l2g_features.py`
Expected: no findings

- [ ] **Step 8: Commit**

```bash
git add pts/pyproject.toml pts/uv.lock pts/src/pts/transformers/l2g pts/tests/test_l2g_features.py
git commit -m "pts: add the L2G feature contract and its imputation"
```

---

### Task 2: Gold-standard parsing and annotation

**Files:**
- Create: `pts/src/pts/transformers/l2g/gold_standard.py`
- Test: `pts/tests/test_l2g_gold_standard.py`

**Interfaces:**
- Consumes: `FEATURES`, `impute_and_cast` from `pts.transformers.l2g.features`.
- Produces: `CURATED_COLUMNS: tuple[str, ...]`, `OTG_CURATION_COLUMNS: frozenset[str]`, `parse_gold_standard(frame: pl.LazyFrame) -> pl.LazyFrame`, `annotate(feature_matrix: pl.LazyFrame, credible_set: pl.LazyFrame, gold_standard: pl.LazyFrame, features: Sequence[str]) -> pl.LazyFrame`.

- [ ] **Step 1: Write the failing test**

Create `pts/tests/test_l2g_gold_standard.py`:

```python
"""Tests for gold-standard parsing and the annotated gold-standard feature matrix."""

import polars as pl
import pytest

from pts.transformers.l2g.gold_standard import annotate, parse_gold_standard

CURATED = {
    'studyLocusId': ['sl1'],
    'geneId': ['g1'],
    'diseaseIds': [['EFO_1']],
    'variantId': ['1_1_A_G'],
    'studyId': ['GCST1'],
    'goldStandardSet': ['positive'],
}


def test_parse_drops_diseaseids_and_keeps_the_curated_columns() -> None:
    out = parse_gold_standard(pl.LazyFrame(CURATED)).collect()
    assert out.columns == ['studyLocusId', 'variantId', 'studyId', 'geneId', 'goldStandardSet']


def test_parse_rejects_the_unported_otg_curation_format() -> None:
    frame = pl.LazyFrame({
        'association_info': [None],
        'gold_standard_info': [None],
        'metadata': [None],
        'sentinel_variant': [None],
        'trait_info': [None],
    })
    with pytest.raises(ValueError, match='OTG curation format'):
        parse_gold_standard(frame)


def test_parse_rejects_a_frame_missing_a_curated_column() -> None:
    frame = pl.LazyFrame({k: v for k, v in CURATED.items() if k != 'goldStandardSet'})
    with pytest.raises(ValueError, match='goldStandardSet'):
        parse_gold_standard(frame)


def _feature_matrix() -> pl.LazyFrame:
    return pl.LazyFrame({
        'studyLocusId': ['sl1', 'sl1', 'sl2'],
        'geneId': ['g1', 'g2', 'g1'],
        'isProteinCoding': [1, 1, 0],
        'vepMaximum': [0.5, 0.25, 0.75],
    })


def _credible_set() -> pl.LazyFrame:
    return pl.LazyFrame({
        'studyLocusId': ['sl1', 'sl2'],
        'variantId': ['1_1_A_G', '2_2_C_T'],
        'studyId': ['GCST1', 'GCST2'],
        'studyType': ['gwas', 'gwas'],
    })


def test_annotate_keeps_only_gold_standard_pairs() -> None:
    gold = pl.LazyFrame({
        'studyLocusId': ['sl1'],
        'variantId': ['1_1_A_G'],
        'studyId': ['GCST1'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    out = annotate(_feature_matrix(), _credible_set(), gold, ['vepMaximum']).collect()
    assert out.select('studyLocusId', 'geneId').rows() == [('sl1', 'g1')]
    assert out.columns == ['studyLocusId', 'geneId', 'goldStandardSet', 'vepMaximum']


def test_annotate_drops_non_protein_coding_rows() -> None:
    gold = pl.LazyFrame({
        'studyLocusId': ['sl2'],
        'variantId': ['2_2_C_T'],
        'studyId': ['GCST2'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    out = annotate(_feature_matrix(), _credible_set(), gold, ['vepMaximum']).collect()
    assert out.height == 0


def test_annotate_deduplicates_identical_rows() -> None:
    gold = pl.LazyFrame({
        'studyLocusId': ['sl1', 'sl1'],
        'variantId': ['1_1_A_G', '1_1_A_G'],
        'studyId': ['GCST1', 'GCST1'],
        'geneId': ['g1', 'g1'],
        'goldStandardSet': ['positive', 'positive'],
    })
    out = annotate(_feature_matrix(), _credible_set(), gold, ['vepMaximum']).collect()
    assert out.height == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_gold_standard.py -rxs`
Expected: FAIL, `ModuleNotFoundError: No module named 'pts.transformers.l2g.gold_standard'`

- [ ] **Step 3: Write the implementation**

Create `pts/src/pts/transformers/l2g/gold_standard.py`:

```python
"""Parse the curated L2G gold standard and annotate it with feature-matrix rows.

gentropy's `_parse_gold_standard` has three branches. On the curated file this pipeline
actually consumes -- `gs://otar001-core/l2g/goldStandard/<version>` -- only the second fires:
the file carries one unexpected column, `diseaseIds`, which is dropped. The OTG-curation branch
needs `find_overlaps`, the variant index and the PPI dataset, and has never run on this data, so
it is NOT ported. A file in that format must fail loudly rather than be mis-parsed.
"""

from collections.abc import Sequence

import polars as pl

from pts.transformers.l2g.features import impute_and_cast

CURATED_COLUMNS: tuple[str, ...] = (
    'studyLocusId',
    'variantId',
    'studyId',
    'geneId',
    'goldStandardSet',
)
"""The columns `L2GGoldStandard` declares as mandatory, in schema order."""

OTG_CURATION_COLUMNS: frozenset[str] = frozenset({
    'association_info',
    'gold_standard_info',
    'metadata',
    'sentinel_variant',
    'trait_info',
})
"""Columns that identify the legacy OTG curation format, which is not supported here."""

LABEL_COLUMN = 'goldStandardSet'


def parse_gold_standard(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Reduce the curated gold standard to its mandatory columns.

    Args:
        frame: the gold-standard file as read, including any extra columns.

    Returns:
        LazyFrame of `CURATED_COLUMNS`.

    Raises:
        ValueError: if the frame is in the unported OTG curation format, or is missing a
            mandatory column.
    """
    columns = set(frame.collect_schema().names())

    if present := OTG_CURATION_COLUMNS & columns:
        msg = (
            f'gold standard is in the OTG curation format (found {sorted(present)}), which is not '
            'supported. Port `L2GGoldStandard.from_otg_curation` before using such a file.'
        )
        raise ValueError(msg)

    if missing := [name for name in CURATED_COLUMNS if name not in columns]:
        msg = f'gold standard is missing mandatory columns {missing}'
        raise ValueError(msg)

    return frame.select(CURATED_COLUMNS)


def annotate(
    feature_matrix: pl.LazyFrame,
    credible_set: pl.LazyFrame,
    gold_standard: pl.LazyFrame,
    features: Sequence[str],
) -> pl.LazyFrame:
    """Build the feature matrix restricted to gold-standard (study locus, gene) pairs.

    Mirrors `L2GGoldStandard.build_feature_matrix`: the credible set supplies `variantId` and
    `studyId` for each study locus, the gold standard is joined on all three keys, non
    protein-coding rows are dropped, and the result is deduplicated and imputed.

    Args:
        feature_matrix: the full feature matrix, including `isProteinCoding`.
        credible_set: credible sets, supplying `studyLocusId`, `variantId` and `studyId`.
        gold_standard: output of `parse_gold_standard`.
        features: feature names, in fitted order.

    Returns:
        LazyFrame of `studyLocusId, geneId, goldStandardSet` followed by the features.
    """
    keyed = feature_matrix.join(
        credible_set.select('studyLocusId', 'variantId', 'studyId'),
        on='studyLocusId',
        how='left',
    )

    matched = (
        keyed.join(gold_standard, on=['studyId', 'variantId', 'geneId'], how='inner')
        .filter(pl.col('isProteinCoding') == 1)
        .drop('studyId', 'variantId', 'isProteinCoding')
        .unique(maintain_order=True)
    )

    return impute_and_cast(matched, features, keep=[LABEL_COLUMN])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_gold_standard.py -rxs`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add pts/src/pts/transformers/l2g/gold_standard.py pts/tests/test_l2g_gold_standard.py
git commit -m "pts: parse the curated L2G gold standard and annotate it"
```

---

### Task 3: The predefined-test split derivation

**Files:**
- Create: `pts/src/pts/transformers/l2g/split.py`
- Test: `pts/tests/test_l2g_split.py`

**Interfaces:**
- Consumes: `LABEL_COLUMN` from `pts.transformers.l2g.gold_standard`.
- Produces: `encode_labels(frame: pl.LazyFrame) -> pl.LazyFrame`, `derive_splits(annotated: pl.LazyFrame, predefined_test: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]`, `split_stats(annotated_total: int, predefined_total: int, train: pl.DataFrame, test: pl.DataFrame) -> dict[str, object]`.

- [ ] **Step 1: Write the failing test**

Create `pts/tests/test_l2g_split.py`:

```python
"""Tests for the pinned train/test split derivation."""

import polars as pl

from pts.transformers.l2g.split import derive_splits, encode_labels, split_stats


def _annotated() -> pl.LazyFrame:
    return pl.LazyFrame({
        'studyLocusId': ['sl1', 'sl1', 'sl2', 'sl3'],
        'geneId': ['g1', 'g2', 'g1', 'g3'],
        'goldStandardSet': ['positive', 'negative', 'negative', 'positive'],
        'vepMaximum': [0.5, 0.25, 0.75, 0.125],
    })


def test_encode_labels_maps_strings_to_integers() -> None:
    out = encode_labels(pl.LazyFrame({'goldStandardSet': ['negative', 'positive']})).collect()
    assert out['goldStandardSet'].to_list() == [0, 1]


def test_encode_labels_passes_already_encoded_integers_through() -> None:
    out = encode_labels(pl.LazyFrame({'goldStandardSet': ['0', '1']})).collect()
    assert out['goldStandardSet'].to_list() == [0, 1]


def test_derive_splits_excludes_every_locus_touching_a_test_positive_gene() -> None:
    # g1 is positive in the test set, so sl1 AND sl2 leave training even though sl2's
    # own row for g1 is labelled negative.
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl1'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    train, test = derive_splits(_annotated(), predefined)
    assert sorted(train['studyLocusId'].unique().to_list()) == ['sl3']
    assert test.select('studyLocusId', 'geneId').rows() == [('sl1', 'g1')]


def test_derive_splits_ignores_the_current_label_when_finding_contamination() -> None:
    # sl2/g1 flipped to negative in this run, but the pair is still in the test set and
    # must not reach training.
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl2'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    train, test = derive_splits(_annotated(), predefined)
    assert 'sl1' not in train['studyLocusId'].to_list()
    assert test.select('studyLocusId', 'geneId').rows() == [('sl2', 'g1')]


def test_derive_splits_re_derives_test_features_from_the_current_matrix() -> None:
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl2'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    _, test = derive_splits(_annotated(), predefined)
    assert test['vepMaximum'].to_list() == [0.75]


def test_derive_splits_accepts_integer_labels_in_the_predefined_test_set() -> None:
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl1'],
        'geneId': ['g1'],
        'goldStandardSet': [1],
    })
    train, test = derive_splits(_annotated(), predefined)
    assert test.height == 1
    assert 'sl1' not in train['studyLocusId'].to_list()


def test_derive_splits_returns_integer_labels() -> None:
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl1'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    train, test = derive_splits(_annotated(), predefined)
    assert train['goldStandardSet'].dtype == pl.Int32
    assert test['goldStandardSet'].to_list() == [1]


def test_split_stats_reports_the_gentropy_key_set() -> None:
    train = pl.DataFrame({
        'studyLocusId': ['sl3'],
        'geneId': ['g3'],
        'goldStandardSet': [1],
    })
    test = pl.DataFrame({
        'studyLocusId': ['sl1'],
        'geneId': ['g1'],
        'goldStandardSet': [0],
    })
    stats = split_stats(4, 1, train, test)
    assert stats == {
        'n_original_total': 4,
        'n_original_test': 1,
        'n_test_new': 1,
        'n_lost_test': 0,
        'n_train': 1,
        'n_lost_total': 2,
        'train': {
            'n_positive': 1,
            'n_negative': 0,
            'n_unique_loci': 1,
            'n_unique_positive_genes': 1,
        },
        'test': {
            'n_positive': 0,
            'n_negative': 1,
            'n_unique_loci': 1,
            'n_unique_positive_genes': 0,
        },
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_split.py -rxs`
Expected: FAIL, `ModuleNotFoundError: No module named 'pts.transformers.l2g.split'`

- [ ] **Step 3: Write the implementation**

Create `pts/src/pts/transformers/l2g/split.py`:

```python
"""Derive the train and test splits from the pinned, predefined test set.

The test set is pinned across releases rather than resampled, so models stay comparable. This
module implements only that path: gentropy's fresh hierarchical split is not ported, because the
pipeline always supplies `predefined_test_parquet_path` and a second split would defeat the point
of pinning the first.
"""

from typing import Any

import polars as pl

from pts.transformers.l2g.gold_standard import LABEL_COLUMN

POSITIVE = 'positive'
NEGATIVE = 'negative'


def encode_labels(frame: pl.LazyFrame, column: str = LABEL_COLUMN) -> pl.LazyFrame:
    """Encode `negative` as 0 and `positive` as 1, tolerating already-encoded values.

    Args:
        frame: frame carrying the label column.
        column: name of the label column.

    Returns:
        The frame with `column` as `Int32`.
    """
    as_string = pl.col(column).cast(pl.String)
    return frame.with_columns(
        pl.when(as_string == NEGATIVE)
        .then(0)
        .when(as_string == POSITIVE)
        .then(1)
        .otherwise(as_string.cast(pl.Int32, strict=False))
        .cast(pl.Int32)
        .alias(column)
    )


def derive_splits(
    annotated: pl.LazyFrame,
    predefined_test: pl.LazyFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the annotated gold-standard matrix against a pinned test set.

    The test partition is re-derived from the current annotated matrix using the pinned
    (study locus, gene) pairs, so its features track the release. The training partition drops
    every study locus that contains a gene which is positive anywhere in the test set.

    The row's own label is deliberately NOT consulted when finding contamination: a locus whose
    gene label flipped from positive to negative between releases is still in the test set via
    the pair join, and must stay out of training.

    Args:
        annotated: output of `gold_standard.annotate`, string or integer labels.
        predefined_test: the pinned test set; only `studyLocusId`, `geneId` and the label are used.

    Returns:
        `(train, test)` as eager DataFrames with `Int32` labels.
    """
    encoded = encode_labels(annotated)
    pinned = encode_labels(predefined_test.select('studyLocusId', 'geneId', LABEL_COLUMN))

    test_positive_genes = pinned.filter(pl.col(LABEL_COLUMN) == 1).select('geneId').unique()

    contaminated = (
        encoded.join(test_positive_genes, on='geneId', how='inner').select('studyLocusId').unique()
    )

    train = encoded.join(contaminated, on='studyLocusId', how='anti')
    test = encoded.join(pinned.select('studyLocusId', 'geneId'), on=['studyLocusId', 'geneId'], how='inner')

    return train.collect(), test.collect()


def _set_stats(frame: pl.DataFrame) -> dict[str, int]:
    """Descriptive statistics for one partition.

    Args:
        frame: a train or test partition with integer labels.

    Returns:
        Counts of positives, negatives, distinct loci and distinct positive genes.
    """
    positive = frame.filter(pl.col(LABEL_COLUMN) == 1)
    return {
        'n_positive': positive.height,
        'n_negative': frame.filter(pl.col(LABEL_COLUMN) == 0).height,
        'n_unique_loci': frame['studyLocusId'].n_unique(),
        'n_unique_positive_genes': positive['geneId'].n_unique(),
    }


def split_stats(
    annotated_total: int,
    predefined_total: int,
    train: pl.DataFrame,
    test: pl.DataFrame,
) -> dict[str, Any]:
    """Build the split statistics document, matching gentropy's key set.

    Args:
        annotated_total: rows in the annotated gold-standard matrix.
        predefined_total: rows in the pinned test set as staged.
        train: the training partition.
        test: the test partition.

    Returns:
        A JSON-serialisable dict.
    """
    return {
        'n_original_total': annotated_total,
        'n_original_test': predefined_total,
        'n_test_new': test.height,
        'n_lost_test': predefined_total - test.height,
        'n_train': train.height,
        'n_lost_total': annotated_total - test.height - train.height,
        'train': _set_stats(train),
        'test': _set_stats(test),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_split.py -rxs`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add pts/src/pts/transformers/l2g/split.py pts/tests/test_l2g_split.py
git commit -m "pts: derive the L2G train/test split from the pinned test set"
```

---

### Task 4: Model fit, evaluation and persistence

**Files:**
- Create: `pts/src/pts/transformers/l2g/model.py`
- Test: `pts/tests/test_l2g_model.py`

**Interfaces:**
- Consumes: `LABEL_COLUMN` from `pts.transformers.l2g.gold_standard`.
- Produces: `DEFAULT_HYPERPARAMETERS: dict[str, object]`, `to_matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray`, `to_labels(frame: pl.DataFrame) -> np.ndarray`, `fit(x, y, hyperparameters) -> XGBClassifier`, `evaluate(model, x, y) -> dict[str, float]`, `save_model(model, path: str) -> None`, `load_model(path: str) -> XGBClassifier`, `missingness(frame: pl.DataFrame, features) -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

Create `pts/tests/test_l2g_model.py`:

```python
"""Tests for L2G model fitting, evaluation and persistence."""

import numpy as np
import polars as pl
import pytest

from pts.transformers.l2g.model import (
    DEFAULT_HYPERPARAMETERS,
    evaluate,
    fit,
    load_model,
    missingness,
    save_model,
    to_labels,
    to_matrix,
)


@pytest.fixture
def separable() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0.0, 0.1, (60, 2)), rng.normal(3.0, 0.1, (60, 2))]).astype(np.float32)
    y = np.concatenate([np.zeros(60), np.ones(60)]).astype(np.int32)
    return x, y


def test_default_hyperparameters_match_the_released_model() -> None:
    assert DEFAULT_HYPERPARAMETERS['n_estimators'] == 300
    assert DEFAULT_HYPERPARAMETERS['random_state'] == 777
    assert DEFAULT_HYPERPARAMETERS['max_delta_step'] == 1
    assert DEFAULT_HYPERPARAMETERS['gamma'] == 0


def test_to_matrix_uses_the_requested_feature_order() -> None:
    frame = pl.DataFrame({'a': [1.0], 'b': [2.0], 'studyLocusId': ['sl1']})
    assert to_matrix(frame, ['b', 'a']).tolist() == [[2.0, 1.0]]


def test_to_matrix_returns_float32() -> None:
    frame = pl.DataFrame({'a': [1.0]})
    assert to_matrix(frame, ['a']).dtype == np.float32


def test_to_labels_returns_integers() -> None:
    frame = pl.DataFrame({'goldStandardSet': [0, 1]})
    assert to_labels(frame).tolist() == [0, 1]


def test_fit_produces_a_model_with_the_configured_number_of_trees(separable) -> None:
    x, y = separable
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    assert model.get_booster().num_boosted_rounds() == 7


def test_fit_is_deterministic_for_a_fixed_seed(separable) -> None:
    x, y = separable
    first = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    second = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))


def test_evaluate_reports_the_six_gentropy_metrics(separable) -> None:
    x, y = separable
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    metrics = evaluate(model, x, y)
    assert set(metrics) == {
        'areaUnderROC',
        'accuracy',
        'weightedPrecision',
        'averagePrecision',
        'weightedRecall',
        'f1',
    }
    assert metrics['accuracy'] == pytest.approx(1.0)


def test_save_and_load_round_trip(separable, tmp_path) -> None:
    x, y = separable
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    path = tmp_path / 'model' / 'classifier.skops'
    save_model(model, str(path))
    assert path.exists()
    np.testing.assert_array_equal(load_model(str(path)).predict_proba(x), model.predict_proba(x))


def test_save_model_rejects_a_path_that_is_not_skops(tmp_path) -> None:
    with pytest.raises(ValueError, match='.skops'):
        save_model(object(), str(tmp_path / 'classifier.json'))


def test_missingness_counts_null_and_zero_as_missing() -> None:
    frame = pl.DataFrame({'a': [0.0, 1.0, 2.0, 3.0], 'b': [1.0, 1.0, 1.0, 1.0]})
    assert missingness(frame, ['a', 'b']) == {'a': 0.25, 'b': 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_model.py -rxs`
Expected: FAIL, `ModuleNotFoundError: No module named 'pts.transformers.l2g.model'`

- [ ] **Step 3: Write the implementation**

Create `pts/src/pts/transformers/l2g/model.py`:

```python
"""Fit, evaluate and persist the L2G classifier.

`DEFAULT_HYPERPARAMETERS` is copied from the model that shipped with 26.09-2, read off
`classifier.skops` itself. Three of its entries -- `n_estimators`, `gamma` and `max_delta_step` --
appear in neither gentropy nor this repository, so reading them from gentropy's config would give
a 100-tree model where the released one has 300. They live here so that never happens again.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import skops.io as sio
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from pts.transformers.l2g.gold_standard import LABEL_COLUMN

DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'random_state': 777,
    'n_estimators': 300,
    'max_depth': 5,
    'min_child_weight': 10,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 0.8,
    'gamma': 0,
    'max_delta_step': 1,
}
"""Read from the 26.09-2 `classifier.skops`; see the module docstring."""


def to_matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Extract the feature matrix in fitted order.

    Args:
        frame: a partition carrying at least the feature columns.
        features: feature names, in the order the model expects.

    Returns:
        A `float32` array of shape `(rows, len(features))`.
    """
    return frame.select(features).to_numpy().astype(np.float32)


def to_labels(frame: pl.DataFrame, column: str = LABEL_COLUMN) -> np.ndarray:
    """Extract the integer label vector.

    Args:
        frame: a partition carrying the label column.
        column: name of the label column.

    Returns:
        An `int32` array.
    """
    return frame[column].to_numpy().astype(np.int32)


def fit(x: np.ndarray, y: np.ndarray, hyperparameters: dict[str, Any]) -> XGBClassifier:
    """Fit the classifier.

    Args:
        x: feature matrix.
        y: integer labels.
        hyperparameters: passed straight to `XGBClassifier`.

    Returns:
        The fitted classifier.
    """
    model = XGBClassifier(**hyperparameters)
    model.fit(X=x, y=y)
    return model


def evaluate(model: XGBClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Evaluate a fitted model, reporting the six metrics gentropy reports.

    Args:
        model: a fitted classifier.
        x: feature matrix.
        y: true integer labels.

    Returns:
        Metric name to value.
    """
    predicted = model.predict(x)
    probabilities = model.predict_proba(x)
    return {
        'areaUnderROC': float(roc_auc_score(y, probabilities[:, 1], average='weighted')),
        'accuracy': float(accuracy_score(y, predicted)),
        'weightedPrecision': float(precision_score(y, predicted, average='weighted', zero_division=0)),
        'averagePrecision': float(average_precision_score(y, probabilities[:, 1], average='weighted')),
        'weightedRecall': float(recall_score(y, predicted, average='weighted', zero_division=0)),
        'f1': float(f1_score(y, predicted, average='weighted', zero_division=0)),
    }


def save_model(model: Any, path: str) -> None:
    """Persist a fitted model in the skops format, at the path the release expects.

    Args:
        model: the fitted classifier.
        path: destination, which must end in `.skops`.

    Raises:
        ValueError: if `path` does not end in `.skops`.
    """
    if not path.endswith('.skops'):
        msg = f'model path must end with .skops, got {path!r}'
        raise ValueError(msg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)


def load_model(path: str) -> XGBClassifier:
    """Load a model persisted by `save_model`.

    `skops` refuses unknown types unless they are named as trusted, so the file's own reported
    types are passed through. That is safe here because the file is written by this pipeline into
    the release it is read back from.

    Args:
        path: a `.skops` file.

    Returns:
        The classifier.
    """
    return sio.load(path, trusted=sio.get_untrusted_types(file=path))


def missingness(frame: pl.DataFrame, features: Sequence[str]) -> dict[str, float]:
    """Fraction of rows where each feature is null or zero.

    Mirrors `L2GFeatureMatrix.calculate_feature_missingness_rate`, which counts zero as missing
    because the imputation has already turned nulls into zeros by the time it runs.

    Args:
        frame: the partition to measure.
        features: feature names.

    Returns:
        Feature name to fraction in [0, 1].
    """
    total = frame.height
    if total == 0:
        return dict.fromkeys(features, 0.0)
    return {
        name: frame.filter(pl.col(name).is_null() | (pl.col(name) == 0)).height / total
        for name in features
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_model.py -rxs`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add pts/src/pts/transformers/l2g/model.py pts/tests/test_l2g_model.py
git commit -m "pts: fit, evaluate and persist the L2G classifier"
```

---

### Task 5: SHAP background and the explainer pool

**Files:**
- Create: `pts/src/pts/transformers/l2g/explain.py`
- Test: `pts/tests/test_l2g_explain.py`

**Interfaces:**
- Consumes: nothing from earlier tasks at runtime; the tests use `fit` and `DEFAULT_HYPERPARAMETERS` from `pts.transformers.l2g.model`.
- Produces: `build_background(frame: pl.DataFrame, features: Sequence[str], size: int, seed: int) -> np.ndarray`, `explain(model, matrix: np.ndarray, background: np.ndarray, *, max_samples: int, workers: int | None = None, chunk_size: int = 50_000) -> tuple[float, np.ndarray]`.

- [ ] **Step 1: Write the failing test**

Create `pts/tests/test_l2g_explain.py`:

```python
"""Tests for SHAP background construction and the explainer pool."""

import numpy as np
import polars as pl
import pytest

from pts.transformers.l2g.explain import build_background, explain
from pts.transformers.l2g.model import DEFAULT_HYPERPARAMETERS, fit


@pytest.fixture
def model_and_data():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0.0, 0.1, (60, 2)), rng.normal(3.0, 0.1, (60, 2))]).astype(np.float32)
    y = np.concatenate([np.zeros(60), np.ones(60)]).astype(np.int32)
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 5, 'max_depth': 2})
    return model, x


def test_build_background_returns_the_requested_size() -> None:
    frame = pl.DataFrame({'a': list(range(100)), 'b': list(range(100))}).cast(pl.Float32)
    assert build_background(frame, ['a', 'b'], 10, 42).shape == (10, 2)


def test_build_background_is_reproducible_for_a_fixed_seed() -> None:
    frame = pl.DataFrame({'a': list(range(100)), 'b': list(range(100))}).cast(pl.Float32)
    first = build_background(frame, ['a', 'b'], 10, 42)
    second = build_background(frame, ['a', 'b'], 10, 42)
    np.testing.assert_array_equal(first, second)


def test_build_background_differs_for_a_different_seed() -> None:
    frame = pl.DataFrame({'a': list(range(100)), 'b': list(range(100))}).cast(pl.Float32)
    assert not np.array_equal(
        build_background(frame, ['a', 'b'], 10, 1), build_background(frame, ['a', 'b'], 10, 2)
    )


def test_build_background_uses_every_row_when_the_frame_is_smaller_than_the_size() -> None:
    frame = pl.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}).cast(pl.Float32)
    assert build_background(frame, ['a', 'b'], 10, 42).shape == (2, 2)


def test_build_background_uses_the_requested_feature_order() -> None:
    frame = pl.DataFrame({'a': [1.0], 'b': [2.0]}).cast(pl.Float32)
    assert build_background(frame, ['b', 'a'], 1, 42).tolist() == [[2.0, 1.0]]


def test_explain_returns_one_shap_value_per_cell(model_and_data) -> None:
    model, x = model_and_data
    _, values = explain(model, x[:20], x[:10], max_samples=10, workers=1)
    assert values.shape == (20, 2)


def test_explain_returns_a_single_base_value(model_and_data) -> None:
    model, x = model_and_data
    base, _ = explain(model, x[:20], x[:10], max_samples=10, workers=1)
    assert isinstance(base, float)


def test_explain_chunking_does_not_change_the_result(model_and_data) -> None:
    model, x = model_and_data
    _, whole = explain(model, x[:20], x[:10], max_samples=10, workers=1, chunk_size=100)
    _, chunked = explain(model, x[:20], x[:10], max_samples=10, workers=1, chunk_size=3)
    np.testing.assert_allclose(whole, chunked, rtol=1e-6, atol=1e-9)


def test_explain_across_workers_matches_a_single_worker(model_and_data) -> None:
    model, x = model_and_data
    _, single = explain(model, x[:20], x[:10], max_samples=10, workers=1, chunk_size=5)
    _, parallel = explain(model, x[:20], x[:10], max_samples=10, workers=2, chunk_size=5)
    np.testing.assert_allclose(single, parallel, rtol=1e-6, atol=1e-9)


def test_explain_handles_an_empty_matrix(model_and_data) -> None:
    model, x = model_and_data
    _, values = explain(model, x[:0], x[:10], max_samples=10, workers=1)
    assert values.shape == (0, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_explain.py -rxs`
Expected: FAIL, `ModuleNotFoundError: No module named 'pts.transformers.l2g.explain'`

- [ ] **Step 3: Write the implementation**

Create `pts/src/pts/transformers/l2g/explain.py`:

```python
"""Interventional TreeSHAP in probability space, spread across processes.

Two things about this are deliberate and easy to get wrong.

First, the masker is constructed explicitly with `max_samples`. Passing a bare array as `data=`
lets `shap` wrap it in an `Independent` masker whose default `max_samples` is 100, which silently
discards most of a larger background -- that is exactly what gentropy does, so its 1000-row sample
has always been a 100-row background. Cost is linear in the size actually used: 317 rows/s/process
at 100, 32 rows/s/process at 1000.

Second, one background serves the whole run. gentropy draws an unseeded sample inside each of its
1000 Batch tasks, so `shapBaseValue` varies across output partitions -- 0.0381 to 0.0668 in
26.09-1. One seeded background gives one base value.
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import polars as pl
import shap

_EXPLAINER: Any = None
"""Per-worker explainer, built once in the pool initialiser."""


def build_background(
    frame: pl.DataFrame,
    features: Sequence[str],
    size: int,
    seed: int,
) -> np.ndarray:
    """Draw the SHAP background from the labelled data.

    Args:
        frame: the rows to sample from, normally train and test concatenated.
        features: feature names, in fitted order.
        size: how many rows to draw; the whole frame is used if it holds fewer.
        seed: seed for the draw, recorded in the run's metrics so it can be reproduced.

    Returns:
        A `float32` array of shape `(min(size, rows), len(features))`.
    """
    matrix = frame.select(features).to_numpy().astype(np.float32)
    if matrix.shape[0] <= size:
        return matrix
    rng = np.random.default_rng(seed)
    return matrix[rng.choice(matrix.shape[0], size, replace=False)]


def _initialise(model: Any, background: np.ndarray, max_samples: int) -> None:
    """Build this worker's explainer.

    Args:
        model: the fitted classifier.
        background: the shared background array.
        max_samples: how many background rows the masker may use.
    """
    global _EXPLAINER  # noqa: PLW0603 -- one explainer per worker process, built once
    masker = shap.maskers.Independent(background, max_samples=max_samples)
    _EXPLAINER = shap.TreeExplainer(
        model,
        data=masker,
        feature_perturbation='interventional',
        model_output='probability',
    )


def _explain_chunk(chunk: np.ndarray) -> np.ndarray:
    """Compute SHAP values for one chunk of rows.

    Args:
        chunk: rows to explain.

    Returns:
        SHAP values, same shape as `chunk`.
    """
    return np.asarray(_EXPLAINER.shap_values(chunk, check_additivity=False))


def explain(
    model: Any,
    matrix: np.ndarray,
    background: np.ndarray,
    *,
    max_samples: int,
    workers: int | None = None,
    chunk_size: int = 50_000,
) -> tuple[float, np.ndarray]:
    """Compute SHAP values for every row, in parallel.

    Args:
        model: the fitted classifier.
        matrix: rows to explain, in fitted feature order.
        background: the background array from `build_background`.
        max_samples: background rows the masker may use.
        workers: process count; defaults to the pool's own default.
        chunk_size: rows per unit of work.

    Returns:
        `(base_value, shap_values)` where `shap_values` has the shape of `matrix`.
    """
    _initialise(model, background, max_samples)
    base_value = float(_EXPLAINER.expected_value)

    if matrix.shape[0] == 0:
        return base_value, np.empty_like(matrix)

    chunks = [matrix[i : i + chunk_size] for i in range(0, matrix.shape[0], chunk_size)]

    if workers == 1:
        return base_value, np.vstack([_explain_chunk(chunk) for chunk in chunks])

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise,
        initargs=(model, background, max_samples),
    ) as pool:
        return base_value, np.vstack(list(pool.map(_explain_chunk, chunks)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_explain.py -rxs`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add pts/src/pts/transformers/l2g/explain.py pts/tests/test_l2g_explain.py
git commit -m "pts: compute L2G SHAP values from one seeded background"
```

---

### Task 6: The `l2g_train` transformer

**Files:**
- Create: `pts/src/pts/transformers/l2g_train.py`
- Modify: `pts/config.yaml` (add the `l2g` step with its first task)
- Test: `pts/tests/test_l2g_train.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `l2g_train(source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], config: Config) -> None`.

- [ ] **Step 1: Write the failing test**

Create `pts/tests/test_l2g_train.py`:

```python
"""End-to-end test for the l2g_train transformer, on tiny local fixtures."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES
from pts.transformers.l2g.model import load_model
from pts.transformers.l2g_train import l2g_train


@pytest.fixture
def workspace(tmp_path):
    """Write a feature matrix, credible set, gold standard and pinned test set to disk."""
    rng = np.random.default_rng(0)
    loci = [f'sl{i}' for i in range(40)]
    genes = [f'ENSG{i % 8:011d}' for i in range(40)]

    matrix = pl.DataFrame(
        {
            'studyLocusId': loci,
            'geneId': genes,
            'isProteinCoding': [1] * 40,
            **{name: rng.random(40) for name in FEATURES},
        }
    )
    (tmp_path / 'fm').mkdir()
    matrix.write_parquet(tmp_path / 'fm' / 'part-0.parquet')

    credible = pl.DataFrame({
        'studyLocusId': loci,
        'variantId': [f'1_{i}_A_G' for i in range(40)],
        'studyId': [f'GCST{i}' for i in range(40)],
        'studyType': ['gwas'] * 40,
    })
    (tmp_path / 'cs').mkdir()
    credible.write_parquet(tmp_path / 'cs' / 'part-0.parquet')

    gold = pl.DataFrame({
        'studyLocusId': loci,
        'geneId': genes,
        'diseaseIds': [['EFO_1']] * 40,
        'variantId': [f'1_{i}_A_G' for i in range(40)],
        'studyId': [f'GCST{i}' for i in range(40)],
        'goldStandardSet': ['positive' if i % 4 == 0 else 'negative' for i in range(40)],
    })
    (tmp_path / 'gs').mkdir()
    gold.write_ndjson(tmp_path / 'gs' / 'part-0.json')

    pinned = pl.DataFrame({
        'studyLocusId': loci[:8],
        'geneId': genes[:8],
        'goldStandardSet': ['positive' if i % 4 == 0 else 'negative' for i in range(8)],
    })
    (tmp_path / 'pt').mkdir()
    pinned.write_parquet(tmp_path / 'pt' / 'part-0.parquet')

    return tmp_path


def _source(workspace) -> dict[str, str]:
    return {
        'feature_matrix': str(workspace / 'fm'),
        'credible_set': str(workspace / 'cs'),
        'gold_standard': str(workspace / 'gs'),
        'predefined_test': str(workspace / 'pt'),
    }


def _destination(workspace) -> dict[str, str]:
    out = workspace / 'out'
    return {
        'model': str(out / 'model' / 'classifier.skops'),
        'background': str(out / 'model' / 'shap_background.parquet'),
        'metrics': str(out / 'model' / 'metrics.json'),
        'train_split': str(out / 'train'),
        'test_split': str(out / 'test'),
        'split_stats': str(out / 'split_stats.json'),
    }


SETTINGS = {
    'features_list': list(FEATURES),
    'hyperparameters': {'n_estimators': 5, 'max_depth': 2, 'random_state': 777},
    'train_on_full_dataset': True,
    'shap_background_size': 10,
    'shap_background_seed': 42,
}


def test_l2g_train_writes_every_declared_artifact(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    for key, path in destination.items():
        assert Path(path).exists(), f'{key} was not written to {path}'


def test_l2g_train_writes_a_loadable_model(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    model = load_model(destination['model'])
    assert model.get_booster().num_boosted_rounds() == 5


def test_l2g_train_metrics_carry_the_six_scores_and_the_run_settings(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    metrics = json.loads(open(destination['metrics']).read())
    assert set(metrics['heldOut']) == {
        'areaUnderROC',
        'accuracy',
        'weightedPrecision',
        'averagePrecision',
        'weightedRecall',
        'f1',
    }
    assert metrics['shapBackgroundSeed'] == 42
    assert metrics['shapBackgroundSize'] == 10
    assert metrics['hyperparameters']['n_estimators'] == 5
    assert metrics['trainOnFullDataset'] is True
    assert set(metrics['featureMissingness']) == set(FEATURES)


def test_l2g_train_split_stats_reconcile(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    stats = json.loads(open(destination['split_stats']).read())
    assert stats['n_train'] + stats['n_test_new'] + stats['n_lost_total'] == stats['n_original_total']


def test_l2g_train_background_has_the_configured_shape(workspace) -> None:
    destination = _destination(workspace)
    l2g_train(_source(workspace), destination, dict(SETTINGS), None)
    background = pl.read_parquet(destination['background'])
    assert background.columns == list(FEATURES)
    assert background.height == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_train.py -rxs`
Expected: FAIL, `ModuleNotFoundError: No module named 'pts.transformers.l2g_train'`

- [ ] **Step 3: Write the implementation**

Create `pts/src/pts/transformers/l2g_train.py`:

```python
"""Train the locus-to-gene classifier from the pinned train/test split.

Replaces gentropy's `LocusToGeneTrainTestSplitStep` and the train mode of `LocusToGeneStep`. Three
things gentropy does are deliberately absent:

* **Cross-validation.** Its folds fed nothing back -- the estimator was cloned, fitted, logged and
  discarded, and the W&B "sweep" grid was built from the model's own single hyperparameter values,
  so it tuned nothing either. It also re-split the training set five times.
* **The Hugging Face Hub upload**, which called `generate_train_test_split` a second time purely to
  attach train/test parquets to the model repo.
* **Weights & Biases.** The metrics it carried are written to `metrics.json` in the release instead,
  versioned with the run that produced them.
"""

import json
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.l2g import explain, gold_standard, model as l2g_model, split
from pts.transformers.l2g.features import FEATURES
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def _write_json(document: dict[str, Any], path: str) -> None:
    """Write a JSON document, creating the parent directory.

    Args:
        document: the object to serialise.
        path: destination file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, indent=2))


def l2g_train(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Build the pinned splits, fit the classifier and write the model and its metrics.

    Args:
        source: keys `feature_matrix`, `credible_set`, `gold_standard`, `predefined_test`.
        destination: keys `model`, `background`, `metrics`, `train_split`, `test_split`,
            `split_stats`.
        settings: keys `features_list`, `hyperparameters`, `train_on_full_dataset`,
            `shap_background_size`, `shap_background_seed`.
        config: otter config; unused, accepted for interface compatibility.
    """
    features = list(settings.get('features_list') or FEATURES)
    hyperparameters = dict(settings['hyperparameters'])
    background_size = int(settings['shap_background_size'])
    background_seed = int(settings['shap_background_seed'])

    logger.info('annotating the gold standard with feature-matrix rows')
    annotated = gold_standard.annotate(
        scan_dataset(source['feature_matrix']),
        scan_dataset(source['credible_set']),
        gold_standard.parse_gold_standard(scan_dataset(source['gold_standard'], format='ndjson')),
        features,
    ).collect()

    predefined = scan_dataset(source['predefined_test']).collect()
    train, test = split.derive_splits(annotated.lazy(), predefined.lazy())
    logger.info(f'split: {train.height} train rows, {test.height} test rows')

    stats = split.split_stats(annotated.height, predefined.height, train, test)
    _write_json(stats, destination['split_stats'])
    write_dataset(train, destination['train_split'])
    write_dataset(test, destination['test_split'])

    x_train = l2g_model.to_matrix(train, features)
    y_train = l2g_model.to_labels(train)
    x_test = l2g_model.to_matrix(test, features)
    y_test = l2g_model.to_labels(test)

    logger.info('fitting the classifier')
    fitted = l2g_model.fit(x_train, y_train, hyperparameters)
    held_out = l2g_model.evaluate(fitted, x_test, y_test)
    logger.info(f'held-out metrics: {held_out}')

    labelled = pl.concat([train, test], how='vertical')
    if settings.get('train_on_full_dataset'):
        # Evaluation above is complete and is not affected by this refit; the saved model simply
        # benefits from the held-out rows too, which is what produced the 26.09-2 model.
        logger.info('refitting on train + held-out for the saved model')
        fitted = l2g_model.fit(
            l2g_model.to_matrix(labelled, features), l2g_model.to_labels(labelled), hyperparameters
        )

    background = explain.build_background(labelled, features, background_size, background_seed)
    Path(destination['background']).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(background, schema=features).write_parquet(destination['background'])

    l2g_model.save_model(fitted, destination['model'])
    _write_json(
        {
            'heldOut': held_out,
            'featureMissingness': l2g_model.missingness(labelled, features),
            'hyperparameters': hyperparameters,
            'features': features,
            'trainOnFullDataset': bool(settings.get('train_on_full_dataset')),
            'shapBackgroundSize': background_size,
            'shapBackgroundSeed': background_seed,
            'split': stats,
        },
        destination['metrics'],
    )
    logger.info('training complete')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_train.py -rxs`
Expected: 5 passed

- [ ] **Step 5: Add the step to `pts/config.yaml`**

Insert before the `VEP_VIEW STEP` block:

```yaml
  #: L2G STEP :#####################################################################################
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
        # Anchored, not a scratchpad sentinel: otter's Scratchpad is string.Template
        # substitution, so `${...}` can only ever produce a string. The order is the
        # order the model is fitted on and the order of the output `features` array.
        features_list: &l2g_features
          - eQtlColocClppMaximum
          - pQtlColocClppMaximum
          - sQtlColocClppMaximum
          - eQtlColocH4Maximum
          - pQtlColocH4Maximum
          - sQtlColocH4Maximum
          - eQtlColocClppMaximumNeighbourhood
          - pQtlColocClppMaximumNeighbourhood
          - sQtlColocClppMaximumNeighbourhood
          - eQtlColocH4MaximumNeighbourhood
          - pQtlColocH4MaximumNeighbourhood
          - sQtlColocH4MaximumNeighbourhood
          - distanceSentinelFootprint
          - distanceSentinelFootprintNeighbourhood
          - distanceFootprintMean
          - distanceFootprintMeanNeighbourhood
          - distanceTssMean
          - distanceTssMeanNeighbourhood
          - distanceSentinelTss
          - distanceSentinelTssNeighbourhood
          - vepMaximum
          - vepMaximumNeighbourhood
          - vepMean
          - vepMeanNeighbourhood
          - e2gMean
          - e2gMeanNeighbourhood
          - geneCount500kb
          - proteinGeneCount500kb
          - credibleSetConfidence
          - transPQtlColocH4Maximum
          - transPQtlColocH4MaximumNeighbourhood
        # Read off the 26.09-2 classifier.skops. n_estimators, gamma and max_delta_step
        # are in neither gentropy nor this repo; do not "restore" them from gentropy's config.
        hyperparameters:
          objective: binary:logistic
          eval_metric: aucpr
          random_state: 777
          n_estimators: 300
          max_depth: 5
          min_child_weight: 10
          eta: 0.05
          subsample: 0.8
          colsample_bytree: 0.8
          reg_alpha: 1
          reg_lambda: 1.0
          scale_pos_weight: 0.8
          gamma: 0
          max_delta_step: 1
        train_on_full_dataset: true
        shap_background_size: 100
        shap_background_seed: 42
  ##################################################################################################
```

- [ ] **Step 6: Run the config suites**

Run: `cd pts && uv run --frozen pytest tests/test_config_specs.py tests/test_config_dataset_paths.py -rxs`
Expected: PASS, including the new `l2g:transform l2g_train` case

- [ ] **Step 7: Commit**

```bash
git add pts/src/pts/transformers/l2g_train.py pts/tests/test_l2g_train.py pts/config.yaml
git commit -m "pts: add the l2g_train transformer and its config step"
```

---

### Task 7: The `l2g_predict` transformer

**Files:**
- Create: `pts/src/pts/transformers/l2g_predict.py`
- Modify: `pts/config.yaml` (add the second task to the `l2g` step)
- Test: `pts/tests/test_l2g_predict.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5, and the artifacts `l2g_train` writes.
- Produces: `build_output(keys: pl.DataFrame, matrix: np.ndarray, shap_values: np.ndarray, base_value: float, features: Sequence[str]) -> pl.DataFrame` and `l2g_predict(source, destination, settings, config) -> None`.

- [ ] **Step 1: Write the failing test**

Create `pts/tests/test_l2g_predict.py`:

```python
"""Tests for the l2g_predict transformer and its output assembly."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from pts.transformers.l2g.features import FEATURES
from pts.transformers.l2g.model import DEFAULT_HYPERPARAMETERS, fit, save_model
from pts.transformers.l2g_predict import build_output, l2g_predict


def test_build_output_pins_the_release_schema() -> None:
    keys = pl.DataFrame({'studyLocusId': ['sl1'], 'geneId': ['g1'], 'score': [0.5]})
    out = build_output(keys, np.array([[1.0, 2.0]], dtype=np.float32),
                       np.array([[0.1, 0.2]], dtype=np.float32), 0.25, ['a', 'b'])
    assert out.columns == ['studyLocusId', 'geneId', 'score', 'features', 'shapBaseValue']
    assert out.schema['studyLocusId'] == pl.String
    assert out.schema['geneId'] == pl.String
    assert out.schema['score'] == pl.Float64
    assert out.schema['shapBaseValue'] == pl.Float32
    assert out.schema['features'] == pl.List(
        pl.Struct({'name': pl.String, 'value': pl.Float32, 'shapValue': pl.Float32})
    )


def test_build_output_keeps_the_features_in_fitted_order() -> None:
    keys = pl.DataFrame({'studyLocusId': ['sl1'], 'geneId': ['g1'], 'score': [0.5]})
    out = build_output(keys, np.array([[1.0, 2.0]], dtype=np.float32),
                       np.array([[0.1, 0.2]], dtype=np.float32), 0.25, ['b', 'a'])
    assert [entry['name'] for entry in out['features'][0]] == ['b', 'a']
    assert [entry['value'] for entry in out['features'][0]] == [1.0, 2.0]
    assert [entry['shapValue'] for entry in out['features'][0]] == pytest.approx([0.1, 0.2])


def test_build_output_gives_every_row_the_same_base_value() -> None:
    keys = pl.DataFrame({'studyLocusId': ['sl1', 'sl2'], 'geneId': ['g1', 'g2'], 'score': [0.5, 0.6]})
    out = build_output(keys, np.zeros((2, 1), dtype=np.float32),
                       np.zeros((2, 1), dtype=np.float32), 0.25, ['a'])
    assert out['shapBaseValue'].unique().to_list() == [pytest.approx(0.25)]


@pytest.fixture
def workspace(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    matrix = pl.DataFrame(
        {
            'studyLocusId': [f'sl{i}' for i in range(n)],
            'geneId': [f'ENSG{i:011d}' for i in range(n)],
            'isProteinCoding': [1] * (n - 5) + [0] * 5,
            **{name: rng.random(n) for name in FEATURES},
        }
    )
    (tmp_path / 'fm').mkdir()
    matrix.write_parquet(tmp_path / 'fm' / 'part-0.parquet')

    credible = pl.DataFrame({
        'studyLocusId': [f'sl{i}' for i in range(n)],
        'studyType': ['gwas'] * (n - 10) + ['eqtl'] * 10,
    })
    (tmp_path / 'cs').mkdir()
    credible.write_parquet(tmp_path / 'cs' / 'part-0.parquet')

    x = rng.random((80, len(FEATURES))).astype(np.float32)
    y = (rng.random(80) > 0.5).astype(np.int32)
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 5, 'max_depth': 2})
    save_model(model, str(tmp_path / 'classifier.skops'))
    pl.DataFrame(x[:10], schema=list(FEATURES)).write_parquet(tmp_path / 'background.parquet')
    return tmp_path


def _source(workspace) -> dict[str, str]:
    return {
        'feature_matrix': str(workspace / 'fm'),
        'credible_set': str(workspace / 'cs'),
        'model': str(workspace / 'classifier.skops'),
        'background': str(workspace / 'background.parquet'),
    }


SETTINGS = {
    'features_list': list(FEATURES),
    'l2g_threshold': 0.0,
    'explain_predictions': True,
    'shap_background_size': 10,
    'shap_workers': 1,
}


def test_l2g_predict_excludes_non_gwas_and_non_protein_coding_rows(workspace) -> None:
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, dict(SETTINGS), None)
    out = pl.read_parquet(f'{destination}/*.parquet')
    # 60 rows, 10 non-gwas at the tail, 5 non-protein-coding at the tail; they overlap.
    assert out.height == 50
    assert 'sl59' not in out['studyLocusId'].to_list()


def test_l2g_predict_applies_the_threshold(workspace) -> None:
    # An empty prediction set is a real production outcome, so this pins whatever
    # `write_dataset` actually does with a zero-row frame. Determine that behaviour and
    # assert it explicitly -- either a readable zero-row dataset or no part files at all.
    # Do NOT weaken the test by choosing a threshold that keeps rows.
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, {**SETTINGS, 'l2g_threshold': 1.1}, None)
    parts = sorted(Path(destination).glob('*.parquet'))
    if parts:
        assert pl.read_parquet(parts).height == 0
    else:
        assert Path(destination).exists()


def test_l2g_predict_output_is_sorted_by_key(workspace) -> None:
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, dict(SETTINGS), None)
    out = pl.read_parquet(f'{destination}/*.parquet')
    assert out['studyLocusId'].to_list() == sorted(out['studyLocusId'].to_list())


def test_l2g_predict_writes_one_feature_entry_per_feature(workspace) -> None:
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, dict(SETTINGS), None)
    out = pl.read_parquet(f'{destination}/*.parquet')
    assert out['features'].list.len().unique().to_list() == [len(FEATURES)]


def test_l2g_predict_without_explanations_leaves_shap_null(workspace) -> None:
    destination = str(workspace / 'out')
    l2g_predict(_source(workspace), destination, {**SETTINGS, 'explain_predictions': False}, None)
    out = pl.read_parquet(f'{destination}/*.parquet')
    assert out['shapBaseValue'].is_null().all()
    # Null, not NaN: the flag says no explanation was computed, and every downstream
    # consumer reads those as different values.
    first = out['features'][0]
    assert all(entry['shapValue'] is None for entry in first)


def test_the_two_config_tasks_share_one_feature_list() -> None:
    """The alias must survive yamlfmt: a drifted copy would fit and score on different orders."""
    config = yaml.safe_load(Path(__file__).parents[1].joinpath('config.yaml').read_text())
    tasks = {task['name']: task for task in config['steps']['l2g']}
    train = tasks['transform l2g_train']['settings']['features_list']
    predict = tasks['transform l2g_predict']['settings']['features_list']
    assert train == predict
    assert len(train) == 31
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_predict.py -rxs`
Expected: FAIL, `ModuleNotFoundError: No module named 'pts.transformers.l2g_predict'`

- [ ] **Step 3: Write the implementation**

Create `pts/src/pts/transformers/l2g_predict.py`:

```python
"""Score every GWAS credible-set/gene pair and explain the survivors.

Replaces the predict mode of gentropy's `LocusToGeneStep`, which ran as 1000 Google Batch tasks.
Each of those started a Spark job on two vCPUs and re-read the ENTIRE feature matrix -- only the
credible-set side was partitioned -- to emit roughly 3,200 rows. The measured work is ~12 s of
scoring for all 61.2M rows plus ~2.8 process-hours of SHAP for the 3.2M that survive the
threshold, so it fits on one VM with room to spare.

Unlike gentropy this never drops the features and re-joins the matrix to get them back: they are
already on the frame that produced the score.
"""

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.l2g import explain as l2g_explain, model as l2g_model
from pts.transformers.l2g.features import FEATURES, impute_and_cast
from pts.transformers.utils.dataset import scan_dataset, write_dataset

OUTPUT_COLUMNS = ('studyLocusId', 'geneId', 'score', 'features', 'shapBaseValue')


def build_output(
    keys: pl.DataFrame,
    matrix: np.ndarray,
    shap_values: np.ndarray | None,
    base_value: float | None,
    features: Sequence[str],
) -> pl.DataFrame:
    """Assemble the release schema from the scored keys and their explanations.

    The `features` array carries one struct per feature, in fitted order. Types are pinned here
    rather than left to inference: `gentropy_l2g_evidence` reads this dataset with an imposed
    schema, so a drift would be coerced silently instead of raised.

    Args:
        keys: `studyLocusId`, `geneId` and `score`, one row per prediction.
        matrix: the feature values behind those rows, in fitted order.
        shap_values: SHAP values with the shape of `matrix`, or None when explanations are off,
            in which case every `shapValue` is null.
        base_value: the model's expected value, or None when explanations are off.
        features: feature names, in fitted order.

    Returns:
        A DataFrame of `OUTPUT_COLUMNS`.
    """
    values = pl.DataFrame(matrix, schema=[(name, pl.Float32) for name in features])
    shap_schema = [(f'shap_{name}', pl.Float32) for name in features]
    shaps = (
        pl.DataFrame(shap_values, schema=shap_schema)
        if shap_values is not None
        else pl.DataFrame(
            {name: [None] * values.height for name, _ in shap_schema}, schema=shap_schema
        )
    )
    wide = pl.concat([keys, values, shaps], how='horizontal')

    return wide.select(
        pl.col('studyLocusId').cast(pl.String),
        pl.col('geneId').cast(pl.String),
        pl.col('score').cast(pl.Float64),
        pl.concat_list(
            pl.struct(
                pl.lit(name, dtype=pl.String).alias('name'),
                pl.col(name).cast(pl.Float32).alias('value'),
                pl.col(f'shap_{name}').cast(pl.Float32).alias('shapValue'),
            )
            for name in features
        ).alias('features'),
        pl.lit(base_value, dtype=pl.Float32).alias('shapBaseValue'),
    )


def l2g_predict(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Score the feature matrix, explain the survivors and write the release dataset.

    Args:
        source: keys `feature_matrix`, `credible_set`, `model`, `background`.
        destination: the output dataset directory.
        settings: keys `features_list`, `l2g_threshold`, `explain_predictions`,
            `shap_background_size`, and optionally `shap_workers`.
        config: otter config; unused, accepted for interface compatibility.
    """
    features = list(settings.get('features_list') or FEATURES)
    threshold = float(settings['l2g_threshold'])

    gwas_loci = (
        scan_dataset(source['credible_set'])
        .filter(pl.col('studyType') == 'gwas')
        .select('studyLocusId')
        .unique()
    )

    logger.info('preparing the prediction matrix')
    prepared = impute_and_cast(
        scan_dataset(source['feature_matrix'])
        .filter(pl.col('isProteinCoding') == 1)
        .join(gwas_loci, on='studyLocusId', how='semi'),
        features,
    ).collect()

    model = l2g_model.load_model(source['model'])
    matrix = l2g_model.to_matrix(prepared, features)
    logger.info(f'scoring {matrix.shape[0]} rows')
    scores = model.predict_proba(matrix)[:, 1] if matrix.shape[0] else np.empty(0, dtype=np.float32)

    keep = scores >= threshold
    keys = prepared.select('studyLocusId', 'geneId').filter(pl.Series(keep)).with_columns(
        pl.Series('score', scores[keep])
    )
    matrix = matrix[keep]
    logger.info(f'{keys.height} rows at or above the {threshold} threshold')

    if settings.get('explain_predictions'):
        background = pl.read_parquet(source['background']).select(features).to_numpy().astype(np.float32)
        workers = settings.get('shap_workers') or os.cpu_count()
        logger.info(f'explaining {matrix.shape[0]} rows on {workers} workers')
        base_value, shap_values = l2g_explain.explain(
            model,
            matrix,
            background,
            max_samples=int(settings['shap_background_size']),
            workers=workers,
        )
    else:
        # Null, not NaN. The flag says no explanation was computed, which is a different
        # statement from "the explanation is not a number", and consumers read them differently.
        base_value, shap_values = None, None

    output = build_output(keys, matrix, shap_values, base_value, features).sort(
        'studyLocusId', 'geneId'
    )
    write_dataset(output, destination)
    logger.info('prediction complete')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_predict.py -rxs`
Expected: 8 passed

- [ ] **Step 5: Add the second task to the `l2g` step in `pts/config.yaml`**

Append inside the `l2g:` block, after the `transform l2g_train` task:

```yaml
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
        shap_background_size: 100
```

- [ ] **Step 6: Run the whole pts suite**

Run: `cd pts && uv run --frozen pytest -rxs -m "not slow and not pgserver"`
Expected: PASS

- [ ] **Step 7: Lint and typecheck**

Run: `cd pts && uv run --frozen ruff check src tests` then `cd pts && uv run --frozen ty check`
Expected: no findings

- [ ] **Step 8: Commit**

```bash
git add pts/src/pts/transformers/l2g_predict.py pts/tests/test_l2g_predict.py pts/config.yaml
git commit -m "pts: add the l2g_predict transformer and its config step"
```

---

### Task 8: Orchestration — retire the three gentropy steps

**Files:**
- Modify: `orchestration/src/orchestration/dags/config/gentropy.yaml`
- Modify: `orchestration/src/orchestration/dags/config/unified_pipeline.yaml`
- Modify: `orchestration/src/orchestration/dags/unified_pipeline.py`
- Delete: `orchestration/src/orchestration/assets/l2g_predict.sh`

**Interfaces:**
- Consumes: the `l2g` step name from `pts/config.yaml` (Tasks 6–7).
- Produces: a DAG where `gentropy_l2g_evidence` depends on `pts_l2g`.

- [ ] **Step 1: Remove the three step blocks from `gentropy.yaml`**

Delete the `l2g_train_test_split:`, `l2g_training:` and `l2g_prediction:` blocks in full (currently lines 209–290). `l2g_feature_matrix:` and `l2g_evidence:` stay. The `l2g_prediction` deletion takes the `cluster: false`, `google_batch_index_specs:` and `google_batch:` sub-blocks with it, including the `hfhub-key` secret mapping.

- [ ] **Step 2: Delete the Batch entrypoint**

```bash
git rm orchestration/src/orchestration/assets/l2g_predict.sh
```

- [ ] **Step 3: Rewire the dependency graph in `unified_pipeline.yaml`**

Replace the `gentropy_l2g_train_test_split`, `gentropy_l2g_training` and `gentropy_l2g_prediction` entries with a single PTS step, and repoint the evidence step:

```yaml
  pts_l2g:
    machine_type: n1-highmem-32
    depends_on:
      - gentropy_l2g_feature_matrix
      - pis_l2g
      - pis_l2g_predefined_test_for_split
  gentropy_l2g_evidence:
    depends_on:
      - pts_l2g
```

`gentropy_l2g_feature_matrix` keeps its own `depends_on` unchanged. Move `pts_l2g` into the PTS block of the file so the steps stay grouped by stage.

**`pts_vep_view` also depends on the step being deleted** (`unified_pipeline.yaml:531`). Repoint it, or the DAG cannot build:

```yaml
  pts_vep_view:
    depends_on:
      - pts_l2g
```

- [ ] **Step 4: Remove the stale wiring in `unified_pipeline.py`**

Delete these three entries from `gentropy_step_outputs` (around lines 371–380):

```python
            'gentropy_l2g_train_test_split': {
                'step.train_parquet_path': gsp('gentropy_l2g_train_test_split', 'step.train_parquet_path'),
                'step.test_parquet_path': gsp('gentropy_l2g_train_test_split', 'step.test_parquet_path'),
            },
            'gentropy_l2g_training': {
                'step.model_path': gsp('gentropy_l2g_training', 'step.model_path'),
            },
            'gentropy_l2g_prediction': {
                'step.predictions_path': gsp('gentropy_l2g_prediction', 'step.predictions_path'),
            },
```

And delete the `l2g_training_version` scratchpad entry (line 90):

```python
                'l2g_training_version': self.run.release_name,
```

- [ ] **Step 5: Verify no reference survives**

Run: `grep -rn --include='*.py' --include='*.yaml' --include='*.sh' "gentropy_l2g_training\|gentropy_l2g_prediction\|gentropy_l2g_train_test_split\|l2g_predict.sh\|l2g_training_version" orchestration/src`
Expected: no matches. Restrict to those extensions: stale `__pycache__/*.pyc` files in the tree still contain the old strings and are not evidence of a live reference.

- [ ] **Step 6: Run the orchestration suite**

Run: `cd orchestration && uv run --frozen pytest -rxs`
Expected: PASS

- [ ] **Step 7: Lint and typecheck**

Run: `cd orchestration && uv run --frozen ruff check src` then `cd orchestration && uv run --frozen ty check`
Expected: no findings

- [ ] **Step 8: Commit**

```bash
git add orchestration/src
git commit -m "orchestration: run L2G training and prediction as one pts step

Retires gentropy_l2g_train_test_split, gentropy_l2g_training and
gentropy_l2g_prediction, including the 1000-task Google Batch job and its
l2g_predict.sh entrypoint. gentropy_l2g_evidence now depends on pts_l2g."
```

---

### Task 9: The parity harness

**Files:**
- Create: `pts/tests/test_l2g_parity.py`

**Interfaces:**
- Consumes: `impute_and_cast`, `FEATURES`, `l2g_model.load_model`, `l2g_model.to_matrix`.
- Produces: nothing importable; this is the acceptance gate.

- [ ] **Step 1: Write the gated parity test**

Create `pts/tests/test_l2g_parity.py`:

```python
"""Fixed-model parity against a published release.

The sharpest available test of the migration: take the release's OWN classifier, run it through
this polars path over the release's own feature matrix, and require the scores back. Because the
model is held fixed, any difference is the migration's doing and not the retraining's.

Row-level SHAP parity is deliberately NOT asserted. Each of gentropy's 1000 Batch tasks drew its
own unseeded background, so `shapBaseValue` varies across the baseline's own partitions -- 0.0381
to 0.0668 in 26.09-1. Re-running gentropy would not reproduce it either.

Gated on an environment variable because it reads a multi-gigabyte GCS dataset:

    L2G_PARITY_RUN=do/platform-2609-1 uv run --frozen pytest tests/test_l2g_parity.py -rxs
"""

import os

import numpy as np
import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES, impute_and_cast
from pts.transformers.l2g.model import load_model, to_matrix

RUN = os.environ.get('L2G_PARITY_RUN')
BUCKET = os.environ.get('L2G_PARITY_BUCKET', 'gs://open-targets-pipeline-runs')

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUN, reason='set L2G_PARITY_RUN to a run prefix to enable'),
]


@pytest.fixture(scope='module')
def scored(tmp_path_factory) -> pl.DataFrame:
    """Score the release's feature matrix with the release's own model."""
    base = f'{BUCKET}/{RUN}'
    local_model = tmp_path_factory.mktemp('model') / 'classifier.skops'
    import gcsfs

    gcsfs.GCSFileSystem().get(
        f'{base}/etc/model/locus_to_gene_model/classifier.skops'.removeprefix('gs://'),
        str(local_model),
    )
    model = load_model(str(local_model))

    gwas = (
        pl.scan_parquet(f'{base}/output/credible_set/*.parquet')
        .filter(pl.col('studyType') == 'gwas')
        .select('studyLocusId')
        .unique()
    )
    prepared = impute_and_cast(
        pl.scan_parquet(f'{base}/intermediate/l2g_feature_matrix/*.parquet')
        .filter(pl.col('isProteinCoding') == 1)
        .join(gwas, on='studyLocusId', how='semi'),
        list(FEATURES),
    ).collect()

    matrix = to_matrix(prepared, list(FEATURES))
    return prepared.select('studyLocusId', 'geneId').with_columns(
        pl.Series('score', model.predict_proba(matrix)[:, 1])
    )


@pytest.fixture(scope='module')
def baseline() -> pl.DataFrame:
    return pl.read_parquet(
        f'{BUCKET}/{RUN}/output/l2g_prediction/*.parquet', columns=['studyLocusId', 'geneId', 'score']
    )


def test_scores_match_the_published_column(scored: pl.DataFrame, baseline: pl.DataFrame) -> None:
    joined = baseline.join(scored, on=['studyLocusId', 'geneId'], how='inner', suffix='_new')
    assert joined.height == baseline.height, 'a published prediction has no row in the new path'
    delta = (joined['score'] - joined['score_new']).abs()
    assert delta.max() < 1e-6, f'max |delta| = {delta.max()}'


def test_the_row_set_at_threshold_matches(scored: pl.DataFrame, baseline: pl.DataFrame) -> None:
    mine = scored.filter(pl.col('score') >= 0.05).select('studyLocusId', 'geneId')
    theirs = baseline.select('studyLocusId', 'geneId')
    assert mine.height == theirs.height
    assert mine.join(theirs, on=['studyLocusId', 'geneId'], how='anti').height == 0


def test_the_baselines_own_base_value_is_not_constant() -> None:
    """Documents why row-level SHAP parity is not asserted anywhere in this file."""
    values = pl.read_parquet(
        f'{BUCKET}/{RUN}/output/l2g_prediction/*.parquet', columns=['shapBaseValue']
    )['shapBaseValue'].unique()
    assert values.len() > 1, 'baseline base value is constant; revisit the SHAP parity decision'
    assert float(values.max()) / float(values.min()) > 1.5
```

- [ ] **Step 2: Confirm it skips by default**

Run: `cd pts && uv run --frozen pytest tests/test_l2g_parity.py -rxs`
Expected: 3 skipped, reason `set L2G_PARITY_RUN to a run prefix to enable`

- [ ] **Step 3: Run it for real**

Run: `cd pts && L2G_PARITY_RUN=do/platform-2609-1 uv run --frozen pytest tests/test_l2g_parity.py -rxs`
Expected: 3 passed. `test_scores_match_the_published_column` is the acceptance gate; it passed at max |Δ| = 0 on a two-partition sample during design.

- [ ] **Step 4: Record the result**

Append the measured `max |delta|`, the row counts, and the wall clock to the spec's Verification section, in both copies (repo and vault).

- [ ] **Step 5: Commit**

```bash
git add pts/tests/test_l2g_parity.py docs/superpowers/specs/2026-09-08-l2g-pts-migration-design.md
git commit -m "pts: add the gated L2G fixed-model parity harness"
```

---

## Self-Review

**Spec coverage.** Every section maps to a task: step and task shape → Tasks 6–7 (config) and 8 (routing); module layout → Tasks 1–5; training → Tasks 2, 3, 4, 6; prediction → Tasks 1, 7; imputation and casting → Task 1; SHAP → Task 5; sizing → Task 8 (`machine_type`); artifacts → Tasks 6–7; orchestration changes → Task 8; divergences → Tasks 1 (cast, null), 2 (`unique(maintain_order=True)`), 7 (sort); verification → Tasks 1–7 (unit) and 9 (parity). Croissant needs no change and has no task, as the spec states.

**Placeholders.** None. Every code step carries the code; no "similar to Task N"; no "add error handling".

**Type consistency.** `LABEL_COLUMN` is defined once in `gold_standard.py` and imported by `split.py` and `model.py`. `FEATURES`, `FIXED_COLUMNS` and `impute_and_cast` are defined in `features.py` and used unchanged in Tasks 2 and 7. `derive_splits` returns eager `pl.DataFrame`s in both its definition (Task 3) and its call site (Task 6). `explain` returns `(float, np.ndarray)` in Task 5 and is unpacked that way in Task 7. `build_output` takes `base_value: float | None` because Task 7's non-explaining branch passes `None`.

**Two gaps found and fixed inline while reviewing.** `build_output`'s `base_value` was typed `float` but Task 7 passes `None` when `explain_predictions` is off — widened to `float | None`. And `missingness` divided by `frame.height` without guarding an empty frame — the guard is now in Task 4's implementation and its behaviour is pinned by the `total == 0` branch.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-08-l2g-pts-migration.md`. Two execution options:

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.
