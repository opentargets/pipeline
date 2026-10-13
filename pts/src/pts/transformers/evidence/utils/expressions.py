"""Per-datasource score and direction-of-effect expressions for `evidence_postprocess`.

Hand-translated, native-polars replacement for the spark SQL strings `pts/config.yaml` carries in
`score_expression` / `direction_on_trait_expression` / `direction_on_target_expression` for each
`evidence_postprocess_*` step. Every expression below was diffed against real spark evaluating the
literal SQL from `config.yaml` -- see `task-registry-report.md` for the full parity run.

Keyed by `settings.datasource_id`, not the step name -- e.g. the `evidence_postprocess_clinvar`
step carries `datasource_id: eva`, so its entry lives under `'eva'`.

Several steps carry byte-for-byte or functionally identical expressions; those are written once as
a module-level function/constant and referenced from every entry that needs them, rather than
copy-pasted:

* `eva` and `eva_somatic` (`evidence_postprocess_clinvar` / `_clinvar_somatic`) share all three
  expressions. The two `score_expression` strings in config.yaml are NOT byte-identical -- one
  entry (`'low penetrance'`) sits at a different position in each map literal -- but `element_at`
  is a keyed lookup, so the two maps are the same function of `confidence`/`clinicalSignificances`
  despite the reordering; see the module's dedicated score comment.
* `element_at(map(...), confidence)` is the shared shape behind `uniprot_variants`,
  `uniprot_literature` and others -- only the
  map differs, so `confidence_score` takes the map as its argument.
* `CASE WHEN diseaseId IS NOT NULL THEN '<value>' END` is the shared shape behind several direction
  expressions across `clinical_precedence`, `intogen` and
  `cancer_gene_census` -- `direction_when_disease_present` takes the literal as its argument.
* `CASE WHEN <col> == a THEN 'GoF' WHEN <col> == b THEN 'LoF' ELSE NULL END` is a 1:1 value lookup,
  behind `cancer_gene_census`'s `direction_on_target` -- expressed
  as `direction_lookup`, a `replace_strict` on the named column.
* The log10-rescaled-score shape (null/NaN stays null, a non-positive value saturates to 1.0,
  everything else linearly interpolated in log10 space and clamped) is shared by `intogen`,
  `gene_burden`, `crispr_screen` and the inner CASE of `expression_atlas` -- only the
  reference points and output column differ, so `log10_rescaled_score` takes them as arguments.

Provenance: `pts/config.yaml` no longer carries the original `score_expression` /
`direction_on_trait_expression` / `direction_on_target_expression` SQL strings -- git commit
`834f1db7` (just before they were stripped) has the last copy, one per `evidence_postprocess_*`
step, if you need to see the source SQL a given entry below was translated from. The translation
was verified 41/41 against real spark: on synthetic data covering the null/edge-case branches of
every expression, and separately on 20/20 real staged datasources end to end -- see
`task-registry-report.md` and `registry-parity-output.txt` for the runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

#: `element_at(map(...), confidence)` maps, one per datasource that uses the shared shape.
_UNIPROT_CONFIDENCE_SCORES = {'high': 1.0, 'medium': 0.5}

#: `eva` (clinvar) / `eva_somatic` (clinvar_somatic) share one score expression -- see module
#: docstring for why the two config.yaml strings differ in key order but not in meaning.
_EVA_SIGNIFICANCE_SCORES = {
    'association not found': 0.0,
    'benign': 0.0,
    'not provided': 0.0,
    'likely benign': 0.0,
    'evidence only': 0.0,
    'likely risk allele': 0.3,
    'low penetrance': 0.3,
    'conflicting interpretations of pathogenicity': 0.3,
    'conflicting data from submitters': 0.3,
    'other': 0.3,
    'uncertain significance': 0.3,
    'uncertain risk allele': 0.3,
    'established risk allele': 0.5,
    'risk factor': 0.5,
    'affects': 0.5,
    'likely pathogenic': 0.7,
    'confers sensitivity': 0.9,
    'association': 0.9,
    'drug response': 0.9,
    'protective': 0.9,
    'pathogenic': 0.9,
}
_EVA_CONFIDENCE_SCORES = {
    'practice guideline': 0.1,
    'reviewed by expert panel': 0.07,
    'criteria provided, multiple submitters, no conflicts': 0.05,
    'criteria provided, conflicting interpretations': 0.02,
    'criteria provided, single submitter': 0.02,
    'no assertion for the individual variant': 0.0,
    'no assertion criteria provided': 0.0,
    'no assertion provided': 0.0,
}
#: `variantFunctionalConsequenceId` values that map to `'LoF'` in the eva direction_on_target CASE.
_EVA_LOF_CONSEQUENCES = [
    'SO_0001589',
    'SO_0001587',
    'SO_0001574',
    'SO_0001575',
    'SO_0002012',
    'SO_0001578',
    'SO_0001893',
]

#: clinical_precedence score: `clinicalStage` map, `trialStopReasonCategories` element map.
_CLINICAL_STAGE_SCORES = {
    'APPROVAL': 1.0,
    'PHASE_4': 1.0,
    'WITHDRAWAL': 1.0,
    'PREAPPROVAL': 0.8,
    'PHASE_3': 0.7,
    'PHASE_2_3': 0.5,
    'PHASE_2': 0.2,
    'PHASE_1_2': 0.15,
    'PHASE_1': 0.1,
    'EARLY_PHASE_1': 0.05,
    'IND': 0.05,
    'PRECLINICAL': 0.01,
    'UNKNOWN': 0.01,
}
_TRIAL_STOP_REASON_SCORES = {
    'Another_Study': 1.0,
    'Business_Administrative': 1.0,
    'Covid19': 1.0,
    'Ethical_Reason': 1.0,
    'Insufficient_Data': 1.0,
    'Insufficient_Enrollment': 1.0,
    'Interim_Analysis': 1.0,
    'Invalid_Reason': 1.0,
    'Logistics_Resources': 1.0,
    'Endpoint_Met': 1.0,
    'Negative': 0.5,
    'No_Context': 1.0,
    'Regulatory': 1.0,
    'Safety_Sideeffects': 0.5,
    'Study_Design': 1.0,
    'Study_Staff_Moved': 1.0,
    'Success': 1.0,
    'Uncategorised': 1.0,
}
#: clinical_precedence direction_on_target: `actionType` values that map to `'LoF'` / `'GoF'`.
_LOF_ACTION_TYPES = [
    'RNAI INHIBITOR',
    'NEGATIVE MODULATOR',
    'NEGATIVE ALLOSTERIC MODULATOR',
    'ANTAGONIST',
    'ANTISENSE INHIBITOR',
    'BLOCKER',
    'INHIBITOR',
    'DEGRADER',
    'INVERSE AGONIST',
    'ALLOSTERIC ANTAGONIST',
    'DISRUPTING AGENT',
    'GENE EDITING NEGATIVE MODULATOR',
]
_GOF_ACTION_TYPES = [
    'PARTIAL AGONIST',
    'ACTIVATOR',
    'POSITIVE ALLOSTERIC MODULATOR',
    'POSITIVE MODULATOR',
    'AGONIST',
    'SEQUESTERING AGENT',
    'STABILISER',
]


def confidence_score(mapping: dict[str, float]) -> pl.Expr:
    """`element_at(map(...), confidence)` -- the shared shape across five datasources.

    `replace_strict` with `default=None` matches spark's `element_at`: a `confidence` value absent
    from `mapping` (or null) becomes null rather than raising.
    """
    return pl.col('confidence').replace_strict(mapping, default=None, return_dtype=pl.Float64)


def direction_when_disease_present(value: str) -> pl.Expr:
    """`CASE WHEN diseaseId IS NOT NULL THEN '<value>' END` -- shared by several direction fields."""
    return pl.when(pl.col('diseaseId').is_not_null()).then(pl.lit(value)).otherwise(None)


def direction_lookup(column: str, mapping: dict[str, str]) -> pl.Expr:
    """`CASE WHEN <column> == a THEN x WHEN <column> == b THEN y ELSE NULL END` as a 1:1 lookup."""
    return pl.col(column).replace_strict(mapping, default=None)


def _true_divide(series: pl.Series, divisor: float) -> pl.Series:
    values = series.cast(pl.Float64).to_numpy() / divisor
    out = pl.Series(series.name, values, dtype=pl.Float64, nan_to_null=False)
    nulls = np.flatnonzero(series.is_null().to_numpy())
    return out.scatter(nulls, None) if len(nulls) else out


def divide(expr: pl.Expr, divisor: float) -> pl.Expr:
    """Divide by a constant with IEEE division, keeping nulls and NaN.

    Polars 2 does not divide exactly: `x / 10` differs from spark in the last bit for about a third
    of values.
    """
    return expr.map_batches(lambda s: _true_divide(s, divisor), return_dtype=pl.Float64, is_elementwise=True)


def log10_rescaled_score(value: pl.Expr, out_min: float, weak_ref: float, strong_ref: float) -> pl.Expr:
    """Rescale a p-value-like `value` onto `[out_min, 1.0]` in log10 space, clamped.

    Common shape behind the intogen/gene_burden/crispr_screen score expressions, and
    expression_atlas's inner CASE:

    * null or NaN input stays null.
    * a non-positive value saturates to the strongest score, 1.0.
    * otherwise, linearly interpolate in log10 space between `weak_ref` (-> `out_min`) and
      `strong_ref` (-> 1.0), then clamp back to `[out_min, 1.0]` for values past either reference
      point -- spark's `LEAST(1.0, GREATEST(out_min, ...))`.

    Args:
        value: the column to rescale.
        out_min: the output score for `value == weak_ref` (and the clamp floor).
        weak_ref: the value that scores `out_min`.
        strong_ref: the value that scores `1.0`.
    """
    log_weak = math.log10(weak_ref)
    log_strong = math.log10(strong_ref)
    rescaled = divide((1.0 - out_min) * (value.log10() - log_weak), log_strong - log_weak) + out_min
    return (
        pl.when(value.is_null() | value.is_nan())
        .then(None)
        .when(value <= 0)
        .then(1.0)
        .otherwise(rescaled.clip(out_min, 1.0))
    )


def linear_rescale(
    value: pl.Expr, in_range_min: float, in_range_max: float, out_range_min: float, out_range_max: float
) -> pl.Expr:
    """Rescale `value` linearly between two ranges, clamped to the output range.

    ```
    CASE
      WHEN value IS NULL OR isnan(value) THEN NULL
      ELSE LEAST(out_max, GREATEST(out_min,
        (out_max - out_min) * (CAST(value AS DOUBLE) - in_min) / (in_max - in_min) + out_min
      ))
    END
    ```
    """
    value = value.cast(pl.Float64)
    rescaled = (
        divide((out_range_max - out_range_min) * (value - in_range_min), in_range_max - in_range_min) + out_range_min
    )
    return pl.when(value.is_null() | value.is_nan()).then(None).otherwise(rescaled.clip(out_range_min, out_range_max))


def _eva_score() -> pl.Expr:
    """`eva` / `eva_somatic` score: best clinical-significance score plus a confidence bonus.

    ```
    coalesce(array_max(transform(clinicalSignificances, x -> element_at(map(...), x))), 0.0)
    + coalesce(element_at(map(...), confidence), 0.0)
    ```

    `list.max()` on an empty or all-unmatched-elements list, and on a null list, is null in polars
    just as spark's `array_max` is -- `fill_null(0.0)` reproduces the `coalesce(..., 0.0)` wrapping
    either way.
    """
    per_significance = (
        pl.col('clinicalSignificances')
        .list.eval(pl.element().replace_strict(_EVA_SIGNIFICANCE_SCORES, default=None, return_dtype=pl.Float64))
        .list.max()
        .fill_null(0.0)
    )
    confidence_bonus = confidence_score(_EVA_CONFIDENCE_SCORES).fill_null(0.0)
    return per_significance + confidence_bonus


def _eva_direction_on_trait() -> pl.Expr:
    """`eva` / `eva_somatic` direction_on_trait: pathogenic/protective terms in `clinicalSignificances`.

    ```
    CASE WHEN CONCAT_WS('', clinicalSignificances) RLIKE '(?i)pathogenic'
     AND CONCAT_WS('', clinicalSignificances) RLIKE '(?i)protect' THEN null
     WHEN ... RLIKE '(?i)pathogenic' THEN 'risk'
     WHEN ... RLIKE '(?i)protect' THEN 'protect'
     ELSE null END
    ```

    `CONCAT_WS('', arr)` skips null elements and never itself returns null; a null list here joins
    to `''`, which matches neither pattern -- the same final `null` result the CASE's `ELSE`
    branch would give, so a null `clinicalSignificances` needs no special casing.
    """
    joined = pl.col('clinicalSignificances').fill_null([]).list.eval(pl.element().fill_null('')).list.join('')
    has_pathogenic = joined.str.contains('(?i)pathogenic')
    has_protect = joined.str.contains('(?i)protect')
    return (
        pl.when(has_pathogenic & has_protect)
        .then(None)
        .when(has_pathogenic)
        .then(pl.lit('risk'))
        .when(has_protect)
        .then(pl.lit('protect'))
        .otherwise(None)
    )


def _eva_direction_on_target() -> pl.Expr:
    """`eva` / `eva_somatic` direction_on_target: LoF-tagged `variantFunctionalConsequenceId`."""
    return (
        pl.when(pl.col('variantFunctionalConsequenceId').is_in(_EVA_LOF_CONSEQUENCES))
        .then(pl.lit('LoF'))
        .otherwise(None)
    )


def _expression_atlas_score() -> pl.Expr:
    """expression_atlas score: log10-rescaled p-value scaled by fold-change magnitude and rank.

    ```
    array_min(array(1.0,
      (CASE ... log10 rescale of resourceScore ... END)
      * (abs(log2FoldChangeValue) / 10) * (log2FoldChangePercentileRank / 100)
    ))
    ```
    """
    base = log10_rescaled_score(pl.col('resourceScore'), 0.0, 1.0, 1e-10)
    magnitude = (
        base
        * divide(pl.col('log2FoldChangeValue').abs(), 10)
        * divide(pl.col('log2FoldChangePercentileRank'), 100)
    )
    # A null or NaN resourceScore makes `base` (and so `magnitude`) null -- min_horizontal, like
    # spark's array_min, skips a null element rather than propagating it, so the row scores 1.0
    # instead of null. That is faithful to spark's `array_min` null-skipping and deliberate, not an
    # oversight; `europepmc`'s score below is shaped the same way and null/NaN-scores 1.0 for the
    # same reason.
    return pl.min_horizontal(pl.lit(1.0), magnitude)


def _clinical_precedence_score() -> pl.Expr:
    """clinical_precedence score: clinical-stage score times a trial-stop-reason multiplier.

    ```
    element_at(map(<clinicalStage scores>), clinicalStage)
    * CASE WHEN trialStopReasonCategories IS NULL THEN double(1.0)
           ELSE array_min(transform(trialStopReasonCategories, x -> element_at(map(...), x))) END
    ```

    `list.min()` skips unmatched (null) elements and, like spark's `array_min`, is null on an empty
    list -- an empty (non-null) `trialStopReasonCategories` therefore multiplies the stage score by
    null, same as spark.
    """
    stage_score = pl.col('clinicalStage').replace_strict(_CLINICAL_STAGE_SCORES, default=None, return_dtype=pl.Float64)
    stop_multiplier = (
        pl.when(pl.col('trialStopReasonCategories').is_null())
        .then(pl.lit(1.0))
        .otherwise(
            pl.col('trialStopReasonCategories')
            .list.eval(pl.element().replace_strict(_TRIAL_STOP_REASON_SCORES, default=None, return_dtype=pl.Float64))
            .list.min()
        )
    )
    return stage_score * stop_multiplier


def _clinical_precedence_direction_on_target() -> pl.Expr:
    """clinical_precedence direction_on_target: LoF/GoF-tagged `actionType`."""
    action_type = pl.col('actionType')
    return (
        pl.when(action_type.is_in(_LOF_ACTION_TYPES))
        .then(pl.lit('LoF'))
        .when(action_type.is_in(_GOF_ACTION_TYPES))
        .then(pl.lit('GoF'))
        .otherwise(None)
    )


def _intogen_direction_on_target() -> pl.Expr:
    """Intogen direction_on_target: LoF/GoF consequence present among `mutatedSamples`.

    ```
    CASE WHEN mutatedSamples IS NOT NULL AND
      exists(transform(mutatedSamples, x -> x.functionalConsequenceId), y -> y IN ('SO_0002054'))
      THEN 'LoF'
      WHEN ... IN ('SO_0002053') THEN 'GoF'
    END
    ```
    """
    consequences = pl.col('mutatedSamples').list.eval(pl.element().struct.field('functionalConsequenceId'))
    # `mutated_present` is unreachable in polars: spark's `exists` is three-valued and would return
    # NULL (not false) on a NULL `mutatedSamples`, which is why the source SQL guards it explicitly.
    # polars' `list.contains` instead returns false for every list shape here -- [x], [null], [],
    # and null itself -- so the guard never actually changes the result; it is kept only because it
    # is faithful to the SQL it was translated from.
    mutated_present = pl.col('mutatedSamples').is_not_null()
    return (
        pl.when(mutated_present & consequences.list.contains('SO_0002054'))
        .then(pl.lit('LoF'))
        .when(mutated_present & consequences.list.contains('SO_0002053'))
        .then(pl.lit('GoF'))
        .otherwise(None)
    )


@dataclass(frozen=True)
class DatasourceExpressions:
    """The score and direction-of-effect expressions for one `datasourceId`."""

    score: pl.Expr
    direction_on_trait: pl.Expr | None = None
    direction_on_target: pl.Expr | None = None


EXPRESSIONS: dict[str, DatasourceExpressions] = {
    'expression_atlas': DatasourceExpressions(score=_expression_atlas_score()),
    'eva': DatasourceExpressions(
        score=_eva_score(),
        direction_on_trait=_eva_direction_on_trait(),
        direction_on_target=_eva_direction_on_target(),
    ),
    'eva_somatic': DatasourceExpressions(
        score=_eva_score(),
        direction_on_trait=_eva_direction_on_trait(),
        direction_on_target=_eva_direction_on_target(),
    ),
    'uniprot_variants': DatasourceExpressions(score=confidence_score(_UNIPROT_CONFIDENCE_SCORES)),
    'uniprot_literature': DatasourceExpressions(score=confidence_score(_UNIPROT_CONFIDENCE_SCORES)),
    'cancer_biomarkers': DatasourceExpressions(score=pl.lit(1.0)),
    'clinical_precedence': DatasourceExpressions(
        score=_clinical_precedence_score(),
        direction_on_trait=direction_when_disease_present('protect'),
        direction_on_target=_clinical_precedence_direction_on_target(),
    ),
    'reactome': DatasourceExpressions(score=pl.lit(1.0)),
    'intogen': DatasourceExpressions(
        score=log10_rescaled_score(pl.col('resourceScore'), 0.25, 0.1, 1e-10),
        direction_on_trait=direction_when_disease_present('risk'),
        direction_on_target=_intogen_direction_on_target(),
    ),
    'crispr_screen': DatasourceExpressions(score=log10_rescaled_score(pl.col('resourceScore'), 0.0, 0.5, 0.005)),
    # Same null/NaN-scores-1.0 shape as `_expression_atlas_score` above, for the same reason: a null
    # or NaN resourceScore makes the first min_horizontal argument null, which min_horizontal skips.
    'europepmc': DatasourceExpressions(score=pl.min_horizontal(divide(pl.col('resourceScore'), 100.0), pl.lit(1.0))),
    'crispr': DatasourceExpressions(score=linear_rescale(pl.col('resourceScore'), 41.5, 100.0, 0.415, 1.0)),
    'cancer_gene_census': DatasourceExpressions(
        score=pl.col('resourceScore'),
        direction_on_trait=direction_when_disease_present('risk'),
        direction_on_target=direction_lookup('TSorOncogene', {'oncogene': 'GoF', 'tsg': 'LoF'}),
    ),
    'ot_crispr': DatasourceExpressions(score=pl.lit(1.0)),
    'ot_crispr_validation': DatasourceExpressions(score=pl.col('resourceScore')),
}
