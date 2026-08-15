"""Tests for the evidence_postprocess score/direction expression registry.

No spark dependency here -- the registry's parity with real spark is verified separately (see
`.superpowers/sdd/plan/registry-parity-output.txt` and `task-registry-report.md`), and the
handful of expected values hard-coded below were measured against that spark run, not guessed.

`pts/config.yaml` no longer carries `score_expression` / `direction_on_trait_expression` /
`direction_on_target_expression` -- those were stripped once the registry below became the sole
source of truth (see `evidence_expressions.py`'s module docstring for where the original SQL a
given entry was translated from still lives in git history). So config.yaml can no longer serve as
this test's oracle for *which* fields a datasource has; the checks below are what config.yaml can
still ground: that every step's `datasource_id` maps to a registry entry and back, and that every
entry is actually usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
import yaml

from pts.transformers.utils.evidence_expressions import EXPRESSIONS

nan = float('nan')

CONFIG = Path(__file__).parents[1] / 'config.yaml'


def _config_datasource_ids() -> set[str]:
    """`datasource_id` for every `evidence_postprocess_*` step in config.yaml."""
    steps = yaml.safe_load(CONFIG.read_text())['steps']
    return {
        tasks[0]['settings']['datasource_id']
        for step_name, tasks in steps.items()
        if step_name.startswith('evidence_postprocess')
    }


CONFIG_DATASOURCE_IDS = _config_datasource_ids()


def test_every_config_datasource_has_a_registry_entry() -> None:
    missing = CONFIG_DATASOURCE_IDS - set(EXPRESSIONS)
    assert not missing


def test_registry_has_no_unexpected_datasources() -> None:
    extra = set(EXPRESSIONS) - CONFIG_DATASOURCE_IDS
    assert not extra


@pytest.mark.parametrize('datasource_id', sorted(EXPRESSIONS))
def test_every_registry_expression_is_a_polars_expr(datasource_id: str) -> None:
    """Every entry's `score` is a usable, non-None polars expression.

    The direction fields are each either absent or a polars expression -- never a leftover string,
    a spark Column, or some other stand-in.
    """
    entry = EXPRESSIONS[datasource_id]
    assert entry.score is not None
    assert isinstance(entry.score, pl.Expr)
    for field in ('direction_on_trait', 'direction_on_target'):
        value = getattr(entry, field)
        assert value is None or isinstance(value, pl.Expr)


# ------------------------------------------------------------------------------------------------
# Hand-picked edge cases. Expected values measured against real spark evaluating the literal SQL
# from config.yaml -- see registry-parity-output.txt for the full run these are drawn from.
# ------------------------------------------------------------------------------------------------


def test_eva_score_sums_significance_and_confidence() -> None:
    df = pl.DataFrame({
        'clinicalSignificances': [
            ['pathogenic'],
            [None],
            ['unknown_value'],
            None,
            ['likely pathogenic', 'pathogenic'],
        ],
        'confidence': [
            'practice guideline',
            'criteria provided, single submitter',
            'unknown_conf',
            None,
            'criteria provided, single submitter',
        ],
    })
    # measured against spark: 0.9+0.1, 0.0+0.02, 0.0+0.0, 0.0+0.0, max(0.7,0.9)+0.02
    # the last row is deliberately two matching, non-equal-scoring elements: array_max picks 0.9 and
    # scores 0.92 -- a sum (0.7+0.9=1.6, +0.02=1.62) would go undetected by every other row here,
    # which each carries only zero or one nonzero-scoring element.
    got = df.with_columns(EXPRESSIONS['eva'].score.alias('score'))['score'].to_list()
    assert got == pytest.approx([1.0, 0.02, 0.0, 0.0, 0.92])


def test_eva_direction_on_trait_null_when_both_pathogenic_and_protective() -> None:
    direction_on_trait = EXPRESSIONS['eva'].direction_on_trait
    assert direction_on_trait is not None
    df = pl.DataFrame({
        'clinicalSignificances': [['pathogenic'], ['protective'], ['pathogenic', 'protective'], [None], ['Pathogenic']]
    })
    got = df.with_columns(direction_on_trait.alias('d'))['d'].to_list()
    # the trailing capitalised 'Pathogenic' pins the '(?i)' case-insensitivity flag: every other row
    # here is already lowercase, so a match on 'pathogenic' would pass with or without that flag.
    assert got == ['risk', 'protect', None, None, 'risk']


def test_log10_rescaled_score_saturates_at_zero_and_null_propagates() -> None:
    # crispr_screen: out_min=0.0, weak_ref=0.5, strong_ref=0.005 -- measured against spark.
    df = pl.DataFrame({'resourceScore': [0.5, 0.005, 0.0, -1.0, None, float('nan')]})
    got = df.with_columns(EXPRESSIONS['crispr_screen'].score.alias('s'))['s'].to_list()
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(1.0)
    assert got[2] == pytest.approx(1.0)  # non-positive input saturates to 1.0
    assert got[3] == pytest.approx(1.0)
    assert got[4] is None
    # NaN pins the '| value.is_nan()' guard separately from is_null(): without it a NaN resourceScore
    # scores `nan` instead of None, and 0.005 lands so close to 1.0 (0.9999999999999998) that
    # pytest.approx would let a broken NaN guard hide behind that row instead.
    assert got[5] is None


def test_expression_atlas_and_europepmc_score_one_for_null_or_nan_resource_score() -> None:
    # Both scores route null/NaN resourceScore through a min_horizontal(..., 1.0) that, like spark's
    # array_min, ignores a null/NaN element rather than propagating it -- so the score saturates to
    # 1.0 instead of going null. Deliberate (see the production comment); pinned here so a refactor
    # that drops the min_horizontal wrapping is caught rather than silently reintroducing nulls.
    df = pl.DataFrame({
        'resourceScore': [None, float('nan')],
        'log2FoldChangeValue': [2.0, 2.0],
        'log2FoldChangePercentileRank': [50.0, 50.0],
    })
    got = df.with_columns(EXPRESSIONS['expression_atlas'].score.alias('s'))['s'].to_list()
    assert got == pytest.approx([1.0, 1.0])

    df2 = pl.DataFrame({'resourceScore': [None, float('nan')]})
    got2 = df2.with_columns(EXPRESSIONS['europepmc'].score.alias('s'))['s'].to_list()
    assert got2 == pytest.approx([1.0, 1.0])


def test_linear_rescale_clamps_project_score() -> None:
    # crispr: linear_rescale(resourceScore, 41.5, 100, 0.415, 1.0) -- measured against spark.
    df = pl.DataFrame({'resourceScore': [0.0, 41.5, 100.0, 150.0]})
    got = df.with_columns(EXPRESSIONS['crispr'].score.alias('s'))['s'].to_list()
    assert got == pytest.approx([0.415, 0.415, 1.0, 1.0])


def test_clinical_precedence_score_multiplies_stage_by_stop_reason_minimum() -> None:
    df = pl.DataFrame({
        'clinicalStage': ['PHASE_2_3', 'PRECLINICAL', 'PHASE_1', 'PHASE_2_3'],
        'trialStopReasonCategories': [['Success', 'Negative'], [], ['unknown_reason'], None],
    })
    # measured against spark: 0.5*0.5, 0.01*null, 0.1*null, 0.5*1.0
    got = df.with_columns(EXPRESSIONS['clinical_precedence'].score.alias('s'))['s'].to_list()
    assert got[0] == pytest.approx(0.25)
    assert got[1] is None
    assert got[2] is None
    # a NULL trialStopReasonCategories (as opposed to an empty, non-null list) takes the
    # `IS NULL -> 1.0` branch and scores the stage score alone -- deleting that branch collapses
    # this case into the empty-list case above and it would score None instead of 0.5.
    assert got[3] == pytest.approx(0.5)


def test_intogen_direction_on_target_needs_a_matching_consequence() -> None:
    direction_on_target = EXPRESSIONS['intogen'].direction_on_target
    assert direction_on_target is not None
    df = pl.DataFrame({
        'mutatedSamples': [
            [{'functionalConsequenceId': 'SO_0002054'}],
            [{'functionalConsequenceId': 'SO_0002053'}],
            [{'functionalConsequenceId': None}],
            None,
            [],
        ],
    })
    got = df.with_columns(direction_on_target.alias('d'))['d'].to_list()
    assert got == ['LoF', 'GoF', None, None, None]


def test_cancer_gene_census_direction_on_target_lookup() -> None:
    direction_on_target = EXPRESSIONS['cancer_gene_census'].direction_on_target
    assert direction_on_target is not None
    df = pl.DataFrame({'TSorOncogene': ['oncogene', 'tsg', 'bivalent', None]})
    got = df.with_columns(direction_on_target.alias('d'))['d'].to_list()
    assert got == ['GoF', 'LoF', None, None]


def test_direction_when_disease_present_is_null_safe() -> None:
    direction_on_trait = EXPRESSIONS['impc'].direction_on_trait
    assert direction_on_trait is not None
    df = pl.DataFrame({'diseaseId': ['MONDO:1', None]})
    got = df.with_columns(direction_on_trait.alias('d'))['d'].to_list()
    assert got == ['risk', None]


# ------------------------------------------------------------------------------------------------
# Exhaustive one-case-per-field table -- all 41 registry fields (23 `score`, 9
# `direction_on_trait`, 9 `direction_on_target`). Each `expected` was derived by running the
# ORIGINAL SQL (git `834f1db7` `pts/config.yaml`, the last commit before the registry replaced it)
# against real spark, not by hand-computing it or reading it off this module -- that would just
# make the test a mirror of the code it exists to pin. `score` fields were run through
# `CAST(... AS DOUBLE)` and, for `crispr`'s `linear_rescale` UDF specifically, with the UDF
# registered exactly as production did (no explicit `returnType`, which pyspark 3.5.7 defaults to
# StringType rather than inferring `DoubleType` from the function's `-> float` hint) -- both match
# `calculate_evidence_score`'s `.withColumn('score', f.expr(score_expression).cast(DoubleType()))`
# (`evidence.py:240`, before the pyspark module was deleted). `direction_on_trait` /
# `direction_on_target` are asserted uncast, matching `assign_direction_on_trait` /
# `assign_direction_on_target`, which apply `f.expr(...)` directly with no cast. Every value here
# was cross-checked against the current registry too (0 mismatches) -- spark and the registry agree
# on all 41.
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExpressionCase:
    """One registry field, a small input frame, and its spark-derived expected output."""

    datasource_id: str
    field: str
    data: dict[str, list]
    expected: list


CASES = [
    _ExpressionCase('gwas_credible_sets', 'score', {'resourceScore': [0.5, 0.0, None]}, [0.5, 0.0, None]),
    _ExpressionCase(
        'expression_atlas',
        'score',
        {
            'resourceScore': [0.001, None, nan, 0.0],
            'log2FoldChangeValue': [-4.0, 2.0, 2.0, 1.0],
            'log2FoldChangePercentileRank': [80.0, 50.0, 50.0, 1.0],
        },
        [0.096, 1.0, 1.0, 0.001],
    ),
    _ExpressionCase(
        'eva',
        'score',
        {
            'clinicalSignificances': [
                ['pathogenic'], [None], ['unknown_value'], None, ['likely pathogenic', 'pathogenic']
            ],
            'confidence': [
                'practice guideline',
                'criteria provided, single submitter',
                'unknown_conf',
                None,
                'criteria provided, single submitter',
            ],
        },
        [1.0, 0.02, 0.0, 0.0, 0.92],
    ),
    _ExpressionCase(
        'eva_somatic',
        'score',
        {
            'clinicalSignificances': [
                ['pathogenic'], [None], ['unknown_value'], None, ['likely pathogenic', 'pathogenic']
            ],
            'confidence': [
                'practice guideline',
                'criteria provided, single submitter',
                'unknown_conf',
                None,
                'criteria provided, single submitter',
            ],
        },
        [1.0, 0.02, 0.0, 0.0, 0.92],
    ),
    _ExpressionCase(
        'eva',
        'direction_on_trait',
        {
            'clinicalSignificances': [
                ['pathogenic'], ['protective'], ['pathogenic', 'protective'], [None], ['Pathogenic']
            ]
        },
        ['risk', 'protect', None, None, 'risk'],
    ),
    _ExpressionCase(
        'eva_somatic',
        'direction_on_trait',
        {
            'clinicalSignificances': [
                ['pathogenic'], ['protective'], ['pathogenic', 'protective'], [None], ['Pathogenic']
            ]
        },
        ['risk', 'protect', None, None, 'risk'],
    ),
    _ExpressionCase(
        'eva',
        'direction_on_target',
        {'variantFunctionalConsequenceId': ['SO_0001589', 'SO_0001893', 'SO_9999999', None]},
        ['LoF', 'LoF', None, None],
    ),
    _ExpressionCase(
        'eva_somatic',
        'direction_on_target',
        {'variantFunctionalConsequenceId': ['SO_0001589', 'SO_0001893', 'SO_9999999', None]},
        ['LoF', 'LoF', None, None],
    ),
    _ExpressionCase(
        'uniprot_variants', 'score', {'confidence': ['high', 'medium', 'low', None]}, [1.0, 0.5, None, None]
    ),
    _ExpressionCase(
        'uniprot_literature', 'score', {'confidence': ['high', 'medium', 'low', None]}, [1.0, 0.5, None, None]
    ),
    _ExpressionCase(
        'clingen', 'score', {'confidence': ['Moderate', 'Strong', 'Unknown', None]}, [0.5, 1.0, None, None]
    ),
    _ExpressionCase('cancer_biomarkers', 'score', {'dummy': ['x', None]}, [1.0, 1.0]),
    _ExpressionCase('reactome', 'score', {'dummy': ['x', None]}, [1.0, 1.0]),
    _ExpressionCase('ot_crispr', 'score', {'dummy': ['x', None]}, [1.0, 1.0]),
    _ExpressionCase(
        'clinical_precedence',
        'score',
        {
            'clinicalStage': ['PHASE_2_3', 'PRECLINICAL', 'PHASE_1', 'PHASE_2_3'],
            'trialStopReasonCategories': [['Success', 'Negative'], [], ['unknown_reason'], None],
        },
        [0.25, None, None, 0.5],
    ),
    _ExpressionCase(
        'clinical_precedence', 'direction_on_trait', {'diseaseId': ['MONDO:1', None]}, ['protect', None]
    ),
    _ExpressionCase(
        'clinical_precedence',
        'direction_on_target',
        {'actionType': ['INHIBITOR', 'AGONIST', 'UNKNOWN', None]},
        ['LoF', 'GoF', None, None],
    ),
    _ExpressionCase('impc', 'score', {'resourceScore': [50.0, 0.0, None]}, [0.5, 0.0, None]),
    _ExpressionCase('impc', 'direction_on_trait', {'diseaseId': ['MONDO:1', None]}, ['risk', None]),
    _ExpressionCase('impc', 'direction_on_target', {'diseaseId': ['MONDO:1', None]}, ['LoF', None]),
    _ExpressionCase(
        'orphanet', 'score', {'confidence': ['Assessed', 'Not yet assessed', 'Unknown', None]}, [1.0, 0.5, None, None]
    ),
    _ExpressionCase('orphanet', 'direction_on_trait', {'diseaseId': ['MONDO:1', None]}, ['risk', None]),
    _ExpressionCase(
        'orphanet',
        'direction_on_target',
        {'variantFunctionalConsequenceId': ['SO_0002053', 'SO_0002054', 'SO_XXXX', None]},
        ['GoF', 'LoF', None, None],
    ),
    _ExpressionCase(
        'gene2phenotype',
        'score',
        {'confidence': ['definitive', 'moderate', 'limited', 'unknown', None]},
        [1.0, 0.5, 0.01, None, None],
    ),
    _ExpressionCase('gene2phenotype', 'direction_on_trait', {'diseaseId': ['MONDO:1', None]}, ['risk', None]),
    _ExpressionCase(
        'gene2phenotype',
        'direction_on_target',
        {'variantFunctionalConsequenceId': ['SO_0002315', 'SO_0002317', 'SO_XXXX', None]},
        ['GoF', 'LoF', None, None],
    ),
    _ExpressionCase(
        'intogen', 'score', {'resourceScore': [0.1, 1e-10, 0.0, None, nan]}, [0.25, 1.0, 1.0, None, None]
    ),
    _ExpressionCase('intogen', 'direction_on_trait', {'diseaseId': ['MONDO:1', None]}, ['risk', None]),
    _ExpressionCase(
        'intogen',
        'direction_on_target',
        {
            'mutatedSamples': [
                [{'functionalConsequenceId': 'SO_0002054'}],
                [{'functionalConsequenceId': 'SO_0002053'}],
                [{'functionalConsequenceId': None}],
                None,
                [],
            ]
        },
        ['LoF', 'GoF', None, None, None],
    ),
    _ExpressionCase(
        'gene_burden', 'score', {'resourceScore': [1e-7, 1e-17, 0.0, None, nan]}, [0.25, 1.0, 1.0, None, None]
    ),
    _ExpressionCase(
        'gene_burden',
        'direction_on_trait',
        {
            'oddsRatio': [2.0, 0.5, None, None, None, 1.0],
            'beta': [None, None, 1.0, -1.0, None, None],
        },
        ['risk', 'protect', 'risk', 'protect', None, None],
    ),
    _ExpressionCase('gene_burden', 'direction_on_target', {'dummy': ['x', None]}, ['LoF', 'LoF']),
    _ExpressionCase(
        'crispr_screen',
        'score',
        {'resourceScore': [0.5, 0.005, 0.0, -1.0, None, nan]},
        [0.0, 1.0, 1.0, 1.0, None, None],
    ),
    _ExpressionCase(
        'europepmc', 'score', {'resourceScore': [50.0, 150.0, 0.0, None, nan]}, [0.5, 1.0, 0.0, 1.0, 1.0]
    ),
    _ExpressionCase(
        'genomics_england', 'score', {'confidence': ['amber', 'green', 'red', None]}, [0.5, 1.0, None, None]
    ),
    _ExpressionCase('crispr', 'score', {'resourceScore': [0.0, 41.5, 100.0, 150.0]}, [0.415, 0.415, 1.0, 1.0]),
    _ExpressionCase('cancer_gene_census', 'score', {'resourceScore': [0.5, None]}, [0.5, None]),
    _ExpressionCase('cancer_gene_census', 'direction_on_trait', {'diseaseId': ['MONDO:1', None]}, ['risk', None]),
    _ExpressionCase(
        'cancer_gene_census',
        'direction_on_target',
        {'TSorOncogene': ['oncogene', 'tsg', 'bivalent', None]},
        ['GoF', 'LoF', None, None],
    ),
    _ExpressionCase(
        'encore',
        'score',
        {'geneticInteractionPValue': [1.0, 0.01, 0.0, None, nan]},
        [0.0, 1.0, 1.0, None, None],
    ),
    _ExpressionCase('ot_crispr_validation', 'score', {'resourceScore': [0.3, None]}, [0.3, None]),
]


def _assert_matches(got: list, expected: list) -> None:
    """Element-wise compare, tolerating float rounding and null."""
    assert len(got) == len(expected)
    for value, exp in zip(got, expected, strict=True):
        if exp is None:
            assert value is None
        elif isinstance(exp, float):
            assert value == pytest.approx(exp)
        else:
            assert value == exp


@pytest.mark.parametrize('case', CASES, ids=lambda c: f'{c.datasource_id}.{c.field}')
def test_registry_expression_matches_spark(case: _ExpressionCase) -> None:
    entry = EXPRESSIONS[case.datasource_id]
    expr = getattr(entry, case.field)
    assert expr is not None
    df = pl.DataFrame(case.data)
    got = df.with_columns(expr.alias('out'))['out'].to_list()
    _assert_matches(got, case.expected)


def test_case_table_covers_every_registry_field() -> None:
    """The table above is exhaustive, not just large.

    Every `(datasource_id, field)` in the registry has exactly one case, and there is nothing in
    the table that isn't in the registry.
    """
    registry_fields = {
        (datasource_id, field)
        for datasource_id, entry in EXPRESSIONS.items()
        for field in ('score', 'direction_on_trait', 'direction_on_target')
        if getattr(entry, field) is not None
    }
    table_fields = {(case.datasource_id, case.field) for case in CASES}
    assert len(CASES) == len(table_fields), 'duplicate (datasource_id, field) in CASES'
    assert table_fields == registry_fields
