"""Tests for the evidence_postprocess score/direction expression registry.

No spark dependency here -- the registry's parity with real spark is verified separately (see
`.superpowers/sdd/plan/registry-parity-output.txt` and `task-registry-report.md`), and the
handful of expected values hard-coded below were measured against that spark run, not guessed.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

from pts.transformers.utils.evidence_expressions import EXPRESSIONS

CONFIG = Path(__file__).parents[1] / 'config.yaml'
EXPRESSION_FIELDS = {
    'score_expression': 'score',
    'direction_on_trait_expression': 'direction_on_trait',
    'direction_on_target_expression': 'direction_on_target',
}


def _config_datasources() -> dict[str, dict[str, str | None]]:
    """`datasource_id -> {field: sql or None}` for every `evidence_postprocess_*` step."""
    steps = yaml.safe_load(CONFIG.read_text())['steps']
    found: dict[str, dict[str, str | None]] = {}
    for step_name, tasks in steps.items():
        if not step_name.startswith('evidence_postprocess'):
            continue
        settings = tasks[0]['settings']
        datasource_id = settings['datasource_id']
        found[datasource_id] = {field: settings.get(cfg_key) for cfg_key, field in EXPRESSION_FIELDS.items()}
    return found


CONFIG_DATASOURCES = _config_datasources()


def test_every_config_datasource_has_a_registry_entry() -> None:
    missing = set(CONFIG_DATASOURCES) - set(EXPRESSIONS)
    assert not missing


def test_registry_has_no_unexpected_datasources() -> None:
    extra = set(EXPRESSIONS) - set(CONFIG_DATASOURCES)
    assert not extra


@pytest.mark.parametrize('datasource_id', sorted(CONFIG_DATASOURCES))
def test_registry_field_presence_matches_config(datasource_id: str) -> None:
    """A field with SQL in config.yaml has a non-None registry expression, and vice versa."""
    entry = EXPRESSIONS[datasource_id]
    for field in EXPRESSION_FIELDS.values():
        has_sql = bool(CONFIG_DATASOURCES[datasource_id][field])
        has_expr = getattr(entry, field) is not None
        assert has_sql == has_expr, f'{datasource_id}.{field}: config has SQL={has_sql}, registry has expr={has_expr}'


@pytest.mark.parametrize('datasource_id', sorted(EXPRESSIONS))
def test_every_registry_expression_is_a_polars_expr(datasource_id: str) -> None:
    entry = EXPRESSIONS[datasource_id]
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
        'clinicalSignificances': [['pathogenic'], [None], ['unknown_value'], None],
        'confidence': ['practice guideline', 'criteria provided, single submitter', 'unknown_conf', None],
    })
    # measured against spark: 0.9+0.1, 0.0+0.02, 0.0+0.0, 0.0+0.0
    got = df.with_columns(EXPRESSIONS['eva'].score.alias('score'))['score'].to_list()
    assert got == pytest.approx([1.0, 0.02, 0.0, 0.0])


def test_eva_direction_on_trait_null_when_both_pathogenic_and_protective() -> None:
    direction_on_trait = EXPRESSIONS['eva'].direction_on_trait
    assert direction_on_trait is not None
    df = pl.DataFrame({'clinicalSignificances': [['pathogenic'], ['protective'], ['pathogenic', 'protective'], [None]]})
    got = df.with_columns(direction_on_trait.alias('d'))['d'].to_list()
    assert got == ['risk', 'protect', None, None]


def test_log10_rescaled_score_saturates_at_zero_and_null_propagates() -> None:
    # crispr_screen: out_min=0.0, weak_ref=0.5, strong_ref=0.005 -- measured against spark.
    df = pl.DataFrame({'resourceScore': [0.5, 0.005, 0.0, -1.0, None]})
    got = df.with_columns(EXPRESSIONS['crispr_screen'].score.alias('s'))['s'].to_list()
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(1.0)
    assert got[2] == pytest.approx(1.0)  # non-positive input saturates to 1.0
    assert got[3] == pytest.approx(1.0)
    assert got[4] is None


def test_linear_rescale_clamps_project_score() -> None:
    # crispr: linear_rescale(resourceScore, 41.5, 100, 0.415, 1.0) -- measured against spark.
    df = pl.DataFrame({'resourceScore': [0.0, 41.5, 100.0, 150.0]})
    got = df.with_columns(EXPRESSIONS['crispr'].score.alias('s'))['s'].to_list()
    assert got == pytest.approx([0.415, 0.415, 1.0, 1.0])


def test_clinical_precedence_score_multiplies_stage_by_stop_reason_minimum() -> None:
    df = pl.DataFrame({
        'clinicalStage': ['PHASE_2_3', 'PRECLINICAL', 'PHASE_1'],
        'trialStopReasonCategories': [['Success', 'Negative'], [], ['unknown_reason']],
    })
    # measured against spark: 0.5*0.5, 0.01*null, 0.1*null
    got = df.with_columns(EXPRESSIONS['clinical_precedence'].score.alias('s'))['s'].to_list()
    assert got[0] == pytest.approx(0.25)
    assert got[1] is None
    assert got[2] is None


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
