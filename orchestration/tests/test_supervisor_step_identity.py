"""Tests for the step identity mapping."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orchestration.supervisor.step_identity import RUN_TASK_PREFIX, identify, is_run_task, step_from_task_id

REPO = Path(__file__).resolve().parents[2]


class TestIdentify:
    def test_splits_on_the_first_underscore_only(self) -> None:
        """`step_config` documents step names as `{stage}_{step_name}` and splits once.

        A step whose own name contains underscores would otherwise lose most of it.
        """
        i = identify('pts_evidence_postprocess_impc')
        assert i.stage == 'pts'
        assert i.config_key == 'evidence_postprocess_impc'

    def test_keeps_the_step_name_as_the_billing_label(self) -> None:
        assert identify('pis_chembl').step == 'pis_chembl'

    def test_handles_every_stage(self) -> None:
        assert identify('gentropy_variant_annotation').stage == 'gentropy'

    def test_a_bare_config_key_is_rejected(self) -> None:
        """`disease` is a config key, not a step name.

        Accepting it silently would produce a task id that matches nothing.
        """
        with pytest.raises(ValueError, match='no config key'):
            identify('disease')

    def test_a_trailing_underscore_is_rejected(self) -> None:
        """`pts_` has a stage but no config key.

        Splitting it silently would yield `config_key=''`, a step with no way to look
        up its task list.
        """
        with pytest.raises(ValueError, match='no config key'):
            identify('pts_')

    def test_a_leading_underscore_is_rejected(self) -> None:
        """`_disease` has a config key but no stage.

        Splitting it silently would yield `stage=''`, which resolves to nothing in a
        `{stage: config}` lookup and drops the step with no error at all.
        """
        with pytest.raises(ValueError, match='no stage'):
            identify('_disease')


class TestStepFromTaskId:
    def test_recovers_the_group(self) -> None:
        assert step_from_task_id('pts_disease.run_pts_disease') == 'pts_disease'

    def test_an_unqualified_id_is_its_own_step(self) -> None:
        assert step_from_task_id('cluster_delete_pts') == 'cluster_delete_pts'


class TestIsRunTask:
    def test_a_task_one_group_deep_is_the_run_task(self) -> None:
        assert is_run_task('pts_target.run_pts_target') is True

    def test_a_task_two_groups_deep_is_still_the_run_task(self) -> None:
        """`cluster: false` nests the run task inside a second, `_batch_jobs` group.

        Reconstructing an id from `RUN_TASK_PREFIX` alone gets this shape wrong; checking
        the last path component does not, because it never has to guess the nesting depth.
        """
        assert is_run_task('gentropy_variant_annotation.gentropy_variant_annotation_batch_jobs.'
                            'run_gentropy_variant_annotation') is True

    def test_a_sibling_task_in_the_group_is_not_the_run_task(self) -> None:
        assert is_run_task('pts_target.diff_pts_target') is False

    def test_a_dotless_id_is_not_mistaken_for_a_run_task(self) -> None:
        """`step_from_task_id` returns a dotless id unchanged, making it its own step.

        `run_{step}` would then equal `run_run_pts_target`, which this id is not.
        """
        assert is_run_task('run_pts_target') is False


class TestIsRunTaskAgainstTheRealDag:
    """`TestIsRunTask` above uses hand-typed ids.

    If the DAG's own naming pattern ever drifted — a stage-level group added around
    every step, say — `is_run_task` would return False for every real task id, and
    `stalled` would silently drop to the ceiling for every step: the exact symptom
    `baseline_from_journal` warns is indistinguishable from an honest first run.
    Building ids from the real step list and the real `RUN_TASK_PREFIX` template,
    rather than from strings typed by hand, is what would actually catch that.
    """

    def test_every_step_run_task_id_is_recognised(self) -> None:
        up = yaml.safe_load((REPO / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml').read_text())
        for step in up['steps']:
            assert is_run_task(f'{step}.{RUN_TASK_PREFIX}{step}') is True, step

    def test_the_batch_nested_steps_are_recognised_two_groups_deep(self) -> None:
        """`gentropy_variant_annotation` and `gentropy_l2g_prediction` set `cluster: false`.

        See `dags/config/gentropy.yaml:120-121,222-223` and the module docstring.
        """
        for step in ('gentropy_variant_annotation', 'gentropy_l2g_prediction'):
            task_id = f'{step}.{step}_batch_jobs.{RUN_TASK_PREFIX}{step}'
            assert is_run_task(task_id) is True, step


class TestAgainstTheRealConfigs:
    """The mapping is worthless if it does not resolve real steps."""

    def test_every_pis_and_pts_step_resolves_to_a_real_config_key(self) -> None:
        up = yaml.safe_load((REPO / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml').read_text())
        configs = {
            stage: yaml.safe_load((REPO / stage / 'config.yaml').read_text()).get('steps', {})
            for stage in ('pis', 'pts')
        }
        unresolved = []
        for step in up['steps']:
            ident = identify(step)
            if ident.stage not in configs:
                continue
            if ident.config_key not in configs[ident.stage]:
                unresolved.append(step)
        assert unresolved == ['pis_enhancer_to_gene'], (
            'pis_enhancer_to_gene is a known stale entry in unified_pipeline.yaml — it has no '
            'enhancer_to_gene step in pis/config.yaml, and gentropy_enhancer_to_gene is declared '
            'separately. Any OTHER unresolved step means the mapping is wrong.'
        )
