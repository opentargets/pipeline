"""Tests for the step identity mapping."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orchestration.supervisor.step_identity import identify, step_from_task_id

REPO = Path(__file__).resolve().parents[2]


class TestIdentify:
    def test_splits_on_the_first_underscore_only(self) -> None:
        """`step_config` documents step names as `{stage}_{step_name}` and splits once.

        A step whose own name contains underscores would otherwise lose most of it.
        """
        i = identify('pts_evidence_postprocess_impc')
        assert i.stage == 'pts'
        assert i.config_key == 'evidence_postprocess_impc'

    def test_qualifies_the_run_task_id_with_its_group(self) -> None:
        """Airflow prefixes group ids onto children, so a bare name never matches."""
        assert identify('pts_disease').run_task_id == 'pts_disease.run_pts_disease'

    def test_keeps_the_step_name_as_the_billing_label(self) -> None:
        assert identify('pis_chembl').step == 'pis_chembl'

    def test_handles_every_stage(self) -> None:
        assert identify('gentropy_variant_annotation').stage == 'gentropy'

    def test_a_bare_config_key_is_rejected(self) -> None:
        """`disease` is a config key, not a step name.

        Accepting it silently would produce a task id that matches nothing.
        """
        with pytest.raises(ValueError, match='no stage prefix'):
            identify('disease')


class TestStepFromTaskId:
    def test_recovers_the_group(self) -> None:
        assert step_from_task_id('pts_disease.run_pts_disease') == 'pts_disease'

    def test_an_unqualified_id_is_its_own_step(self) -> None:
        assert step_from_task_id('cluster_delete_pts') == 'cluster_delete_pts'

    def test_round_trips_with_identify(self) -> None:
        assert step_from_task_id(identify('pts_target').run_task_id) == 'pts_target'


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
