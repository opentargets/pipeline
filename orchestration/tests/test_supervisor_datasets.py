"""Tests for extracting a step's output datasets from the stage configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from orchestration.supervisor.datasets import destinations_for

REPO = Path(__file__).resolve().parents[2]  # tests -> orchestration -> pipeline


def _config(tasks: list[dict[str, object]]) -> dict[str, object]:
    return {'steps': {'disease': tasks}}


class TestDestinationsFor:
    def test_reads_a_scalar_destination(self) -> None:
        cfg = _config([{'name': 't', 'destination': 'output/disease'}])
        assert destinations_for('pts_disease', cfg) == ['output/disease']

    def test_reads_a_list_destination(self) -> None:
        """Defensive only — no list-valued destination exists in either config today."""
        cfg = _config([{'name': 't', 'destination': ['output/a', 'view/b']}])
        assert destinations_for('pts_disease', cfg) == ['output/a', 'view/b']

    def test_recurses_into_a_foreach_do_block(self) -> None:
        """`foreach:` nests its task under `do:`, one level below the step's task list.

        Twenty-eight real tasks live there. A walk that only reads the step's own list
        never reaches them and every other test in this file still passes.
        """
        cfg = _config([
            {'name': 'plain', 'destination': 'output/plain'},
            {'name': 'fan', 'foreach': ['a', 'b'], 'do': [{'name': 'inner', 'destination': 'output/inner'}]},
        ])
        assert destinations_for('pts_disease', cfg) == ['output/plain', 'output/inner']

    def test_reads_a_mapping_destination(self) -> None:
        cfg = _config([{'name': 't', 'destination': {'x': 'output/a', 'y': 'view/b'}}])
        assert destinations_for('pts_disease', cfg) == ['output/a', 'view/b']

    def test_keeps_output_and_view_only(self) -> None:
        """`intermediate/` is scratch between steps, not a release artifact."""
        cfg = _config([
            {'name': 'a', 'destination': 'output/keep'},
            {'name': 'b', 'destination': 'view/keep'},
            {'name': 'c', 'destination': 'intermediate/drop'},
            {'name': 'd', 'destination': 'input/drop'},
        ])
        assert destinations_for('pts_disease', cfg) == ['output/keep', 'view/keep']

    def test_drops_templated_paths(self) -> None:
        """A path containing `${each}` resolves only at run time.

        Emitting it verbatim would match nothing in GCS and read as a missing dataset.
        """
        cfg = _config([
            {'name': 'a', 'destination': 'output/fine'},
            {'name': 'b', 'destination': 'output/${each}/parquet'},
        ])
        assert destinations_for('pts_disease', cfg) == ['output/fine']

    def test_deduplicates(self) -> None:
        cfg = _config([
            {'name': 'a', 'destination': 'output/same'},
            {'name': 'b', 'destination': 'output/same'},
        ])
        assert destinations_for('pts_disease', cfg) == ['output/same']

    def test_a_step_with_no_config_yields_nothing(self) -> None:
        """`pis_enhancer_to_gene` is declared in unified_pipeline.yaml with no backing config.

        It must yield an empty list rather than raising.
        """
        assert destinations_for('pis_enhancer_to_gene', {'steps': {}}) == []


class TestAgainstTheRealConfigs:
    """Measured against both configs on 2026-08-24. A disagreement here is a finding."""

    def test_a_known_step_resolves_to_its_real_datasets(self) -> None:
        cfg = yaml.safe_load((REPO / 'pts/config.yaml').read_text())
        assert 'output/disease' in destinations_for('pts_disease', cfg)

    def test_every_step_yields_only_release_namespaces(self) -> None:
        cfg = yaml.safe_load((REPO / 'pts/config.yaml').read_text())
        up = yaml.safe_load((REPO / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml').read_text())
        checked = 0
        for step in (s for s in up['steps'] if s.startswith('pts_')):
            for dest in destinations_for(step, cfg):
                checked += 1
                assert dest.startswith(('output/', 'view/')), f'{step} yielded {dest}'
                assert '${' not in dest, f'{step} yielded a templated path {dest}'
        assert checked > 0, 'asserted over nothing, which proves nothing'

    def test_both_configs_together_yield_the_measured_release_datasets(self) -> None:
        """Pins the totals so a regression in the `do:` recursion cannot pass silently.

        Counting only what a walk actually reaches is exactly the bug this guards: an
        implementation that never enters `foreach`/`do` produces a plausible-looking
        subset and every other assertion in this file still holds.
        """
        up = yaml.safe_load((REPO / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml').read_text())
        configs = {s: yaml.safe_load((REPO / s / 'config.yaml').read_text()) for s in ('pis', 'pts')}
        found: set[str] = set()
        producing = 0
        for step in up['steps']:
            stage = step.split('_', 1)[0]
            if stage not in configs:
                continue
            dests = destinations_for(step, configs[stage])
            producing += bool(dests)
            found.update(dests)
        assert len(found) == 71, f'expected 71 unique release datasets, got {len(found)}'
        assert sum(d.startswith('view/') for d in found) == 13
        assert producing == 57, f'expected 57 steps producing release datasets, got {producing}'
