"""Live checks against the real pipeline-runs and pre-releases buckets.

Skipped unless RUN_GCS_TESTS is set, because these read real GCS buckets and need
application-default credentials. `check.yaml` runs a bare `pytest` with no such
variable set, so this suite never runs in CI -- it is a documented manual procedure,
not a guarantee. Run with:

    RUN_GCS_TESTS=1 uv run --frozen pytest tests/test_supervisor_diff_live.py -rxs

RUN and REFERENCE are pinned to the pair verified on 2026-08-24, together with the
row/byte/file counts pinned below; do not substitute another run or release without
re-verifying its own known values first.

The default suite is scoped to `pts_disease` alone, whose three datasets are all a
single consolidated file -- deliberately the cheap case -- plus two direct
`read_stats` reads of `output/association_overall_direct`, the spark-partitioned
case (43 part files), so the whole default run stays a matter of seconds rather than
the minutes a full release costs. The whole-release comparison itself, with row
counts, is gated behind a second variable, RUN_GCS_FULL_TESTS, and only reports its
wall time rather than asserting on it -- see TestWholeRelease.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any

import pytest
from google.cloud import storage

from orchestration.supervisor.datasets import stage_configs, unified_pipeline_steps
from orchestration.supervisor.gcs import Footer, collect_diffs, footer_reader, read_stats
from orchestration.utils.common import GCS_PIPELINE_RUNS_BUCKET, GCS_PRE_RELEASES_BUCKET

pytestmark = pytest.mark.skipif(
    not os.environ.get('RUN_GCS_TESTS'),
    reason='needs GCS credentials, set RUN_GCS_TESTS=1 to run',
)

RUN = 'ds/26.06.0-dev2'
REFERENCE = '26.03'

DISEASE_STEP = ['pts_disease']
"""A deliberately small step: three datasets, each a single consolidated file."""

DISEASE_FILES = 1
DISEASE_BYTES = 7_312_633
DISEASE_ROWS = 47_030
DISEASE_COLUMNS = 18

ASSOCIATION_DATASET = 'output/association_overall_direct'
ASSOCIATION_FILES = 43
ASSOCIATION_BYTES = 633_763_096
ASSOCIATION_ROWS = 4_508_002

FULL_RELEASE = pytest.mark.skipif(
    not os.environ.get('RUN_GCS_FULL_TESTS'),
    reason='whole-release comparison, minutes even pooled; set RUN_GCS_FULL_TESTS=1 to run',
)


# `storage.Client().bucket(...)` is left unannotated rather than typed `storage.Bucket`:
# the real client's `list_blobs` signature does not structurally match `gcs.Bucket`'s
# minimal Protocol (it carries extra parameters ahead of `prefix`), so a precise
# annotation here would make `ty` reject every real call into `collect_diffs`/`read_stats`
# below, even though they work -- Python's duck typing does not require the Protocol's
# exact signature, only that a `prefix=` keyword call succeeds.
@pytest.fixture(scope='module')
def run_bucket() -> Any:
    return storage.Client().bucket(GCS_PIPELINE_RUNS_BUCKET)


@pytest.fixture(scope='module')
def reference_bucket() -> Any:
    return storage.Client().bucket(GCS_PRE_RELEASES_BUCKET)


@pytest.fixture(scope='module')
def reference_footer() -> Callable[[str], Footer]:
    return footer_reader(GCS_PRE_RELEASES_BUCKET)


@pytest.fixture(scope='module')
def run_footer() -> Callable[[str], Footer]:
    return footer_reader(GCS_PIPELINE_RUNS_BUCKET)


class TestKnownDatasets:
    """Pins the recorded figures directly, independent of `collect_diffs`'s step walk."""

    def test_disease_matches_the_recorded_reference_figures(
        self, reference_bucket: Any, reference_footer: Callable[[str], Footer]
    ) -> None:
        stats = read_stats(reference_bucket, f'{REFERENCE}/output/disease', reference_footer)
        assert stats is not None
        assert stats.files == DISEASE_FILES
        assert stats.total_bytes == DISEASE_BYTES
        assert stats.rows == DISEASE_ROWS
        assert len(stats.column_types) == DISEASE_COLUMNS

    def test_association_overall_direct_matches_the_recorded_reference_figures(
        self, reference_bucket: Any, reference_footer: Callable[[str], Footer]
    ) -> None:
        stats = read_stats(reference_bucket, f'{REFERENCE}/{ASSOCIATION_DATASET}', reference_footer)
        assert stats is not None
        assert stats.files == ASSOCIATION_FILES
        assert stats.total_bytes == ASSOCIATION_BYTES
        assert stats.rows == ASSOCIATION_ROWS

    def test_the_spark_partitioned_layout_is_still_countable(
        self, reference_bucket: Any, reference_footer: Callable[[str], Footer]
    ) -> None:
        """`part-00000-<uuid>-c000.snappy.parquet` must still end in `.parquet`.

        Break this by imagining `countable` computed off a suffix check that failed to
        see past the spark UUID/`-c000` tail: this dataset -- the classic part-file
        layout, not the single-file one the other test above already covers -- would
        report `countable=False` and lose its row count, and so would every other
        partitioned dataset in the release.
        """
        stats = read_stats(reference_bucket, f'{REFERENCE}/{ASSOCIATION_DATASET}', reference_footer)
        assert stats is not None
        assert stats.countable is True
        assert stats.rows is not None


class TestAbsentDataset:
    """`do/test_pharma` is a real pis-only run with no `output/` at all."""

    def test_a_pis_only_run_reports_no_output_dataset(self, run_bucket: Any) -> None:
        """Must be None, not an empty/zero-row `DatasetStats`.

        A dataset genuinely produced with zero rows and a dataset never produced at
        all are different findings, and `read_stats` telling them apart is the whole
        point of returning None here rather than a `DatasetStats(rows=0)`.
        """
        stats = read_stats(run_bucket, 'do/test_pharma/output/disease', None)
        assert stats is None


class TestScopedComparison:
    """Runs `collect_diffs` for real, scoped to `pts_disease` to stay cheap by default."""

    def test_at_least_one_dataset_is_compared(
        self,
        run_bucket: Any,
        reference_bucket: Any,
        run_footer: Callable[[str], Footer],
        reference_footer: Callable[[str], Footer],
    ) -> None:
        diffs, _ = collect_diffs(
            run_bucket, RUN, reference_bucket, REFERENCE, DISEASE_STEP, stage_configs(), run_footer, reference_footer
        )
        assert diffs, 'an empty comparison proves nothing about the plumbing that produced it'
        assert {d.dataset for d in diffs} >= {'output/disease'}

    def test_every_dataset_path_carries_its_namespace(
        self,
        run_bucket: Any,
        reference_bucket: Any,
        run_footer: Callable[[str], Footer],
        reference_footer: Callable[[str], Footer],
    ) -> None:
        diffs, _ = collect_diffs(
            run_bucket, RUN, reference_bucket, REFERENCE, DISEASE_STEP, stage_configs(), run_footer, reference_footer
        )
        assert diffs
        assert all(d.dataset.startswith(('output/', 'view/')) for d in diffs)

    def test_row_counts_are_present_for_parquet_and_never_zero(
        self,
        run_bucket: Any,
        reference_bucket: Any,
        run_footer: Callable[[str], Footer],
        reference_footer: Callable[[str], Footer],
    ) -> None:
        diffs, _ = collect_diffs(
            run_bucket, RUN, reference_bucket, REFERENCE, DISEASE_STEP, stage_configs(), run_footer, reference_footer
        )
        countable = [d for d in diffs if d.countable]
        assert countable, 'every pts_disease dataset is parquet; an empty set here means read_stats regressed'
        for d in countable:
            if d.side in ('both', 'run_only'):
                assert d.run_rows
            if d.side in ('both', 'reference_only'):
                assert d.reference_rows

    def test_the_disease_dataset_matches_the_recorded_reference_figures(
        self,
        run_bucket: Any,
        reference_bucket: Any,
        run_footer: Callable[[str], Footer],
        reference_footer: Callable[[str], Footer],
    ) -> None:
        diffs, _ = collect_diffs(
            run_bucket, RUN, reference_bucket, REFERENCE, DISEASE_STEP, stage_configs(), run_footer, reference_footer
        )
        disease = next(d for d in diffs if d.dataset == 'output/disease')
        assert disease.reference_rows == DISEASE_ROWS
        assert disease.reference_bytes == DISEASE_BYTES
        assert disease.reference_files == DISEASE_FILES


class TestNoRowsMatchesTheFullPathMinusRows:
    """Pins that `--no-rows` differs from the full path only in what it omits."""

    def test_no_rows_returns_the_same_datasets_and_bytes_with_rows_none(
        self,
        run_bucket: Any,
        reference_bucket: Any,
        run_footer: Callable[[str], Footer],
        reference_footer: Callable[[str], Footer],
    ) -> None:
        with_rows, _ = collect_diffs(
            run_bucket, RUN, reference_bucket, REFERENCE, DISEASE_STEP, stage_configs(), run_footer, reference_footer
        )
        without_rows, _ = collect_diffs(
            run_bucket, RUN, reference_bucket, REFERENCE, DISEASE_STEP, stage_configs(), None, None
        )

        assert with_rows, 'nothing to compare the fast path against'
        # the full path must actually have rows to omit, or "every row count is None"
        # below is true of both paths and proves nothing about what --no-rows skipped.
        assert any(d.run_rows is not None or d.reference_rows is not None for d in with_rows)

        by_dataset_with = {d.dataset: d for d in with_rows}
        by_dataset_without = {d.dataset: d for d in without_rows}
        assert set(by_dataset_with) == set(by_dataset_without)
        for name, full in by_dataset_with.items():
            fast = by_dataset_without[name]
            assert fast.run_bytes == full.run_bytes
            assert fast.reference_bytes == full.reference_bytes
            assert fast.run_files == full.run_files
            assert fast.reference_files == full.reference_files
            assert fast.run_rows is None
            assert fast.reference_rows is None


@FULL_RELEASE
class TestWholeRelease:
    """The comparison `pipeline-supervisor diff` actually runs in production.

    Gated separately from RUN_GCS_TESTS because this is the minutes-long comparison
    the module docstring warns about -- exactly what the plan says gets switched off
    if it were the default. Only the wall time is reported; nothing here asserts on
    it, since bucket contents and network conditions vary run to run.
    """

    def test_the_whole_release_compares_with_rows_and_reports_its_wall_time(
        self,
        run_bucket: Any,
        reference_bucket: Any,
        run_footer: Callable[[str], Footer],
        reference_footer: Callable[[str], Footer],
    ) -> None:
        steps = unified_pipeline_steps()
        start = time.monotonic()
        diffs, _ = collect_diffs(
            run_bucket, RUN, reference_bucket, REFERENCE, steps, stage_configs(), run_footer, reference_footer
        )
        elapsed = time.monotonic() - start
        assert diffs, 'an empty whole-release comparison proves nothing'
        assert all(d.dataset.startswith(('output/', 'view/')) for d in diffs)
        sys.stdout.write(f'\nwhole-release comparison (with rows): {len(diffs)} datasets, {elapsed:.1f}s\n')

    def test_the_whole_release_no_rows_path_reports_its_wall_time(
        self, run_bucket: Any, reference_bucket: Any
    ) -> None:
        steps = unified_pipeline_steps()
        start = time.monotonic()
        diffs, _ = collect_diffs(run_bucket, RUN, reference_bucket, REFERENCE, steps, stage_configs(), None, None)
        elapsed = time.monotonic() - start
        assert diffs, 'an empty whole-release comparison proves nothing'
        assert all(d.run_rows is None and d.reference_rows is None for d in diffs)
        sys.stdout.write(f'whole-release comparison (--no-rows): {len(diffs)} datasets, {elapsed:.1f}s\n')
