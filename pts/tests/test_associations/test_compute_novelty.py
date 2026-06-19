"""Regression tests for Association.compute_novelty algorithmic refactor.

Pins specific output values from the current implementation so that the
array-based rewrite of _get_novelty can be validated for equivalence.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.pyspark.associations_utils.association import Association


@pytest.mark.slow
class TestComputeNoveltyRegression:
    """Regression suite for compute_novelty.

    Three groups exercise: (a) a single peak followed by flat years,
    (b) two peaks within the novelty window, (c) the 'overall' aggregation
    path with a null aggregationValue (NA-fill / NA-strip round-trip).
    """

    # diseaseId, targetId, year, yearlyEvidenceScores, associationScore,
    # aggregationType, aggregationValue, yearlyEvidenceCount
    DATASET = [
        # Group A: one peak in 2010
        ('d1', 't1', 2010, [0.5], 0.5, 'datasourceId', 'ds1', 1),
        # Group B: peak in 2010, second peak in 2014
        ('d2', 't2', 2010, [0.4], 0.4, 'datasourceId', 'ds1', 1),
        ('d2', 't2', 2014, [0.7], 0.7, 'datasourceId', 'ds1', 1),
        # Group C: 'overall' aggregation with null aggregationValue
        ('d3', 't3', 2010, [0.6], 0.6, 'overall', None, 1),
    ]

    SCHEMA = (
        'diseaseId STRING, '
        'targetId STRING, '
        'year INTEGER, '
        'yearlyEvidenceScores ARRAY<FLOAT>, '
        'associationScore FLOAT, '
        'aggregationType STRING, '
        'aggregationValue STRING, '
        'yearlyEvidenceCount INTEGER'
    )

    @pytest.fixture(autouse=True)
    def _setup(self: TestComputeNoveltyRegression, spark: SparkSession) -> None:
        self.association = Association(spark.createDataFrame(self.DATASET, self.SCHEMA))

    def test_output_row_count(self: TestComputeNoveltyRegression) -> None:
        """compute_novelty produces one row per (target, disease, aggregationValue) group."""
        result = self.association.compute_novelty()
        assert result.count() == 3  # three groups in the fixture

    def test_output_columns_present(self: TestComputeNoveltyRegression) -> None:
        """Output schema includes the public-facing columns."""
        cols = set(self.association.compute_novelty().columns)
        for required in (
            'targetId',
            'diseaseId',
            'aggregationType',
            'aggregationValue',
            'associationScore',
            'evidenceCount',
            'timeseries',
            'currentNovelty',
        ):
            assert required in cols, f'missing column: {required}'

    def test_currentnovelty_for_group_a_is_zero_long_after_peak(
        self: TestComputeNoveltyRegression,
    ) -> None:
        # Group A has its peak in 2010; with novelty_window=10 the peak's
        # contribution to currentNovelty (current_year) decays to ~0 long after.
        current_year = datetime.now().year
        # Only meaningful if current_year is well past 2020 (peak + window)
        assert current_year >= 2021
        row = (
            self.association
            .compute_novelty()
            .filter((f.col('targetId') == 't1') & (f.col('diseaseId') == 'd1'))
            .first()
        )
        assert row is not None
        # currentNovelty: novelty in current_year. Group A peak is 2010 (peak_value=0.5).
        # window=10 → peak only contributes for years 2010..2020. After that, 0.
        assert row.currentNovelty == pytest.approx(0.0, abs=1e-9)

    def test_timeseries_array_contains_expected_years(
        self: TestComputeNoveltyRegression,
    ) -> None:
        """Pin the exact set of years in the timeseries for both groups.

        _back_fill_missing_years generates years [first_evidence_year - 5, current_year + 1].
        compute_novelty then nullifies the year on entries beyond current_year.

        Both groups have first_evidence_year = 2010, so:
        - Total timeseries entries per group: (current_year + 1) - 2005 + 1
        - Non-null years per group: {2005, 2006, ..., current_year}
        """
        current_year = datetime.now().year
        expected_non_null_years = set(range(2005, current_year + 1))
        expected_entry_count = (current_year + 1) - 2005 + 1

        rows = self.association.compute_novelty().collect()
        by_key = {(r.targetId, r.diseaseId): r for r in rows}

        for key in [('t1', 'd1'), ('t2', 'd2'), ('t3', 'd3')]:
            row = by_key[key]
            assert row is not None, f'missing group {key}'

            # Total entry count includes the nullified-year entry
            assert len(row.timeseries) == expected_entry_count, (
                f'group {key}: expected {expected_entry_count} timeseries entries, got {len(row.timeseries)}'
            )

            # The set of non-null years must exactly match the back-filled range
            non_null_years = {t.year for t in row.timeseries if t.year is not None}
            assert non_null_years == expected_non_null_years, (
                f'group {key}: non-null years mismatch.\n'
                f'  missing: {expected_non_null_years - non_null_years}\n'
                f'  extra:   {non_null_years - expected_non_null_years}'
            )

    def test_associationscore_nonnull_for_groups_with_evidence(
        self: TestComputeNoveltyRegression,
    ) -> None:
        """Output associationScore is non-null for groups with evidence."""
        rows = self.association.compute_novelty().collect()
        for row in rows:
            assert row.associationScore is not None
            assert row.associationScore >= 0.0

    def test_evidencecount_matches_input(self: TestComputeNoveltyRegression) -> None:
        """evidenceCount sums input yearlyEvidenceCount across the group.

        Note: _back_fill_missing_years does NOT propagate yearlyEvidenceCount —
        filled rows carry NULL — so the sum equals the input total. If a future
        refactor populates yearlyEvidenceCount on filled rows, this expectation
        would change and that change should be intentional.
        """  # noqa: D403
        rows = self.association.compute_novelty().collect()
        by_key = {(r.targetId, r.diseaseId): r for r in rows}
        # Group A: one row with yearlyEvidenceCount=1
        assert by_key['t1', 'd1'].evidenceCount == 1
        # Group B: two rows with yearlyEvidenceCount=1+1
        assert by_key['t2', 'd2'].evidenceCount == 2
        # Group C: one row with yearlyEvidenceCount=1
        assert by_key['t3', 'd3'].evidenceCount == 1

    def test_overall_group_aggregationvalue_round_trip_to_null(
        self: TestComputeNoveltyRegression,
    ) -> None:
        """Group C ('overall' aggregation) has null aggregationValue in input.

        compute_novelty fills nulls with 'NA' for back-fill purposes and then
        strips 'NA' back to null in the final output. This pins the round-trip
        so a refactor that changes how nulls are handled cannot silently flip
        the public contract of overall-aggregation outputs.
        """
        rows = self.association.compute_novelty().collect()
        by_key = {(r.targetId, r.diseaseId): r for r in rows}
        group_c = by_key['t3', 'd3']
        assert group_c.aggregationType == 'overall'
        assert group_c.aggregationValue is None, f'expected null aggregationValue, got {group_c.aggregationValue!r}'
