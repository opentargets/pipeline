"""Tests for the association OTF enrichment module."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.association_otf import (
    _compute_facet_classes,
    _compute_facet_therapeutic_areas,
    _compute_facet_tractability,
)


@pytest.mark.slow
class TestComputeFacetClasses:
    """Tests for _compute_facet_classes."""

    TARGET_CLASS_SCHEMA = (
        t
        .StructType()
        .add('targetId', t.StringType())
        .add('targetData', t.StringType())
        .add(
            'targetClass',
            t.ArrayType(
                t.StructType().add('id', t.LongType()).add('label', t.StringType()).add('level', t.StringType())
            ),
        )
    )

    DATASET = [
        (
            'T1',
            'T1 gene1 G1',
            [
                {'id': 1, 'label': 'Enzyme', 'level': 'l1'},
                {'id': 1, 'label': 'Kinase', 'level': 'l2'},
                {'id': 2, 'label': 'Receptor', 'level': 'l1'},
                {'id': 2, 'label': 'GPCR', 'level': 'l2'},
                {'id': 3, 'label': 'SomeOther', 'level': 'l3'},
            ],
        ),
        (
            'T2',
            'T2 gene2 G2',
            [
                {'id': 10, 'label': 'Transporter', 'level': 'l1'},
                {'id': 10, 'label': 'ABC transporter', 'level': 'l2'},
            ],
        ),
        ('T3', 'T3 gene3 G3', None),
    ]

    @pytest.fixture(autouse=True)
    def _setup(self: TestComputeFacetClasses, spark: SparkSession) -> None:
        self.df = spark.createDataFrame(self.DATASET, schema=self.TARGET_CLASS_SCHEMA)
        self.result = _compute_facet_classes(self.df)

    def test_target_class_column_dropped(self: TestComputeFacetClasses) -> None:
        """TargetClass column should be replaced by facetClasses."""
        assert 'targetClass' not in self.result.columns
        assert 'facetClasses' in self.result.columns

    def test_other_columns_preserved(self: TestComputeFacetClasses) -> None:
        """Non-targetClass columns should remain untouched."""
        assert 'targetId' in self.result.columns
        assert 'targetData' in self.result.columns

    def test_row_count_preserved(self: TestComputeFacetClasses) -> None:
        """Left outer join should preserve all original rows."""
        assert self.result.count() == len(self.DATASET)

    def test_l3_entries_filtered_out(self: TestComputeFacetClasses) -> None:
        """Only l1 and l2 levels should appear in facetClasses."""
        t1_row = self.result.filter(f.col('targetId') == 'T1').first()
        facet_classes = t1_row['facetClasses']  # ty:ignore[not-subscriptable]
        # T1 has 2 valid id groups (id=1, id=2); id=3 is l3-only and filtered
        assert len(facet_classes) == 2

    def test_struct_has_l1_l2_fields(self: TestComputeFacetClasses) -> None:
        """Each facet class struct should have l1 and l2 fields."""
        t1_row = self.result.filter(f.col('targetId') == 'T1').first()
        for fc in t1_row['facetClasses']:  # ty:ignore[not-subscriptable]
            assert 'l1' in fc.asDict()
            assert 'l2' in fc.asDict()

    def test_correct_l1_l2_pairing(self: TestComputeFacetClasses) -> None:
        """l1 and l2 labels should be correctly paired by id."""
        t2_row = self.result.filter(f.col('targetId') == 'T2').first()
        fc = t2_row['facetClasses'][0]  # ty:ignore[not-subscriptable]
        assert fc['l1'] == 'Transporter'
        assert fc['l2'] == 'ABC transporter'

    def test_null_target_class_produces_null_facet(self: TestComputeFacetClasses) -> None:
        """Targets with null targetClass should have null facetClasses."""
        t3_row = self.result.filter(f.col('targetId') == 'T3').first()
        assert t3_row['facetClasses'] is None  # ty:ignore[not-subscriptable]


@pytest.mark.slow
class TestComputeFacetTherapeuticAreas:
    """Tests for _compute_facet_therapeutic_areas."""

    DATASET = [
        ('D1', 'D1 Cancer', ['D1', 'D2'], 'Cancer'),
        ('D2', 'D2 Neoplasm', ['D1'], 'Neoplasm'),
        ('D3', 'D3 Infection', None, 'Infection'),
    ]

    @pytest.fixture(autouse=True)
    def _setup(self: TestComputeFacetTherapeuticAreas, spark: SparkSession) -> None:
        self.df = spark.createDataFrame(
            self.DATASET,
            'diseaseId STRING, diseaseData STRING, therapeuticAreas ARRAY<STRING>, name STRING',
        )
        self.result = _compute_facet_therapeutic_areas(self.df, 'diseaseId', 'name', 'therapeuticAreas')

    def test_output_columns(self: TestComputeFacetTherapeuticAreas) -> None:
        """Output should contain the key column and the vec column."""
        assert set(self.result.columns) == {'diseaseId', 'therapeuticAreas'}

    def test_d1_resolves_both_ta_labels(self: TestComputeFacetTherapeuticAreas) -> None:
        """D1 has therapeuticAreas [D1, D2] which should resolve to {Cancer, Neoplasm}."""
        d1_row = self.result.filter(f.col('diseaseId') == 'D1').first()
        assert set(d1_row['therapeuticAreas']) == {'Cancer', 'Neoplasm'}  # ty:ignore[not-subscriptable]

    def test_d2_resolves_single_ta_label(self: TestComputeFacetTherapeuticAreas) -> None:
        """D2 has therapeuticAreas [D1] which should resolve to {Cancer}."""
        d2_row = self.result.filter(f.col('diseaseId') == 'D2').first()
        assert set(d2_row['therapeuticAreas']) == {'Cancer'}  # ty:ignore[not-subscriptable]

    def test_null_therapeutic_areas_handled(self: TestComputeFacetTherapeuticAreas) -> None:
        """D3 has null therapeuticAreas and should still appear in results."""
        d3_row = self.result.filter(f.col('diseaseId') == 'D3').first()
        assert d3_row is not None


@pytest.mark.slow
class TestComputeFacetTractability:
    """Tests for _compute_facet_tractability."""

    TRACTABILITY_SCHEMA = (
        t
        .StructType()
        .add('targetId', t.StringType())
        .add(
            'tractability',
            t.ArrayType(
                t.StructType().add('id', t.StringType()).add('modality', t.StringType()).add('value', t.BooleanType())
            ),
        )
    )

    DATASET = [
        (
            'T1',
            [
                {'id': 'sm_1', 'modality': 'SM', 'value': True},
                {'id': 'sm_2', 'modality': 'SM', 'value': False},
                {'id': 'ab_1', 'modality': 'AB', 'value': True},
                {'id': 'pr_1', 'modality': 'PR', 'value': True},
                {'id': 'oc_1', 'modality': 'OC', 'value': True},
            ],
        ),
        (
            'T2',
            [
                {'id': 'sm_x', 'modality': 'SM', 'value': False},
                {'id': 'ab_x', 'modality': 'AB', 'value': False},
            ],
        ),
        ('T3', None),
    ]

    @pytest.fixture(autouse=True)
    def _setup(self: TestComputeFacetTractability, spark: SparkSession) -> None:
        self.df = spark.createDataFrame(self.DATASET, schema=self.TRACTABILITY_SCHEMA)
        self.result = _compute_facet_tractability(self.df)

    def test_facet_columns_added(self: TestComputeFacetTractability) -> None:
        """All four modality facet columns should be present."""
        expected = {
            'facetTractabilitySmallmolecule',
            'facetTractabilityAntibody',
            'facetTractabilityProtac',
            'facetTractabilityOthermodalities',
        }
        assert expected.issubset(set(self.result.columns))

    def test_miid_column_removed(self: TestComputeFacetTractability) -> None:
        """The temporary monotonically_increasing_id column should be dropped."""
        assert 'miid' not in self.result.columns

    def test_row_count_preserved(self: TestComputeFacetTractability) -> None:
        """All original rows should be preserved."""
        assert self.result.count() == len(self.DATASET)

    def test_only_true_values_kept(self: TestComputeFacetTractability) -> None:
        """Only tractability entries with value=True should appear."""
        t1_row = self.result.filter(f.col('targetId') == 'T1').first()
        # SM: sm_1 is True, sm_2 is False → only sm_1
        assert t1_row['facetTractabilitySmallmolecule'] == ['sm_1']  # ty:ignore[not-subscriptable]
        assert t1_row['facetTractabilityAntibody'] == ['ab_1']  # ty:ignore[not-subscriptable]
        assert t1_row['facetTractabilityProtac'] == ['pr_1']  # ty:ignore[not-subscriptable]
        assert t1_row['facetTractabilityOthermodalities'] == ['oc_1']  # ty:ignore[not-subscriptable]

    def test_all_false_produces_empty_arrays(self: TestComputeFacetTractability) -> None:
        """When all entries have value=False, facet arrays should be empty."""
        t2_row = self.result.filter(f.col('targetId') == 'T2').first()
        assert t2_row['facetTractabilitySmallmolecule'] == []  # ty:ignore[not-subscriptable]
        assert t2_row['facetTractabilityAntibody'] == []  # ty:ignore[not-subscriptable]

    def test_null_tractability_handled(self: TestComputeFacetTractability) -> None:
        """Targets with null tractability should still be present."""
        t3_row = self.result.filter(f.col('targetId') == 'T3').first()
        assert t3_row is not None
