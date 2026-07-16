"""Tests for the target_prioritisation pyspark module.

Covers the query functions that were migrated off output/target's removed
tractability, safetyLiabilities, and chemicalProbes fields onto the
standalone target_tractability, safety_liability, and chemical_probes
datasets, so that _ligand_pocket_query/_safety_query/_chemical_probes_query
work against the current output/target schema instead of crashing on a
missing column.
"""

from pyspark.sql import Row
from pyspark.sql.types import (
    BooleanType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.target_prioritisation import (
    _chemical_probes_query,
    _ligand_pocket_query,
    _safety_query,
)

TRACTABILITY_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('modality', StringType()),
    StructField('id', StringType()),
    StructField('value', BooleanType()),
])

SAFETY_LIABILITY_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('event', StringType()),
])

CHEMICAL_PROBES_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('isHighQuality', BooleanType()),
])


def _queryset(spark, target_ids):
    return spark.createDataFrame([(t,) for t in target_ids], 'targetid STRING')


# ---------------------------------------------------------------------------
# _ligand_pocket_query
# ---------------------------------------------------------------------------


def test_ligand_pocket_query_flags_high_quality_ligand(spark):
    """A target with a True High-Quality Ligand assessment gets Nr_Ligand=1."""
    tractability = spark.createDataFrame(
        [Row(targetId='T1', modality='SM', id='High-Quality Ligand', value=True)],
        TRACTABILITY_SCHEMA,
    )
    result = _ligand_pocket_query(_queryset(spark, ['T1']), tractability)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_Ligand == 1
    assert row.Nr_Pocket == 0
    assert row.Nr_sMBinder == 0


def test_ligand_pocket_query_ignores_false_values(spark):
    """A False assessment does not flag the corresponding facet."""
    tractability = spark.createDataFrame(
        [Row(targetId='T1', modality='SM', id='High-Quality Pocket', value=False)],
        TRACTABILITY_SCHEMA,
    )
    result = _ligand_pocket_query(_queryset(spark, ['T1']), tractability)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_Pocket == 0


def test_ligand_pocket_query_target_with_no_rows_is_kept(spark):
    """A target absent from the tractability dataset is still present via the left join."""
    tractability = spark.createDataFrame([], TRACTABILITY_SCHEMA)
    result = _ligand_pocket_query(_queryset(spark, ['T1']), tractability)
    assert result.count() == 1


def test_ligand_pocket_query_ignores_unrelated_modality_ids(spark):
    """Assessment ids outside the three tracked categories don't produce extra pivot columns."""
    tractability = spark.createDataFrame(
        [
            Row(targetId='T1', modality='SM', id='High-Quality Ligand', value=True),
            Row(targetId='T1', modality='AB', id='Clinical Precedence', value=True),
        ],
        TRACTABILITY_SCHEMA,
    )
    result = _ligand_pocket_query(_queryset(spark, ['T1']), tractability)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_Ligand == 1
    assert 'Clinical Precedence' not in result.columns


# ---------------------------------------------------------------------------
# _safety_query
# ---------------------------------------------------------------------------


def test_safety_query_flags_targets_with_events(spark):
    """A target with at least one safety liability row gets hasSafetyEvent=Yes."""
    safety = spark.createDataFrame([Row(targetId='T1', event='hepatotoxicity')], SAFETY_LIABILITY_SCHEMA)
    result = _safety_query(_queryset(spark, ['T1']), safety)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.hasSafetyEvent == 'Yes'
    assert row.Nr_Event == -1


def test_safety_query_target_with_no_events_left_null(spark):
    """A target with no safety liability rows gets null hasSafetyEvent via the left join."""
    safety = spark.createDataFrame([], SAFETY_LIABILITY_SCHEMA)
    result = _safety_query(_queryset(spark, ['T1']), safety)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.hasSafetyEvent is None


def test_safety_query_counts_distinct_events(spark):
    """Multiple safety liability rows for one target are counted and collected."""
    safety = spark.createDataFrame(
        [
            Row(targetId='T1', event='hepatotoxicity'),
            Row(targetId='T1', event='cardiotoxicity'),
        ],
        SAFETY_LIABILITY_SCHEMA,
    )
    result = _safety_query(_queryset(spark, ['T1']), safety)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.nEvents == 2
    assert set(row.events) == {'hepatotoxicity', 'cardiotoxicity'}


# ---------------------------------------------------------------------------
# _chemical_probes_query
# ---------------------------------------------------------------------------


def test_chemical_probes_query_flags_high_quality(spark):
    """A target with a high-quality probe gets Nr_chprob=1."""
    probes = spark.createDataFrame([Row(targetId='T1', isHighQuality=True)], CHEMICAL_PROBES_SCHEMA)
    result = _chemical_probes_query(_queryset(spark, ['T1']), probes)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_chprob == 1


def test_chemical_probes_query_low_quality_only(spark):
    """A target with only non-high-quality probes gets Nr_chprob=0."""
    probes = spark.createDataFrame([Row(targetId='T1', isHighQuality=False)], CHEMICAL_PROBES_SCHEMA)
    result = _chemical_probes_query(_queryset(spark, ['T1']), probes)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_chprob == 0


def test_chemical_probes_query_target_with_no_probes_left_null(spark):
    """A target with no chemical probe rows gets null Nr_chprob via the left join."""
    probes = spark.createDataFrame([], CHEMICAL_PROBES_SCHEMA)
    result = _chemical_probes_query(_queryset(spark, ['T1']), probes)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_chprob is None


def test_chemical_probes_query_drops_unresolved_target_id(spark):
    """Probe rows with a null targetId (unresolved during ENSG lookup) are excluded."""
    probes = spark.createDataFrame([Row(targetId=None, isHighQuality=True)], CHEMICAL_PROBES_SCHEMA)
    result = _chemical_probes_query(_queryset(spark, ['T1']), probes)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_chprob is None
