"""Tests for the target_prioritisation pyspark module.

Covers the query functions that were migrated off output/target's removed
tractability, safetyLiabilities, chemicalProbes, and homologues fields onto
the standalone target_tractability, safety_liability, chemical_probes, and
homologues datasets, so that _ligand_pocket_query/_safety_query/
_chemical_probes_query/_paralogs_query/_orthologs_mouse_query work against
the current output/target schema instead of crashing on a missing column.
"""

from pyspark.sql import Row
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.target_prioritisation import (
    _chemical_probes_query,
    _ligand_pocket_query,
    _orthologs_mouse_query,
    _paralogs_query,
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

HOMOLOGUES_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('speciesId', StringType()),
    StructField('speciesName', StringType()),
    StructField('homologyType', StringType()),
    StructField('targetGeneId', StringType()),
    StructField('isHighConfidence', StringType()),
    StructField('targetGeneSymbol', StringType()),
    StructField('queryPercentageIdentity', DoubleType()),
    StructField('targetPercentageIdentity', DoubleType()),
    StructField('priority', IntegerType()),
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


# ---------------------------------------------------------------------------
# _paralogs_query
# ---------------------------------------------------------------------------


def _homologue_row(**kwargs):
    defaults = {
        'targetId': 'T1',
        'speciesId': '9606',
        'speciesName': 'Human',
        'homologyType': 'other_paralog',
        'targetGeneId': 'T2',
        'isHighConfidence': '1',
        'targetGeneSymbol': 'GENE2',
        'queryPercentageIdentity': 70.0,
        'targetPercentageIdentity': 65.0,
        'priority': 0,
    }
    defaults.update(kwargs)
    return Row(**defaults)


def test_paralogs_query_scores_high_identity_paralog(spark):
    """A paralog with queryPercentageIdentity above 60 produces a negative Nr_paralogs score."""
    homologues = spark.createDataFrame(
        [_homologue_row(homologyType='other_paralog', queryPercentageIdentity=80.0)], HOMOLOGUES_SCHEMA
    )
    result = _paralogs_query(_queryset(spark, ['T1']), homologues)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_paralogs == -0.5


def test_paralogs_query_low_identity_paralog_scores_zero(spark):
    """A paralog with queryPercentageIdentity below 60 produces Nr_paralogs=0."""
    homologues = spark.createDataFrame(
        [_homologue_row(homologyType='within_species_paralog', queryPercentageIdentity=40.0)], HOMOLOGUES_SCHEMA
    )
    result = _paralogs_query(_queryset(spark, ['T1']), homologues)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_paralogs == 0


def test_paralogs_query_ignores_orthologs(spark):
    """A non-paralog homologyType (ortholog) is excluded from the paralog score."""
    homologues = spark.createDataFrame(
        [_homologue_row(homologyType='ortholog_one2one', queryPercentageIdentity=95.0)], HOMOLOGUES_SCHEMA
    )
    result = _paralogs_query(_queryset(spark, ['T1']), homologues)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_paralogs is None


def test_paralogs_query_target_with_no_rows_is_kept(spark):
    """A target absent from the homologues dataset is still present via the left join."""
    homologues = spark.createDataFrame([], HOMOLOGUES_SCHEMA)
    result = _paralogs_query(_queryset(spark, ['T1']), homologues)
    assert result.count() == 1


# ---------------------------------------------------------------------------
# _orthologs_mouse_query
# ---------------------------------------------------------------------------


def test_orthologs_mouse_query_scores_high_identity_mouse_ortholog(spark):
    """A mouse ortholog with queryPercentageIdentity above 80 produces a positive Nr_ortholog score."""
    homologues = spark.createDataFrame(
        [
            _homologue_row(
                homologyType='ortholog_one2one',
                speciesName='Mouse',
                queryPercentageIdentity=90.0,
            )
        ],
        HOMOLOGUES_SCHEMA,
    )
    result = _orthologs_mouse_query(_queryset(spark, ['T1']), homologues)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_ortholog == 0.5


def test_orthologs_mouse_query_ignores_non_mouse_species(spark):
    """An ortholog in a species other than mouse is excluded."""
    homologues = spark.createDataFrame(
        [
            _homologue_row(
                homologyType='ortholog_one2one',
                speciesName='Zebrafish',
                queryPercentageIdentity=95.0,
            )
        ],
        HOMOLOGUES_SCHEMA,
    )
    result = _orthologs_mouse_query(_queryset(spark, ['T1']), homologues)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_ortholog is None


def test_orthologs_mouse_query_ignores_mouse_paralogs(spark):
    """A mouse row that is a paralog rather than an ortholog is excluded."""
    homologues = spark.createDataFrame(
        [
            _homologue_row(
                homologyType='other_paralog',
                speciesName='Mouse',
                queryPercentageIdentity=95.0,
            )
        ],
        HOMOLOGUES_SCHEMA,
    )
    result = _orthologs_mouse_query(_queryset(spark, ['T1']), homologues)
    row = result.filter('targetid = "T1"').first()
    assert row is not None
    assert row.Nr_ortholog is None


def test_orthologs_mouse_query_target_with_no_rows_is_kept(spark):
    """A target absent from the homologues dataset is still present via the left join."""
    homologues = spark.createDataFrame([], HOMOLOGUES_SCHEMA)
    result = _orthologs_mouse_query(_queryset(spark, ['T1']), homologues)
    assert result.count() == 1
