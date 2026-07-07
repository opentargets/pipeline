"""Tests for the ENSG resolution functions in the chemical_probes module."""

from pyspark.sql import Row
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from pts.pyspark.chemical_probes import _build_ensg_lookup, _resolve_targets

# ---------------------------------------------------------------------------
# Shared schemas and helpers
# ---------------------------------------------------------------------------

TARGET_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('approvedSymbol', StringType()),
    StructField('proteinIds', ArrayType(StructType([
        StructField('id', StringType()),
        StructField('source', StringType()),
    ]))),
])

EVIDENCE_SCHEMA = StructType([
    StructField('targetFromSourceId', StringType()),
    StructField('id', StringType()),
    StructField('drugFromSourceId', StringType()),
    StructField('drugId', StringType()),
])


def _target_row(ensg, symbol, protein_ids=None):
    return Row(
        id=ensg,
        approvedSymbol=symbol,
        proteinIds=[Row(id=p, source='uniprot') for p in (protein_ids or [])],
    )


def _evidence_row(source_id, compound='probe1', drug_source='CP001', drug_id=None):
    return Row(
        targetFromSourceId=source_id,
        id=compound,
        drugFromSourceId=drug_source,
        drugId=drug_id,
    )


# ---------------------------------------------------------------------------
# _build_ensg_lookup
# ---------------------------------------------------------------------------


def test_build_ensg_lookup_output_columns(spark):
    """Output has exactly ensgId and name columns."""
    rows = [_target_row('ENSG00000001', 'GENE1')]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    assert set(lut.columns) == {'ensgId', 'name'}


def test_build_ensg_lookup_includes_symbol(spark):
    """ApprovedSymbol appears in the name array."""
    rows = [_target_row('ENSG00000001', 'GENE1')]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.first()
    assert row is not None
    assert 'GENE1' in row.name


def test_build_ensg_lookup_includes_protein_id(spark):
    """Protein accession IDs appear in the name array."""
    rows = [_target_row('ENSG00000001', 'GENE1', protein_ids=['P12345'])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.first()
    assert row is not None
    assert 'P12345' in row.name


def test_build_ensg_lookup_handles_empty_protein_ids(spark):
    """Empty proteinIds does not cause an error; symbol still present."""
    rows = [_target_row('ENSG00000002', 'GENE2', protein_ids=[])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.first()
    assert row is not None
    assert 'GENE2' in row.name


# ---------------------------------------------------------------------------
# _resolve_targets
# ---------------------------------------------------------------------------


def test_resolve_targets_output_has_target_id(spark):
    """Output contains a targetId column."""
    evidence = spark.createDataFrame([_evidence_row('GENE1')], EVIDENCE_SCHEMA)
    target = spark.createDataFrame([_target_row('ENSG00000001', 'GENE1')], TARGET_SCHEMA)
    lut = _build_ensg_lookup(target)
    result = _resolve_targets(evidence, lut)
    assert 'targetId' in result.columns


def test_resolve_targets_resolves_symbol_to_ensg(spark):
    """TargetFromSourceId matching an approvedSymbol maps to the correct ENSG."""
    evidence = spark.createDataFrame([_evidence_row('GENE1')], EVIDENCE_SCHEMA)
    target = spark.createDataFrame([_target_row('ENSG00000001', 'GENE1')], TARGET_SCHEMA)
    lut = _build_ensg_lookup(target)
    result = _resolve_targets(evidence, lut)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG00000001'


def test_resolve_targets_resolves_protein_id_to_ensg(spark):
    """TargetFromSourceId matching a protein accession maps to the correct ENSG."""
    evidence = spark.createDataFrame([_evidence_row('P12345')], EVIDENCE_SCHEMA)
    target = spark.createDataFrame([_target_row('ENSG00000002', 'GENE2', protein_ids=['P12345'])], TARGET_SCHEMA)
    lut = _build_ensg_lookup(target)
    result = _resolve_targets(evidence, lut)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG00000002'


def test_resolve_targets_keeps_unresolvable_rows_with_null_target_id(spark):
    """Rows whose targetFromSourceId matches no target are kept with a null targetId.

    drugId/drugFromSourceId on this dataset are consumed independently of
    target resolution (see drug_molecule), so unresolvable rows must not be
    dropped here.
    """
    evidence = spark.createDataFrame([_evidence_row('UNKNOWN')], EVIDENCE_SCHEMA)
    target = spark.createDataFrame([_target_row('ENSG00000001', 'GENE1')], TARGET_SCHEMA)
    lut = _build_ensg_lookup(target)
    result = _resolve_targets(evidence, lut)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId is None


def test_resolve_targets_retains_target_from_source_id(spark):
    """TargetFromSourceId is preserved alongside the resolved targetId."""
    evidence = spark.createDataFrame([_evidence_row('GENE1')], EVIDENCE_SCHEMA)
    target = spark.createDataFrame([_target_row('ENSG00000001', 'GENE1')], TARGET_SCHEMA)
    lut = _build_ensg_lookup(target)
    result = _resolve_targets(evidence, lut)
    assert 'targetFromSourceId' in result.columns
    row = result.first()
    assert row is not None
    assert row.targetFromSourceId == 'GENE1'


def test_resolve_targets_multiple_probes_same_target(spark):
    """Multiple probes for the same target each produce their own row."""
    evidence = spark.createDataFrame([
        _evidence_row('GENE1', compound='probe_a'),
        _evidence_row('GENE1', compound='probe_b'),
    ], EVIDENCE_SCHEMA)
    target = spark.createDataFrame([_target_row('ENSG00000001', 'GENE1')], TARGET_SCHEMA)
    lut = _build_ensg_lookup(target)
    result = _resolve_targets(evidence, lut)
    assert result.count() == 2
    assert result.filter('targetId = "ENSG00000001"').count() == 2
