"""Tests for the gene_essentiality PySpark module."""

from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.gene_essentiality import (
    _build_ensg_lookup,
    _resolve_target_ids,
)

# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

TARGET_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('approvedSymbol', StringType()),
    StructField('proteinIds', ArrayType(StructType([
        StructField('id', StringType()),
        StructField('source', StringType()),
    ]))),
])

ESSENTIALITY_SCHEMA = StructType([
    StructField('targetSymbol', StringType()),
    StructField('isEssential', BooleanType()),
    StructField('depMapEssentiality', ArrayType(StringType())),
])


# ---------------------------------------------------------------------------
# _build_ensg_lookup
# ---------------------------------------------------------------------------


def test_build_ensg_lookup_includes_approved_symbol(spark):
    """ApprovedSymbol appears in the name array."""
    rows = [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.filter('ensgId = "ENSG00000001"').first()
    assert row is not None
    assert 'GENE1' in row.name


def test_build_ensg_lookup_output_columns(spark):
    """Output has exactly ensgId and name columns."""
    rows = [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    assert set(lut.columns) == {'ensgId', 'name'}


def test_build_ensg_lookup_handles_null_protein_ids(spark):
    """Null proteinIds (non-coding genes, e.g. microRNAs) does not wipe out the symbol.

    Regression test: flatten(array(proteinIds.id, [approvedSymbol])) returns NULL for
    the whole array if proteinIds.id is NULL rather than an empty array, silently
    dropping approvedSymbol too and breaking symbol-based ENSG resolution for every
    non-coding gene.
    """
    rows = [Row(id='ENSG00000003', approvedSymbol='MIR122', proteinIds=None)]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.filter('ensgId = "ENSG00000003"').first()
    assert row is not None
    assert row.name == ['MIR122']


# ---------------------------------------------------------------------------
# _resolve_target_ids
# ---------------------------------------------------------------------------


def test_resolve_target_ids_output_columns(spark):
    """Output has exactly targetId, isEssential and depMapEssentiality columns."""
    essentiality = spark.createDataFrame(
        [Row(targetSymbol='GENE1', isEssential=True, depMapEssentiality=[])], ESSENTIALITY_SCHEMA
    )
    target = spark.createDataFrame(
        [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[])], TARGET_SCHEMA
    )
    lut = _build_ensg_lookup(target)
    result = _resolve_target_ids(essentiality, lut)
    assert set(result.columns) == {'targetId', 'isEssential', 'depMapEssentiality'}


def test_resolve_target_ids_resolves_symbol_to_ensg(spark):
    """A targetSymbol matching an approvedSymbol is resolved to its ENSG id."""
    essentiality = spark.createDataFrame(
        [Row(targetSymbol='GENE1', isEssential=True, depMapEssentiality=[])], ESSENTIALITY_SCHEMA
    )
    target = spark.createDataFrame(
        [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[])], TARGET_SCHEMA
    )
    lut = _build_ensg_lookup(target)
    result = _resolve_target_ids(essentiality, lut)
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG00000001'


def test_resolve_target_ids_drops_unresolvable_rows(spark):
    """A targetSymbol absent from output/target is dropped (validation)."""
    essentiality = spark.createDataFrame(
        [Row(targetSymbol='UNKNOWN_SYMBOL', isEssential=True, depMapEssentiality=[])], ESSENTIALITY_SCHEMA
    )
    target = spark.createDataFrame(
        [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[])], TARGET_SCHEMA
    )
    lut = _build_ensg_lookup(target)
    result = _resolve_target_ids(essentiality, lut)
    assert result.count() == 0


def test_resolve_target_ids_resolves_symbol_for_non_coding_gene(spark):
    """Essentiality rows for non-coding genes (null proteinIds) still resolve by symbol.

    Regression test for the MIR122-class bug: a non-coding target has
    proteinIds=None rather than [], which previously wiped out the whole
    ensg_lookup name array (including approvedSymbol) via flatten(array(...)),
    dropping this entry.
    """
    essentiality = spark.createDataFrame(
        [Row(targetSymbol='MIR122', isEssential=True, depMapEssentiality=[])], ESSENTIALITY_SCHEMA
    )
    target = spark.createDataFrame(
        [Row(id='ENSG_MIR122', approvedSymbol='MIR122', proteinIds=None)], TARGET_SCHEMA
    )
    lut = _build_ensg_lookup(target)
    result = _resolve_target_ids(essentiality, lut)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG_MIR122'


def test_resolve_target_ids_merges_multiple_entries_per_target(spark):
    """Multiple essentiality rows resolving to the same target are merged into one row."""
    essentiality = spark.createDataFrame([
        Row(targetSymbol='GENE1', isEssential=True, depMapEssentiality=['a']),
        Row(targetSymbol='GENE1', isEssential=False, depMapEssentiality=['b']),
    ], ESSENTIALITY_SCHEMA)
    target = spark.createDataFrame(
        [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[])], TARGET_SCHEMA
    )
    lut = _build_ensg_lookup(target)
    result = _resolve_target_ids(essentiality, lut)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.isEssential is True
    assert set(row.depMapEssentiality) == {'a', 'b'}
