"""Tests for the safety_liability PySpark module."""

from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.safety_liability import (
    _build_ensg_lookup,
    _build_safety_liabilities,
)

# ---------------------------------------------------------------------------
# Shared schemas and helpers
# ---------------------------------------------------------------------------

SAFETY_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('targetFromSourceId', StringType()),
    StructField('event', StringType()),
    StructField('eventId', StringType()),
    StructField('datasource', StringType()),
    StructField('effects', ArrayType(StructType([
        StructField('direction', StringType()),
        StructField('dosing', StringType()),
    ]))),
    StructField('literature', StringType()),
    StructField('url', StringType()),
    StructField('biosamples', ArrayType(StructType([
        StructField('tissueLabel', StringType()),
        StructField('tissueId', StringType()),
        StructField('cellLabel', StringType()),
        StructField('cellFormat', StringType()),
        StructField('cellId', StringType()),
    ]))),
    StructField('studies', ArrayType(StructType([
        StructField('description', StringType()),
        StructField('name', StringType()),
        StructField('type', StringType()),
    ]))),
])

TARGET_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('approvedSymbol', StringType()),
    StructField('proteinIds', ArrayType(StructType([
        StructField('id', StringType()),
        StructField('source', StringType()),
    ]))),
])

DISEASE_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('obsoleteTerms', ArrayType(StringType())),
])


def _safety_row(*, ensg=None, source_id=None, event='hepatotoxicity', event_id='EFO_0001234',
                datasource='AstraZeneca', literature=None, url=None):
    return Row(
        id=ensg,
        targetFromSourceId=source_id,
        event=event,
        eventId=event_id,
        datasource=datasource,
        effects=None,
        literature=literature,
        url=url,
        biosamples=None,
        studies=None,
    )


# ---------------------------------------------------------------------------
# _build_ensg_lookup
# ---------------------------------------------------------------------------


def test_build_ensg_lookup_includes_approved_symbol(spark):
    """ApprovedSymbol appears in the name array."""
    rows = [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[Row(id='P12345', source='uniprot')])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.filter('ensgId = "ENSG00000001"').first()
    assert 'GENE1' in row.name


def test_build_ensg_lookup_includes_protein_id(spark):
    """Protein accession IDs appear in the name array."""
    rows = [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[Row(id='P12345', source='uniprot')])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.filter('ensgId = "ENSG00000001"').first()
    assert 'P12345' in row.name


def test_build_ensg_lookup_handles_empty_protein_ids(spark):
    """Empty proteinIds array does not cause an error; symbol still in name."""
    rows = [Row(id='ENSG00000002', approvedSymbol='GENE2', proteinIds=[])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.filter('ensgId = "ENSG00000002"').first()
    assert 'GENE2' in row.name


def test_build_ensg_lookup_output_columns(spark):
    """Output has exactly ensgId and name columns."""
    rows = [Row(id='ENSG00000003', approvedSymbol='G3', proteinIds=[])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    assert set(lut.columns) == {'ensgId', 'name'}


# ---------------------------------------------------------------------------
# _build_safety_liabilities
# ---------------------------------------------------------------------------


def test_build_safety_liabilities_output_columns(spark):
    """Output has one row per liability record with flat columns."""
    safety = spark.createDataFrame(
        [_safety_row(ensg='ENSG00000001')], SAFETY_SCHEMA
    )
    target = spark.createDataFrame(
        [Row(id='ENSG00000001', approvedSymbol='G1', proteinIds=[])], TARGET_SCHEMA
    )
    diseases = spark.createDataFrame(
        [Row(id='EFO_9999', obsoleteTerms=[])], DISEASE_SCHEMA
    )
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases)
    assert set(result.columns) == {
        'targetId', 'event', 'eventId', 'effects', 'biosamples',
        'datasource', 'literature', 'url', 'studies',
    }


def test_build_safety_liabilities_one_row_per_liability(spark):
    """Multiple liabilities for the same target produce separate rows."""
    safety = spark.createDataFrame([
        _safety_row(ensg='ENSG00000001', event='hepatotoxicity', event_id='EFO_0001'),
        _safety_row(ensg='ENSG00000001', event='cardiotoxicity', event_id='EFO_0002'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame(
        [Row(id='ENSG00000001', approvedSymbol='G1', proteinIds=[])], TARGET_SCHEMA
    )
    diseases = spark.createDataFrame(
        [Row(id='EFO_9999', obsoleteTerms=[])], DISEASE_SCHEMA
    )
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases)
    assert result.count() == 2
    assert result.filter('targetId = "ENSG00000001"').count() == 2


def test_build_safety_liabilities_resolves_symbol_to_ensg(spark):
    """ToxCast rows with null id but known symbol get their ENSG resolved."""
    safety = spark.createDataFrame([
        _safety_row(ensg=None, source_id='GENE1', datasource='ToxCast'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame([
        Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[]),
    ], TARGET_SCHEMA)
    diseases = spark.createDataFrame(
        [Row(id='EFO_9999', obsoleteTerms=[])], DISEASE_SCHEMA
    )
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases)
    assert result.count() == 1
    assert result.first().targetId == 'ENSG00000001'


def test_build_safety_liabilities_resolves_protein_id_to_ensg(spark):
    """Entries keyed by protein accession are resolved via proteinIds lookup."""
    safety = spark.createDataFrame([
        _safety_row(ensg=None, source_id='P12345', datasource='AstraZeneca'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame([
        Row(id='ENSG00000002', approvedSymbol='GENE2', proteinIds=[Row(id='P12345', source='uniprot')]),
    ], TARGET_SCHEMA)
    diseases = spark.createDataFrame(
        [Row(id='EFO_9999', obsoleteTerms=[])], DISEASE_SCHEMA
    )
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases)
    assert result.count() == 1
    assert result.first().targetId == 'ENSG00000002'


def test_build_safety_liabilities_drops_unresolvable_rows(spark):
    """Rows with null id that cannot be resolved to an ENSG are dropped."""
    safety = spark.createDataFrame([
        _safety_row(ensg=None, source_id='UNKNOWN_SYMBOL', datasource='ToxCast'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame([
        Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[]),
    ], TARGET_SCHEMA)
    diseases = spark.createDataFrame(
        [Row(id='EFO_9999', obsoleteTerms=[])], DISEASE_SCHEMA
    )
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases)
    assert result.count() == 0


def test_build_safety_liabilities_remaps_obsolete_efo(spark):
    """Obsolete EFO disease IDs in eventId are replaced with current IDs."""
    safety = spark.createDataFrame([
        _safety_row(ensg='ENSG00000001', event_id='EFO_OBSOLETE'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame([
        Row(id='ENSG00000001', approvedSymbol='G1', proteinIds=[]),
    ], TARGET_SCHEMA)
    diseases = spark.createDataFrame([
        Row(id='EFO_CURRENT', obsoleteTerms=['EFO_OBSOLETE']),
    ], DISEASE_SCHEMA)
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases)
    assert result.first().eventId == 'EFO_CURRENT'


def test_build_safety_liabilities_keeps_current_efo_unchanged(spark):
    """EFO IDs that are not obsolete are left untouched."""
    safety = spark.createDataFrame([
        _safety_row(ensg='ENSG00000001', event_id='EFO_CURRENT'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame([
        Row(id='ENSG00000001', approvedSymbol='G1', proteinIds=[]),
    ], TARGET_SCHEMA)
    diseases = spark.createDataFrame([
        Row(id='EFO_OTHER', obsoleteTerms=['EFO_SOMETHINGELSE']),
    ], DISEASE_SCHEMA)
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases)
    assert result.first().eventId == 'EFO_CURRENT'


def test_build_safety_liabilities_multiple_targets(spark):
    """Liabilities for different targets produce separate rows."""
    safety = spark.createDataFrame([
        _safety_row(ensg='ENSG00000001', event='hepatotoxicity'),
        _safety_row(ensg='ENSG00000002', event='cardiotoxicity'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame([
        Row(id='ENSG00000001', approvedSymbol='G1', proteinIds=[]),
        Row(id='ENSG00000002', approvedSymbol='G2', proteinIds=[]),
    ], TARGET_SCHEMA)
    diseases = spark.createDataFrame(
        [Row(id='EFO_9999', obsoleteTerms=[])], DISEASE_SCHEMA
    )
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases)
    assert result.count() == 2
