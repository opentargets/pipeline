"""Tests for the safety_liability PySpark module."""

import pyspark.sql.functions as f
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
    _harmonize_safety_evidence,
    clean_phenotype_to_describe_safety_event,
    process_adverse_events,
    process_toxcast,
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

ADVERSE_EVENTS_SCHEMA = StructType([
    StructField('biologicalSystem', StringType()),
    StructField('effect', StringType()),
    StructField('efoId', StringType()),
    StructField('ensemblId', StringType()),
    StructField('pmid', StringType()),
    StructField('ref', StringType()),
    StructField('symptom', StringType()),
    StructField('target', StringType()),
    StructField('uberonCode', StringType()),
    StructField('url', StringType()),
])

TOXCAST_SCHEMA = StructType([
    StructField('assay_component_endpoint_name', StringType()),
    StructField('assay_component_desc', StringType()),
    StructField('biological_process_target', StringType()),
    StructField('tissue', StringType()),
    StructField('cell_format', StringType()),
    StructField('cell_short_name', StringType()),
    StructField('assay_format_type', StringType()),
    StructField('official_symbol', StringType()),
    StructField('eventId', StringType()),
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
# process_* source harmonizers and _harmonize_safety_evidence
# ---------------------------------------------------------------------------


def test_process_adverse_events_maps_columns(spark):
    """Raw adverse-events columns are renamed/restructured into the common evidence schema."""
    row = Row(
        biologicalSystem='gastrointestinal', effect='activation_general', efoId='EFO_0009836',
        ensemblId='ENSG00000133019', pmid='23197038', ref='Bowes et al. (2012)',
        symptom='bronchoconstriction', target='CHRM3', uberonCode='UBERON_0005409', url=None,
    )
    result = process_adverse_events(spark.createDataFrame([row], ADVERSE_EVENTS_SCHEMA))
    out = result.first()
    assert out is not None
    assert out.id == 'ENSG00000133019'
    assert out.event == 'bronchoconstriction'
    assert out.eventId == 'EFO_0009836'
    assert out.datasource == 'Bowes et al. (2012)'
    assert out.effects[0].direction == 'Activation/Increase/Upregulation'


def test_process_toxcast_maps_columns(spark):
    """ToxCast rows are keyed by targetFromSourceId (symbol), not an ENSG id."""
    row = Row(
        assay_component_endpoint_name='ACEA_ER_80hr', assay_component_desc='some assay',
        biological_process_target='cell proliferation', tissue=None, cell_format='cell line',
        cell_short_name='T47D', assay_format_type='cell-based', official_symbol=' ESR1 ', eventId=None,
    )
    result = process_toxcast(spark.createDataFrame([row], TOXCAST_SCHEMA))
    out = result.first()
    assert out is not None
    assert out.targetFromSourceId == 'ESR1'
    assert out.event == 'cell proliferation'
    assert out.datasource == 'ToxCast'


def test_clean_phenotype_maps_toxicity_to_drug_toxicity(spark):
    """The phenotype-cleaning column expression normalizes known phrases to a fixed vocabulary."""
    schema = StructType([StructField('phenotypeText', StringType())])
    df = spark.createDataFrame([Row(phenotypeText='toxicity')], schema)
    result = df.withColumn('cleaned', clean_phenotype_to_describe_safety_event(f.col('phenotypeText')))
    assert result.first().cleaned == 'drug toxicity'


def test_harmonize_safety_evidence_dedupes_identical_source_rows(spark):
    """Two identical raw adverse-event rows collapse into a single harmonized record."""
    row = Row(
        biologicalSystem='gastrointestinal', effect='activation_general', efoId='EFO_0009836',
        ensemblId='ENSG00000133019', pmid='23197038', ref='Bowes et al. (2012)',
        symptom='bronchoconstriction', target='CHRM3', uberonCode='UBERON_0005409', url=None,
    )
    processed = process_adverse_events(spark.createDataFrame([row, row], ADVERSE_EVENTS_SCHEMA))
    # An empty (but fully-columned) second source lets unionByName(allowMissingColumns=True)
    # fill in columns process_adverse_events doesn't produce (e.g. targetFromSourceId),
    # mirroring how the real pipeline unions six heterogeneous sources together.
    empty_toxcast = process_toxcast(spark.createDataFrame([], TOXCAST_SCHEMA))
    result = _harmonize_safety_evidence([processed, empty_toxcast])
    assert result.count() == 1


# ---------------------------------------------------------------------------
# _build_ensg_lookup
# ---------------------------------------------------------------------------


def test_build_ensg_lookup_includes_approved_symbol(spark):
    """ApprovedSymbol appears in the name array."""
    rows = [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[Row(id='P12345', source='uniprot')])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.filter('ensgId = "ENSG00000001"').first()
    assert row is not None
    assert 'GENE1' in row.name


def test_build_ensg_lookup_includes_protein_id(spark):
    """Protein accession IDs appear in the name array."""
    rows = [Row(id='ENSG00000001', approvedSymbol='GENE1', proteinIds=[Row(id='P12345', source='uniprot')])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.filter('ensgId = "ENSG00000001"').first()
    assert row is not None
    assert 'P12345' in row.name


def test_build_ensg_lookup_handles_empty_protein_ids(spark):
    """Empty proteinIds array does not cause an error; symbol still in name."""
    rows = [Row(id='ENSG00000002', approvedSymbol='GENE2', proteinIds=[])]
    lut = _build_ensg_lookup(spark.createDataFrame(rows, TARGET_SCHEMA))
    row = lut.filter('ensgId = "ENSG00000002"').first()
    assert row is not None
    assert 'GENE2' in row.name


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
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
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
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
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
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG00000001'


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
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG00000002'


def test_build_safety_liabilities_resolves_symbol_for_non_coding_gene(spark):
    """ToxCast rows for non-coding genes (null proteinIds) still resolve by symbol.

    Regression test for the MIR122 bug: a non-coding target has proteinIds=None
    rather than [], which previously wiped out the whole ensg_lookup name array
    (including approvedSymbol) via flatten(array(...)), dropping this event.
    """
    safety = spark.createDataFrame([
        _safety_row(ensg=None, source_id='MIR122', datasource='ToxCast'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame([
        Row(id='ENSG_MIR122', approvedSymbol='MIR122', proteinIds=None),
    ], TARGET_SCHEMA)
    diseases = spark.createDataFrame(
        [Row(id='EFO_9999', obsoleteTerms=[])], DISEASE_SCHEMA
    )
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG_MIR122'


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
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
    assert result.count() == 0


def test_build_safety_liabilities_drops_ids_not_in_target(spark):
    """Rows whose id doesn't match any target in output/target are dropped, even if non-null."""
    safety = spark.createDataFrame([
        _safety_row(ensg='ENSG00000001', event='hepatotoxicity'),
        _safety_row(ensg='ENSG_STALE_ID', event='cardiotoxicity'),
    ], SAFETY_SCHEMA)
    target = spark.createDataFrame(
        [Row(id='ENSG00000001', approvedSymbol='G1', proteinIds=[])], TARGET_SCHEMA
    )
    diseases = spark.createDataFrame(
        [Row(id='EFO_9999', obsoleteTerms=[])], DISEASE_SCHEMA
    )
    ensg_lut = _build_ensg_lookup(target)
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG00000001'


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
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
    row = result.first()
    assert row is not None
    assert row.eventId == 'EFO_CURRENT'


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
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
    row = result.first()
    assert row is not None
    assert row.eventId == 'EFO_CURRENT'


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
    result = _build_safety_liabilities(safety, ensg_lut, diseases, target)
    assert result.count() == 2
