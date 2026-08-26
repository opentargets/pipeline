"""Tests for the chemical_probes module."""

import inspect
from pathlib import Path

from pyspark.sql import Row
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

from pts.pyspark.chemical_probes import (
    PROBES_SETS,
    _build_ensg_lookup,
    _resolve_targets,
    collapse_cols_data_in_array,
    generate_chemical_probes_evidence,
)

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


def test_resolve_targets_drops_unresolvable_rows(spark):
    """Rows whose targetFromSourceId matches no target are dropped (validation)."""
    evidence = spark.createDataFrame([_evidence_row('UNKNOWN')], EVIDENCE_SCHEMA)
    target = spark.createDataFrame([_target_row('ENSG00000001', 'GENE1')], TARGET_SCHEMA)
    lut = _build_ensg_lookup(target)
    result = _resolve_targets(evidence, lut)
    assert result.count() == 0


def test_resolve_targets_resolves_symbol_for_non_coding_gene(spark):
    """Probes for non-coding genes (null proteinIds) still resolve by symbol.

    Regression test for the MIR122-class bug: a non-coding target has
    proteinIds=None rather than [], which previously wiped out the whole
    ensg_lookup name array (including approvedSymbol) via flatten(array(...)),
    dropping this probe.
    """
    evidence = spark.createDataFrame([_evidence_row('MIR122')], EVIDENCE_SCHEMA)
    target = spark.createDataFrame(
        [Row(id='ENSG_MIR122', approvedSymbol='MIR122', proteinIds=None)], TARGET_SCHEMA
    )
    lut = _build_ensg_lookup(target)
    result = _resolve_targets(evidence, lut)
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.targetId == 'ENSG_MIR122'


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


# ---------------------------------------------------------------------------
# collapse_cols_data_in_array / PROBES_SETS
# ---------------------------------------------------------------------------

# One-hot datasource columns of the PROBES sheet, read from the upstream Probes &
# Drugs spreadsheet used by the 2026.09 release
# (input/target/chemicalprobes/probes.xlsx). Kept verbatim so that a name drifting
# upstream shows up here as a failure rather than as an UNRESOLVED_COLUMN crash in
# production.
PROBES_SHEET_DATASOURCE_COLUMNS = [
    'Bromodomains chemical toolbox',
    'Chemical Probes for Understudied Kinases',
    'Chemical Probes.org',
    'Gray Laboratory Probes',
    'High-quality chemical probes',
    'MLP Probes',
    'Nature Chemical Biology Probes',
    'Open Science Probes',
    'opnMe Portal',
    'Protein methyltransferases chemical toolbox',
    'SGC Probes',
    'A Collection of Useful Nuisance Compounds (CONS) for Interrogation of Bioassay Integrity',
    'Concise Guide to Pharmacology 2025/26',
    'Kinase Chemogenomic Set (KCGS)',
    'Kinase Inhibitors (best-in-class)',
    'Novartis Chemogenetic Library (NIBR MoA Box)',
    'Nuisance compounds in cellular assays',
]


def _probes_sheet_df(spark, memberships):
    """Build a PROBES-sheet-shaped frame with the real upstream column names.

    Args:
        spark: Spark session.
        memberships: Datasource column names the single probe row belongs to.

    Returns:
        DataFrame with one row, ``pdid`` plus every one-hot datasource column.
    """
    schema = StructType(
        [StructField('pdid', StringType())]
        + [StructField(c, DoubleType()) for c in PROBES_SHEET_DATASOURCE_COLUMNS]
    )
    row = [1.0 if c in memberships else None for c in PROBES_SHEET_DATASOURCE_COLUMNS]
    return spark.createDataFrame([['PD000001', *row]], schema)


def test_probes_sets_are_all_columns_of_the_probes_sheet(spark):
    """Every PROBES_SETS entry resolves against the upstream PROBES sheet columns.

    Regression test for upstream header drift: 'Probe Miner (suitable probes)' and
    'Tool Compound Set' were withdrawn and 'Concise Guide to Pharmacology 2019/20'
    was renamed to the 2025/26 edition, so the stale list made
    collapse_cols_data_in_array raise UNRESOLVED_COLUMN and failed the whole step.
    """
    df = _probes_sheet_df(spark, memberships=['SGC Probes'])
    result = collapse_cols_data_in_array(df, PROBES_SETS, 'datasourceIds')
    row = result.first()
    assert row is not None
    assert row.datasourceIds == ['SGC Probes']


def test_collapse_cols_data_in_array_collects_every_membership(spark):
    """A probe in several sets collects all of them, and only them."""
    memberships = ['Chemical Probes.org', 'High-quality chemical probes', 'Concise Guide to Pharmacology 2025/26']
    df = _probes_sheet_df(spark, memberships=memberships)
    result = collapse_cols_data_in_array(df, PROBES_SETS, 'datasourceIds')
    row = result.first()
    assert row is not None
    assert sorted(row.datasourceIds) == sorted(memberships)


def test_the_cons_set_is_collected_as_its_own_datasource(spark):
    """CONS is adopted as a datasourceId, and is not a rename of the older nuisance set.

    'A Collection of Useful Nuisance Compounds (CONS) for Interrogation of Bioassay
    Integrity' is new in the 01_2026 export. The pre-existing 'Nuisance compounds in
    cellular assays' column is still present upstream and holds different compounds, so
    both belong in PROBES_SETS and a probe can be in one, the other, or both.

    This adds a datasourceId that did not appear in previous releases, which is why it
    is pinned here rather than left to the length of the list.
    """
    cons = 'A Collection of Useful Nuisance Compounds (CONS) for Interrogation of Bioassay Integrity'
    older = 'Nuisance compounds in cellular assays'
    assert cons in PROBES_SHEET_DATASOURCE_COLUMNS
    assert cons in PROBES_SETS
    assert older in PROBES_SETS, 'adopting CONS must not displace the set it sits alongside'

    df = _probes_sheet_df(spark, memberships=[cons])
    row = collapse_cols_data_in_array(df, PROBES_SETS, 'datasourceIds').first()
    assert row is not None
    assert row.datasourceIds == [cons]

    both = _probes_sheet_df(spark, memberships=[cons, older])
    row_both = collapse_cols_data_in_array(both, PROBES_SETS, 'datasourceIds').first()
    assert row_both is not None
    assert sorted(row_both.datasourceIds) == sorted([cons, older]), (
        'the two nuisance sets must be collected independently, not collapsed into one'
    )


def test_probe_miner_score_is_not_produced():
    """probeMinerScore was removed from the dataset, not nulled.

    Probe Miner was retired upstream: neither the PROBES sheet nor PROBES TARGETS
    carries a Probe Miner column any more, so the field can only ever be null. It is
    dropped from `target.chemicalProbes` rather than published as a column that is
    permanently empty.

    This pins the removal at the point a reader would notice it — the grouping columns
    that shape the output — because reinstating it there is exactly how it would come
    back. `croissant`'s chemical_probes recordset and `target_view._build_chemical_probes`
    must stay in step with this; all three were changed together.
    """
    source = Path(inspect.getfile(generate_chemical_probes_evidence)).read_text()
    grouping = source.split('grouping_cols = [', 1)[1].split(']', 1)[0]
    assert 'probeMinerScore' not in grouping, (
        'probeMinerScore is back in the chemical probes output. It has no upstream '
        'source, so it can only be null; if it is genuinely returning, croissant and '
        'target_view need the column back too.'
    )
