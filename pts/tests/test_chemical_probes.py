"""Tests for the chemical_probes module."""

import pandas as pd
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

from pts.pyspark.chemical_probes import (
    PROBES_SETS,
    _build_ensg_lookup,
    _resolve_targets,
    collapse_cols_data_in_array,
    process_probes_targets_data,
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

    Regression test for header drift: an entry naming a column the spreadsheet no
    longer has makes collapse_cols_data_in_array raise UNRESOLVED_COLUMN and fails the
    whole step, on Dataproc, after cluster spin-up.
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
    """CONS is a set of its own, not a rename of 'Nuisance compounds in cellular assays'.

    Both columns exist upstream and their memberships do not overlap, so a probe can be
    in either or both. Pinned because the two names invite being collapsed into one.
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


# ---------------------------------------------------------------------------
# PROBES TARGETS sheet
# ---------------------------------------------------------------------------


def _probes_targets_xlsx(path, extra_columns=None):
    """Write a minimal PROBES TARGETS workbook, shaped like the upstream export.

    The first column becomes the index, as the step reads the sheet with index_col=0.
    """
    frame = pd.DataFrame({
        'pdid': ['PD-1', 'PD-2', 'PD-3'],
        # PD-3 is dropped by the step's own gene_name filter. The surviving rows must
        # keep both shapes of any mixed column, or there is nothing left to merge.
        'gene_name': ['BRD4', 'EGFR', '-'],
        'organism': ['Homo sapiens', 'Homo sapiens', 'Homo sapiens'],
        'target': ['BRD4', 'EGFR', 'KRAS'],
        'action': ['inhibitor', '-', '-'],
        'control_smiles': ['CC', 'CC', 'CC'],
        'P&D probe-likeness score': [1.0, 2.0, 3.0],
        'Cells score (Chemical Probes.org)': [1.0, 2.0, 3.0],
        'Organisms score (Chemical Probes.org)': [1.0, 2.0, 3.0],
        **(extra_columns or {}),
    })
    frame.to_excel(path, sheet_name='PROBES TARGETS', index=False)
    return path


def test_probes_targets_survives_a_mixed_type_column(spark, tmp_path):
    """An unused column mixing booleans and blanks must not reach spark.

    Upstream 01_2026 changed `covalent` from a string column to real booleans with
    blank cells. Pandas types the blanks as float, and spark's schema inference
    cannot merge DoubleType with BooleanType, so the step died on a column it
    never reads. Handing spark only the columns the step uses keeps that class of
    upstream drift out of inference.
    """
    xlsx = _probes_targets_xlsx(
        tmp_path / 'probes.xlsx',
        # Upstream writes the literal text 'True' and leaves the rest blank. Pandas
        # turns that text into real bools and the blanks into NaN, giving one object
        # column holding both -- writing a python True instead round-trips to 1.0 and
        # reproduces nothing.
        extra_columns={'covalent': ['True', None, None]},
    )

    # The shared fixture enables arrow, whose converter tolerates the mixed column.
    # Production sets no arrow config, so spark falls back to row-wise schema
    # inference -- the path that actually failed. Match production here, and put the
    # session-scoped setting back for everyone else.
    arrow = spark.conf.get('spark.sql.execution.arrow.pyspark.enabled')
    spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', 'false')
    try:
        df = process_probes_targets_data(spark, str(xlsx))
        rows = df.collect()
    finally:
        spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', arrow)

    assert 'covalent' not in df.columns
    assert sorted(r.pdid for r in rows) == ['PD-1', 'PD-2'], (
        'only the row without a gene_name should be dropped'
    )
