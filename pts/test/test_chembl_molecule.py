"""Tests for the chembl_molecule module."""

import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.chembl_molecule import _molecule_preprocess, process_molecules

# --- Schemas matching the raw ChEMBL molecule input ---

MOLECULE_STRUCTURES = StructType([
    StructField('canonical_smiles', StringType()),
    StructField('standard_inchi_key', StringType()),
    StructField('molfile', StringType()),
])

MOLECULE_HIERARCHY = StructType([
    StructField('parent_chembl_id', StringType()),
])

CROSS_REFERENCE = StructType([
    StructField('xref_id', StringType()),
    StructField('xref_src', StringType()),
])

MOLECULE_SYNONYM = StructType([
    StructField('molecule_synonym', StringType()),
    StructField('syn_type', StringType()),
])

RAW_MOLECULE_SCHEMA = StructType([
    StructField('molecule_chembl_id', StringType()),
    StructField('molecule_structures', MOLECULE_STRUCTURES),
    StructField('molecule_type', StringType()),
    StructField('pref_name', StringType()),
    StructField('cross_references', ArrayType(CROSS_REFERENCE)),
    StructField('molecule_hierarchy', MOLECULE_HIERARCHY),
    StructField('molecule_synonyms', ArrayType(MOLECULE_SYNONYM)),
])

# drugbank lookup as already renamed inside process_molecules
DRUGBANK_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('drugbank_id', StringType()),
])

# raw drugbank lookup with ChEMBL's source column names
RAW_DRUGBANK_SCHEMA = StructType([
    StructField("From src:'1'", StringType()),
    StructField("To src:'2'", StringType()),
])

# A short but structurally valid MDL molblock (single carbon atom), terminated
# by the `M  END` line. This is what PTS should emit.
SAMPLE_MOLBLOCK = (
    '\n     RDKit          2D\n\n'
    '  1  0  0  0  0  0  0  0  0  0999 V2000\n'
    '    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n'
    'M  END\n'
)

# ChEMBL ships `molfile` as a full SD-file record: the molblock plus appended
# SDF property tags. PTS truncates this back to the bare molblock.
SAMPLE_MOLFILE = SAMPLE_MOLBLOCK + (
    '> <chembl_id>\nCHEMBL1\n\n'
    '> <chembl_pref_name>\nDRUG A\n\n'
    '$$$$\n'
)

# A molfile-shaped string with no `M  END` terminator. PTS has nothing to
# truncate here, so it must pass through unchanged.
MOLFILE_NO_TERMINATOR = 'malformed molfile content\nwith no terminator line\n'


# --- Fixtures ---


@pytest.fixture(scope='module')
def raw_molecule_df(spark):
    """Raw ChEMBL molecule rows: an SD-file molfile, a missing one, a malformed one."""
    data = [
        Row(
            molecule_chembl_id='CHEMBL1',
            molecule_structures=Row(
                canonical_smiles='C',
                standard_inchi_key='INCHI1',
                molfile=SAMPLE_MOLFILE,
            ),
            molecule_type='Small molecule',
            pref_name='Drug A',
            cross_references=[],
            molecule_hierarchy=Row(parent_chembl_id='CHEMBL1'),
            molecule_synonyms=[],
        ),
        Row(
            molecule_chembl_id='CHEMBL2',
            molecule_structures=Row(
                canonical_smiles=None,
                standard_inchi_key=None,
                molfile=None,
            ),
            molecule_type='Antibody',
            pref_name='Drug B',
            cross_references=[],
            molecule_hierarchy=Row(parent_chembl_id='CHEMBL2'),
            molecule_synonyms=[],
        ),
        Row(
            molecule_chembl_id='CHEMBL3',
            molecule_structures=Row(
                canonical_smiles='CC',
                standard_inchi_key='INCHI3',
                molfile=MOLFILE_NO_TERMINATOR,
            ),
            molecule_type='Small molecule',
            pref_name='Drug C',
            cross_references=[],
            molecule_hierarchy=Row(parent_chembl_id='CHEMBL3'),
            molecule_synonyms=[],
        ),
    ]
    return spark.createDataFrame(data, schema=RAW_MOLECULE_SCHEMA)


# Two drugbank fixtures because the two entry points expect different shapes:
# _molecule_preprocess takes the already-renamed lookup (id, drugbank_id), while
# process_molecules takes the raw lookup and renames the columns itself.
@pytest.fixture(scope='module')
def drugbank_df(spark):
    """Renamed drugbank lookup as consumed by _molecule_preprocess."""
    return spark.createDataFrame([], schema=DRUGBANK_SCHEMA)


@pytest.fixture(scope='module')
def raw_drugbank_df(spark):
    """Raw drugbank lookup with ChEMBL's source column names."""
    return spark.createDataFrame([], schema=RAW_DRUGBANK_SCHEMA)


# --- Tests for _molecule_preprocess ---


class TestMoleculePreprocess:
    def test_molblock_truncated_at_m_end(self, raw_molecule_df, drugbank_df):
        """molblock is the source molfile truncated at `M  END`."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL1'] == SAMPLE_MOLBLOCK

    def test_molblock_sdf_tags_stripped(self, raw_molecule_df, drugbank_df):
        """The SDF property tags appended after `M  END` are removed."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        molblock = {r['id']: r['molblock'] for r in result.collect()}['CHEMBL1']
        assert molblock.endswith('M  END\n')
        assert '> <chembl_id>' not in molblock
        assert '$$$$' not in molblock

    def test_molblock_null_when_molfile_absent(self, raw_molecule_df, drugbank_df):
        """molblock is null when the source molecule has no molfile."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL2'] is None

    def test_molfile_without_terminator_passed_through(self, raw_molecule_df, drugbank_df):
        """A source molfile with no `M  END` terminator is left unchanged."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL3'] == MOLFILE_NO_TERMINATOR

    def test_molblock_is_string_column(self, raw_molecule_df, drugbank_df):
        """molblock is exposed as a string column."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        assert result.schema['molblock'].dataType == StringType()


# --- Tests for process_molecules ---


class TestProcessMolecules:
    def test_molblock_preserved(self, raw_molecule_df, raw_drugbank_df):
        """The truncated molblock survives process_molecules into the output."""
        result = process_molecules(raw_molecule_df, raw_drugbank_df)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL1'] == SAMPLE_MOLBLOCK
        assert rows['CHEMBL2'] is None

    def test_row_count_unchanged(self, raw_molecule_df, raw_drugbank_df):
        """Adding molblock does not change the row count."""
        result = process_molecules(raw_molecule_df, raw_drugbank_df)
        assert result.count() == raw_molecule_df.count()
