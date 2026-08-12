"""Tests for chembl_molecule, which now joins raw ChEMBL tables."""

import json

import pytest
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.chembl_molecule import _molecule_preprocess, process_molecules

# A short but structurally valid MDL molblock (single carbon atom), with no
# terminator newline of its own -- the raw variants below add zero, one, or
# two trailing newlines on top of this.
_BARE_MOLBLOCK = (
    '\n     RDKit          2D\n\n'
    '  1  0  0  0  0  0  0  0  0  0999 V2000\n'
    '    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n'
    'M  END'
)

# PTS always emits exactly one trailing newline after `M  END`: every one of
# the 2,897,819 molblocks in the 26.06 release ends "M  END\n", so this is
# the truncated shape regardless of how the raw value was terminated.
SAMPLE_MOLBLOCK = _BARE_MOLBLOCK + '\n'

# ChEMBL ships some `molfile` values as a full SD-file record: the molblock plus
# appended SDF property tags, separated by a single newline (the old
# Elasticsearch shape). PTS truncates this back to the bare molblock.
SAMPLE_MOLFILE_WITH_SDF_TAGS = (
    _BARE_MOLBLOCK + '\n> <chembl_id>\nCHEMBL1\n\n> <chembl_pref_name>\nDRUG A\n\n$$$$\n'
)

# The raw column is otherwise inconsistent about a trailing newline after the
# terminator: most real values have none, a minority have one, and some are
# malformed with more than one. All must truncate to the same SAMPLE_MOLBLOCK.
MOLBLOCK_ZERO_TRAILING_NEWLINES = _BARE_MOLBLOCK
MOLBLOCK_ONE_TRAILING_NEWLINE = _BARE_MOLBLOCK + '\n'
MOLBLOCK_TWO_TRAILING_NEWLINES = _BARE_MOLBLOCK + '\n\n'

# A molfile-shaped string with no `M  END` terminator. PTS has nothing to
# truncate here, so it must pass through unchanged.
MOLFILE_NO_TERMINATOR = 'malformed molfile content\nwith no terminator line\n'

# raw drugbank lookup with ChEMBL's source column names -- these aren't valid
# bare DDL-string identifiers, so the schema is declared explicitly.
RAW_DRUGBANK_SCHEMA = StructType([
    StructField("From src:'1'", StringType()),
    StructField("To src:'2'", StringType()),
])


@pytest.fixture
def tables(spark: SparkSession) -> dict:
    """Raw ChEMBL molecule tables covering the structural edge cases under test."""
    molecule_dictionary = spark.createDataFrame(
        [
            (1, 'CHEMBL1', 'Drug A', 'Small molecule'),
            (2, 'CHEMBL2', 'Drug B', 'Antibody'),
            (3, 'CHEMBL3', '  lone drug ', 'Small molecule'),
            (4, 'CHEMBL4', 'Drug D', 'Small molecule'),
            (5, 'CHEMBL5', 'Drug E', 'Small molecule'),
            (6, 'CHEMBL6', 'Drug F', 'Small molecule'),
            (7, 'CHEMBL7', 'Drug G', 'Small molecule'),
        ],
        'molregno int, chembl_id string, pref_name string, molecule_type string',
    )
    compound_structures = spark.createDataFrame(
        [
            (1, 'C', 'INCHI1', SAMPLE_MOLFILE_WITH_SDF_TAGS),
            # molregno 2 (CHEMBL2, an antibody) has no structure row at all.
            (3, 'CC', 'INCHI3', None),
            (4, None, None, MOLFILE_NO_TERMINATOR),
            (5, None, None, MOLBLOCK_ZERO_TRAILING_NEWLINES),
            (6, None, None, MOLBLOCK_ONE_TRAILING_NEWLINE),
            (7, None, None, MOLBLOCK_TWO_TRAILING_NEWLINES),
        ],
        'molregno int, canonical_smiles string, standard_inchi_key string, molfile string',
    )
    # Every molecule is its own parent: _molecule_preprocess must null that out
    # rather than treat it as a real parent.
    molecule_hierarchy = spark.createDataFrame(
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        'molregno int, parent_molregno int',
    )
    molecule_synonyms = spark.createDataFrame(
        [], 'molsyn_id int, molregno int, synonyms string, syn_type string'
    )
    drugbank_lookup = spark.createDataFrame([], RAW_DRUGBANK_SCHEMA)
    return {
        'molecule_dictionary': molecule_dictionary,
        'compound_structures': compound_structures,
        'molecule_hierarchy': molecule_hierarchy,
        'molecule_synonyms': molecule_synonyms,
        'drugbank_lookup': drugbank_lookup,
    }


@pytest.fixture
def drugbank(spark: SparkSession) -> DataFrame:
    """Renamed drugbank lookup as consumed by _molecule_preprocess (empty by default)."""
    return spark.createDataFrame([], 'id string, drugbank_id string')


def rows_by_id(df: DataFrame) -> dict:
    return {r['id']: r.asDict(recursive=True) for r in df.collect()}


def _preprocess(tables: dict, drugbank: DataFrame) -> DataFrame:
    return _molecule_preprocess(
        tables['molecule_dictionary'],
        tables['compound_structures'],
        tables['molecule_hierarchy'],
        tables['molecule_synonyms'],
        drugbank,
    )


def _process(tables: dict, aact_batch: DataFrame | None = None) -> DataFrame:
    return process_molecules(
        tables['molecule_dictionary'],
        tables['compound_structures'],
        tables['molecule_hierarchy'],
        tables['molecule_synonyms'],
        tables['drugbank_lookup'],
        aact_batch,
    )


# --- Tests for _molecule_preprocess ---


class TestMoleculePreprocess:
    def test_molblock_truncated_and_sdf_tags_stripped(self, tables, drugbank):
        """molblock is the source molfile truncated at `M  END`, SDF tags dropped."""  # noqa: D403
        result = _preprocess(tables, drugbank)
        molblock = {r['id']: r['molblock'] for r in result.collect()}['CHEMBL1']
        assert molblock == SAMPLE_MOLBLOCK
        assert '> <chembl_id>' not in molblock
        assert '$$$$' not in molblock

    def test_molblock_null_when_no_compound_structure(self, tables, drugbank):
        """molblock is null when the molecule has no compound_structures row."""  # noqa: D403
        result = _preprocess(tables, drugbank)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL2'] is None

    def test_molfile_without_terminator_passed_through(self, tables, drugbank):
        """A source molfile with no `M  END` terminator is left unchanged."""
        result = _preprocess(tables, drugbank)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL4'] == MOLFILE_NO_TERMINATOR

    def test_molblock_is_string_column(self, tables, drugbank):
        """molblock is exposed as a string column."""  # noqa: D403
        result = _preprocess(tables, drugbank)
        assert result.schema['molblock'].dataType == StringType()

    def test_pref_name_is_trimmed(self, tables, drugbank):
        """Leading and trailing whitespace on pref_name is stripped from name."""
        # 18 ChEMBL 37 values carry a trailing space that ChEMBL's own indexer
        # trims; left untrimmed they would reach the Platform as the drug name.
        result = _preprocess(tables, drugbank)
        rows = {r['id']: r['name'] for r in result.collect()}
        assert rows['CHEMBL3'] == 'lone drug'

    def test_molblock_terminator_variants_all_truncate(self, tables, drugbank):
        """Zero, one, or two trailing newlines after `M  END` all truncate to exactly one."""
        # Every molblock in the 26.06 release ends "M  END\n" -- dropping that
        # newline would change a published column, so this must be exact, not lenient.
        result = _preprocess(tables, drugbank)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        for chembl_id in ('CHEMBL5', 'CHEMBL6', 'CHEMBL7'):
            assert rows[chembl_id] == SAMPLE_MOLBLOCK
            assert rows[chembl_id].endswith('M  END\n')
            assert '> <chembl_id>' not in rows[chembl_id]


# --- Tests for process_molecules ---


class TestProcessMolecules:
    def test_molblock_preserved(self, tables):
        """The truncated molblock survives process_molecules into the output."""
        rows = rows_by_id(_process(tables))
        assert rows['CHEMBL1']['molblock'] == SAMPLE_MOLBLOCK
        assert rows['CHEMBL2']['molblock'] is None

    def test_row_count_unchanged(self, tables):
        """Adding molblock does not change the row count."""
        result = _process(tables)
        assert result.count() == tables['molecule_dictionary'].count()


class TestCrossReferences:
    def test_drugbank_survives_while_chembl_cross_references_are_gone(self, tables, spark):
        """DrugBank cross references are unaffected by the removal of ChEMBL's own."""
        drugbank_lookup = spark.createDataFrame([('CHEMBL1', 'DB00001')], RAW_DRUGBANK_SCHEMA)
        result = _process({**tables, 'drugbank_lookup': drugbank_lookup})
        row = rows_by_id(result)['CHEMBL1']
        assert {x['source'] for x in row['crossReferences']} == {'drugbank'}
        assert row['crossReferences'] == [{'source': 'drugbank', 'ids': ['DB00001']}]

    def test_no_drugbank_id_means_no_cross_references(self, tables):
        """With no ChEMBL source left, a molecule with no DrugBank id has none at all."""
        row = rows_by_id(_process(tables))['CHEMBL1']
        assert not row.get('crossReferences')


class TestSynonymStructs:
    def test_synonyms_are_label_source_structs(self, spark, tables):
        """ChEMBL synonyms become {label, source:'ChEMBL'} structs, sorted."""
        molecule_dictionary = spark.createDataFrame(
            [(10, 'CHEMBL10', 'Aspirin', 'Small molecule')],
            'molregno int, chembl_id string, pref_name string, molecule_type string',
        )
        compound_structures = spark.createDataFrame([], tables['compound_structures'].schema)
        molecule_hierarchy = spark.createDataFrame([(10, 10)], 'molregno int, parent_molregno int')
        molecule_synonyms = spark.createDataFrame(
            [(1, 10, 'ASA', 'OTHER'), (2, 10, 'Bayer', 'TRADE_NAME')],
            'molsyn_id int, molregno int, synonyms string, syn_type string',
        )
        result = process_molecules(
            molecule_dictionary,
            compound_structures,
            molecule_hierarchy,
            molecule_synonyms,
            tables['drugbank_lookup'],
        )
        row = rows_by_id(result)['CHEMBL10']
        assert [(s['label'], s['source']) for s in row['synonyms']] == [('ASA', 'ChEMBL')]
        assert [(t['label'], t['source']) for t in row['tradeNames']] == [('Bayer', 'ChEMBL')]

    def test_empty_synonyms_are_empty_struct_array(self, tables):
        """Molecules with no synonyms get an empty (not null) struct array."""
        row = rows_by_id(_process(tables))['CHEMBL1']
        assert row['synonyms'] == []
        assert row['tradeNames'] == []

    def test_synonyms_schema_is_struct(self, tables):
        """Synonyms column type is array<struct<label,source>>."""
        result = _process(tables)
        field = result.schema['synonyms'].dataType
        assert isinstance(field, ArrayType)
        element_type = field.elementType
        assert isinstance(element_type, StructType)
        assert {sub.name for sub in element_type.fields} == {'label', 'source'}


class TestMergeAndTwoSource:
    def test_two_source_molecule(self, spark, tables):
        molecule_dictionary = spark.createDataFrame(
            [(1, 'CHEMBL1', 'Filgrastim', 'Protein')],
            'molregno int, chembl_id string, pref_name string, molecule_type string',
        )
        compound_structures = spark.createDataFrame([], tables['compound_structures'].schema)
        molecule_hierarchy = spark.createDataFrame([(1, 1)], 'molregno int, parent_molregno int')
        molecule_synonyms = spark.createDataFrame(
            [(1, 1, 'Neupogen', 'TRADE_NAME')],
            'molsyn_id int, molregno int, synonyms string, syn_type string',
        )

        outer_schema = StructType([
            StructField('custom_id', StringType()),
            StructField(
                'response',
                StructType([
                    StructField(
                        'body',
                        StructType([
                            StructField(
                                'output',
                                ArrayType(
                                    StructType([
                                        StructField('type', StringType()),
                                        StructField(
                                            'content',
                                            ArrayType(StructType([StructField('text', StringType())])),
                                        ),
                                    ])
                                ),
                            ),
                        ]),
                    )
                ]),
            ),
        ])
        payload = json.dumps({
            'investigated_drugs': [{'drug': 'Filgrastim', 'synonyms': ['G-CSF']}],
            'comparator_drugs': [],
            'supportive_drugs': [],
        })
        content = [Row(text=payload)]
        output = [Row(type='message', content=content)]
        batch = spark.createDataFrame(
            [
                Row(custom_id='NCT1', response=Row(body=Row(output=output))),
                Row(custom_id='NCT2', response=Row(body=Row(output=output))),
            ],
            outer_schema,
        )

        result = process_molecules(
            molecule_dictionary,
            compound_structures,
            molecule_hierarchy,
            molecule_synonyms,
            tables['drugbank_lookup'],
            batch,
        )
        row = rows_by_id(result)['CHEMBL1']
        sources = {s['source'] for s in row['synonyms']}
        labels = {s['label'] for s in row['synonyms']}
        assert 'AACT' in sources
        assert 'g-csf' in labels
        assert row['name'] == 'Filgrastim'  # AACT label never becomes name

    def test_process_molecules_without_aact_batch_still_works(self, tables):
        """process_molecules without a batch arg behaves as before (no AACT)."""
        result = _process(tables)
        assert result.count() == tables['molecule_dictionary'].count()
