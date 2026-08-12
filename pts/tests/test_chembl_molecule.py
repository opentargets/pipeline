"""Tests for chembl_molecule, which joins raw ChEMBL tables straight into polars."""

import polars as pl

from pts.transformers.chembl_molecule import _molecule_preprocess, process_molecules

# A short but structurally valid MDL molblock (single carbon atom), with no
# terminator newline of its own -- the raw variants below add zero, one, or two
# trailing newlines on top of this.
_BARE_MOLBLOCK = (
    '\n     RDKit          2D\n\n'
    '  1  0  0  0  0  0  0  0  0  0999 V2000\n'
    '    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n'
    'M  END'
)

# PTS always emits exactly one trailing newline after `M  END`: every one of the
# 2,897,819 molblocks in the 26.06 release ends "M  END\n", so this is the
# truncated shape regardless of how the raw value was terminated.
SAMPLE_MOLBLOCK = _BARE_MOLBLOCK + '\n'

# ChEMBL ships some `molfile` values as a full SD-file record: the molblock plus
# appended SDF property tags, separated by a single newline (the old Elasticsearch
# shape). PTS truncates this back to the bare molblock.
SAMPLE_MOLFILE_WITH_SDF_TAGS = (
    _BARE_MOLBLOCK + '\n> <chembl_id>\nCHEMBL1\n\n> <chembl_pref_name>\nDRUG A\n\n$$$$\n'
)

# The raw column is otherwise inconsistent about a trailing newline after the
# terminator: most real values have none, a minority have one, and some are
# malformed with more than one. All must truncate to the same SAMPLE_MOLBLOCK.
MOLBLOCK_ZERO_TRAILING_NEWLINES = _BARE_MOLBLOCK
MOLBLOCK_ONE_TRAILING_NEWLINE = _BARE_MOLBLOCK + '\n'
MOLBLOCK_TWO_TRAILING_NEWLINES = _BARE_MOLBLOCK + '\n\n'

# A molfile-shaped string with no `M  END` terminator. PTS has nothing to truncate
# here, so it must pass through unchanged.
MOLFILE_NO_TERMINATOR = 'malformed molfile content\nwith no terminator line\n'

# Raw drugbank lookup with ChEMBL's source column names -- these aren't valid bare
# DDL-string identifiers, so the schema is declared explicitly.
RAW_DRUGBANK_SCHEMA = {"From src:'1'": pl.Utf8, "To src:'2'": pl.Utf8}


def tables() -> dict[str, pl.DataFrame]:
    """Raw ChEMBL molecule tables covering the structural edge cases under test."""
    molecule_dictionary = pl.DataFrame(
        [
            (1, 'CHEMBL1', 'Drug A', 'Small molecule'),
            (2, 'CHEMBL2', 'Drug B', 'Antibody'),
            (3, 'CHEMBL3', '  lone drug ', 'Small molecule'),
            (4, 'CHEMBL4', 'Drug D', 'Small molecule'),
            (5, 'CHEMBL5', 'Drug E', 'Small molecule'),
            (6, 'CHEMBL6', 'Drug F', 'Small molecule'),
            (7, 'CHEMBL7', 'Drug G', 'Small molecule'),
        ],
        schema=['molregno', 'chembl_id', 'pref_name', 'molecule_type'],
        orient='row',
    )
    compound_structures = pl.DataFrame(
        [
            (1, 'C', 'INCHI1', SAMPLE_MOLFILE_WITH_SDF_TAGS),
            # molregno 2 (CHEMBL2, an antibody) has no structure row at all.
            (3, 'CC', 'INCHI3', None),
            (4, None, None, MOLFILE_NO_TERMINATOR),
            (5, None, None, MOLBLOCK_ZERO_TRAILING_NEWLINES),
            (6, None, None, MOLBLOCK_ONE_TRAILING_NEWLINE),
            (7, None, None, MOLBLOCK_TWO_TRAILING_NEWLINES),
        ],
        schema=['molregno', 'canonical_smiles', 'standard_inchi_key', 'molfile'],
        orient='row',
    )
    # Every molecule is its own parent: _molecule_preprocess must null that out
    # rather than treat it as a real parent.
    molecule_hierarchy = pl.DataFrame(
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        schema=['molregno', 'parent_molregno'],
        orient='row',
    )
    molecule_synonyms = pl.DataFrame(
        schema={'molsyn_id': pl.Int64, 'molregno': pl.Int64, 'synonyms': pl.Utf8, 'syn_type': pl.Utf8}
    )
    drugbank_lookup = pl.DataFrame(schema=RAW_DRUGBANK_SCHEMA)
    return {
        'molecule_dictionary': molecule_dictionary,
        'compound_structures': compound_structures,
        'molecule_hierarchy': molecule_hierarchy,
        'molecule_synonyms': molecule_synonyms,
        'drugbank_lookup': drugbank_lookup,
    }


def drugbank() -> pl.DataFrame:
    """Renamed drugbank lookup as consumed by _molecule_preprocess (empty by default)."""
    return pl.DataFrame(schema={'id': pl.Utf8, 'drugbank_id': pl.Utf8})


def rows_by_id(df: pl.DataFrame) -> dict:
    return {r['id']: r for r in df.to_dicts()}


def _preprocess(t: dict, db: pl.DataFrame) -> pl.DataFrame:
    return _molecule_preprocess(
        t['molecule_dictionary'],
        t['compound_structures'],
        t['molecule_hierarchy'],
        t['molecule_synonyms'],
        db,
    )


def _process(t: dict, aact_batch: pl.DataFrame | None = None) -> pl.DataFrame:
    return process_molecules(
        t['molecule_dictionary'],
        t['compound_structures'],
        t['molecule_hierarchy'],
        t['molecule_synonyms'],
        t['drugbank_lookup'],
        aact_batch,
    )


# --- Tests for _molecule_preprocess ---


class TestMoleculePreprocess:
    def test_molblock_truncated_and_sdf_tags_stripped(self):
        """molblock is the source molfile truncated at `M  END`, SDF tags dropped."""  # noqa: D403
        t = tables()
        result = _preprocess(t, drugbank())
        molblock = rows_by_id(result)['CHEMBL1']['molblock']
        assert molblock == SAMPLE_MOLBLOCK
        assert '> <chembl_id>' not in molblock
        assert '$$$$' not in molblock

    def test_molblock_null_when_no_compound_structure(self):
        """molblock is null when the molecule has no compound_structures row."""  # noqa: D403
        t = tables()
        result = _preprocess(t, drugbank())
        assert rows_by_id(result)['CHEMBL2']['molblock'] is None

    def test_molfile_without_terminator_passed_through(self):
        """A source molfile with no `M  END` terminator is left unchanged."""
        t = tables()
        result = _preprocess(t, drugbank())
        assert rows_by_id(result)['CHEMBL4']['molblock'] == MOLFILE_NO_TERMINATOR

    def test_molblock_is_string_column(self):
        """molblock is exposed as a string column."""  # noqa: D403
        t = tables()
        result = _preprocess(t, drugbank())
        assert result.schema['molblock'] == pl.Utf8

    def test_pref_name_is_trimmed(self):
        """Leading and trailing whitespace on pref_name is stripped from name."""
        # 18 ChEMBL 37 values carry a trailing space that ChEMBL's own indexer
        # trims; left untrimmed they would reach the Platform as the drug name.
        t = tables()
        result = _preprocess(t, drugbank())
        assert rows_by_id(result)['CHEMBL3']['name'] == 'lone drug'

    def test_molblock_terminator_variants_all_truncate(self):
        """Zero, one, or two trailing newlines after `M  END` all truncate to exactly one."""
        # Every molblock in the 26.06 release ends "M  END\n" -- dropping that
        # newline would change a published column, so this must be exact, not lenient.
        t = tables()
        result = _preprocess(t, drugbank())
        rows = rows_by_id(result)
        for chembl_id in ('CHEMBL5', 'CHEMBL6', 'CHEMBL7'):
            molblock = rows[chembl_id]['molblock']
            assert molblock == SAMPLE_MOLBLOCK
            assert molblock.endswith('M  END\n')
            assert '> <chembl_id>' not in molblock


# --- Tests for process_molecules ---


class TestProcessMolecules:
    def test_molblock_preserved(self):
        """The truncated molblock survives process_molecules into the output."""
        t = tables()
        rows = rows_by_id(_process(t))
        assert rows['CHEMBL1']['molblock'] == SAMPLE_MOLBLOCK
        assert rows['CHEMBL2']['molblock'] is None

    def test_row_count_unchanged(self):
        """Adding molblock does not change the row count."""
        t = tables()
        result = _process(t)
        assert result.height == t['molecule_dictionary'].height


class TestCrossReferences:
    def test_drugbank_survives_while_chembl_cross_references_are_gone(self):
        """DrugBank cross references are unaffected by the removal of ChEMBL's own."""
        t = tables()
        t = {**t, 'drugbank_lookup': pl.DataFrame([('CHEMBL1', 'DB00001')], schema=RAW_DRUGBANK_SCHEMA, orient='row')}
        row = rows_by_id(_process(t))['CHEMBL1']
        assert {x['source'] for x in row['crossReferences']} == {'drugbank'}
        assert row['crossReferences'] == [{'source': 'drugbank', 'ids': ['DB00001']}]

    def test_no_drugbank_id_means_no_cross_references(self):
        """With no ChEMBL source left, a molecule with no DrugBank id has none at all."""
        t = tables()
        row = rows_by_id(_process(t))['CHEMBL1']
        assert not row.get('crossReferences')


class TestSynonymStructs:
    def test_synonyms_are_label_source_structs(self):
        """ChEMBL synonyms become {label, source:'ChEMBL'} structs, sorted."""
        molecule_dictionary = pl.DataFrame(
            [(10, 'CHEMBL10', 'Aspirin', 'Small molecule')],
            schema=['molregno', 'chembl_id', 'pref_name', 'molecule_type'],
            orient='row',
        )
        compound_structures = pl.DataFrame(schema=tables()['compound_structures'].schema)
        molecule_hierarchy = pl.DataFrame([(10, 10)], schema=['molregno', 'parent_molregno'], orient='row')
        molecule_synonyms = pl.DataFrame(
            [(1, 10, 'ASA', 'OTHER'), (2, 10, 'Bayer', 'TRADE_NAME')],
            schema=['molsyn_id', 'molregno', 'synonyms', 'syn_type'],
            orient='row',
        )
        result = process_molecules(
            molecule_dictionary,
            compound_structures,
            molecule_hierarchy,
            molecule_synonyms,
            tables()['drugbank_lookup'],
        )
        row = rows_by_id(result)['CHEMBL10']
        assert [(s['label'], s['source']) for s in row['synonyms']] == [('ASA', 'ChEMBL')]
        assert [(t['label'], t['source']) for t in row['tradeNames']] == [('Bayer', 'ChEMBL')]

    def test_empty_synonyms_are_empty_struct_array(self):
        """Molecules with no synonyms get an empty (not null) struct array."""
        t = tables()
        row = rows_by_id(_process(t))['CHEMBL1']
        assert row['synonyms'] == []
        assert row['tradeNames'] == []

    def test_synonyms_schema_is_struct(self):
        """Synonyms column type is list[struct[label,source]]."""
        t = tables()
        result = _process(t)
        field = result.schema['synonyms']
        assert isinstance(field, pl.List)
        element_type = field.inner
        assert isinstance(element_type, pl.Struct)
        assert {sub.name for sub in element_type.fields} == {'label', 'source'}

    def test_null_synonym_text_does_not_survive_inside_the_synonyms_list(self):
        """A molecule_synonyms row with a null label is dropped, not shipped as `None`."""
        # pyspark's collect_set drops nulls; a bare polars list aggregation does not,
        # so this must be exercised explicitly -- Task 7 shipped this exact defect
        # for a different aggregation and it reached a Critical.
        molecule_dictionary = pl.DataFrame(
            [(20, 'CHEMBL20', 'Nullosyn', 'Small molecule')],
            schema=['molregno', 'chembl_id', 'pref_name', 'molecule_type'],
            orient='row',
        )
        compound_structures = pl.DataFrame(schema=tables()['compound_structures'].schema)
        molecule_hierarchy = pl.DataFrame([(20, 20)], schema=['molregno', 'parent_molregno'], orient='row')
        molecule_synonyms = pl.DataFrame(
            [(1, 20, None, 'OTHER'), (2, 20, 'Realonym', 'OTHER'), (3, 20, None, 'TRADE_NAME')],
            schema=['molsyn_id', 'molregno', 'synonyms', 'syn_type'],
            orient='row',
        )
        result = process_molecules(
            molecule_dictionary,
            compound_structures,
            molecule_hierarchy,
            molecule_synonyms,
            tables()['drugbank_lookup'],
        )
        row = rows_by_id(result)['CHEMBL20']
        assert row['synonyms'] == [{'label': 'Realonym', 'source': 'ChEMBL'}]
        assert row['tradeNames'] == []
        assert None not in [s['label'] for s in row['synonyms']]


class TestMergeAndTwoSource:
    def test_two_source_molecule(self):
        molecule_dictionary = pl.DataFrame(
            [(1, 'CHEMBL1', 'Filgrastim', 'Protein')],
            schema=['molregno', 'chembl_id', 'pref_name', 'molecule_type'],
            orient='row',
        )
        compound_structures = pl.DataFrame(schema=tables()['compound_structures'].schema)
        molecule_hierarchy = pl.DataFrame([(1, 1)], schema=['molregno', 'parent_molregno'], orient='row')
        molecule_synonyms = pl.DataFrame(
            [(1, 1, 'Neupogen', 'TRADE_NAME')],
            schema=['molsyn_id', 'molregno', 'synonyms', 'syn_type'],
            orient='row',
        )

        batch = pl.DataFrame(
            {
                'id': ['NCT1', 'NCT2'],
                'investigated_drugs': [
                    [{'drug': 'Filgrastim', 'synonyms': ['G-CSF']}],
                    [{'drug': 'Filgrastim', 'synonyms': ['G-CSF']}],
                ],
                'comparator_drugs': [[], []],
                'supportive_drugs': [[], []],
            },
            schema={
                'id': pl.Utf8,
                'investigated_drugs': pl.List(pl.Struct({'drug': pl.Utf8, 'synonyms': pl.List(pl.Utf8)})),
                'comparator_drugs': pl.List(pl.Struct({'drug': pl.Utf8, 'synonyms': pl.List(pl.Utf8)})),
                'supportive_drugs': pl.List(pl.Struct({'drug': pl.Utf8, 'synonyms': pl.List(pl.Utf8)})),
            },
        )

        result = process_molecules(
            molecule_dictionary,
            compound_structures,
            molecule_hierarchy,
            molecule_synonyms,
            tables()['drugbank_lookup'],
            batch,
        )
        row = rows_by_id(result)['CHEMBL1']
        sources = {s['source'] for s in row['synonyms']}
        labels = {s['label'] for s in row['synonyms']}
        assert 'AACT' in sources
        assert 'g-csf' in labels
        assert row['name'] == 'Filgrastim'  # AACT label never becomes name

    def test_process_molecules_without_aact_batch_still_works(self):
        """process_molecules without a batch arg behaves as before (no AACT)."""
        t = tables()
        result = _process(t)
        assert result.height == t['molecule_dictionary'].height
