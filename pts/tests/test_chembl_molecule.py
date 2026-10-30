"""Tests for chembl_molecule, which joins raw ChEMBL tables straight into polars."""

import random

import polars as pl
import pytest

from pts.schemas.drug_molecule import molecule_properties_schema
from pts.transformers.chembl_molecule import (
    PEPTIDE_MAX_MW,
    _molecule_preprocess,
    add_compound_properties,
    process_molecules,
)

# A short but structurally valid MDL molblock (single carbon atom), with no
# terminator newline of its own -- the raw variants below add zero, one, or two
# trailing newlines on top of this.
_BARE_MOLBLOCK = (
    '\n     RDKit          2D\n\n'
    '  1  0  0  0  0  0  0  0  0  0999 V2000\n'
    '    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n'
    'M  END'
)

# PTS always emits exactly one trailing newline after `M  END`, which is how the
# published molblocks are terminated, regardless of how the raw value ended.
SAMPLE_MOLBLOCK = _BARE_MOLBLOCK + '\n'

# ChEMBL ships some `molfile` values as a full SD-file record: the molblock plus
# appended SDF property tags, separated by a single newline. PTS truncates this
# back to the bare molblock.
SAMPLE_MOLFILE_WITH_SDF_TAGS = _BARE_MOLBLOCK + '\n> <chembl_id>\nCHEMBL1\n\n> <chembl_pref_name>\nDRUG A\n\n$$$$\n'

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
        """molblock is the source molfile truncated at `M  END`, SDF tags dropped."""
        t = tables()
        result = _preprocess(t, drugbank())
        molblock = rows_by_id(result)['CHEMBL1']['molblock']
        assert molblock == SAMPLE_MOLBLOCK
        assert '> <chembl_id>' not in molblock
        assert '$$$$' not in molblock

    def test_molblock_null_when_no_compound_structure(self):
        """molblock is null when the molecule has no compound_structures row."""
        t = tables()
        result = _preprocess(t, drugbank())
        assert rows_by_id(result)['CHEMBL2']['molblock'] is None

    def test_molfile_without_terminator_passed_through(self):
        """A source molfile with no `M  END` terminator is left unchanged."""
        t = tables()
        result = _preprocess(t, drugbank())
        assert rows_by_id(result)['CHEMBL4']['molblock'] == MOLFILE_NO_TERMINATOR

    def test_molblock_is_string_column(self):
        """molblock is exposed as a string column."""
        t = tables()
        result = _preprocess(t, drugbank())
        assert result.schema['molblock'] == pl.Utf8

    def test_pref_name_is_trimmed(self):
        """Leading and trailing whitespace on pref_name is stripped from name."""
        # A handful of pref_name values carry a trailing space; left untrimmed
        # they would reach the Platform as the drug name.
        t = tables()
        result = _preprocess(t, drugbank())
        assert rows_by_id(result)['CHEMBL3']['name'] == 'lone drug'

    def test_molblock_terminator_variants_all_truncate(self):
        """Zero, one, or two trailing newlines after `M  END` all truncate to exactly one."""
        # The published molblock ends "M  END\n" -- dropping that newline would
        # change the column, so this must be exact rather than lenient.
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

    def test_childless_molecule_has_null_childchemblids_not_empty_list(self):
        """A molecule with no children gets a null childChemblIds, not []."""
        # synonyms/tradeNames are coalesced to [] but childChemblIds is deliberately
        # left null: most molecules have no children, so filling it would rewrite
        # the column on nearly every row.
        t = tables()
        row = rows_by_id(_process(t))['CHEMBL1']
        assert row['childChemblIds'] is None

    def test_parent_with_children_gets_populated_childchemblids(self):
        """A molecule with children still gets the real childChemblIds list."""
        molecule_dictionary = pl.DataFrame(
            [(30, 'CHEMBLPARENT', 'Parent', 'Small molecule'), (31, 'CHEMBLCHILD', 'Child', 'Small molecule')],
            schema=['molregno', 'chembl_id', 'pref_name', 'molecule_type'],
            orient='row',
        )
        compound_structures = pl.DataFrame(schema=tables()['compound_structures'].schema)
        molecule_hierarchy = pl.DataFrame([(30, 30), (31, 30)], schema=['molregno', 'parent_molregno'], orient='row')
        molecule_synonyms = pl.DataFrame(
            schema={'molsyn_id': pl.Int64, 'molregno': pl.Int64, 'synonyms': pl.Utf8, 'syn_type': pl.Utf8}
        )
        result = process_molecules(
            molecule_dictionary,
            compound_structures,
            molecule_hierarchy,
            molecule_synonyms,
            tables()['drugbank_lookup'],
        )
        rows = rows_by_id(result)
        assert rows['CHEMBLPARENT']['childChemblIds'] == ['CHEMBLCHILD']
        assert rows['CHEMBLCHILD']['childChemblIds'] is None


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
        # a bare polars list aggregation keeps nulls,
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


class TestOrderIsDeterministic:
    """`childChemblIds` is published, and nothing upstream pins the order it is built in.

    The rows feeding the aggregation come from a `SELECT DISTINCT` against a restored
    dump, which promises no order, and polars' `group_by`/`unique` promise none either.
    The aggregation therefore has to impose one itself.

    Both tests need a parent with many children: with only a handful, `unique` happens
    to return them in a stable order and the tests pass whether or not the code sorts,
    which would make them decoration rather than a guard.
    """

    CHILDREN = 200

    @classmethod
    def _tables(cls, order: list[int]) -> dict:
        """One parent (molregno 1) with `CHILDREN` children, rows in `order`.

        `order` permutes BOTH tables. The molecule_dictionary order is the one that
        actually reaches the aggregation -- `_molecule_preprocess` joins the hierarchy
        onto it, so the join output follows the dictionary, not the hierarchy read.
        Permuting only the hierarchy leaves the result stable and proves nothing.
        """
        molecule_dictionary = pl.DataFrame(
            [(m, f'CHEMBL{m:04d}', f'Drug {m}', 'Small molecule') for m in [1, *order]],
            schema=['molregno', 'chembl_id', 'pref_name', 'molecule_type'],
            orient='row',
        )
        return {
            'molecule_dictionary': molecule_dictionary,
            'compound_structures': pl.DataFrame(
                schema={
                    'molregno': pl.Int64,
                    'canonical_smiles': pl.Utf8,
                    'standard_inchi_key': pl.Utf8,
                    'molfile': pl.Utf8,
                }
            ),
            'molecule_hierarchy': pl.DataFrame(
                [(1, 1), *[(child, 1) for child in order]],
                schema=['molregno', 'parent_molregno'],
                orient='row',
            ),
            'molecule_synonyms': pl.DataFrame(
                schema={'molsyn_id': pl.Int64, 'molregno': pl.Int64, 'synonyms': pl.Utf8, 'syn_type': pl.Utf8}
            ),
            'drugbank_lookup': pl.DataFrame(schema=RAW_DRUGBANK_SCHEMA),
        }

    @classmethod
    def _permutations(cls) -> list[list[int]]:
        children = list(range(2, cls.CHILDREN + 2))
        rng = random.Random(0)
        shuffled = []
        for _ in range(3):
            p = children[:]
            rng.shuffle(p)
            shuffled.append(p)
        return [children, children[::-1], *shuffled]

    def test_child_order_does_not_depend_on_row_order(self):
        """Element order inside the published `childChemblIds` array."""
        results = [rows_by_id(_process(self._tables(p)))['CHEMBL0001']['childChemblIds'] for p in self._permutations()]
        assert results[0] == sorted(results[0])
        assert all(r == results[0] for r in results), 'childChemblIds varied with input row order'

    def test_the_same_input_gives_the_same_output(self):
        """`unique(subset=...)` keeps one row per id and may pick any of them.

        Output row order deliberately follows the input's -- that is what
        `maintain_order` preserves -- so this asserts reproducibility, not
        independence: the same frame in must give the same frame out. Pinning the
        input order itself is a separate matter, and this module's reads are not
        ordered.
        """
        t = self._tables(list(range(2, self.CHILDREN + 2)))
        runs = [_process(t) for _ in range(3)]
        assert all(r.equals(runs[0]) for r in runs), 'repeated runs on one input differed'


# --- Tests for add_compound_properties ---

DECIMAL = pl.Decimal(38, 10)


def property_tables() -> dict[str, pl.DataFrame]:
    """Raw ChEMBL property and alert tables, typed as `read_dump_tables` returns them."""
    molecule_dictionary = pl.DataFrame(
        [
            (1, 'CHEMBL1', 'Small molecule', 'MOL', 1),
            (2, 'CHEMBL2', 'Protein', 'MOL', 0),
            (3, 'CHEMBL3', 'Protein', 'MOL', 0),
            (4, 'CHEMBL4', 'Protein', 'SEQ', 0),
            (5, 'CHEMBL5', None, 'MOL', None),
            (6, 'CHEMBL6', 'Antibody', 'SEQ', 0),
        ],
        schema={
            'molregno': pl.Int64,
            'chembl_id': pl.String,
            'molecule_type': pl.String,
            'structure_type': pl.String,
            'natural_product': pl.Int16,
        },
        orient='row',
    )
    compound_properties = pl.DataFrame(
        [
            (1, '493.62', '4.59', 4, 37, '0.39', 0),
            # the peptide rule's edge: exactly PEPTIDE_MAX_MW, and just above it
            (2, f'{PEPTIDE_MAX_MW}', None, None, 100, None, 2),
            (3, f'{PEPTIDE_MAX_MW}.01', None, None, 101, None, 2),
            # molregno 4 (a Protein) has no properties row, so no molecular weight
            (5, '120.10', '-0.50', 0, 9, '0.45', 0),
        ],
        schema=[
            'molregno',
            'mw_freebase',
            'alogp',
            'aromatic_rings',
            'heavy_atoms',
            'qed_weighted',
            'num_ro5_violations',
        ],
        orient='row',
    ).with_columns(
        pl.col('mw_freebase', 'alogp', 'qed_weighted').cast(DECIMAL),
        pl.col('aromatic_rings', 'heavy_atoms').cast(pl.Int32),
        pl.col('num_ro5_violations').cast(pl.Int16),
    )
    compound_structural_alerts = pl.DataFrame(
        {'molregno': [1, 1, 1, 1, 5], 'alert_id': [40, 7, 12, 7, 12]},
    )
    structural_alerts = pl.DataFrame({
        'alert_id': [7, 12, 40],
        'alert_set_id': [4, 1, 1],
        'alert_name': ['quinone_A(370)', 'Michael acceptor', 'thiol'],
    })
    structural_alert_sets = pl.DataFrame({'alert_set_id': [1, 4], 'set_name': ['Glaxo', 'PAINS']})
    return {
        'molecule_dictionary': molecule_dictionary,
        'compound_properties': compound_properties,
        'compound_structural_alerts': compound_structural_alerts,
        'structural_alerts': structural_alerts,
        'structural_alert_sets': structural_alert_sets,
    }


def _with_properties(**overrides: pl.DataFrame) -> pl.DataFrame:
    t = property_tables() | overrides
    molecules = pl.DataFrame({'id': [f'CHEMBL{i}' for i in range(1, 7)], 'name': [f'drug {i}' for i in range(1, 7)]})
    return add_compound_properties(
        molecules,
        t['molecule_dictionary'],
        t['compound_properties'],
        t['compound_structural_alerts'],
        t['structural_alerts'],
        t['structural_alert_sets'],
    )


class TestAddCompoundProperties:
    def test_appends_the_property_columns_with_their_types(self):
        result = _with_properties()
        assert result.columns == ['id', 'name', *molecule_properties_schema.keys()]
        assert result.select(molecule_properties_schema.keys()).schema == molecule_properties_schema

    def test_keeps_every_molecule_in_order(self):
        assert _with_properties()['id'].to_list() == [f'CHEMBL{i}' for i in range(1, 7)]

    def test_decimal_properties_become_floats(self):
        row = rows_by_id(_with_properties())['CHEMBL1']
        assert (row['mwFreebase'], row['alogp'], row['qedWeighted']) == (493.62, 4.59, 0.39)
        assert (row['aromaticRings'], row['heavyAtoms'], row['numRo5Violations']) == (4, 37, 0)

    def test_molecule_without_properties_row_has_null_properties(self):
        row = rows_by_id(_with_properties())['CHEMBL4']
        assert row['mwFreebase'] is None
        assert row['alogp'] is None
        assert row['heavyAtoms'] is None

    def test_alerts_from_every_set_sorted_by_set_then_id(self):
        alerts = rows_by_id(_with_properties())['CHEMBL1']['structuralAlerts']
        assert alerts == [
            {'alertId': 12, 'alertName': 'Michael acceptor', 'setName': 'Glaxo'},
            {'alertId': 40, 'alertName': 'thiol', 'setName': 'Glaxo'},
            {'alertId': 7, 'alertName': 'quinone_A(370)', 'setName': 'PAINS'},
        ]

    def test_molecule_without_alerts_has_an_empty_list(self):
        assert rows_by_id(_with_properties())['CHEMBL2']['structuralAlerts'] == []

    def test_peptide_rule_edge(self):
        rows = rows_by_id(_with_properties())
        assert rows['CHEMBL2']['moleculeClass'] == 'Peptide'
        assert rows['CHEMBL3']['moleculeClass'] == 'Protein'

    def test_protein_without_molecular_weight_stays_protein(self):
        assert rows_by_id(_with_properties())['CHEMBL4']['moleculeClass'] == 'Protein'

    def test_other_types_keep_their_chembl_type_and_null_reads_unknown(self):
        rows = rows_by_id(_with_properties())
        assert rows['CHEMBL1']['moleculeClass'] == 'Small molecule'
        assert rows['CHEMBL6']['moleculeClass'] == 'Antibody'
        assert rows['CHEMBL5']['moleculeClass'] == 'Unknown'

    def test_structure_type_and_natural_product(self):
        rows = rows_by_id(_with_properties())
        assert (rows['CHEMBL1']['structureType'], rows['CHEMBL1']['naturalProduct']) == ('MOL', True)
        assert (rows['CHEMBL4']['structureType'], rows['CHEMBL4']['naturalProduct']) == ('SEQ', False)
        assert rows['CHEMBL5']['naturalProduct'] is None

    @pytest.mark.parametrize('table', ['molecule_dictionary', 'compound_properties'])
    def test_duplicate_molregno_fails_instead_of_duplicating_molecules(self, table):
        duplicated = pl.concat([property_tables()[table], property_tables()[table].head(1)])
        with pytest.raises(pl.exceptions.ComputeError):
            _with_properties(**{table: duplicated})
