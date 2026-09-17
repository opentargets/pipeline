"""Tests for drug_mechanism_of_action, which reads the raw ChEMBL tables straight from the dump."""

import polars as pl
import pytest

from pts.transformers.drug_mechanism_of_action import (
    ORDER_BY,
    TABLES,
    _chembl_target,
    _consolidate_duplicate_references,
    _with_target_chembl_id,
    process_mechanism_of_action,
)

REFERENCE_SCHEMA = pl.Struct({'source': pl.String, 'ids': pl.List(pl.String), 'urls': pl.List(pl.String)})

MECHANISM_SCHEMA = {
    'mechanismOfAction': pl.String,
    'actionType': pl.String,
    'chemblIds': pl.List(pl.String),
    'parentChemblId': pl.String,
    'references': pl.List(REFERENCE_SCHEMA),
    'targetName': pl.String,
    'targetType': pl.String,
    'targets': pl.List(pl.String),
}


@pytest.fixture
def tables() -> dict:
    drug_mechanism = pl.DataFrame(
        {
            'mec_id': [100, 101],
            'record_id': [1000, 1001],
            'molregno': [1, 3],
            'mechanism_of_action': ['Inhibits enzyme X', 'Blocks receptor'],
            'tid': [20, 21],
            'action_type': ['INHIBITOR', 'ANTAGONIST'],
        }
    )
    mechanism_refs = pl.DataFrame({
        'mecref_id': [1, 2],
        'mec_id': [100, 100],
        'ref_type': ['PMID', 'DOI'],
        'ref_id': ['12345', '10.1/xyz'],
        'ref_url': ['http://pmid/12345', 'http://doi/xyz'],
    })
    molecule_dictionary = pl.DataFrame({
        'molregno': [1, 2, 3],
        'chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'pref_name': ['MolA', 'ParentA', 'MolB'],
        'molecule_type': ['Small molecule', 'Small molecule', 'Small molecule'],
    })
    molecule_hierarchy = pl.DataFrame({'molregno': [1, 2, 3], 'parent_molregno': [2, 2, 3]})
    target_dictionary = pl.DataFrame({
        'tid': [20, 21],
        'chembl_id': ['CHEMBL_T20', 'CHEMBL_T21'],
        'pref_name': ['Target Twenty', 'Target TwentyOne'],
        'target_type': ['SINGLE PROTEIN', 'SINGLE PROTEIN'],
    })
    target_components = pl.DataFrame({
        'targcomp_id': [2001, 2002, 2003],
        'tid': [20, 20, 21],
        'component_id': [300, 301, 302],
    })
    component_sequences = pl.DataFrame({
        'component_id': [300, 301, 302],
        'accession': ['P100', 'P200', 'P300'],
    })
    gene_df = pl.DataFrame(
        {
            'id': ['ENSG1', 'ENSG2', 'ENSG3'],
            'uniprot_trembl': [['P100'], None, None],
            'uniprot_swissprot': [None, ['P200'], ['P300']],
        },
        schema={'id': pl.String, 'uniprot_trembl': pl.List(pl.String), 'uniprot_swissprot': pl.List(pl.String)},
    )
    return {
        'drug_mechanism': drug_mechanism,
        'mechanism_refs': mechanism_refs,
        'molecule_dictionary': molecule_dictionary,
        'molecule_hierarchy': molecule_hierarchy,
        'target_dictionary': target_dictionary,
        'target_components': target_components,
        'component_sequences': component_sequences,
        'gene_df': gene_df,
    }


def rows_by_target(df: pl.DataFrame) -> dict:
    return {r['targetName']: r for r in df.to_dicts()}


class TestProcessMechanismOfAction:
    def test_one_row_per_mec_id(self, tables: dict) -> None:
        result = process_mechanism_of_action(**tables)
        # assert on the raw count BEFORE keying, so a fan-out through target_components cannot hide
        assert result.height == 2

    def test_scalar_fields(self, tables: dict) -> None:
        rows = rows_by_target(process_mechanism_of_action(**tables))
        assert rows['Target Twenty']['mechanismOfAction'] == 'Inhibits enzyme X'
        assert rows['Target Twenty']['actionType'] == 'INHIBITOR'
        assert rows['Target TwentyOne']['mechanismOfAction'] == 'Blocks receptor'
        assert rows['Target TwentyOne']['actionType'] == 'ANTAGONIST'

    def test_chembl_ids_has_molecule_and_parent(self, tables: dict) -> None:
        # Order is [molecule, parent] by construction; assert it directly rather
        # than sorting, so swapping the concat_list(...) arguments would fail here.
        rows = rows_by_target(process_mechanism_of_action(**tables))
        assert rows['Target Twenty']['chemblIds'] == ['CHEMBL1', 'CHEMBL2']

    def test_chembl_ids_deduplicates_a_self_parent(self, tables: dict) -> None:
        rows = rows_by_target(process_mechanism_of_action(**tables))
        assert rows['Target TwentyOne']['chemblIds'] == ['CHEMBL3']

    def test_references_grouped_by_ref_type(self, tables: dict) -> None:
        refs = rows_by_target(process_mechanism_of_action(**tables))['Target Twenty']['references']
        by_source = {r['source']: r for r in refs}
        assert by_source['PMID']['ids'] == ['12345']
        assert by_source['PMID']['urls'] == ['http://pmid/12345']
        assert by_source['DOI']['ids'] == ['10.1/xyz']
        assert by_source['DOI']['urls'] == ['http://doi/xyz']

    def test_no_references_is_an_empty_array_not_null(self, tables: dict) -> None:
        rows = rows_by_target(process_mechanism_of_action(**tables))
        assert rows['Target TwentyOne']['references'] == []

    def test_null_ref_id_and_url_do_not_survive_inside_the_list(self, tables: dict) -> None:
        """A reference with no id/url gets ``ids: []`` / ``urls: []``, not ``[None]``.

        Reproduces published references such as 'Expert' or 'KEGG' entries. ``collect_list``
        on the pyspark side drops nulls, and the polars aggregation must too.
        """
        tables = dict(tables)
        tables['mechanism_refs'] = pl.concat([
            tables['mechanism_refs'],
            pl.DataFrame(
                {'mecref_id': [3], 'mec_id': [100], 'ref_type': ['Expert'], 'ref_id': [None], 'ref_url': [None]},
                schema=tables['mechanism_refs'].schema,
            ),
        ])
        refs = rows_by_target(process_mechanism_of_action(**tables))['Target Twenty']['references']
        by_source = {r['source']: r for r in refs}
        assert by_source['Expert']['ids'] == []
        assert by_source['Expert']['urls'] == []

    def test_target_components_join_produces_all_component_accessions(self, tables: dict) -> None:
        rows = rows_by_target(process_mechanism_of_action(**tables))
        # tid=20 has two components (accessions P100 and P200); both must resolve to genes
        # without multiplying the mechanism row.
        assert sorted(rows['Target Twenty']['targets']) == ['ENSG1', 'ENSG2']
        assert rows['Target Twenty']['targetType'] == 'single protein'


class TestCrossDrugReferenceCrosstalk:
    """Two unrelated drugs sharing a mechanism must not pool their references.

    Goes through `process_mechanism_of_action` on purpose. The unit tests below build
    their own frame, so they cannot see a defect that lives in the column set: if
    nothing identifying the drug survives into the consolidation, the grouping key is
    display information alone and a whole pharmacological class merges into one row.
    """

    @pytest.fixture
    def shared_mechanism(self, tables: dict) -> dict:
        """CHEMBL1 (a salt of CHEMBL2) and CHEMBL3 both inhibit the same target."""
        tables = dict(tables)
        tables['drug_mechanism'] = pl.DataFrame({
            'mec_id': [100, 101],
            'record_id': [1000, 1001],
            'molregno': [1, 3],
            'mechanism_of_action': ['Inhibits enzyme X', 'Inhibits enzyme X'],
            'tid': [20, 20],
            'action_type': ['INHIBITOR', 'INHIBITOR'],
        })
        tables['mechanism_refs'] = pl.DataFrame({
            'mecref_id': [1, 2],
            'mec_id': [100, 101],
            'ref_type': ['FDA', 'FDA'],
            'ref_id': ['drug-one-label', 'drug-three-label'],
            'ref_url': ['http://fda/drug-one', 'http://fda/drug-three'],
        })
        return tables

    def test_the_two_drugs_stay_two_rows(self, shared_mechanism: dict) -> None:
        result = process_mechanism_of_action(**shared_mechanism)
        assert result.height == 2

    def test_neither_drug_inherits_the_other_s_references(self, shared_mechanism: dict) -> None:
        result = process_mechanism_of_action(**shared_mechanism)
        exploded = result.explode('chemblIds').rename({'chemblIds': 'drugId'})
        urls = {
            r['drugId']: sorted({u for ref in r['references'] for u in ref['urls']})
            for r in exploded.to_dicts()
        }
        assert urls['CHEMBL1'] == ['http://fda/drug-one']
        assert urls['CHEMBL3'] == ['http://fda/drug-three']

    def test_the_parent_of_the_salt_is_unaffected_too(self, shared_mechanism: dict) -> None:
        """CHEMBL2 reaches the mechanism through its salt, and only through its salt."""
        result = process_mechanism_of_action(**shared_mechanism)
        exploded = result.explode('chemblIds').rename({'chemblIds': 'drugId'})
        parent = exploded.filter(pl.col('drugId') == 'CHEMBL2').to_dicts()
        assert len(parent) == 1
        assert sorted({u for ref in parent[0]['references'] for u in ref['urls']}) == ['http://fda/drug-one']


class TestWithTargetChemblId:
    def test_null_tid_yields_null_target_chembl_id_without_dropping_the_row(self) -> None:
        mechanism = pl.DataFrame(
            {'mec_id': [100, 101], 'tid': [20, None]}, schema={'mec_id': pl.Int64, 'tid': pl.Int64}
        )
        target_dictionary = pl.DataFrame({
            'tid': [20],
            'chembl_id': ['CHEMBL_T20'],
            'pref_name': ['Target Twenty'],
            'target_type': ['SINGLE PROTEIN'],
        })
        result = _with_target_chembl_id(mechanism, target_dictionary)
        rows = {r['mec_id']: r['target_chembl_id'] for r in result.to_dicts()}
        assert rows == {100: 'CHEMBL_T20', 101: None}


class TestChemblTarget:
    def test_two_components_yield_one_row_with_both_genes(self, tables: dict) -> None:
        result = _chembl_target(
            tables['target_dictionary'],
            tables['target_components'],
            tables['component_sequences'],
            tables['gene_df'],
        )
        rows = rows_by_target(result)
        assert len(rows) == 2
        assert rows['Target Twenty']['target_chembl_id'] == 'CHEMBL_T20'
        assert sorted(rows['Target Twenty']['targets']) == ['ENSG1', 'ENSG2']
        assert rows['Target TwentyOne']['targets'] == ['ENSG3']


class TestConsolidateDuplicateReferences:
    def test_parent_and_child_mechanism_are_merged(self) -> None:
        """A parent's native mechanism and its child's rolled-up copy collapse into one row.

        Reproduces the CHEMBL479 case: the mechanism is recorded once for
        CHEMBL479 alone, and once for CHEMBL1200916, whose ChEMBL
        `_metadata.all_molecule_chembl_ids` rolls it up to include CHEMBL479 too.
        """
        refs = [{'source': 'PubMed', 'ids': ['111'], 'urls': ['u1']}]
        data = [
            {
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist', 'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL479'], 'parentChemblId': 'CHEMBL479',
                'references': refs, 'targetName': '5-HT2a',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist', 'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL1200916', 'CHEMBL479'], 'parentChemblId': 'CHEMBL479',
                'references': refs, 'targetName': '5-HT2a',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)
        rows = result.to_dicts()

        assert len(rows) == 1
        assert sorted(rows[0]['chemblIds']) == ['CHEMBL1200916', 'CHEMBL479']

    def test_the_drug_anchor_does_not_reach_the_output(self) -> None:
        """`parentChemblId` groups the rows and is then dropped.

        It is internal: a new published column would need a croissant recordset entry.
        """
        df = pl.DataFrame(
            [
                {
                    'mechanismOfAction': 'Inhibits X', 'actionType': 'INHIBITOR',
                    'chemblIds': ['CHEMBL1'], 'parentChemblId': 'CHEMBL1',
                    'references': [], 'targetName': 'X',
                    'targetType': 'single protein', 'targets': ['ENSG1'],
                }
            ],
            schema=MECHANISM_SCHEMA,
        )

        result = _consolidate_duplicate_references(df)

        assert 'parentChemblId' not in result.columns
        assert result.columns == [c for c in MECHANISM_SCHEMA if c != 'parentChemblId']

    def test_unrelated_drugs_sharing_a_mechanism_are_not_merged(self) -> None:
        """Two different drugs with the same mechanism and target stay two rows.

        Same class, same target, so every column except `chemblIds`, `parentChemblId`
        and `references` is identical -- a key built from the display fields alone
        would collapse them into one drug.
        """
        drug_a_refs = [{'source': 'FDA', 'ids': ['label-a'], 'urls': ['url-a']}]
        drug_b_refs = [{'source': 'FDA', 'ids': ['label-b'], 'urls': ['url-b']}]
        data = [
            {
                'mechanismOfAction': 'Inhibits enzyme X', 'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_A'], 'parentChemblId': 'CHEMBL_A',
                'references': drug_a_refs, 'targetName': 'Enzyme X',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Inhibits enzyme X', 'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_B_SALT', 'CHEMBL_B'], 'parentChemblId': 'CHEMBL_B',
                'references': drug_b_refs, 'targetName': 'Enzyme X',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)

        assert result.height == 2
        by_drug = {
            r['drugId']: r
            for r in result.explode('chemblIds').rename({'chemblIds': 'drugId'}).to_dicts()
        }
        assert by_drug['CHEMBL_A']['references'] == drug_a_refs
        assert by_drug['CHEMBL_B']['references'] == drug_b_refs
        assert by_drug['CHEMBL_B_SALT']['references'] == drug_b_refs

    def test_one_drug_does_not_inherit_another_s_references(self) -> None:
        """A drug must never carry a reference belonging to a different drug."""
        data = [
            {
                'mechanismOfAction': 'Inhibits enzyme X', 'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_A'], 'parentChemblId': 'CHEMBL_A',
                'references': [{'source': 'FDA', 'ids': ['label-a'], 'urls': ['url-a']}],
                'targetName': 'Enzyme X',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Inhibits enzyme X', 'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_B'], 'parentChemblId': 'CHEMBL_B',
                'references': [{'source': 'FDA', 'ids': ['label-b'], 'urls': ['url-b']}],
                'targetName': 'Enzyme X',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)
        exploded = result.explode('chemblIds').rename({'chemblIds': 'drugId'})
        urls = {
            r['drugId']: sorted({u for ref in r['references'] for u in ref['urls']})
            for r in exploded.to_dicts()
        }

        assert urls['CHEMBL_A'] == ['url-a']
        assert urls['CHEMBL_B'] == ['url-b']

    def test_two_salts_of_the_same_parent_are_merged(self) -> None:
        """The anchor is the parent, so sibling salts still collapse onto one row."""
        data = [
            {
                'mechanismOfAction': 'Inhibits X', 'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_SALT_A', 'CHEMBL_PARENT'], 'parentChemblId': 'CHEMBL_PARENT',
                'references': [{'source': 'PubMed', 'ids': ['1'], 'urls': ['u1']}],
                'targetName': 'X', 'targetType': 'single protein', 'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Inhibits X', 'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_SALT_B', 'CHEMBL_PARENT'], 'parentChemblId': 'CHEMBL_PARENT',
                'references': [{'source': 'PubMed', 'ids': ['2'], 'urls': ['u2']}],
                'targetName': 'X', 'targetType': 'single protein', 'targets': ['ENSG1'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)

        assert result.height == 1
        assert sorted(result.to_dicts()[0]['chemblIds']) == ['CHEMBL_PARENT', 'CHEMBL_SALT_A', 'CHEMBL_SALT_B']

    def test_distinct_mechanisms_are_not_merged(self) -> None:
        """Two genuinely different mechanisms on the same drug must both survive."""
        refs = [{'source': 'PubMed', 'ids': ['111'], 'urls': ['u1']}]
        data = [
            {
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist', 'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL479'], 'parentChemblId': 'CHEMBL479',
                'references': refs, 'targetName': '5-HT2a',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Dopamine D2 receptor antagonist', 'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL479'], 'parentChemblId': 'CHEMBL479',
                'references': refs, 'targetName': 'D2',
                'targetType': 'single protein', 'targets': ['ENSG2'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)

        assert result.height == 2
        assert set(result['mechanismOfAction']) == {
            'Serotonin 2a (5-HT2a) receptor antagonist',
            'Dopamine D2 receptor antagonist',
        }

    def test_child_drug_still_sees_the_mechanism(self) -> None:
        """The child molecule's own page must still resolve the merged mechanism."""
        refs = [{'source': 'PubMed', 'ids': ['111'], 'urls': ['u1']}]
        data = [
            {
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist', 'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL479'], 'parentChemblId': 'CHEMBL479',
                'references': refs, 'targetName': '5-HT2a',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist', 'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL1200916', 'CHEMBL479'], 'parentChemblId': 'CHEMBL479',
                'references': refs, 'targetName': '5-HT2a',
                'targetType': 'single protein', 'targets': ['ENSG1'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)
        exploded = result.explode('chemblIds').rename({'chemblIds': 'drugId'})

        assert exploded.filter(pl.col('drugId') == 'CHEMBL1200916').height == 1


class TestOrderIsDeterministic:
    """This step has to produce the same answer on every run, and needs help to.

    Polars promises nothing about row order out of a join, a `group_by` or a
    `unique`, and the reads are `SELECT DISTINCT` against a restored dump, so
    without both `ORDER_BY` and `maintain_order` the output differs run to run on
    byte-identical inputs. Both halves are pinned; this is the guard on them.
    """

    def test_order_by_covers_the_tables_whose_order_reaches_the_output(self) -> None:
        assert ORDER_BY == {
            'drug_mechanism': ['mec_id'],
            'mechanism_refs': ['mecref_id'],
            'target_dictionary': ['tid'],
            'target_components': ['tid', 'component_id'],
            'component_sequences': ['component_id', 'accession'],
        }

    def test_every_ordered_column_is_in_that_table_s_projection(self) -> None:
        """A SELECT DISTINCT cannot order by a column it does not select."""
        for table, columns in ORDER_BY.items():
            assert set(columns) <= set(TABLES[table]), table

    def test_mecref_id_is_read_only_so_the_refs_can_be_ordered(self) -> None:
        """It is in the projection for no other reason, so say so where it would be removed."""
        assert 'mecref_id' in TABLES['mechanism_refs']

    def test_the_output_does_not_depend_on_the_order_the_rows_arrived_in(self, tables: dict) -> None:
        """Shuffle every input, re-apply ORDER_BY as the read does, and get the same frame."""
        key = {
            'drug_mechanism': 'drug_mechanism',
            'mechanism_refs': 'mechanism_refs',
            'target_dictionary': 'target_dictionary',
            'target_components': 'target_components',
            'component_sequences': 'component_sequences',
        }

        def prepare(seed: int | None) -> dict:
            prepared = {}
            for name, df in tables.items():
                if seed is not None and isinstance(df, pl.DataFrame):
                    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
                if name in key:
                    df = df.sort(ORDER_BY[key[name]])
                prepared[name] = df
            return prepared

        baseline = process_mechanism_of_action(**prepare(None))
        for seed in (1, 2, 3):
            assert process_mechanism_of_action(**prepare(seed)).equals(baseline)
