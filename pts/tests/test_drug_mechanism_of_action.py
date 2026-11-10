"""Tests for drug_mechanism_of_action, which reads the raw ChEMBL tables straight from the dump."""

import random
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from loguru import logger

from pts.transformers import drug_mechanism_of_action as module
from pts.transformers.clinical_mira.record_store import publish_snapshot
from pts.transformers.clinical_mira.staged_outputs import MOA, MOA_LITERATURE
from pts.transformers.drug_mechanism_of_action import (
    ORDER_BY,
    TABLES,
    _chembl_target,
    _consolidate_duplicate_references,
    _with_target_chembl_id,
    drug_mechanism_of_action,
    merge_inferred,
    process_mechanism_of_action,
)
from pts.transformers.utils.dataset import scan_dataset, write_dataset

PROTEIN_ID = pl.Struct({'id': pl.String, 'source': pl.String})

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
    drug_mechanism = pl.DataFrame({
        'mec_id': [100, 101],
        'record_id': [1000, 1001],
        'molregno': [1, 3],
        'mechanism_of_action': ['Inhibits enzyme X', 'Blocks receptor'],
        'tid': [20, 21],
        'action_type': ['INHIBITOR', 'ANTAGONIST'],
    })
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
            'proteinIds': [
                [{'id': 'P100', 'source': 'uniprot_trembl'}],
                [{'id': 'P200', 'source': 'uniprot_swissprot'}, {'id': 'ENSP2', 'source': 'ensembl_PRO'}],
                [{'id': 'P300', 'source': 'uniprot_swissprot'}],
            ],
        },
        schema={'id': pl.String, 'proteinIds': pl.List(PROTEIN_ID)},
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
        exploded = result.explode('chemblIds', empty_as_null=True).rename({'chemblIds': 'drugId'})
        urls = {r['drugId']: sorted({u for ref in r['references'] for u in ref['urls']}) for r in exploded.to_dicts()}
        assert urls['CHEMBL1'] == ['http://fda/drug-one']
        assert urls['CHEMBL3'] == ['http://fda/drug-three']

    def test_the_parent_of_the_salt_is_unaffected_too(self, shared_mechanism: dict) -> None:
        """CHEMBL2 reaches the mechanism through its salt, and only through its salt."""
        result = process_mechanism_of_action(**shared_mechanism)
        exploded = result.explode('chemblIds', empty_as_null=True).rename({'chemblIds': 'drugId'})
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


class TestTargetLookup:
    """Accessions resolve through the target dataset's `proteinIds`."""

    @staticmethod
    def _targets(tables: dict, gene_df: pl.DataFrame) -> dict:
        result = _chembl_target(
            tables['target_dictionary'], tables['target_components'], tables['component_sequences'], gene_df
        )
        return {r['targetName']: r['targets'] for r in result.to_dicts()}

    def test_only_swissprot_and_trembl_accessions_resolve(self, tables: dict) -> None:
        gene_df = pl.DataFrame(
            {
                'id': ['ENSG1', 'ENSG2', 'ENSG3'],
                'proteinIds': [
                    [{'id': 'P100', 'source': 'uniprot_obsolete'}],
                    [{'id': 'P200', 'source': 'ensembl_PRO'}],
                    None,
                ],
            },
            schema={'id': pl.String, 'proteinIds': pl.List(PROTEIN_ID)},
        )
        assert self._targets(tables, gene_df) == {'Target Twenty': [], 'Target TwentyOne': []}

    def test_a_target_named_by_gene_id_resolves(self, tables: dict) -> None:
        sequences = tables['component_sequences'].with_columns(pl.col('accession').replace({'P300': 'ENSG3'}))
        result = _chembl_target(tables['target_dictionary'], tables['target_components'], sequences, tables['gene_df'])
        assert {r['targetName']: r['targets'] for r in result.to_dicts()}['Target TwentyOne'] == ['ENSG3']

    def test_genes_sharing_an_accession_come_in_id_order(self, tables: dict) -> None:
        gene_df = pl.DataFrame(
            {
                'id': ['ENSG9', 'ENSG4', 'ENSG7'],
                'proteinIds': [[{'id': 'P300', 'source': 'uniprot_swissprot'}]] * 3,
            },
            schema={'id': pl.String, 'proteinIds': pl.List(PROTEIN_ID)},
        )
        assert self._targets(tables, gene_df)['Target TwentyOne'] == ['ENSG4', 'ENSG7', 'ENSG9']
        assert self._targets(tables, gene_df.reverse())['Target TwentyOne'] == ['ENSG4', 'ENSG7', 'ENSG9']


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
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist',
                'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL479'],
                'parentChemblId': 'CHEMBL479',
                'references': refs,
                'targetName': '5-HT2a',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist',
                'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL1200916', 'CHEMBL479'],
                'parentChemblId': 'CHEMBL479',
                'references': refs,
                'targetName': '5-HT2a',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
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
                    'mechanismOfAction': 'Inhibits X',
                    'actionType': 'INHIBITOR',
                    'chemblIds': ['CHEMBL1'],
                    'parentChemblId': 'CHEMBL1',
                    'references': [],
                    'targetName': 'X',
                    'targetType': 'single protein',
                    'targets': ['ENSG1'],
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
                'mechanismOfAction': 'Inhibits enzyme X',
                'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_A'],
                'parentChemblId': 'CHEMBL_A',
                'references': drug_a_refs,
                'targetName': 'Enzyme X',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Inhibits enzyme X',
                'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_B_SALT', 'CHEMBL_B'],
                'parentChemblId': 'CHEMBL_B',
                'references': drug_b_refs,
                'targetName': 'Enzyme X',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)

        assert result.height == 2
        by_drug = {
            r['drugId']: r
            for r in result.explode('chemblIds', empty_as_null=True).rename({'chemblIds': 'drugId'}).to_dicts()
        }
        assert by_drug['CHEMBL_A']['references'] == drug_a_refs
        assert by_drug['CHEMBL_B']['references'] == drug_b_refs
        assert by_drug['CHEMBL_B_SALT']['references'] == drug_b_refs

    def test_one_drug_does_not_inherit_another_s_references(self) -> None:
        """A drug must never carry a reference belonging to a different drug."""
        data = [
            {
                'mechanismOfAction': 'Inhibits enzyme X',
                'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_A'],
                'parentChemblId': 'CHEMBL_A',
                'references': [{'source': 'FDA', 'ids': ['label-a'], 'urls': ['url-a']}],
                'targetName': 'Enzyme X',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Inhibits enzyme X',
                'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_B'],
                'parentChemblId': 'CHEMBL_B',
                'references': [{'source': 'FDA', 'ids': ['label-b'], 'urls': ['url-b']}],
                'targetName': 'Enzyme X',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)
        exploded = result.explode('chemblIds', empty_as_null=True).rename({'chemblIds': 'drugId'})
        urls = {r['drugId']: sorted({u for ref in r['references'] for u in ref['urls']}) for r in exploded.to_dicts()}

        assert urls['CHEMBL_A'] == ['url-a']
        assert urls['CHEMBL_B'] == ['url-b']

    def test_two_salts_of_the_same_parent_are_merged(self) -> None:
        """The anchor is the parent, so sibling salts still collapse onto one row."""
        data = [
            {
                'mechanismOfAction': 'Inhibits X',
                'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_SALT_A', 'CHEMBL_PARENT'],
                'parentChemblId': 'CHEMBL_PARENT',
                'references': [{'source': 'PubMed', 'ids': ['1'], 'urls': ['u1']}],
                'targetName': 'X',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Inhibits X',
                'actionType': 'INHIBITOR',
                'chemblIds': ['CHEMBL_SALT_B', 'CHEMBL_PARENT'],
                'parentChemblId': 'CHEMBL_PARENT',
                'references': [{'source': 'PubMed', 'ids': ['2'], 'urls': ['u2']}],
                'targetName': 'X',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
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
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist',
                'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL479'],
                'parentChemblId': 'CHEMBL479',
                'references': refs,
                'targetName': '5-HT2a',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Dopamine D2 receptor antagonist',
                'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL479'],
                'parentChemblId': 'CHEMBL479',
                'references': refs,
                'targetName': 'D2',
                'targetType': 'single protein',
                'targets': ['ENSG2'],
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
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist',
                'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL479'],
                'parentChemblId': 'CHEMBL479',
                'references': refs,
                'targetName': '5-HT2a',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
            {
                'mechanismOfAction': 'Serotonin 2a (5-HT2a) receptor antagonist',
                'actionType': 'ANTAGONIST',
                'chemblIds': ['CHEMBL1200916', 'CHEMBL479'],
                'parentChemblId': 'CHEMBL479',
                'references': refs,
                'targetName': '5-HT2a',
                'targetType': 'single protein',
                'targets': ['ENSG1'],
            },
        ]
        df = pl.DataFrame(data, schema=MECHANISM_SCHEMA)

        result = _consolidate_duplicate_references(df)
        exploded = result.explode('chemblIds', empty_as_null=True).rename({'chemblIds': 'drugId'})

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


CANDIDATE_SCHEMA = {
    'parentChemblId': pl.String,
    'chemblIds': pl.List(pl.String),
    'maximumClinicalStage': pl.String,
    'drugType': pl.String,
    'targetFromSource': pl.String,
    'targets': pl.List(pl.String),
    'targetSymbols': pl.List(pl.String),
    'targetType': pl.String,
    'actionType': pl.String,
    'mechanismOfAction': pl.String,
    'confidence': pl.Float64,
    'statementType': pl.String,
    'source': pl.String,
    'docIds': pl.List(pl.String),
    'quote': pl.String,
    'quoteDocId': pl.String,
}
EXCLUDE_APPROVED = {'aact': ['APPROVAL'], 'literature': ['APPROVAL']}


def candidate(**overrides: object) -> dict:
    row = {
        'parentChemblId': 'CHEMBL90',
        'chemblIds': ['CHEMBL90'],
        'maximumClinicalStage': 'PHASE_2',
        'drugType': 'Small molecule',
        'targetFromSource': 'Kinase K',
        'targets': ['ENSG9'],
        'targetSymbols': ['K'],
        'targetType': 'single protein',
        'actionType': 'INHIBITOR',
        'mechanismOfAction': 'Kinase K inhibitor',
        'confidence': 0.9,
        'statementType': 'explicit',
        'source': 'aact',
        'docIds': ['NCT01', 'NCT02'],
        'quote': 'inhibits K',
        'quoteDocId': 'NCT01',
    }
    return row | overrides


def candidates_frame(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=CANDIDATE_SCHEMA)


def paper(**overrides: object) -> dict:
    return candidate(source='literature', docIds=['pmid22'], quoteDocId='pmid22') | overrides


class TestMergeInferred:
    @pytest.fixture
    def chembl(self, tables: dict) -> pl.DataFrame:
        return process_mechanism_of_action(**tables)

    def merge(
        self,
        chembl: pl.DataFrame,
        *rows: dict,
        minimum: float = 0.8,
        exclude_stages: dict[str, list[str]] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        excluded = EXCLUDE_APPROVED if exclude_stages is None else exclude_stages
        return merge_inferred(chembl, candidates_frame(*rows), minimum, excluded)

    def test_without_aact_only_curation_is_added(self, tables: dict) -> None:
        result = process_mechanism_of_action(**tables)
        assert result.columns[-1] == 'curation'
        assert result['curation'].to_list() == ['manual'] * result.height
        assert set(result.columns) == set(MECHANISM_SCHEMA) - {'parentChemblId'} | {'curation'}

    def test_drug_with_a_chembl_mechanism_gets_no_aact_row(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(chembl, candidate(parentChemblId='CHEMBL3', chemblIds=['CHEMBL3']))
        assert merged.equals(chembl)

    def test_a_sibling_salt_with_a_chembl_mechanism_blocks_the_parent(self, chembl: pl.DataFrame) -> None:
        # CHEMBL1 is a salt of CHEMBL2 and carries the mechanism; the candidate lists the parent and another salt.
        merged, report = self.merge(chembl, candidate(parentChemblId='CHEMBL2', chemblIds=['CHEMBL2', 'CHEMBL7']))
        assert merged.equals(chembl)
        assert report['category'].to_list() == ['extra_target']
        assert report['chemblCurrentTargets'].to_list() == [['ENSG1', 'ENSG2']]

    def test_a_parent_with_a_chembl_mechanism_blocks_a_candidate_with_null_chembl_ids(
        self, chembl: pl.DataFrame
    ) -> None:
        merged, report = self.merge(chembl, candidate(parentChemblId='CHEMBL3', chemblIds=None))
        assert merged.equals(chembl)
        assert report['category'].to_list() == ['extra_target']

    def test_a_chembl_mechanism_on_a_listed_salt_blocks_the_parent(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(chembl, candidate(parentChemblId='CHEMBL77', chemblIds=['CHEMBL77', 'CHEMBL1']))
        assert merged.equals(chembl)

    def test_approved_and_low_confidence_candidates_stay_out_but_reach_the_side_file(
        self, chembl: pl.DataFrame
    ) -> None:
        merged, report = self.merge(
            chembl,
            candidate(parentChemblId='CHEMBL91', chemblIds=['CHEMBL91'], maximumClinicalStage='APPROVAL'),
            candidate(parentChemblId='CHEMBL92', chemblIds=['CHEMBL92'], confidence=0.75),
        )
        assert merged.equals(chembl)
        assert sorted(report['parentChemblId'].to_list()) == ['CHEMBL91', 'CHEMBL92']
        assert set(report['category'].to_list()) == {'new_moa'}
        assert report['chemblCurrentTargets'].to_list() == [[], []]
        assert {'quote', 'confidence', 'docIds', 'source'} <= set(report.columns)

    def test_aact_row_cites_its_trials(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(chembl, candidate(chemblIds=['CHEMBL90', 'CHEMBL93']))
        row = merged.tail(1).to_dicts()[0]
        assert row['references'] == [
            {
                'source': 'ClinicalTrials',
                'ids': ['NCT01', 'NCT02'],
                'urls': ['https://clinicaltrials.gov/study/NCT01', 'https://clinicaltrials.gov/study/NCT02'],
            }
        ]
        assert row['curation'] == 'electronic'
        assert row['chemblIds'] == ['CHEMBL90', 'CHEMBL93']
        assert row['targetName'] == 'Kinase K'
        assert merged.schema == chembl.schema
        assert merged.height == chembl.height + 1
        assert merged.head(chembl.height).equals(chembl)

    def test_duplicate_gene_set_keeps_the_most_confident(self, chembl: pl.DataFrame) -> None:
        merged, report = self.merge(
            chembl,
            candidate(confidence=0.85, mechanismOfAction='weaker', targets=['ENSG9', 'ENSG8']),
            candidate(confidence=0.95, mechanismOfAction='stronger', targets=['ENSG8', 'ENSG9']),
        )
        assert report['mechanismOfAction'].to_list() == ['stronger']
        assert merged.tail(1)['mechanismOfAction'].to_list() == ['stronger']
        assert merged.height == chembl.height + 1

    def test_extra_target_is_reported_only_when_no_gene_overlaps(self, chembl: pl.DataFrame) -> None:
        _, report = self.merge(
            chembl,
            candidate(parentChemblId='CHEMBL3', chemblIds=['CHEMBL3'], targets=['ENSG3', 'ENSG9']),
            candidate(parentChemblId='CHEMBL3', chemblIds=['CHEMBL3'], targets=['ENSG9']),
        )
        assert report['category'].to_list() == ['extra_target']
        assert report['targets'].to_list() == [['ENSG9']]
        assert report['chemblCurrentTargets'].to_list() == [['ENSG3']]

    def test_primary_key_is_unique_after_merge(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(
            chembl,
            candidate(parentChemblId='CHEMBL91', chemblIds=['CHEMBL91'], targets=['ENSG8']),
            candidate(parentChemblId='CHEMBL92', chemblIds=['CHEMBL92'], targets=['ENSG8']),
            candidate(parentChemblId='CHEMBL91', chemblIds=['CHEMBL91'], targets=['ENSG7'], mechanismOfAction='m2'),
        )
        assert merged.height == chembl.height + 3
        assert not merged.select('chemblIds', 'mechanismOfAction', 'targets').is_duplicated().any()

    def test_merge_without_literature_source_is_unchanged(self, chembl: pl.DataFrame) -> None:
        merged, report = self.merge(
            chembl,
            candidate(),
            candidate(parentChemblId='CHEMBL3', chemblIds=['CHEMBL3'], targets=['ENSG9']),
            exclude_stages={'aact': ['APPROVAL']},
        )
        assert merged.head(chembl.height).equals(chembl)
        assert merged.tail(1)['references'].to_list() == [
            [
                {
                    'source': 'ClinicalTrials',
                    'ids': ['NCT01', 'NCT02'],
                    'urls': ['https://clinicaltrials.gov/study/NCT01', 'https://clinicaltrials.gov/study/NCT02'],
                }
            ]
        ]
        assert report.select('source', 'category').rows() == [('aact', 'new_moa'), ('aact', 'extra_target')]

    def test_merge_unions_trial_and_paper_references(self, chembl: pl.DataFrame) -> None:
        merged, report = self.merge(
            chembl,
            candidate(confidence=0.85, docIds=['NCT02'], mechanismOfAction='trial'),
            paper(confidence=0.9, docIds=['pmid3', 'pmc7', 'pmid22'], targets=['ENSG9'], mechanismOfAction='paper'),
            paper(confidence=0.8, docIds=['pmid4'], maximumClinicalStage='APPROVAL', mechanismOfAction='approved'),
            candidate(confidence=0.5, docIds=['NCT77'], mechanismOfAction='weak'),
        )
        assert merged.height == chembl.height + 1
        row = merged.tail(1).to_dicts()[0]
        assert row['mechanismOfAction'] == 'paper'
        assert row['curation'] == 'electronic'
        assert row['references'] == [
            {'source': 'ClinicalTrials', 'ids': ['NCT02'], 'urls': ['https://clinicaltrials.gov/study/NCT02']},
            {
                'source': 'PubMed',
                'ids': ['3', '22'],
                'urls': ['http://europepmc.org/abstract/MED/3', 'http://europepmc.org/abstract/MED/22'],
            },
            {'source': 'PMC', 'ids': ['PMC7'], 'urls': ['http://europepmc.org/article/PMC/PMC7']},
        ]
        assert sorted(report.select('source', 'mechanismOfAction').rows()) == [
            ('aact', 'trial'),
            ('literature', 'paper'),
        ]

    def test_merge_emits_pmc_reference_for_pmc_only_doc(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(chembl, paper(docIds=['pmc7'], quoteDocId='pmc7'))
        assert merged.tail(1)['references'].to_list() == [
            [{'source': 'PMC', 'ids': ['PMC7'], 'urls': ['http://europepmc.org/article/PMC/PMC7']}]
        ]
        assert merged.schema == chembl.schema

    def test_exclude_stages_per_source(self, chembl: pl.DataFrame) -> None:
        approved = paper(maximumClinicalStage='APPROVAL')
        excluded, report = self.merge(chembl, approved)
        published, _ = self.merge(chembl, approved, exclude_stages={'aact': ['APPROVAL'], 'literature': []})
        assert excluded.equals(chembl)
        assert report['source'].to_list() == ['literature']
        assert published.height == chembl.height + 1
        assert published.tail(1)['references'].to_list() == [
            [{'source': 'PubMed', 'ids': ['22'], 'urls': ['http://europepmc.org/abstract/MED/22']}]
        ]

    def test_filtering_comes_before_deduplication(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(
            chembl,
            candidate(confidence=0.95, maximumClinicalStage='APPROVAL', mechanismOfAction='trial'),
            paper(confidence=0.85, maximumClinicalStage='APPROVAL', mechanismOfAction='paper'),
            exclude_stages={'aact': ['APPROVAL'], 'literature': []},
        )
        row = merged.tail(1).to_dicts()[0]
        assert row['mechanismOfAction'] == 'paper'
        assert [reference['source'] for reference in row['references']] == ['PubMed']

    def test_a_source_without_exclude_stages_raises(self, chembl: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match='literature'):
            self.merge(chembl, candidate(), paper(), exclude_stages={'aact': ['APPROVAL']})

    def test_trial_candidates_of_one_gene_set_union_their_trials(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(
            chembl,
            candidate(confidence=0.9, docIds=['NCT02'], mechanismOfAction='strong'),
            candidate(confidence=0.85, docIds=['NCT01', 'NCT03'], mechanismOfAction='weaker'),
        )
        row = merged.tail(1).to_dicts()[0]
        assert row['mechanismOfAction'] == 'strong'
        assert row['references'] == [
            {
                'source': 'ClinicalTrials',
                'ids': ['NCT01', 'NCT02', 'NCT03'],
                'urls': [f'https://clinicaltrials.gov/study/NCT0{i}' for i in (1, 2, 3)],
            }
        ]

    def test_pubmed_ids_sort_numerically(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(chembl, paper(docIds=['pmid22', 'pmid3', 'pmc10', 'pmc9']))
        references = merged.tail(1)['references'].to_list()[0]
        assert [(r['source'], r['ids']) for r in references] == [('PubMed', ['3', '22']), ('PMC', ['PMC10', 'PMC9'])]
        assert references[0]['urls'] == ['http://europepmc.org/abstract/MED/3', 'http://europepmc.org/abstract/MED/22']

    @staticmethod
    def inferred(merged: pl.DataFrame, chembl: pl.DataFrame) -> list[tuple[str, list[str]]]:
        return sorted(merged.slice(chembl.height).select('mechanismOfAction', 'targets').rows())

    def test_a_lower_confidence_row_contained_in_a_higher_one_is_dropped(self, chembl: pl.DataFrame) -> None:
        messages: list[str] = []
        handler = logger.add(lambda message: messages.append(message.record['message']))
        try:
            merged, report = self.merge(
                chembl,
                candidate(confidence=0.95, targets=['ENSG1', 'ENSG2'], mechanismOfAction='broad'),
                paper(confidence=0.85, targets=['ENSG1'], mechanismOfAction='narrow'),
            )
        finally:
            logger.remove(handler)
        assert self.inferred(merged, chembl) == [('broad', ['ENSG1', 'ENSG2'])]
        row = merged.tail(1).to_dicts()[0]
        assert [r['source'] for r in row['references']] == ['ClinicalTrials']
        assert sorted(report['mechanismOfAction'].to_list()) == ['broad', 'narrow']
        assert any(m.startswith('1 inferred rows not published') and 'subset' in m for m in messages), messages

    def test_partially_overlapping_rows_are_both_kept(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(
            chembl,
            candidate(confidence=0.95, targets=['ENSG1', 'ENSG2'], mechanismOfAction='a'),
            paper(confidence=0.85, targets=['ENSG2', 'ENSG3'], mechanismOfAction='b'),
        )
        assert self.inferred(merged, chembl) == [('a', ['ENSG1', 'ENSG2']), ('b', ['ENSG2', 'ENSG3'])]

    def test_a_narrower_higher_row_does_not_drop_a_broader_lower_one(self, chembl: pl.DataFrame) -> None:
        broad = ['ENSG1', 'ENSG2', 'ENSG3', 'ENSG4', 'ENSG5']
        merged, _ = self.merge(
            chembl,
            candidate(confidence=0.95, targets=['ENSG1'], mechanismOfAction='MC1R'),
            paper(confidence=0.85, targets=broad, mechanismOfAction='MC1R-MC5R'),
        )
        assert self.inferred(merged, chembl) == [('MC1R', ['ENSG1']), ('MC1R-MC5R', broad)]

    def test_disjoint_rows_of_one_drug_are_both_kept(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(
            chembl,
            candidate(confidence=0.95, targets=['ENSG1'], mechanismOfAction='a'),
            paper(confidence=0.85, targets=['ENSG2'], mechanismOfAction='b'),
        )
        assert self.inferred(merged, chembl) == [('a', ['ENSG1']), ('b', ['ENSG2'])]

    def test_contained_rows_of_different_drugs_are_both_kept(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(
            chembl,
            candidate(confidence=0.95, targets=['ENSG1', 'ENSG2'], mechanismOfAction='a'),
            candidate(
                parentChemblId='CHEMBL91',
                chemblIds=['CHEMBL91'],
                confidence=0.85,
                targets=['ENSG1'],
                mechanismOfAction='b',
            ),
        )
        assert self.inferred(merged, chembl) == [('a', ['ENSG1', 'ENSG2']), ('b', ['ENSG1'])]

    def test_a_chain_drops_only_the_contained_rows(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(
            chembl,
            paper(confidence=0.8, docIds=['pmid4'], targets=['ENSG4'], mechanismOfAction='D'),
            paper(confidence=0.85, docIds=['pmid3'], targets=['ENSG3', 'ENSG4'], mechanismOfAction='C'),
            candidate(confidence=0.95, targets=['ENSG1', 'ENSG2', 'ENSG3'], mechanismOfAction='A'),
            paper(confidence=0.9, docIds=['pmid2'], targets=['ENSG2', 'ENSG3'], mechanismOfAction='B'),
        )
        assert self.inferred(merged, chembl) == [('A', ['ENSG1', 'ENSG2', 'ENSG3']), ('C', ['ENSG3', 'ENSG4'])]

    def test_a_row_without_a_reference_does_not_drop_a_contained_row(self, chembl: pl.DataFrame) -> None:
        merged, _ = self.merge(
            chembl,
            paper(confidence=0.95, docIds=['doi10.1/x'], targets=['ENSG1', 'ENSG2'], mechanismOfAction='unreferenced'),
            candidate(confidence=0.85, targets=['ENSG1'], mechanismOfAction='referenced'),
        )
        assert self.inferred(merged, chembl) == [('referenced', ['ENSG1'])]

    def test_an_equal_confidence_containment_is_resolved_the_same_way_under_shuffle(self, chembl: pl.DataFrame) -> None:
        rows = [
            paper(
                confidence=0.9, docIds=['pmid1', 'pmid2'], targets=['ENSG1', 'ENSG2'], mechanismOfAction='wide two docs'
            ),
            paper(confidence=0.9, docIds=['pmid3'], targets=['ENSG1'], mechanismOfAction='narrow one doc'),
            paper(confidence=0.9, docIds=['pmid5', 'pmid6'], targets=['ENSG5'], mechanismOfAction='narrow two docs'),
            paper(confidence=0.9, docIds=['pmid7'], targets=['ENSG5', 'ENSG6'], mechanismOfAction='wide one doc'),
            paper(confidence=0.9, docIds=['pmid8'], targets=['ENSG3', 'ENSG4'], mechanismOfAction='wide key'),
            paper(confidence=0.9, docIds=['pmid9'], targets=['ENSG4'], mechanismOfAction='narrow later key'),
        ]
        expected = [
            ('narrow two docs', ['ENSG5']),
            ('wide key', ['ENSG3', 'ENSG4']),
            ('wide one doc', ['ENSG5', 'ENSG6']),
            ('wide two docs', ['ENSG1', 'ENSG2']),
        ]
        for seed in range(8):
            shuffled = rows[:]
            random.Random(seed).shuffle(shuffled)
            merged, _ = self.merge(chembl, *shuffled)
            assert self.inferred(merged, chembl) == expected

    def test_literature_doc_ids_are_classified_by_what_follows_the_prefix(self, chembl: pl.DataFrame) -> None:
        messages: list[str] = []
        handler = logger.add(lambda message: messages.append(message.record['message']))
        try:
            merged, _ = self.merge(
                chembl,
                paper(docIds=['NCT05', 'pmidAGR123', 'pmc777', 'pmidppr540062', 'pmidPMC777', 'pmid12345']),
            )
        finally:
            logger.remove(handler)
        assert merged.tail(1)['references'].to_list() == [
            [
                {'source': 'ClinicalTrials', 'ids': ['NCT05'], 'urls': ['https://clinicaltrials.gov/study/NCT05']},
                {'source': 'PubMed', 'ids': ['12345'], 'urls': ['http://europepmc.org/abstract/MED/12345']},
                {'source': 'PMC', 'ids': ['PMC777'], 'urls': ['http://europepmc.org/article/PMC/PMC777']},
                {
                    'source': 'EuropePMC',
                    'ids': ['PPR540062'],
                    'urls': ['https://europepmc.org/article/PPR/PPR540062'],
                },
            ]
        ]
        assert any(m.startswith('1 doc ids') and 'not referenced' in m for m in messages), messages

    @pytest.mark.parametrize(
        ('doc_id', 'source', 'ids'),
        [
            ('pmid12345', 'PubMed', ['12345']),
            ('pmidppr540062', 'EuropePMC', ['PPR540062']),
            ('pmidPPR540062', 'EuropePMC', ['PPR540062']),
            ('pmidPMC777', 'PMC', ['PMC777']),
            ('pmidpmc777', 'PMC', ['PMC777']),
            ('pmc777', 'PMC', ['PMC777']),
            ('PMC777', 'PMC', ['PMC777']),
        ],
    )
    def test_one_literature_doc_id_gives_one_reference(
        self, chembl: pl.DataFrame, doc_id: str, source: str, ids: list[str]
    ) -> None:
        merged, _ = self.merge(chembl, paper(docIds=[doc_id], quoteDocId=doc_id))
        references = merged.tail(1)['references'].to_list()[0]
        assert [(r['source'], r['ids']) for r in references] == [(source, ids)]

    @pytest.mark.parametrize('doc_id', ['pmidAGR123', 'pmid12a', 'pmidPMC', 'pmcPPR1', 'pmid'])
    def test_a_literature_doc_id_of_no_known_kind_is_not_published(self, chembl: pl.DataFrame, doc_id: str) -> None:
        merged, _ = self.merge(chembl, paper(docIds=[doc_id], quoteDocId=doc_id))
        assert merged.equals(chembl)

    def test_a_candidate_without_a_known_doc_id_is_not_published(self, chembl: pl.DataFrame) -> None:
        messages: list[str] = []
        handler = logger.add(lambda message: messages.append(message.record['message']))
        try:
            merged, report = self.merge(chembl, paper(docIds=['doi10.1/x'], quoteDocId='doi10.1/x'))
        finally:
            logger.remove(handler)
        assert merged.equals(chembl)
        assert report['docIds'].to_list() == [['doi10.1/x']]
        assert any('1 candidates' in m and 'no reference' in m for m in messages), messages


LABELS = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))


def _snapshot_moa(columns: dict) -> pl.DataFrame:
    return pl.DataFrame({name: [None] for name in columns}, schema=columns)


class TestEntryPoint:
    @pytest.fixture
    def staged(self, tables: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
        monkeypatch.setattr(module, 'read_dump_tables', lambda *_a, **_k: tables)
        target = tables['gene_df'].with_columns(
            approvedSymbol=pl.lit('S'),
            biotype=pl.lit('protein_coding'),
            genomicLocation=pl.lit({'chromosome': '1', 'start': 1, 'end': 2, 'strand': 1}),
            approvedName=pl.lit('n'),
            symbolSynonyms=pl.lit([], dtype=LABELS),
            obsoleteSymbols=pl.lit([], dtype=LABELS),
            nameSynonyms=pl.lit([], dtype=LABELS),
        )
        write_dataset(target, str(tmp_path / 'target'))
        write_dataset(
            pl.DataFrame(
                {'id': ['NCT1'], 'source': ['x'], 'drugs': [[]], 'clinicalStage': ['PHASE_1']},
                schema={
                    'id': pl.String,
                    'source': pl.String,
                    'drugs': pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String})),
                    'clinicalStage': pl.String,
                },
            ),
            str(tmp_path / 'clinical_report'),
        )
        write_dataset(
            pl.DataFrame(
                {'id': ['CHEMBL90'], 'parentId': [None], 'childChemblIds': [[]], 'drugType': ['Small molecule']},
                schema={
                    'id': pl.String,
                    'parentId': pl.String,
                    'childChemblIds': pl.List(pl.String),
                    'drugType': pl.String,
                },
            ),
            str(tmp_path / 'chembl_molecule'),
        )
        publish_snapshot(_snapshot_moa({'drugName': pl.String}), str(tmp_path / 'aact'), MOA.metadata('x'), None)
        publish_snapshot(
            _snapshot_moa({'drugId': pl.String}), str(tmp_path / 'literature'), MOA_LITERATURE.metadata('x'), None
        )
        calls: dict[str, dict] = {}

        def fake(name: str, row: dict):
            def build(moa: pl.DataFrame, *args: object, **kwargs: object) -> pl.DataFrame:
                calls[name] = {'columns': moa.columns, 'args': args, 'kwargs': kwargs}
                return candidates_frame(row)

            return build

        monkeypatch.setattr(module, 'build_trial_candidates', fake('aact', candidate()))
        monkeypatch.setattr(
            module,
            'build_literature_candidates',
            fake('literature', paper(parentChemblId='CHEMBL95', chemblIds=['CHEMBL95'], targets=['ENSG5'])),
        )
        return {
            'chembl': Path('dump'),
            'target': tmp_path / 'target',
            'clinical_report': tmp_path / 'clinical_report',
            'chembl_molecule': tmp_path / 'chembl_molecule',
            'aact': tmp_path / 'aact',
            'literature': tmp_path / 'literature',
            'root': tmp_path,
            'calls': calls,
        }

    @staticmethod
    def settings() -> dict:
        return {
            'release_min_confidence': 0.8,
            'exclude_stages': EXCLUDE_APPROVED,
            'report_min_confidence': 0.7,
            'link_share': 0.8,
            'class_level_genes': 20,
            'suggestion_min_confidence': 0.8,
        }

    @staticmethod
    def sources(staged: dict, *names: str) -> dict:
        return {name: staged[name] for name in ('chembl', 'target', 'clinical_report', 'chembl_molecule', *names)}

    @staticmethod
    def destinations(root: Path) -> dict:
        return {'output': root / 'out', 'chembl_report': root / 'report', 'candidates': root / 'candidates'}

    def test_a_plain_destination_without_aact_writes_the_chembl_rows(self, staged: dict) -> None:
        out = staged['root'] / 'out'
        drug_mechanism_of_action(
            {'chembl': Path('dump'), 'target': staged['target']}, out, {}, SimpleNamespace(work_path=staged['root'])
        )
        result = scan_dataset(str(out)).collect()
        assert result.height == 2
        assert set(result['curation']) == {'manual'}

    def test_without_either_source_no_candidates_are_written(self, staged: dict) -> None:
        root = staged['root']
        drug_mechanism_of_action(
            self.sources(staged), self.destinations(root), self.settings(), SimpleNamespace(work_path=root)
        )
        assert set(scan_dataset(str(root / 'out')).collect()['curation']) == {'manual'}
        assert not (root / 'candidates').exists()
        assert not (root / 'report').exists()
        assert staged['calls'] == {}

    def test_both_sources_write_candidates_report_and_output(self, staged: dict) -> None:
        root = staged['root']
        drug_mechanism_of_action(
            self.sources(staged, 'aact', 'literature'),
            self.destinations(root),
            self.settings(),
            SimpleNamespace(work_path=root),
        )
        out = scan_dataset(str(root / 'out')).collect()
        assert sorted(out['curation'].to_list()) == ['electronic', 'electronic', 'manual', 'manual']
        assert sorted(scan_dataset(str(root / 'candidates')).collect()['source'].to_list()) == ['aact', 'literature']
        assert sorted(scan_dataset(str(root / 'report')).collect()['source'].to_list()) == ['aact', 'literature']
        assert staged['calls']['aact']['columns'] == ['drugName']
        assert staged['calls']['aact']['kwargs']['link_share'] == 0.8
        assert staged['calls']['literature']['columns'] == ['drugId']
        assert 'link_share' not in staged['calls']['literature']['kwargs']

    def test_the_trial_source_alone_publishes_trial_rows(self, staged: dict) -> None:
        root = staged['root']
        drug_mechanism_of_action(
            self.sources(staged, 'aact'),
            self.destinations(root),
            self.settings() | {'exclude_stages': {'aact': ['APPROVAL']}},
            SimpleNamespace(work_path=root),
        )
        out = scan_dataset(str(root / 'out')).collect()
        assert sorted(out['curation'].to_list()) == ['electronic', 'manual', 'manual']
        assert scan_dataset(str(root / 'candidates')).collect()['source'].to_list() == ['aact']

    def test_a_snapshot_of_the_wrong_kind_is_refused(self, staged: dict) -> None:
        root = staged['root']
        with pytest.raises(ValueError, match='kind'):
            drug_mechanism_of_action(
                self.sources(staged) | {'literature': staged['aact']},
                {'output': root / 'out'},
                self.settings(),
                SimpleNamespace(work_path=root),
            )
