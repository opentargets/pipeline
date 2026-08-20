"""Tests for drug_warning, which reads the raw ChEMBL tables straight from the dump."""

import polars as pl
import pytest

from pts.transformers.drug_warning import ORDER_BY, _deduplicate_warnings, process_drug_warnings

REFERENCE_SCHEMA = pl.Struct({'id': pl.String, 'source': pl.String, 'url': pl.String})

WARNING_SCHEMA = {
    'chemblIds': pl.List(pl.String),
    'toxicityClass': pl.String,
    'country': pl.String,
    'description': pl.String,
    'id': pl.Int64,
    'references': pl.List(REFERENCE_SCHEMA),
    'warningType': pl.String,
    'year': pl.Int32,
    'efoTerm': pl.String,
    'efoId': pl.String,
    'efoIdForWarningClass': pl.String,
}


@pytest.fixture
def tables() -> dict:
    warnings = pl.DataFrame(
        {
            'warning_id': [10, 11],
            'molregno': [1, 3],
            'warning_type': ['Withdrawn', 'Warning'],
            'warning_class': ['Cardiotoxicity', None],
            'warning_country': ['France', 'US'],
            'warning_description': ['bad things', None],
            'warning_year': [2009, None],
            'efo_term': ['term', None],
            'efo_id': ['EFO_1', None],
            'efo_id_for_warning_class': ['EFO_2', None],
        },
        schema_overrides={'warning_year': pl.Int32},
    )
    refs = pl.DataFrame({
        'warnref_id': [1, 2],
        'warning_id': [10, 10],
        'ref_type': ['ISBN', 'DOI'],
        'ref_id': ['ref-a', 'ref-b'],
        'ref_url': ['http://a', 'http://b'],
    })
    molecules = pl.DataFrame({
        'molregno': [1, 2, 3],
        'chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
    })
    hierarchy = pl.DataFrame({'molregno': [1, 2, 3], 'parent_molregno': [2, 2, 3]})
    return {'warnings': warnings, 'refs': refs, 'molecules': molecules, 'hierarchy': hierarchy}


def rows_by_id(df: pl.DataFrame) -> dict:
    return {r['id']: r for r in df.to_dicts()}


class TestProcessDrugWarnings:
    def test_one_row_per_warning(self, tables: dict) -> None:
        result = process_drug_warnings(**tables)
        # assert on the raw count BEFORE keying, so a fan-out cannot hide
        assert result.height == 2
        assert sorted(rows_by_id(result)) == [10, 11]

    def test_scalar_fields(self, tables: dict) -> None:
        w = rows_by_id(process_drug_warnings(**tables))[10]
        assert w['warningType'] == 'Withdrawn'
        assert w['toxicityClass'] == 'Cardiotoxicity'
        assert w['country'] == 'France'
        assert w['description'] == 'bad things'
        assert w['year'] == 2009
        assert w['efoTerm'] == 'term'
        assert w['efoId'] == 'EFO_1'
        assert w['efoIdForWarningClass'] == 'EFO_2'

    def test_year_is_int32(self, tables: dict) -> None:
        """`year` is int32, the width `warning_year` has in ChEMBL. Pins the published type."""
        assert process_drug_warnings(**tables).schema['year'] == pl.Int32

    def test_chembl_ids_has_molecule_and_parent(self, tables: dict) -> None:
        # Order is [molecule, parent] by construction; assert it directly rather
        # than sorting, so swapping the concat_list(...) arguments would fail here.
        assert rows_by_id(process_drug_warnings(**tables))[10]['chemblIds'] == ['CHEMBL1', 'CHEMBL2']

    def test_chembl_ids_deduplicates_a_self_parent(self, tables: dict) -> None:
        assert rows_by_id(process_drug_warnings(**tables))[11]['chemblIds'] == ['CHEMBL3']

    def test_references(self, tables: dict) -> None:
        refs = rows_by_id(process_drug_warnings(**tables))[10]['references']
        assert {r['source'] for r in refs} == {'ISBN', 'DOI'}
        assert {r['id'] for r in refs} == {'ref-a', 'ref-b'}
        assert {r['url'] for r in refs} == {'http://a', 'http://b'}

    def test_no_references_is_an_empty_array_not_null(self, tables: dict) -> None:
        assert rows_by_id(process_drug_warnings(**tables))[11]['references'] == []


class TestDeduplicateWarnings:
    def test_parent_and_child_warning_are_merged(self) -> None:
        """A parent's native warning and its child's rolled-up copy collapse into one row.

        Reproduces the CHEMBL479 case: warning 3961 belongs to CHEMBL479 alone,
        warning 4517 belongs to CHEMBL1200916 but ChEMBL's own
        `_metadata.all_molecule_chembl_ids` rolls it up to include CHEMBL479 too.
        """
        ref = [{'id': '22941581', 'source': 'PubMed', 'url': 'http://europepmc.org/abstract/MED/22941581'}]
        data = [
            {
                'chemblIds': ['CHEMBL479'], 'toxicityClass': 'cardiotoxicity', 'country': 'Worldwide',
                'description': 'Cardiac arrythmias', 'id': 3961, 'references': ref, 'warningType': 'Withdrawn',
                'year': 2005, 'efoTerm': 'cardiac arrhythmia', 'efoId': 'EFO:0004269',
                'efoIdForWarningClass': 'EFO:1001482',
            },
            {
                'chemblIds': ['CHEMBL1200916', 'CHEMBL479'], 'toxicityClass': 'cardiotoxicity', 'country': 'Worldwide',
                'description': 'Cardiac arrythmias', 'id': 4517, 'references': ref, 'warningType': 'Withdrawn',
                'year': 2005, 'efoTerm': 'cardiac arrhythmia', 'efoId': 'EFO:0004269',
                'efoIdForWarningClass': 'EFO:1001482',
            },
        ]
        df = pl.DataFrame(data, schema=WARNING_SCHEMA)

        result = _deduplicate_warnings(df)
        rows = result.to_dicts()

        assert len(rows) == 1
        assert rows[0]['id'] == 3961
        assert sorted(rows[0]['chemblIds']) == ['CHEMBL1200916', 'CHEMBL479']

    def test_distinct_warnings_are_not_merged(self) -> None:
        """Two genuinely different warnings on the same drug must both survive."""
        data = [
            {
                'chemblIds': ['CHEMBL479'], 'toxicityClass': 'cardiotoxicity', 'country': 'Worldwide',
                'description': 'Cardiac arrythmias', 'id': 3961, 'references': [], 'warningType': 'Withdrawn',
                'year': 2005, 'efoTerm': 'cardiac arrhythmia', 'efoId': 'EFO:0004269',
                'efoIdForWarningClass': 'EFO:1001482',
            },
            {
                'chemblIds': ['CHEMBL479'], 'toxicityClass': None, 'country': 'United States',
                'description': None, 'id': 100, 'references': [], 'warningType': 'Black Box Warning',
                'year': None, 'efoTerm': None, 'efoId': None, 'efoIdForWarningClass': None,
            },
        ]
        df = pl.DataFrame(data, schema=WARNING_SCHEMA)

        result = _deduplicate_warnings(df)

        assert result.height == 2
        assert set(result['warningType']) == {'Withdrawn', 'Black Box Warning'}

    def test_child_drug_still_sees_the_warning(self) -> None:
        """The child molecule's own page must still resolve the merged warning."""
        ref = [{'id': '22941581', 'source': 'PubMed', 'url': 'http://europepmc.org/abstract/MED/22941581'}]
        data = [
            {
                'chemblIds': ['CHEMBL479'], 'toxicityClass': 'cardiotoxicity', 'country': 'Worldwide',
                'description': 'Cardiac arrythmias', 'id': 3961, 'references': ref, 'warningType': 'Withdrawn',
                'year': 2005, 'efoTerm': 'cardiac arrhythmia', 'efoId': 'EFO:0004269',
                'efoIdForWarningClass': 'EFO:1001482',
            },
            {
                'chemblIds': ['CHEMBL1200916', 'CHEMBL479'], 'toxicityClass': 'cardiotoxicity', 'country': 'Worldwide',
                'description': 'Cardiac arrythmias', 'id': 4517, 'references': ref, 'warningType': 'Withdrawn',
                'year': 2005, 'efoTerm': 'cardiac arrhythmia', 'efoId': 'EFO:0004269',
                'efoIdForWarningClass': 'EFO:1001482',
            },
        ]
        df = pl.DataFrame(data, schema=WARNING_SCHEMA)

        result = _deduplicate_warnings(df)
        exploded = result.explode('chemblIds').rename({'chemblIds': 'drugId'})

        assert exploded.filter(pl.col('drugId') == 'CHEMBL1200916').height == 1


class TestOrderIsDeterministic:
    """`references` is a published array whose element order must not float.

    Two things have to hold together: the read is ordered (``ORDER_BY``), and the
    polars pipeline keeps that order through its joins and group_bys. Either one
    alone leaves the array at the mercy of the query plan.
    """

    def test_order_by_covers_the_tables_whose_order_reaches_the_output(self) -> None:
        """`warning_refs` becomes the references array; `drug_warning` drives row order."""
        assert ORDER_BY == {'drug_warning': ['warning_id'], 'warning_refs': ['warnref_id']}

    def test_every_ordered_column_is_in_that_table_s_projection(self) -> None:
        """A SELECT DISTINCT cannot order by a column it does not select."""
        from pts.transformers.drug_warning import TABLES

        for table, columns in ORDER_BY.items():
            assert set(columns) <= set(TABLES[table]), table

    def test_the_output_does_not_depend_on_the_order_the_rows_arrived_in(self, tables: dict) -> None:
        """Shuffle the inputs, re-apply ORDER_BY as the read does, and get the same frame."""
        ordered = {
            'warnings': tables['warnings'].sort(ORDER_BY['drug_warning']),
            'refs': tables['refs'].sort(ORDER_BY['warning_refs']),
            'molecules': tables['molecules'],
            'hierarchy': tables['hierarchy'],
        }
        baseline = _deduplicate_warnings(process_drug_warnings(**ordered))

        for seed in (1, 2, 3):
            shuffled = {
                'warnings': tables['warnings'].sample(fraction=1.0, shuffle=True, seed=seed).sort(
                    ORDER_BY['drug_warning']
                ),
                'refs': tables['refs'].sample(fraction=1.0, shuffle=True, seed=seed).sort(ORDER_BY['warning_refs']),
                'molecules': tables['molecules'].sample(fraction=1.0, shuffle=True, seed=seed),
                'hierarchy': tables['hierarchy'].sample(fraction=1.0, shuffle=True, seed=seed),
            }
            assert _deduplicate_warnings(process_drug_warnings(**shuffled)).equals(baseline)
