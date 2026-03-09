"""Tests for replace_obsolete_terms and its helper _replace_obsolete_in_column."""

from __future__ import annotations

import polars as pl
import pytest

from pts.transformers.disease import _replace_obsolete_in_column, replace_obsolete_terms

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

LIST_STR = pl.List(pl.String)


@pytest.fixture
def base_df() -> pl.DataFrame:
    """Minimal disease DataFrame with one obsolete term per relationship column.

    Layout
    ------
    - EFO_1  : the current disease; its old term was 'EFO_OLD'
    - EFO_2  : has EFO_OLD in parents  -> should become EFO_1
    - EFO_3  : has EFO_OLD in children -> should become EFO_1
    - EFO_4  : has EFO_OLD in ancestors -> should become EFO_1
    - EFO_5  : has EFO_OLD in descendants -> should become EFO_1
    - EFO_6  : has no obsolete references at all -> must remain unchanged
    """
    return pl.DataFrame(
        {
            'id': ['EFO_1', 'EFO_2', 'EFO_3', 'EFO_4', 'EFO_5', 'EFO_6'],
            'obsoleteTerms': [
                ['EFO_OLD'],
                [],
                [],
                [],
                [],
                [],
            ],
            'parents': [
                [],
                ['EFO_OLD'],
                [],
                [],
                [],
                ['EFO_2'],
            ],
            'children': [
                [],
                [],
                ['EFO_OLD'],
                [],
                [],
                [],
            ],
            'ancestors': [
                [],
                [],
                [],
                ['EFO_OLD'],
                [],
                ['EFO_2', 'EFO_1'],
            ],
            'descendants': [
                [],
                [],
                [],
                [],
                ['EFO_OLD'],
                [],
            ],
        },
        schema={
            'id': pl.String,
            'obsoleteTerms': LIST_STR,
            'parents': LIST_STR,
            'children': LIST_STR,
            'ancestors': LIST_STR,
            'descendants': LIST_STR,
        },
    )


# ---------------------------------------------------------------------------
# Tests for replace_obsolete_terms
# ---------------------------------------------------------------------------


class TestReplaceObsoleteTerms:
    """Tests for the public replace_obsolete_terms function."""

    @pytest.fixture(autouse=True)
    def _run(self, base_df: pl.DataFrame) -> None:
        self.input_df = base_df
        self.result = replace_obsolete_terms(base_df)

    def test_returns_dataframe(self) -> None:
        """Return type must be a polars DataFrame."""
        assert isinstance(self.result, pl.DataFrame)

    def test_schema_preserved(self) -> None:
        """Output schema must be identical to input schema."""
        assert self.result.schema == self.input_df.schema

    def test_row_count_preserved(self) -> None:
        """Row count must not change."""
        assert self.result.height == self.input_df.height

    @pytest.mark.parametrize(
        'col_name',
        ['parents', 'children', 'ancestors', 'descendants'],
    )
    def test_obsolete_term_replaced_in_column(self, col_name: str) -> None:
        """EFO_OLD must not appear in any relationship column after replacement."""
        all_values = self.result[col_name].explode().drop_nulls().to_list()
        assert 'EFO_OLD' not in all_values, f"Obsolete term 'EFO_OLD' still present in column '{col_name}'"

    @pytest.mark.parametrize(
        'col_name',
        ['parents', 'children', 'ancestors', 'descendants'],
    )
    def test_obsolete_term_replaced_with_current_id(self, col_name: str) -> None:
        """Every occurrence of EFO_OLD must be replaced by its current id EFO_1."""
        # Find the row that originally had EFO_OLD in this column.
        row_id = self.input_df.filter(pl.col(col_name).list.contains('EFO_OLD'))['id'].to_list()
        assert len(row_id) == 1, 'Fixture should have exactly one row with EFO_OLD per column'
        fixed_values = self.result.filter(pl.col('id') == row_id[0])[col_name][0].to_list()
        assert 'EFO_1' in fixed_values

    def test_non_obsolete_references_untouched(self) -> None:
        """Valid references (EFO_2, EFO_1) in EFO_6 must remain unchanged."""
        efo6_ancestors_before = set(self.input_df.filter(pl.col('id') == 'EFO_6')['ancestors'][0].to_list())
        efo6_ancestors_after = set(self.result.filter(pl.col('id') == 'EFO_6')['ancestors'][0].to_list())
        assert efo6_ancestors_before == efo6_ancestors_after

    def test_obsolete_terms_column_unchanged(self) -> None:
        """The obsoleteTerms column itself must not be modified."""
        before = self.input_df.select(['id', 'obsoleteTerms']).sort('id')
        after = self.result.select(['id', 'obsoleteTerms']).sort('id')
        assert before.equals(after)

    def test_id_column_unchanged(self) -> None:
        """The id column must remain identical."""
        assert self.result['id'].sort().to_list() == self.input_df['id'].sort().to_list()


# ---------------------------------------------------------------------------
# Tests for _replace_obsolete_in_column
# ---------------------------------------------------------------------------


class TestReplaceObsoleteInColumn:
    """Tests for the private helper _replace_obsolete_in_column."""

    @pytest.fixture
    def obsolete_map(self) -> pl.DataFrame:
        """Simple mapping: EFO_OLD -> EFO_1."""
        return pl.DataFrame(
            {'obsolete_term': ['EFO_OLD'], 'current_id': ['EFO_1']},
            schema={'obsolete_term': pl.String, 'current_id': pl.String},
        )

    def test_replaces_single_match(self, base_df: pl.DataFrame, obsolete_map: pl.DataFrame) -> None:
        """A single obsolete term inside a list is replaced correctly."""
        result = _replace_obsolete_in_column(base_df, 'parents', obsolete_map)
        parents_efo2 = result.filter(pl.col('id') == 'EFO_2')['parents'][0].to_list()
        assert 'EFO_OLD' not in parents_efo2
        assert 'EFO_1' in parents_efo2

    def test_schema_preserved(self, base_df: pl.DataFrame, obsolete_map: pl.DataFrame) -> None:
        """Schema of the returned DataFrame must have the same columns and types (order may differ)."""
        result = _replace_obsolete_in_column(base_df, 'parents', obsolete_map)
        assert dict(result.schema) == dict(base_df.schema)

    def test_rows_without_match_unchanged(self, base_df: pl.DataFrame, obsolete_map: pl.DataFrame) -> None:
        """Rows that do not reference any obsolete term must be unchanged."""
        result = _replace_obsolete_in_column(base_df, 'parents', obsolete_map)
        # EFO_6 has ['EFO_2'] in parents - no obsolete reference.
        before = base_df.filter(pl.col('id') == 'EFO_6')['parents'][0].to_list()
        after = result.filter(pl.col('id') == 'EFO_6')['parents'][0].to_list()
        assert sorted(before) == sorted(after)

    def test_empty_lists_stay_empty(self, base_df: pl.DataFrame, obsolete_map: pl.DataFrame) -> None:
        """Rows with an empty list in the target column stay empty."""
        result = _replace_obsolete_in_column(base_df, 'parents', obsolete_map)
        # EFO_1 has [] in parents.
        parents_efo1 = result.filter(pl.col('id') == 'EFO_1')['parents'][0].to_list()
        assert parents_efo1 == []

    def test_multiple_obsolete_terms_in_one_row(self, obsolete_map: pl.DataFrame) -> None:
        """Multiple obsolete terms in the same list cell are all replaced."""
        extra_map = pl.concat([
            obsolete_map,
            pl.DataFrame(
                {'obsolete_term': ['EFO_OLD2'], 'current_id': ['EFO_2']},
                schema={'obsolete_term': pl.String, 'current_id': pl.String},
            ),
        ])
        df = pl.DataFrame(
            {
                'id': ['EFO_X'],
                'parents': [['EFO_OLD', 'EFO_OLD2', 'EFO_3']],
            },
            schema={'id': pl.String, 'parents': LIST_STR},
        )
        result = _replace_obsolete_in_column(df, 'parents', extra_map)
        parents = result['parents'][0].to_list()
        assert 'EFO_OLD' not in parents
        assert 'EFO_OLD2' not in parents
        assert 'EFO_1' in parents
        assert 'EFO_2' in parents
        assert 'EFO_3' in parents
