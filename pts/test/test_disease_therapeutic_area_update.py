"""Tests for update_therapeutic_areas."""

from __future__ import annotations

import polars as pl
import pytest

from pts.transformers.disease import update_therapeutic_areas

LIST_STR = pl.List(pl.String)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_df() -> pl.DataFrame:
    """Minimal disease DataFrame to exercise TA propagation.

    Layout
    ------
    - TA_1      : therapeutic area; self-references itself in therapeuticAreas
    - TA_2      : second therapeutic area; self-references itself
    - DISEASE_A : direct child of TA_1; already has correct TA
    - DISEASE_B : child of DISEASE_A; missing TA (should inherit TA_1 via ancestor)
    - DISEASE_C : child of both TA_1 and TA_2; currently only has TA_1 (should gain TA_2)
    - DISEASE_D : no ancestors, not a TA; should remain empty
    """
    return pl.DataFrame(
        {
            'id': ['TA_1', 'TA_2', 'DISEASE_A', 'DISEASE_B', 'DISEASE_C', 'DISEASE_D'],
            'ancestors': [
                [],
                [],
                ['TA_1'],
                ['DISEASE_A', 'TA_1'],
                ['TA_1', 'TA_2'],
                [],
            ],
            'therapeuticAreas': [
                ['TA_1'],
                ['TA_2'],
                ['TA_1'],
                [],           # missing — should be filled in
                ['TA_1'],     # incomplete — should gain TA_2
                [],           # no TA ancestors — should remain empty
            ],
        },
        schema={
            'id': pl.String,
            'ancestors': LIST_STR,
            'therapeuticAreas': LIST_STR,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpdateTherapeuticAreas:
    """Tests for update_therapeutic_areas."""

    @pytest.fixture(autouse=True)
    def _run(self, base_df: pl.DataFrame) -> None:
        self.input_df = base_df
        self.result = update_therapeutic_areas(base_df)

    # -- structural checks ---------------------------------------------------

    def test_returns_dataframe(self) -> None:
        """Return type must be a polars DataFrame."""
        assert isinstance(self.result, pl.DataFrame)

    def test_schema_preserved(self) -> None:
        """Output schema must match input schema."""
        assert dict(self.result.schema) == dict(self.input_df.schema)

    def test_row_count_preserved(self) -> None:
        """Row count must not change."""
        assert self.result.height == self.input_df.height

    def test_ids_unchanged(self) -> None:
        """The id column must be identical after the update."""
        assert sorted(self.result['id'].to_list()) == sorted(self.input_df['id'].to_list())

    # -- propagation correctness ---------------------------------------------

    def test_missing_ta_filled_from_ancestor(self) -> None:
        """DISEASE_B has no TAs but its ancestor chain includes TA_1 — must gain TA_1."""
        tas = self.result.filter(pl.col('id') == 'DISEASE_B')['therapeuticAreas'][0].to_list()
        assert 'TA_1' in tas

    def test_incomplete_ta_filled_from_ancestor(self) -> None:
        """DISEASE_C had only TA_1 but also has TA_2 in its ancestor chain — must gain TA_2."""
        tas = self.result.filter(pl.col('id') == 'DISEASE_C')['therapeuticAreas'][0].to_list()
        assert 'TA_1' in tas
        assert 'TA_2' in tas

    def test_no_ta_when_no_ta_ancestors(self) -> None:
        """DISEASE_D has no ancestors and is not a TA — therapeuticAreas must stay empty."""
        tas = self.result.filter(pl.col('id') == 'DISEASE_D')['therapeuticAreas'][0].to_list()
        assert tas == []

    def test_therapeutic_area_retains_self_reference(self) -> None:
        """A therapeutic area node must keep itself in therapeuticAreas."""
        for ta_id in ['TA_1', 'TA_2']:
            tas = self.result.filter(pl.col('id') == ta_id)['therapeuticAreas'][0].to_list()
            assert ta_id in tas, f'{ta_id} missing from its own therapeuticAreas'

    def test_already_correct_tas_unchanged(self) -> None:
        """DISEASE_A already has the correct TA — it must not gain spurious entries."""
        tas = self.result.filter(pl.col('id') == 'DISEASE_A')['therapeuticAreas'][0].to_list()
        assert sorted(tas) == ['TA_1']

    # -- data-quality checks -------------------------------------------------

    def test_no_duplicate_tas(self) -> None:
        """therapeuticAreas lists must not contain duplicates for any row."""
        for row in self.result.iter_rows(named=True):
            tas = row['therapeuticAreas']
            assert len(tas) == len(set(tas)), f"Duplicate TAs found for id={row['id']}: {tas}"

    def test_no_null_in_therapeutic_areas(self) -> None:
        """No row should have a null therapeuticAreas list (empty lists are fine)."""
        assert self.result['therapeuticAreas'].null_count() == 0

    def test_ancestors_column_unchanged(self) -> None:
        """The ancestors column must not be modified by this function."""
        before = self.input_df.select(['id', 'ancestors']).sort('id')
        after = self.result.select(['id', 'ancestors']).sort('id')
        assert before.equals(after)


class TestUpdateTherapeuticAreasEdgeCases:
    """Edge-case tests for update_therapeutic_areas."""

    def test_all_diseases_are_therapeutic_areas(self) -> None:
        """When every disease is a TA, each should reference itself."""
        df = pl.DataFrame(
            {
                'id': ['TA_A', 'TA_B'],
                'ancestors': [[], ['TA_A']],
                'therapeuticAreas': [['TA_A'], ['TA_A', 'TA_B']],
            },
            schema={'id': pl.String, 'ancestors': LIST_STR, 'therapeuticAreas': LIST_STR},
        )
        result = update_therapeutic_areas(df)
        assert 'TA_A' in result.filter(pl.col('id') == 'TA_A')['therapeuticAreas'][0].to_list()
        assert 'TA_B' in result.filter(pl.col('id') == 'TA_B')['therapeuticAreas'][0].to_list()

    def test_no_therapeutic_areas_anywhere(self) -> None:
        """When no disease belongs to any TA, all therapeuticAreas must stay empty."""
        df = pl.DataFrame(
            {
                'id': ['D1', 'D2'],
                'ancestors': [[], ['D1']],
                'therapeuticAreas': [[], []],
            },
            schema={'id': pl.String, 'ancestors': LIST_STR, 'therapeuticAreas': LIST_STR},
        )
        result = update_therapeutic_areas(df)
        for row in result.iter_rows(named=True):
            assert row['therapeuticAreas'] == [], f"Expected empty TAs for {row['id']}"

    def test_single_row_is_ta(self) -> None:
        """A single therapeutic area disease should reference only itself."""
        df = pl.DataFrame(
            {
                'id': ['TA_ONLY'],
                'ancestors': [[]],
                'therapeuticAreas': [['TA_ONLY']],
            },
            schema={'id': pl.String, 'ancestors': LIST_STR, 'therapeuticAreas': LIST_STR},
        )
        result = update_therapeutic_areas(df)
        assert result['therapeuticAreas'][0].to_list() == ['TA_ONLY']

    def test_extra_columns_preserved(self) -> None:
        """Any columns beyond the three used by the function must pass through untouched."""
        df = pl.DataFrame(
            {
                'id': ['D1'],
                'ancestors': [[]],
                'therapeuticAreas': [[]],
                'name': ['some disease'],
            },
            schema={'id': pl.String, 'ancestors': LIST_STR, 'therapeuticAreas': LIST_STR, 'name': pl.String},
        )
        result = update_therapeutic_areas(df)
        assert 'name' in result.columns
        assert result['name'][0] == 'some disease'
