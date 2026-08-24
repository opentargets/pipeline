"""Tests for the dataset diff core: schema classification and the materiality rule."""

from __future__ import annotations

from typing import Any

from orchestration.supervisor.diff import ColumnChange, DatasetDiff, compare_schemas, is_material, schema_of


class TestCompareSchemas:
    def test_classifies_and_orders_added_removed_retyped(self) -> None:
        """One schema comparison exercising all three kinds at once.

        Column names are chosen so that alphabetical order differs from both insertion
        order and reverse-alphabetical order within every group — `zeta`/`alpha` and
        `omega`/`beta` would come out wrong under either. `same` is present on both
        sides with the same type and must not appear at all.
        """
        run = {
            'zeta': 'int64',
            'alpha': 'int64',
            'shared_z': 'string',
            'shared_a': 'float64',
            'same': 'int64',
        }
        reference = {
            'omega': 'int64',
            'beta': 'int64',
            'shared_z': 'int64',
            'shared_a': 'string',
            'same': 'int64',
        }

        changes = compare_schemas(run, reference)

        assert changes == [
            ColumnChange(column='alpha', kind='added', run_type='int64'),
            ColumnChange(column='zeta', kind='added', run_type='int64'),
            ColumnChange(column='beta', kind='removed', reference_type='int64'),
            ColumnChange(column='omega', kind='removed', reference_type='int64'),
            ColumnChange(column='shared_a', kind='retyped', run_type='float64', reference_type='string'),
            ColumnChange(column='shared_z', kind='retyped', run_type='string', reference_type='int64'),
        ]

    def test_identical_schemas_yield_nothing(self) -> None:
        schema = {'a': 'int64', 'b': 'string'}
        assert compare_schemas(schema, dict(schema)) == []

    def test_a_column_present_on_both_sides_with_the_same_type_is_not_reported(self) -> None:
        """Guards the retyped filter: presence on both sides is not itself a change."""
        assert compare_schemas({'a': 'int64'}, {'a': 'int64'}) == []


def _diff(**overrides: Any) -> DatasetDiff:
    base: dict[str, Any] = {
        'dataset': 'output/disease',
        'side': 'both',
        'run_rows': 1000,
        'reference_rows': 1000,
        'run_bytes': 1000,
        'reference_bytes': 1000,
        'columns': [],
    }
    base.update(overrides)
    return DatasetDiff(**base)


class TestIsMaterial:
    def test_a_one_sided_dataset_is_always_material(self) -> None:
        """Materiality from `side` must not be gated by the threshold at all.

        A threshold of 1.0 would suppress any size or row move, however large. Only
        `side` can be responsible for this returning True.
        """
        diff = _diff(side='run_only', reference_rows=None, reference_bytes=0)
        assert is_material(diff, threshold=1.0)

    def test_a_schema_change_is_material_regardless_of_counts(self) -> None:
        """Rows and bytes are unchanged; only `columns` is non-empty.

        With a generous threshold that a real count move could never cross, the only
        thing that can make this True is the schema change.
        """
        diff = _diff(columns=[ColumnChange(column='x', kind='added', run_type='int64')])
        assert is_material(diff, threshold=1.0)

    def test_a_sub_threshold_row_move_is_not_material(self) -> None:
        """4% against a 5% threshold. A 50%-style test would pass under an inverted comparison too."""
        diff = _diff(reference_rows=1000, run_rows=1040)
        assert not is_material(diff, threshold=0.05)

    def test_a_supra_threshold_row_move_is_material(self) -> None:
        """6% against the same 5% threshold — the pair brackets the boundary tightly."""
        diff = _diff(reference_rows=1000, run_rows=1060)
        assert is_material(diff, threshold=0.05)

    def test_a_byte_move_is_judged_independently_of_rows(self) -> None:
        """Rows are unchanged (0% move); only bytes cross the threshold.

        A comparison that only ever looked at rows would call this not material.
        """
        diff = _diff(run_rows=1000, reference_rows=1000, run_bytes=1200, reference_bytes=1000)
        assert is_material(diff, threshold=0.05)

    def test_a_sub_threshold_byte_move_is_not_material(self) -> None:
        diff = _diff(run_rows=1000, reference_rows=1000, run_bytes=1040, reference_bytes=1000)
        assert not is_material(diff, threshold=0.05)

    def test_a_zero_reference_row_count_does_not_divide_by_zero(self) -> None:
        """Both directions of the zero-reference case: no rows on either side is not a move."""
        diff = _diff(reference_rows=0, run_rows=0, reference_bytes=0, run_bytes=0)
        assert not is_material(diff, threshold=0.05)

    def test_a_zero_reference_row_count_with_rows_on_the_run_side_is_material(self) -> None:
        """A dataset going from zero rows to non-zero can never clear a fractional threshold."""
        diff = _diff(reference_rows=0, run_rows=5, reference_bytes=0, run_bytes=0)
        assert is_material(diff, threshold=0.05)

    def test_an_uncountable_dataset_is_not_treated_as_zero_rows(self) -> None:
        """`run_rows`/`reference_rows` are None, not 0, when the format has no footer.

        If None were compared as 0 against 0, `previous == 0` would fire on the row
        check same as the zero-reference case above, which happens to also return
        False here — so the byte side is pinned with a real, non-zero move to make
        sure it is actually the None-skip firing and not a coincidence of that case.
        """
        diff = _diff(
            countable=False,
            run_rows=None,
            reference_rows=None,
            run_bytes=1000,
            reference_bytes=1000,
        )
        assert not is_material(diff, threshold=0.05)


class TestDatasetDiffDeltas:
    def test_row_delta_is_run_minus_reference(self) -> None:
        assert _diff(run_rows=1200, reference_rows=1000).row_delta == 200

    def test_row_delta_is_none_when_either_side_has_no_count(self) -> None:
        assert _diff(run_rows=None, reference_rows=1000).row_delta is None
        assert _diff(run_rows=1000, reference_rows=None).row_delta is None

    def test_byte_delta_is_run_minus_reference(self) -> None:
        assert _diff(run_bytes=900, reference_bytes=1000).byte_delta == -100


class _FakeType:
    """Stands in for a pyarrow type, whose `str()` differs from its `repr()`."""

    def __str__(self) -> str:
        return 'int64'

    def __repr__(self) -> str:
        return 'FakeType()'


class _FakeField:
    def __init__(self, name: str, field_type: object) -> None:
        self.name = name
        self.type = field_type


class TestSchemaOf:
    def test_maps_field_names_to_stringified_types(self) -> None:
        """Asserts `str()` is applied rather than the type object kept as-is.

        `repr()` would produce `FakeType()` here, so a swap of `str` for `repr` — or
        for the bare object — is caught.
        """
        footer = [_FakeField('a', _FakeType()), _FakeField('b', _FakeType())]
        assert schema_of(footer) == {'a': 'int64', 'b': 'int64'}

    def test_empty_footer_yields_an_empty_schema(self) -> None:
        assert schema_of([]) == {}
