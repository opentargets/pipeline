"""Tests for pts.transformers.evidence.identifiers."""

import hashlib

import polars as pl

from pts.transformers.evidence import flags
from pts.transformers.evidence.identifiers import assign_evidence_identifier, validate_uniqueness


class TestAssignEvidenceIdentifier:
    def test_id_matches_sha1_of_concatenated_fields(self) -> None:
        schema = {'a': pl.String, 'b': pl.String, 'c': pl.String}
        df = pl.DataFrame({'a': ['x'], 'b': ['y'], 'c': [None]}, schema=schema)
        result = assign_evidence_identifier(df, ['a', 'b', 'c'])
        expected = hashlib.sha1(b'xynull').hexdigest()
        assert result.to_dicts()[0]['id'] == expected

    def test_ignores_fields_absent_from_the_dataframe(self) -> None:
        df = pl.DataFrame({'a': ['x']})
        result = assign_evidence_identifier(df, ['a', 'does_not_exist'])
        expected = hashlib.sha1(b'x').hexdigest()
        assert result.to_dicts()[0]['id'] == expected

    def test_same_fields_produce_the_same_id(self) -> None:
        df = pl.DataFrame({'a': ['x', 'x'], 'b': ['y', 'y']})
        result = assign_evidence_identifier(df, ['a', 'b'])
        ids = result['id'].to_list()
        assert ids[0] == ids[1]

    def test_handles_a_list_of_structs_like_encores_disease_cell_lines(self) -> None:
        """Unlike Spark, Polars refuses to `.cast(String)` a list-of-structs column outright.

        Reproduces a real failure hit against ENCORE evidence, whose `diseaseCellLines` (a
        `unique_fields` entry) is a `list[struct]`.
        """
        cell_line = [{'id': 'SIDM001', 'name': 'A', 'tissue': 'Breast', 'tissueId': 'X'}]
        df = pl.DataFrame({'a': ['x', 'x'], 'diseaseCellLines': [cell_line, cell_line]})

        result = assign_evidence_identifier(df, ['a', 'diseaseCellLines'])

        ids = result['id'].to_list()
        assert ids[0] == ids[1]
        assert ids[0] != assign_evidence_identifier(
            pl.DataFrame({'a': ['x'], 'diseaseCellLines': [[]]}), ['a', 'diseaseCellLines']
        )['id'][0]


class TestValidateUniqueness:
    def test_flags_all_but_one_row_per_duplicate_id(self) -> None:
        df = pl.DataFrame({'id': ['1', '1', '2'], 'value': ['a', 'b', 'c']})
        result = validate_uniqueness(df)

        flagged = result.filter(pl.col('id') == '1')
        assert sorted(flagged['qualityControls'].to_list()) == [[], [flags.DUPLICATED]]
        assert result.filter(pl.col('id') == '2').to_dicts()[0]['qualityControls'] == []

    def test_survivor_is_stable_across_row_order(self) -> None:
        df_a = pl.DataFrame({'id': ['1', '1'], 'value': ['a', 'b']})
        df_b = pl.DataFrame({'id': ['1', '1'], 'value': ['b', 'a']})

        def survivor(df: pl.DataFrame) -> str:
            result = validate_uniqueness(df)
            return result.filter(pl.col('qualityControls').list.len() == 0).to_dicts()[0]['value']

        assert survivor(df_a) == survivor(df_b)

    def test_handles_a_list_column_like_literature(self) -> None:
        """Unlike Spark, Polars refuses to `.cast(String)` a List column outright.

        Reproduces a real failure hit against production GWAS evidence data, which always
        carries a `literature` (List(String)) column.
        """
        df = pl.DataFrame({
            'id': ['1', '1', '2'],
            'literature': [['111'], ['111'], None],
        })
        result = validate_uniqueness(df)
        assert result.filter(pl.col('id') == '2').to_dicts()[0]['qualityControls'] == []
        flagged = result.filter(pl.col('id') == '1')
        assert sorted(flagged['qualityControls'].to_list()) == [[], [flags.DUPLICATED]]

    def test_handles_a_list_of_structs_column_like_encores_disease_cell_lines(self) -> None:
        """Every column feeds the content hash, not just the ones in `unique_fields` -- so a
        list-of-structs column (e.g. ENCORE's `diseaseCellLines`) must not crash it either.
        """
        cell_line = [{'id': 'SIDM001', 'name': 'A', 'tissue': 'Breast', 'tissueId': 'X'}]
        df = pl.DataFrame({
            'id': ['1', '1', '2'],
            'diseaseCellLines': [cell_line, cell_line, None],
        })
        result = validate_uniqueness(df)
        assert result.filter(pl.col('id') == '2').to_dicts()[0]['qualityControls'] == []
        flagged = result.filter(pl.col('id') == '1')
        assert sorted(flagged['qualityControls'].to_list()) == [[], [flags.DUPLICATED]]
