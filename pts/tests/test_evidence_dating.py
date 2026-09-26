"""Tests for pts.transformers.evidence.dating."""

import polars as pl

from pts.transformers.evidence.utils.dating import resolve_evidence_date, resolve_publication_date


class TestResolvePublicationDate:
    def test_no_op_when_literature_column_is_absent(self) -> None:
        df = pl.DataFrame({'id': ['1']})
        publication_lut = pl.DataFrame({'publicationId': ['111'], 'publicationDate': ['2020-01-01']})
        result = resolve_publication_date(df, publication_lut)
        assert result.to_dicts() == df.to_dicts()

    def test_takes_the_earliest_matching_publication_date(self) -> None:
        df = pl.DataFrame({'id': ['1'], 'literature': [['111', '222']]})
        publication_lut = pl.DataFrame({
            'publicationId': ['111', '222'],
            'publicationDate': ['2020-06-01', '2019-01-01'],
        })
        result = resolve_publication_date(df, publication_lut)
        assert result.to_dicts()[0]['publicationDate'] == '2019-01-01'

    def test_null_when_no_reference_matches(self) -> None:
        df = pl.DataFrame({'id': ['1'], 'literature': [['UNKNOWN']]})
        publication_lut = pl.DataFrame({'publicationId': ['111'], 'publicationDate': ['2020-01-01']})
        result = resolve_publication_date(df, publication_lut)
        assert result.to_dicts()[0]['publicationDate'] is None

    def test_matches_case_insensitively_and_trims_whitespace(self) -> None:
        df = pl.DataFrame({'id': ['1'], 'literature': [[' med-1 ']]})
        publication_lut = pl.DataFrame({'publicationId': ['MED-1'], 'publicationDate': ['2020-01-01']})
        result = resolve_publication_date(df, publication_lut)
        assert result.to_dicts()[0]['publicationDate'] == '2020-01-01'


class TestResolveEvidenceDate:
    def test_takes_the_earliest_of_the_present_date_columns(self) -> None:
        df = pl.DataFrame({'publicationDate': ['2020-06-01'], 'curationDate': ['2019-01-01']})
        result = resolve_evidence_date(df)
        assert result.to_dicts()[0]['evidenceDate'] == '2019-01-01'

    def test_null_when_no_date_columns_present(self) -> None:
        df = pl.DataFrame({'unrelated': [1]})
        result = resolve_evidence_date(df)
        assert result.to_dicts()[0]['evidenceDate'] is None

    def test_ignores_nulls_among_present_columns(self) -> None:
        df = pl.DataFrame(
            {'publicationDate': [None], 'curationDate': ['2019-01-01']},
            schema={'publicationDate': pl.String, 'curationDate': pl.String},
        )
        result = resolve_evidence_date(df)
        assert result.to_dicts()[0]['evidenceDate'] == '2019-01-01'
