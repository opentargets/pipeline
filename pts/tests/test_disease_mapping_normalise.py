"""Tests for the polars port of OnToma's query preparation and label normalisation.

`test_data/ontoma/normalisation.parquet` holds what OnToma's spark expressions and spark-nlp
pipeline return for a sample of disease index labels and fixed edge cases; see the generator next
to it. The suite has no spark-nlp, so that recorded output is the reference.
"""

from pathlib import Path

import polars as pl
import pytest
from ontoma.nlp_pipeline import NLPPipeline

from pts.transformers.utils.disease_mapping.normalise import (
    STOP_WORDS,
    normalise,
    normalise_symbol,
    normalise_term,
    stem,
)
from pts.transformers.utils.disease_mapping.query import (
    alternative_translations,
    clean_disease_label,
    determine_track,
    format_identifier,
    spark_trim,
)

FIXTURE = Path(__file__).parent / 'test_data' / 'disease_mapping' / 'normalisation.parquet'


@pytest.fixture(scope='module')
def reference() -> pl.DataFrame:
    return pl.read_parquet(FIXTURE)


def _mismatches(df: pl.DataFrame, expected: str, got: str) -> list[tuple]:
    return df.filter(pl.col(expected).ne_missing(pl.col(got))).select('label', expected, got).rows()


@pytest.mark.parametrize(('track', 'function'), [('term', normalise_term), ('symbol', normalise_symbol)])
def test_normalisation_matches_spark_nlp(reference: pl.DataFrame, track: str, function) -> None:
    got = reference.with_columns(pl.Series('got', [function(label) for label in reference['label']]))
    assert _mismatches(got, track, 'got') == []


@pytest.mark.parametrize(
    ('column', 'expression'),
    [
        ('track', determine_track(pl.col('label'))),
        ('translations', alternative_translations(spark_trim(pl.col('label')))),
        ('cleanedDiseaseLabel', clean_disease_label(pl.col('label'))),
        ('formattedIdentifier', format_identifier(spark_trim(pl.col('label')).str.to_uppercase())),
    ],
)
def test_query_expressions_match_spark(reference: pl.DataFrame, column: str, expression: pl.Expr) -> None:
    got = reference.with_columns(expression.alias('got'))
    assert _mismatches(got, column, 'got') == []


@pytest.mark.parametrize(
    ('word', 'expected'),
    [
        # spark-nlp's Porter variant, not nltk's: `ed` shadows `eed`
        ('agreed', 'agr'),
        ('feed', 'fe'),
        ('hopping', 'hop'),
        ('falling', 'fall'),
        ('replacement', 'replac'),
        ('generalization', 'gener'),
        ('s', ''),
    ],
)
def test_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_term_track_peels_context_characters_before_splitting() -> None:
    # `of.` is not a stop word, so `of` survives; a split before peeling would have dropped it
    assert normalise_term('of.-x') == 'ofx'
    # peeled off a whole candidate, `(of)` leaves the stop word; inside `(of)-x` it leaves `of)`
    assert normalise_term('(of) x') == 'x'
    assert normalise_term('(of)-x') == 'ofx'


def test_stop_words_are_case_sensitive() -> None:
    assert normalise_term('the') == ''
    assert normalise_term('The') == ''
    assert normalise_term('THE') == 'the'


def test_null_label_or_track_is_not_normalised() -> None:
    assert normalise(None, 'term') is None
    assert normalise('label', None) is None


def test_unknown_track_is_rejected() -> None:
    with pytest.raises(ValueError, match='unknown NLP pipeline track'):
        normalise('label', 'other')


def test_null_label_expressions_stay_null() -> None:
    df = pl.DataFrame({'label': [None]}, schema={'label': pl.String}).select(
        track=determine_track(pl.col('label')),
        cleaned=clean_disease_label(pl.col('label')),
        identifier=format_identifier(pl.col('label')),
    )
    assert df.row(0) == ('term', None, None)


def test_stop_words_match_ontoma() -> None:
    # the lookup tables were normalised with OnToma's list; a change there must be mirrored here
    assert set(NLPPipeline.all_stop_words) == STOP_WORDS
