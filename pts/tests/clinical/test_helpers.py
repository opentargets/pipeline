import polars as pl

from pts.transformers.clinical import filter_df, union_dfs
from pts.transformers.clinical.helpers import extract_per_distinct_text, sanitise_text


def test_union_dfs_fills_missing_columns_with_null() -> None:
    out = union_dfs([pl.DataFrame({'a': [1]}), pl.DataFrame({'b': ['x']})])

    assert out.to_dicts() == [{'a': 1, 'b': None}, {'a': None, 'b': 'x'}]


def test_filter_df_takes_sql_or_expression() -> None:
    df = pl.DataFrame({'source': ['ClinicalTrials', 'FDA', None]})

    assert filter_df(df, "source != 'ClinicalTrials'")['source'].to_list() == ['FDA']
    assert filter_df(df, pl.col('source') == 'FDA')['source'].to_list() == ['FDA']


def test_extract_per_distinct_text_runs_once_per_text_and_keeps_rows() -> None:
    df = pl.DataFrame({'key': [1, 2, 3, 4], 'text': ['b', None, 'a', 'b']})
    calls: list[list[str]] = []

    def extract(texts: pl.Series) -> pl.Series:
        calls.append(texts.to_list())
        return texts.map_elements(lambda t: [t.upper()], return_dtype=pl.List(pl.String))

    out = extract_per_distinct_text(df, 'text', 'entities', extract)

    assert calls == [['a', 'b']]
    assert out.to_dicts() == [
        {'key': 1, 'text': 'b', 'entities': ['B']},
        {'key': 2, 'text': None, 'entities': []},
        {'key': 3, 'text': 'a', 'entities': ['A']},
        {'key': 4, 'text': 'b', 'entities': ['B']},
    ]


class TestSanitiseText:
    """Unit tests for sanitise_text."""

    def test_null_byte_reconstructs_accented_e(self):
        assert sanitise_text('S\x00e9zary syndrome') == 'Sézary syndrome'

    def test_null_byte_reconstructs_accented_o(self):
        assert sanitise_text('Sj\x00f6gren') == 'Sjögren'

    def test_null_byte_reconstructs_registered_trademark(self):
        assert sanitise_text('Technospiron\x00AE') == 'Technospiron®'

    def test_orphan_null_byte_removed(self):
        assert sanitise_text('te\x00st') == 'test'

    def test_aact_unescapes_left_square_bracket(self):
        assert sanitise_text(r'pain \[hyperalgesia\]') == 'pain [hyperalgesia]'

    def test_aact_unescapes_less_than(self):
        assert sanitise_text(r'platelets \<= 30x109/L') == 'platelets <= 30x109/L'

    def test_aact_unescapes_greater_than(self):
        assert sanitise_text(r'\> 5 cm') == '> 5 cm'

    def test_aact_unescapes_ampersand(self):
        assert sanitise_text(r'K\&L grade 3') == 'K&L grade 3'

    def test_aact_unescapes_tilde(self):
        assert sanitise_text(r'foo \~ bar') == 'foo ~ bar'

    def test_aact_unescapes_underscore(self):
        assert sanitise_text(r'foo \_ bar') == 'foo _ bar'

    def test_aact_unescapes_asterisk(self):
        assert sanitise_text(r'foo \* bar') == 'foo * bar'

    def test_already_clean_text_is_idempotent(self):
        assert sanitise_text('Sézary') == 'Sézary'
        assert sanitise_text('pain [hyperalgesia]') == 'pain [hyperalgesia]'

    def test_garbled_null_byte_removed_but_no_reconstruction(self):
        assert sanitise_text('s\x00zary') == 'szary'
