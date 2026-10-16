r"""Polars ports of the spark expressions OnToma prepares a query label or id with.

Ports of `ontoma.common.utils` and of the extraction in `OnToma._extract_query_entity_*`. Where
spark and polars disagree on a primitive the port follows spark, and each case is named here:

* spark's `trim` removes the space character only, not whitespace in general.
* spark's `regexp_extract` returns an empty string when the pattern does not match, polars null.
* The regular expressions run on Java's engine in spark. Java's `.` excludes every line
  terminator (`\n`, `\r`, `\u0085`, `\u2028`, `\u2029`), not only `\n`, and its `$` also
  matches just before a final line terminator. `JAVA_DOT` and `JAVA_END` spell that out; without
  them a label such as `#Name\n` extracts to an empty string instead of `Name`, which shows up
  on real disease labels.
"""

from __future__ import annotations

import polars as pl

#: `.` as Java's regex engine reads it.
JAVA_DOT = r'[^\n\r\x{85}\x{2028}\x{2029}]'
#: `$` as Java's regex engine reads it outside multiline mode.
JAVA_END = r'(?:\r\n|[\n\r\x{85}\x{2028}\x{2029}])?\z'

#: `_translate_greek_alphabet`: lowercase greek letters only.
GREEK_TRANSLATION = ('αβγδεζηικλμνξπρτυω', 'abgdezhiklmnxprtuo')
#: `_translate_special_characters`: accented latin letters.
SPECIAL_CHARACTER_TRANSLATION = (
    'àèìòùáéíóúâêîôûäëïöüÀÈÌÒÙÁÉÍÓÚÂÊÎÔÛÄËÏÖÜãåõøÃÅÕØçñýÇÑÝ',
    'aeiouaeiouaeiouaeiouAEIOUAEIOUAEIOUAEIOUaaooAAOOcnyCNY',
)

_DISEASE_LABEL_PREFIX = rf'^(?:[A-Z]{{1}}[0-9]{{2}}[-.A-Z0-9]* |Chapter [IVX]+ )?({JAVA_DOT}+){JAVA_END}'
_NESTED_IDENTIFIER = rf'^{JAVA_DOT}+:({JAVA_DOT}+_{JAVA_DOT}+){JAVA_END}'


def spark_trim(expr: pl.Expr) -> pl.Expr:
    """Spark's `trim`: strip the space character from both ends."""
    return expr.str.strip_chars(' ')


def translate(expr: pl.Expr, table: tuple[str, str]) -> pl.Expr:
    """Spark's `translate` for a table mapping each character to exactly one other."""
    source, target = table
    if len(source) != len(target) or len(set(source)) != len(source):
        raise ValueError('translation table must map distinct characters one to one')
    return expr.str.replace_many(list(source), list(target))


def alternative_translations(label: pl.Expr) -> pl.Expr:
    """`get_alternative_translations`: the label with greek and accented letters translated, then with greek only.

    Both are kept because accented characters in some labels are encoding damage rather than
    accents, and translating those would not help.
    """
    greek = translate(label, GREEK_TRANSLATION)
    return pl.concat_list(translate(greek, SPECIAL_CHARACTER_TRANSLATION), greek)


def determine_track(label: pl.Expr) -> pl.Expr:
    """`determine_track`: `symbol` for an acronym-like label, otherwise `term`.

    A label is acronym-like when it has no space and is at most 4 characters long, or at most 11
    and more than half of its letters uppercase. A label without letters has no uppercase
    proportion, which leaves only the length test, null-propagating exactly as in spark.
    """
    length = label.str.len_chars()
    uppercase = label.str.replace_all('[^A-Z]', '').str.len_chars()
    letters = label.str.replace_all('[^A-Za-z]', '').str.len_chars()
    proportion = pl.when(letters == 0).then(None).otherwise(uppercase / letters)
    is_symbol = ~label.str.contains(' ', literal=True) & ((length <= 4) | ((length <= 11) & (proportion > 0.5)))
    return pl.when(is_symbol).then(pl.lit('symbol')).otherwise(pl.lit('term'))


def clean_disease_label(label: pl.Expr) -> pl.Expr:
    """`clean_disease_label`: of a label containing `#`, the part after the last one, less a code prefix."""
    last = label.str.split('#').list.last()
    cleaned = last.str.extract(_DISEASE_LABEL_PREFIX, 1).fill_null('')
    return pl.when(label.str.contains('#', literal=True)).then(cleaned).otherwise(label)


def format_identifier(identifier: pl.Expr) -> pl.Expr:
    """`format_identifier`: unnest `X:A_B` to `A_B`, then write `_` as `:` and Orphanet prefixes as `ORPHANET:`."""
    nested = identifier.str.extract(_NESTED_IDENTIFIER, 1).fill_null('')
    identifier = pl.when(nested.str.len_chars() > 1).then(nested).otherwise(identifier)
    return identifier.str.replace_all('_', ':', literal=True).str.replace_all('ORDO:|ORPHA:', 'ORPHANET:')
