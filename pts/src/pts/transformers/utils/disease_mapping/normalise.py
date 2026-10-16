r"""Label normalisation of OnToma's NLP pipeline, without spark-nlp.

OnToma normalises every label with a spark-nlp pipeline (`ontoma.nlp_pipeline.NLPPipeline`) on
one of two tracks, and the lookup tables hold labels normalised that way. A query only resolves
when it is normalised exactly as the table was, so this module reproduces spark-nlp 7.0.0
character for character rather than approximating it. Read off the Scala sources of that tag and
diffed against the real pipeline over 484,017 labels -- every disease name, synonym, cross
reference and id of the disease index, the 236k words of `/usr/share/dict/words` and hand-made
edge cases -- with no mismatch on either track.

What the pipeline stages reduce to:

* **Tokenizer.** Candidates are runs of non-whitespace, where whitespace is Java's ASCII `\s`
  (a no-break space is NOT a separator). The tokenizer rule peels a run of context characters
  off either end of a candidate, and only then are the pieces split on the split characters.
  The order matters: `of.-x` yields `of.` and `x`, so `of` survives as a stop word would not.
* **Normalizer.** Its cleanup patterns run BEFORE lowercasing and Java's `\w` is ASCII, so a
  token keeps `[A-Za-z0-9_]` only, lowercased, and is dropped when nothing is left.
* **StopWordsCleaner.** Case-sensitive, on the raw tokens, so `The` goes and `THE` stays.
* **Stemmer.** spark-nlp's own Porter variant, ported literally below. It is NOT the Porter
  algorithm nltk implements: rules are tried in list order and fall through to the next one when
  a condition fails, and the `eed` rule is shadowed by `ed`, so `agreed` stems to `agr` and
  `feed` to `fe`.

The symbol track tokenizes on fewer characters and skips stop words and stemming, and since the
normalizer removes every separator anyway it comes down to filtering the characters of the label.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache

#: `ontoma.nlp_pipeline.NLPPipeline.google_stop_words`, in OnToma's form so the two compare by eye;
#: a test keeps the copy in step.
_GOOGLE_STOP_WORDS_TEXT = (
    "about above after again against all am an and any are aren't as at be because "
    "been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing "
    "don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's "
    "her here here's hers herself him himself his how how's i'd i'll i'm i've if in into is isn't it it's its "
    "itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours "
    "ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's "
    "the their theirs them themselves then there there's these they they'd they'll they're they've this those "
    "through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when "
    "when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll "
    "you're you've your yours yourself yourselves"
)
GOOGLE_STOP_WORDS: tuple[str, ...] = tuple(_GOOGLE_STOP_WORDS_TEXT.split())

#: Matched case-sensitively, as `NLPPipeline.all_stop_words` is: each word and its capitalised form.
STOP_WORDS: frozenset[str] = frozenset(GOOGLE_STOP_WORDS) | {w.capitalize() for w in GOOGLE_STOP_WORDS}

#: The tokenizer's default context characters, peeled off either end of a candidate.
CONTEXT_CHARS = '.,;:!?*-()"\''

#: Split characters of each track, as `NLPPipeline.generate_pipeline` configures them.
TERM_SPLIT_CHARS = '-/:,;'
SYMBOL_SPLIT_CHARS = ':,;'

#: Java's `\s`, which unlike Python's is ASCII only.
_JAVA_WHITESPACE = re.compile(r'[ \t\n\x0b\f\r]+')
_NON_WORD = re.compile(r'[^A-Za-z0-9_]')
_TERM_SPLIT = re.compile(f'[{re.escape(TERM_SPLIT_CHARS)}]')


def _term_tokens(label: str) -> list[str]:
    """Tokens of the term track, before stop words are removed.

    The context characters peeled off a candidate's ends become tokens of their own, which the
    normalizer then empties; dropping them here instead is equivalent, as no stop word is made of
    context characters alone. A candidate of context characters only is kept whole by spark-nlp
    and emptied the same way.
    """
    tokens = []
    for candidate in _JAVA_WHITESPACE.split(label):
        middle = candidate.strip(CONTEXT_CHARS)
        tokens.extend(t for t in _TERM_SPLIT.split(middle) if t)
    return tokens


def _clean_token(token: str) -> str:
    return _NON_WORD.sub('', token).lower()


# --- spark-nlp EnglishStemmer -----------------------------------------------------------------
# A literal port of `com.johnsnowlabs.nlp.annotators.EnglishStemmer` (Stemmer.scala, 7.0.0).
# Rules are (condition on the stem left once the suffix is cut, suffix, rebuild of that stem), and
# `_replace` applies the FIRST rule whose suffix matches and whose condition holds, as the Scala
# `applyReplaces` does. Keep the shapes aligned with the Scala source rather than tidying them.

_Condition = Callable[[str], bool]
_Rule = tuple[_Condition, str, Callable[[str], str]]


def _consonant_at(word: str, pos: int) -> bool:
    if not 0 <= pos < len(word):
        return False
    char = word[pos]
    if char in 'aeiou':
        return False
    return not (char == 'y' and _consonant_at(word, pos - 1))


def _vowel_at(word: str, pos: int) -> bool:
    # true out of range too, as in the Scala source; `_measure` relies on it
    return not _consonant_at(word, pos)


def _measure(word: str) -> int:
    return sum(1 for pos in range(len(word)) if _vowel_at(word, pos) and _consonant_at(word, pos + 1))


def _contains_vowel(word: str) -> bool:
    return any(_vowel_at(word, pos) for pos in range(len(word)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) > 1 and word[-1] == word[-2] and _consonant_at(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    n = len(word)
    return (
        n > 2
        and _consonant_at(word, n - 1)
        and _vowel_at(word, n - 2)
        and _consonant_at(word, n - 3)
        and word[n - 2] not in 'wxy'
    )


def _always(_: str) -> bool:
    return True


def _m_gt(n: int) -> _Condition:
    return lambda stem: _measure(stem) > n


def _append(suffix: str) -> Callable[[str], str]:
    return lambda stem: stem + suffix


def _drop_last(stem: str) -> str:
    return stem[:-1]


def _replace(word: str, rules: list[_Rule], common: _Condition = _always) -> str:
    for condition, suffix, build in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if common(stem) and condition(stem):
                return build(stem)
    return word


def _suffixes(pairs: list[tuple[str, str]]) -> list[_Rule]:
    return [(_always, suffix, _append(replacement)) for suffix, replacement in pairs]


_STEP_1A = _suffixes([('sses', 'ss'), ('ies', 'i'), ('ss', 'ss'), ('s', '')])
_STEP_1B_SUFFIX = _suffixes([('ed', ''), ('ing', '')])
_STEP_1B_FIXUP: list[_Rule] = [
    *_suffixes([('at', 'ate'), ('bl', 'ble'), ('iz', 'ize')]),
    (lambda s: _ends_double_consonant(s) and not s.endswith(('l', 's', 'z')), '', _drop_last),
    (lambda s: _measure(s) == 1 and _ends_cvc(s), '', _append('e')),
]
_STEP_1B_EED: list[_Rule] = [(_m_gt(0), 'eed', _append('ee'))]
_STEP_1C: list[_Rule] = [(_contains_vowel, 'y', _append('i'))]
_STEP_2 = _suffixes([
    ('ational', 'ate'),
    ('tional', 'tion'),
    ('enci', 'ence'),
    ('anci', 'ance'),
    ('izer', 'ize'),
    ('abli', 'able'),
    ('alli', 'al'),
    ('entli', 'ent'),
    ('eli', 'e'),
    ('ousli', 'ous'),
    ('ization', 'ize'),
    ('ation', 'ate'),
    ('ator', 'ate'),
    ('alism', 'al'),
    ('iveness', 'ive'),
    ('fulness', 'ful'),
    ('ousness', 'ous'),
    ('aliti', 'al'),
    ('iviti', 'ive'),
    ('biliti', 'ble'),
])
_STEP_3 = _suffixes([
    ('icate', 'ic'),
    ('ative', ''),
    ('alize', 'al'),
    ('iciti', 'ic'),
    ('ical', 'ic'),
    ('ful', ''),
    ('ness', ''),
])
_STEP_4: list[_Rule] = [
    *_suffixes([(s, '') for s in ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent']]),
    (lambda s: s.endswith(('s', 't')), 'ion', _append('')),
    *_suffixes([(s, '') for s in ['ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize']]),
]
_STEP_5A: list[_Rule] = [
    (_m_gt(1), 'e', _append('')),
    (lambda s: _measure(s) == 1 and not _ends_cvc(s), 'e', _append('')),
]
_STEP_5B: list[_Rule] = [
    (lambda s: _measure(s) > 1 and _ends_double_consonant(s) and s.endswith('l'), '', _drop_last),
]


@cache
def stem(word: str) -> str:
    """Stem a word as spark-nlp's `Stemmer` does; may return an empty string (`s` -> ``)."""
    word = _replace(word.lower(), _STEP_1A)
    if (word.endswith('ed') and _contains_vowel(word[:-2])) or (word.endswith('ing') and _contains_vowel(word[:-3])):
        word = _replace(word, _STEP_1B_SUFFIX, _contains_vowel)
        word = _replace(word, _STEP_1B_FIXUP)
    else:
        word = _replace(word, _STEP_1B_EED)
    word = _replace(word, _STEP_1C)
    word = _replace(word, _STEP_2, _m_gt(0))
    word = _replace(word, _STEP_3, _m_gt(0))
    word = _replace(word, _STEP_4, _m_gt(1))
    word = _replace(word, _STEP_5A)
    return _replace(word, _STEP_5B)


# --- tracks -----------------------------------------------------------------------------------


def normalise_term(label: str) -> str:
    """Normalise a label on the term track: the distinct stems, sorted and concatenated."""
    tokens = (_clean_token(t) for t in _term_tokens(label) if t not in STOP_WORDS)
    stems = {stem(t) for t in tokens if t}
    stems.discard('')
    return ''.join(sorted(stems))


def normalise_symbol(label: str) -> str:
    """Normalise a label on the symbol track: its ASCII word characters, lowercased, in order.

    Tokenizing first is skipped because it cannot change the result: tokens only drop
    whitespace and split characters, which the normalizer would remove anyway.
    """
    return _clean_token(label)


def normalise(label: str | None, track: str | None) -> str | None:
    """Normalise a label on the given track (`term` or `symbol`); null for a null label or track.

    A null label is NOT normalised to an empty string here, although spark-nlp would do so: OnToma
    joins the normalised labels back on the raw label, and that null key never matches.
    """
    if label is None or track is None:
        return None
    if track == 'term':
        return normalise_term(label)
    if track == 'symbol':
        return normalise_symbol(label)
    raise ValueError(f'unknown NLP pipeline track: {track!r}')
