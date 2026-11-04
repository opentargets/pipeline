"""Ground extracted target names on Ensembl gene ids through the target view's labels.

A label is looked up field by field, from the most to the least specific. The first
field that knows the label decides, and a label that field gives to several genes is
left unresolved rather than guessed.
"""

from __future__ import annotations

import re
from collections import defaultdict

import polars as pl

FIELDS = ('approvedSymbol', 'symbolSynonyms', 'obsoleteSymbols', 'approvedName', 'nameSynonyms')
"""Label fields of the target view, most specific first."""

CLASS_LEVEL_GENES = 20
"""A prediction grounding to more genes than this names a class, not a target."""

Lookup = dict[str, list[tuple[int, str]]]


def normalise_label(label: str) -> str:
    """Lowercase, drop hyphens, collapse whitespace."""
    return re.sub(r'\s+', ' ', label.lower().replace('-', '')).strip()


def build_label_index(target_view: pl.DataFrame) -> pl.DataFrame:
    """Return every label of every target, with the field it came from.

    Returns:
        pl.DataFrame: ``label`` (normalised), ``field``, ``targetId``.
    """
    parts = []
    for field in FIELDS:
        column = pl.col(field)
        if isinstance(target_view.schema[field], pl.List):
            column = column.list.eval(pl.element().struct.field('label'))
        else:
            column = pl.concat_list(column)
        parts.append(
            target_view
            .select(pl.col('id').alias('targetId'), column.alias('label'))
            .explode('label')
            .drop_nulls('label')
            .with_columns(field=pl.lit(field))
        )
    return (
        pl
        .concat(parts)
        .with_columns(label=pl.col('label').map_elements(normalise_label, return_dtype=pl.String))
        .filter(pl.col('label').str.len_chars() > 0)
        .unique()
        .select('label', 'field', 'targetId')
    )


def _to_lookup(index: pl.DataFrame) -> Lookup:
    lookup: Lookup = defaultdict(list)
    priority = {field: i for i, field in enumerate(FIELDS)}
    for label, field, target_id in index.iter_rows():
        lookup[label].append((priority[field], target_id))
    return lookup


def _candidates(label: str, lookup: Lookup) -> list[str]:
    hits = lookup.get(normalise_label(label), [])
    if not hits:
        return []
    best = min(p for p, _ in hits)
    return sorted({t for p, t in hits if p == best})


def _best_field(label: str, lookup: Lookup) -> str | None:
    hits = lookup.get(normalise_label(label), [])
    return FIELDS[min(p for p, _ in hits)] if hits else None


APPROVED_FIELDS = frozenset({'approvedSymbol', 'approvedName'})
"""Fields whose match is trusted over the model's own suggestion."""


def _resolve(label: str, lookup: Lookup) -> list[str]:
    genes = _candidates(label, lookup)
    return genes if len(genes) == 1 else []


def _labels_by_gene(index: pl.DataFrame) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    for label, _, target_id in index.iter_rows():
        labels[target_id].add(label)
    return labels


GENERIC_WORDS = frozenset({
    'a',
    'activated',
    'activity',
    'agent',
    'alpha',
    'and',
    'antagonist',
    'agonist',
    'antibody',
    'antigen',
    'associated',
    'b',
    'beta',
    'binding',
    'blocker',
    'c',
    'cell',
    'cells',
    'channel',
    'chain',
    'complex',
    'containing',
    'delta',
    'dependent',
    'dna',
    'domain',
    'enzyme',
    'factor',
    'family',
    'gamma',
    'gene',
    'growth',
    'human',
    'i',
    'ii',
    'iii',
    'inhibitor',
    'iv',
    'kinase',
    'kinases',
    'like',
    'member',
    'modulator',
    'of',
    'protein',
    'proteins',
    'receptor',
    'receptors',
    'related',
    'rna',
    'serine',
    'site',
    'subunit',
    'the',
    'threonine',
    'to',
    'type',
    'tyrosine',
    'v',
})
"""Words too common in gene labels to tie a target name to one gene."""


def _words(text: str) -> set[str]:
    return {w for w in re.sub(r'[^a-z0-9]+', ' ', text.lower()).split() if not w.isdigit()}


def _named_in(name: str, labels: set[str]) -> bool:
    """Whether the name shares a specific (non-generic) word with any of a gene's labels."""
    name_words = (_words(name) | _words(name.replace('-', ''))) - GENERIC_WORDS
    return any(name_words & _words(label) for label in labels)


def resolve(label: str, index: pl.DataFrame) -> list[str]:
    """Return the gene a label names at its most specific field, or none if ambiguous there."""
    return _resolve(label, _to_lookup(index.filter(pl.col('label') == normalise_label(label))))


def ground_mechanisms(mechanisms: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """Add ``targetIds``, ``grounding`` and ``class_level`` to the mechanisms.

    ``grounding`` is ``source`` when the name the text uses resolves, ``suggestion`` when only
    the model's symbol suggestions do, ``unresolved`` when there is nothing to resolve, and
    ``skipped`` for mechanisms that failed the quote check or name a non-human target.

    A suggestion may only normalise the name the text uses: when that name is ambiguous it
    must be one of the name's genes, and when the name is unknown one of its labels must
    occur in the name. Suggestions that resolve but fail this are ``suggestion_unsupported``.

    A name that matches one gene only through a synonym, while the model's suggestions
    resolve to other genes, is a ``conflict`` and is left ungrounded: short aliases (``MEK``,
    ``TORC1``, ``A2A``) are often synonyms of an unrelated gene.
    """
    lookup = _to_lookup(index)
    labels = _labels_by_gene(index)
    target_ids, grounding = [], []
    for row in mechanisms.select('targetFromSource', 'targetSymbolSuggestion', 'kept', 'isHumanTarget').iter_rows(
        named=True
    ):
        if not (row['kept'] and row['isHumanTarget']):
            target_ids.append([])
            grounding.append('skipped')
            continue
        source = _resolve(row['targetFromSource'], lookup)
        resolved = {g for s in row['targetSymbolSuggestion'] or [] for g in _resolve(s, lookup)}
        if source:
            trusted = _best_field(row['targetFromSource'], lookup) in APPROVED_FIELDS
            if trusted or not resolved or set(source) <= resolved:
                target_ids.append(source)
                grounding.append('source')
            else:
                target_ids.append([])
                grounding.append('conflict')
            continue
        candidates = _candidates(row['targetFromSource'], lookup)
        if candidates:
            supported = resolved & set(candidates)
        else:
            supported = {g for g in resolved if _named_in(row['targetFromSource'], labels[g])}
        target_ids.append(sorted(supported))
        if supported:
            grounding.append('suggestion')
        elif resolved:
            grounding.append('suggestion_unsupported')
        else:
            grounding.append('unresolved')
    return mechanisms.with_columns(
        targetIds=pl.Series(target_ids, dtype=pl.List(pl.String)),
        grounding=pl.Series(grounding, dtype=pl.String),
    ).with_columns(class_level=pl.col('targetIds').list.len() > CLASS_LEVEL_GENES)
