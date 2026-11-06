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

GROUNDED = ('source', 'suggestion', 'receptor', 'suggestion_fallback')
"""Grounding outcomes that give a mechanism its genes."""

LOCATION_FIELDS = ('biotype', 'genomicLocation')
"""Target view fields that settle which copy of a shared approved symbol is labelled."""

CLASS_LEVEL_GENES = 20
"""A prediction grounding to more genes than this names a class, not a target."""

Lookup = dict[str, list[tuple[int, str]]]


def normalise_label(label: str) -> str:
    """Lowercase, drop hyphens, collapse whitespace."""
    return re.sub(r'\s+', ' ', label.lower().replace('-', '')).strip()


def _preferred_copies(target_view: pl.DataFrame) -> pl.DataFrame:
    """Keep, of the genes sharing an approved symbol, the protein-coding ones off chromosome Y.

    A copy is dropped only when another gene with its symbol ranks higher: protein-coding
    first, then not on Y. Genes with a symbol of their own are kept whatever they are.
    """
    rank = (pl.col('biotype') != 'protein_coding').fill_null(True).cast(pl.Int8) * 2 + (
        pl.col('genomicLocation').struct.field('chromosome') == 'Y'
    ).fill_null(False).cast(pl.Int8)
    return (
        target_view
        .with_columns(_rank=rank)
        .filter(pl.col('approvedSymbol').is_null() | (pl.col('_rank') == pl.col('_rank').min().over('approvedSymbol')))
        .drop('_rank')
    )


def build_label_index(target_view: pl.DataFrame) -> pl.DataFrame:
    """Return every label of every target, with the field it came from.

    When the target view carries ``biotype`` and ``genomicLocation``, genes that share an
    approved symbol with a better copy are left out (``_preferred_copies``), so that a
    pseudoautosomal gene's Y copy or a pseudogene does not make the symbol ambiguous.

    Returns:
        pl.DataFrame: ``label`` (normalised), ``field``, ``targetId``.
    """
    if all(f in target_view.columns for f in LOCATION_FIELDS):
        target_view = _preferred_copies(target_view)
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


STEM_MIN = 3
STEM_SUFFIX_MAX = 2


def _stem_of(word: str, label_word: str) -> bool:
    """Whether ``word`` is a family stem of ``label_word`` (``fgfr`` of ``fgfr1``, ``gnrh`` of ``gnrhr``)."""
    return len(word) >= STEM_MIN and label_word.startswith(word) and 0 < len(label_word) - len(word) <= STEM_SUFFIX_MAX


def _named_in(name: str, labels: set[str]) -> bool:
    """Whether the name shares a specific word, or a family stem, with any of a gene's labels."""
    name_words = (_words(name) | _words(name.replace('-', ''))) - GENERIC_WORDS
    label_words = {w for label in labels for w in _words(label)}
    return bool(name_words & label_words) or any(_stem_of(w, lw) for w in name_words for lw in label_words)


def resolve(label: str, index: pl.DataFrame) -> list[str]:
    """Return the gene a label names at its most specific field, or none if ambiguous there."""
    return _resolve(label, _to_lookup(index.filter(pl.col('label') == normalise_label(label))))


def _ground_one(row: dict, lookup: Lookup, labels: dict[str, set[str]]) -> tuple[list[str], str, set[str]]:
    """Ground one mechanism: its genes, its outcome and the genes its suggestions resolve to."""
    if not (row['kept'] and row['isHumanTarget']):
        return [], 'skipped', set()
    resolved = {g for s in row['targetSymbolSuggestion'] or [] for g in _resolve(s, lookup)}
    if row['actsOnReceptor']:
        receptors = {g for g in resolved if _named_in(row['targetFromSource'], labels[g])}
        outcome = 'receptor' if receptors else 'suggestion_unsupported' if resolved else 'unresolved'
        return sorted(receptors), outcome, resolved
    source = _resolve(row['targetFromSource'], lookup)
    if source:
        trusted = _best_field(row['targetFromSource'], lookup) in APPROVED_FIELDS
        if not trusted and len(resolved) > 1 and set(source) <= resolved:
            return sorted(resolved), 'suggestion', resolved
        if trusted or not resolved or set(source) <= resolved:
            return source, 'source', resolved
        return [], 'conflict', resolved
    candidates = _candidates(row['targetFromSource'], lookup)
    if candidates:
        supported = resolved & set(candidates)
    else:
        supported = {g for g in resolved if _named_in(row['targetFromSource'], labels[g])}
    if supported:
        return sorted(supported), 'suggestion', resolved
    return [], 'suggestion_unsupported' if resolved else 'unresolved', resolved


FALLBACK_FROM = frozenset({'suggestion_unsupported', 'conflict'})
"""Outcomes a confident mechanism may ground from its suggestions alone."""


def ground_mechanisms(
    mechanisms: pl.DataFrame,
    index: pl.DataFrame,
    class_level_genes: int = CLASS_LEVEL_GENES,
    suggestion_min_confidence: float | None = None,
) -> pl.DataFrame:
    """Add ``targetIds``, ``grounding`` and ``class_level`` to the mechanisms.

    ``grounding`` is ``source`` when the name the text uses resolves, ``suggestion`` when only
    the model's symbol suggestions do, ``receptor`` when the text names a ligand and the drug
    acts on its receptor (the receptor's symbols must share a word or family stem with the
    ligand's name), ``unresolved`` when there is nothing to resolve, and
    ``skipped`` for mechanisms that failed the quote check or name a non-human target.

    A suggestion may only normalise the name the text uses: when that name is ambiguous it
    must be one of the name's genes, and when the name is unknown one of its labels must
    occur in the name. Suggestions that resolve but fail this are ``suggestion_unsupported``.

    A name that matches one gene only through a synonym, while the model's suggestions
    resolve to other genes, is a ``conflict`` and is left ungrounded: short aliases (``MEK``,
    ``TORC1``, ``A2A``) are often synonyms of an unrelated gene. A synonym match that is one of
    several suggested genes names a family (``HDAC``), and grounds to all of them.

    With ``suggestion_min_confidence`` set, a ``suggestion_unsupported`` or ``conflict``
    mechanism whose ``confidence`` is at least that grounds to the genes its suggestions
    resolve to, as ``suggestion_fallback``. The mechanisms then need a ``confidence`` column.
    """
    lookup = _to_lookup(index)
    labels = _labels_by_gene(index)
    target_ids, grounding = [], []
    columns = ['targetFromSource', 'targetSymbolSuggestion', 'kept', 'isHumanTarget', 'actsOnReceptor']
    if suggestion_min_confidence is not None:
        columns.append('confidence')
    for row in mechanisms.select(columns).iter_rows(named=True):
        genes, outcome, resolved = _ground_one(row, lookup, labels)
        if (
            suggestion_min_confidence is not None
            and outcome in FALLBACK_FROM
            and (row['confidence'] or 0) >= suggestion_min_confidence
        ):
            genes, outcome = sorted(resolved), 'suggestion_fallback'
        target_ids.append(genes)
        grounding.append(outcome)
    symbol_of = dict(index.filter(pl.col('field') == 'approvedSymbol').select('targetId', 'label').iter_rows())
    return mechanisms.with_columns(
        targetIds=pl.Series(target_ids, dtype=pl.List(pl.String)),
        targetSymbols=pl.Series(
            [[symbol_of.get(g, g).upper() for g in genes] for genes in target_ids], dtype=pl.List(pl.String)
        ),
        grounding=pl.Series(grounding, dtype=pl.String),
    ).with_columns(class_level=pl.col('targetIds').list.len() > class_level_genes)
