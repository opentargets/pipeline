"""Comparing identifiers and names across ontologies.

The disease index carries terms from EFO, MONDO, HP and ORDO, and those sources
do not agree on how to write a cross-reference: the same SNOMED code appears as
``SCTID``, ``SNOMEDCT_US`` or ``SNOMED`` depending on which of them wrote the
line.  Nor is every cross-reference an assertion of identity.  Both kinds of
knowledge live here, apart from the rules that use them.
"""

import polars as pl

# Namespaces that appear in dbXRefs but do not assert identity at all.  The
# merge arms select by allow-list, so this is belt-and-braces for any other
# caller of normalised_xrefs -- but it takes precedence, so a namespace named
# here can never be re-admitted by adding it to an allow-list below.
_EXCLUDED_NAMESPACES = frozenset({
    'PMID', 'UNIPROT', 'HTTP', 'HTTPS', 'WIKIPEDIA', 'HGNC', 'HMDB',
    'KEGG', 'KEGG COMPOUND', 'METACYC', 'REACTOME', 'GTR', 'CSP',
})  # fmt: skip

# Classification codes are many-to-one by design, so sharing one says nothing
# about identity and collapses unrelated terms into a single group.
_COARSE_NAMESPACES = frozenset({'ICD9', 'ICD10', 'ICD11', 'MEDDRA', 'FYLER'})

# Vocabularies whose codes denote a single clinical concept, so sharing one is
# evidence of equivalence when a name corroborates it.
FINE_NAMESPACES = frozenset({
    'OMIM', 'UMLS', 'SNOMED', 'MESH', 'MEDGEN', 'DOID', 'NCIT', 'ORPHANET',
    'MONDO', 'EFO', 'HP', 'GARD', 'NANDO', 'NORD', 'ICDO', 'ONCOTREE',
})  # fmt: skip

# Namespaces written as an ontology id, used to resolve a direct cross-reference
# to the term it names.
ONTOLOGY_NAMESPACES = frozenset({'MONDO', 'EFO', 'HP', 'ORPHANET', 'OBA', 'DOID', 'NCIT', 'GO', 'MP'})

# The same authority is spelled differently by MONDO, HP and ORDO.
_NAMESPACE_ALIASES = {
    'SCTID': 'SNOMED',
    'SNOMEDCT': 'SNOMED',
    'SNOMEDCT_US': 'SNOMED',
    'MSH': 'MESH',
    'ICD10CM': 'ICD10',
    'ICD10WHO': 'ICD10',
    'ICD11.FOUNDATION': 'ICD11',
    'ORPHA': 'ORPHANET',
}

_EXACT_SYNONYM = 'hasExactSynonym'


def normalised_xrefs(active: pl.DataFrame) -> pl.DataFrame:
    """Explode dbXRefs into one canonical ``NAMESPACE:code`` row per term.

    Drops the namespaces that do not assert identity (publications, proteins,
    links) and the coarse classification codes, and folds the spellings the
    source ontologies disagree on (``SCTID`` / ``SNOMEDCT_US`` / ``SNOMED``).

    Args:
        active: Node frame carrying ``short_id`` and the raw ``meta`` struct.

    Returns:
        One row per term and distinct cross-reference, with ``namespace``,
        ``code`` and ``canonical`` columns.
    """
    return (
        active
        .select('short_id', pl.col('meta').struct['xrefs'].alias('xrefs'))
        .explode('xrefs')
        .select('short_id', pl.col('xrefs').struct['val'].alias('val'))
        .filter(pl.col('val').is_not_null(), pl.col('val').str.contains(':'))
        .with_columns(
            namespace=pl.col('val').str.split(':').list.first().str.strip_chars().str.to_uppercase(),
            code=pl.col('val').str.split(':').list.last().str.strip_chars(),
        )
        .with_columns(namespace=pl.col('namespace').replace(_NAMESPACE_ALIASES))
        .filter(
            ~pl.col('namespace').is_in(_EXCLUDED_NAMESPACES),
            ~pl.col('namespace').is_in(_COARSE_NAMESPACES),
            pl.col('code') != '',
        )
        .with_columns(canonical=pl.col('namespace') + pl.lit(':') + pl.col('code'))
        .unique(['short_id', 'canonical'])
    )


def name_bags(active: pl.DataFrame) -> dict[str, set[str]]:
    """Map each term to its label plus exact synonyms, lowercased.

    Only exact synonyms count: related and broad synonyms are what let a term
    be listed under a name it is not equivalent to.

    Args:
        active: Node frame carrying ``short_id``, ``lbl`` and the ``meta`` struct.

    Returns:
        Term id to the set of names that term answers to.
    """
    exact_synonyms = (
        active
        .select('short_id', pl.col('meta').struct['synonyms'].alias('synonyms'))
        .explode('synonyms')
        .filter(pl.col('synonyms').struct['pred'] == _EXACT_SYNONYM)
        .select('short_id', pl.col('synonyms').struct['val'].alias('val'))
    )
    names = (
        pl
        .concat([active.select('short_id', pl.col('lbl').alias('val')), exact_synonyms])
        .filter(pl.col('val').is_not_null())
        .with_columns(val=pl.col('val').str.replace_all('\n', '').str.to_lowercase().str.strip_chars())
        .group_by('short_id')
        .agg(pl.col('val').unique().alias('names'))
    )
    return {row['short_id']: set(row['names']) for row in names.iter_rows(named=True)}
