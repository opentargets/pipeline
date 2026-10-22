"""Merge the per-source frames into the target dataset."""

from __future__ import annotations

import polars as pl

from pts.schemas.target import TARGET
from pts.transformers.target.lists import flatten_distinct, union

#: Lists `uniprot_by_gene` gathers from a gene's UniProt entries.
UNIPROT_LISTS = [
    'synonyms',
    'symbolSynonyms',
    'nameSynonyms',
    'functionDescriptions',
    'proteinIds',
    'subcellularLocations',
    'dbXrefs',
    'targetClass',
]

DEDUPLICATED = ['synonyms', 'symbolSynonyms', 'nameSynonyms', 'obsoleteNames', 'obsoleteSymbols']


def merge_hgnc_ensembl(hgnc: pl.DataFrame, ensembl: pl.DataFrame) -> pl.DataFrame:
    """Ensembl genes with their HGNC record; HGNC's symbol and name win, then Ensembl's, then the id."""
    return (
        ensembl
        .rename({'approvedName': '_an', 'approvedSymbol': '_as'})
        .join(hgnc, left_on='id', right_on='ensemblId', how='left', maintain_order='left')
        .with_columns(
            pl.coalesce('approvedName', '_an', pl.lit('')).alias('approvedName'),
            pl.coalesce('approvedSymbol', '_as', 'id').alias('approvedSymbol'),
        )
        .drop('_an', '_as')
    )


def add_protein_classification(uniprot: pl.DataFrame, protein_class: pl.DataFrame) -> pl.DataFrame:
    """Each UniProt entry with the classes of its accessions, `[]` when there are none."""
    per_accession = (
        uniprot
        .select(
            'uniprotId',
            union(
                pl.concat_list(pl.col('uniprotId')),
                pl.col('proteinIds').list.eval(pl.element().struct.field('id')),
            ).alias('pid'),
        )
        .explode('pid', empty_as_null=False, keep_nulls=False)
        .with_columns(pl.col('pid').str.strip_chars(' '))
        .join(protein_class, left_on='pid', right_on='accession', how='left', maintain_order='left')
    )
    return uniprot.join(
        flatten_distinct(per_accession, 'uniprotId', 'targetClass'), on='uniprotId', how='left', maintain_order='left'
    )


def uniprot_by_gene(hgnc: pl.DataFrame, uniprot: pl.DataFrame) -> pl.DataFrame:
    """UniProt entries gathered per gene through HGNC's UniProt ids.

    Each list is the concatenation of the gene's distinct entry lists, as spark's
    `flatten(collect_set(...))`, so an element shared by two different lists appears twice.
    """
    mapped = (
        hgnc
        .select('ensemblId', pl.col('uniprotIds').alias('uniprotId'))
        .explode('uniprotId', empty_as_null=False, keep_nulls=False)
        .join(uniprot, on='uniprotId', how='inner')
        .sort('ensemblId', 'uniprotId', nulls_last=False, maintain_order=True)
        .with_columns(
            pl.struct(pl.col('uniprotId').alias('id'), pl.lit('uniprot_obsolete').alias('source')).alias(
                'uniprotProteinId'
            )
        )
    )
    genes = mapped.select('ensemblId').unique(maintain_order=True)
    for column in UNIPROT_LISTS:
        genes = genes.join(
            flatten_distinct(mapped, 'ensemblId', column),
            on='ensemblId',
            how='left',
            maintain_order='left',
            nulls_equal=True,
        )
    ids = mapped.group_by('ensemblId', maintain_order=True).agg(pl.col('uniprotProteinId').unique(maintain_order=True))
    return (
        genes
        .join(ids, on='ensemblId', how='left', maintain_order='left', nulls_equal=True)
        .with_columns(
            pl.when(pl.col('targetClass').list.len() > 0).then(pl.col('targetClass')).alias('targetClass'),
            union(pl.col('proteinIds'), pl.col('uniprotProteinId')).alias('proteinIds'),
        )
        .drop('uniprotProteinId')
        .rename({'ensemblId': 'id'})
    )


def promote_hgnc_swissprot(ensembl: pl.DataFrame, hgnc: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame:
    """Ensembl genes, with HGNC's reviewed UniProt accession added as `uniprot_swissprot` where Ensembl gives none.

    Ensembl links some protein-coding genes to no Swiss-Prot entry although HGNC maps them to a current,
    reviewed one, e.g. NLRP3 and Q96P20, whose sequences disagree on the start codon
    (opentargets/issues#4604). Through HGNC alone the accession is only labelled `uniprot_obsolete`, so
    everything keyed on Swiss-Prot -- GO, ChEMBL mechanisms, coding variants -- misses the gene.

    An accession is added only for a protein-coding gene with no Swiss-Prot accession from Ensembl, only
    when it is the primary accession of a reviewed entry, and only when no other gene holds it: neither as
    Ensembl's Swiss-Prot accession nor as another promotion. Other genes are returned unchanged.

    Args:
        ensembl: the ensembl builder's genes; `id`, `biotype` and `proteinIds` are read.
        hgnc: the hgnc builder's [ensemblId, uniprotIds, ...].
        entries: `read_uniprot`'s entries; `reviewed` and `accessions` (primary first) are read.

    Returns:
        `ensembl`, with the promoted accessions appended to `proteinIds`.
    """
    # every accession trimmed as `filter_and_sort_protein_ids` trims it later, so ` P1 ` and `P1` are one accession
    # in each check here, as they will be in the output
    source = pl.element().struct.field('source')
    swissprot = ensembl.select(
        'id',
        pl
        .col('proteinIds')
        .list.eval(pl.element().filter(source == 'uniprot_swissprot').struct.field('id').str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ''))
        .alias('sp'),
    )
    held = swissprot.explode('sp', empty_as_null=False, keep_nulls=False).select(pl.col('sp').alias('uniprotId'))
    lacking = ensembl.filter(pl.col('biotype') == 'protein_coding').join(
        swissprot.filter(pl.col('sp').list.len().fill_null(0) == 0), on='id', how='semi'
    )
    reviewed = entries.filter(pl.col('reviewed')).select(
        pl.col('accessions').list.first().str.strip_chars().alias('uniprotId')
    )
    candidates = (
        hgnc
        .select(pl.col('ensemblId').alias('id'), pl.col('uniprotIds').alias('uniprotId'))
        .explode('uniprotId', empty_as_null=False, keep_nulls=False)
        .with_columns(pl.col('uniprotId').str.strip_chars())
        .join(lacking, on='id', how='semi')
        .join(reviewed, on='uniprotId', how='semi')
        .join(held, on='uniprotId', how='anti')
        .unique(maintain_order=True)
    )
    promoted = (
        candidates
        .filter(pl.len().over('uniprotId') == 1)
        .group_by('id', maintain_order=True)
        .agg(pl.struct(pl.col('uniprotId').alias('id'), pl.lit('uniprot_swissprot').alias('source')).alias('_promoted'))
    )
    return (
        ensembl
        .join(promoted, on='id', how='left', maintain_order='left')
        .with_columns(
            pl
            .when(pl.col('_promoted').is_not_null())
            .then(pl.concat_list(pl.col('proteinIds').fill_null([]), '_promoted'))
            .otherwise(pl.col('proteinIds'))
            .alias('proteinIds')
        )
        .drop('_promoted')
    )


def protein_source_priority(source: pl.Expr) -> pl.Expr:
    """1 swissprot, 2 trembl, 3 uniprot, 4 ensembl, 5 anything else, e.g. uniprot_obsolete."""
    return (
        pl
        .when(source.str.contains('swiss', literal=True))
        .then(1)
        .when(source.str.contains('trembl', literal=True))
        .then(2)
        .when(source.str.ends_with('uniprot'))
        .then(3)
        .when(source.str.contains('ensembl', literal=True))
        .then(4)
        .otherwise(5)
    )


def filter_and_sort_protein_ids(df: pl.DataFrame) -> pl.DataFrame:
    """One entry per accession, its highest-priority source, sorted by priority then accession.

    A gene left with no entries gets null.
    """
    entries = (
        df
        .select('id', pl.col('proteinIds').alias('p'))
        .explode('p', empty_as_null=False, keep_nulls=False)
        .unnest('p', separator='.')
        .select(
            'id',
            pl.col('p.id').str.strip_chars().alias('pid'),
            pl.col('p.source').alias('source'),
        )
        .drop_nulls(['pid', 'source'])
        .with_columns(protein_source_priority(pl.col('source')).alias('priority'))
        .sort('id', 'pid', 'priority', 'source')
        .unique(subset=['id', 'pid'], keep='first', maintain_order=True)
        .sort('id', 'priority', 'pid')
        .group_by('id', maintain_order=True)
        .agg(pl.struct(pl.col('pid').alias('id'), 'source').alias('proteinIds'))
    )
    return df.drop('proteinIds').join(entries, on='id', how='left', maintain_order='left')


def remove_redundant_xrefs(df: pl.DataFrame) -> pl.DataFrame:
    """Drop GO and Ensembl entries from `dbXrefs`."""
    source = pl.element().struct.field('source')
    return df.with_columns(pl.col('dbXrefs').list.eval(pl.element().filter((source != 'GO') & (source != 'Ensembl'))))


def add_ncbi_synonyms(df: pl.DataFrame, ncbi: pl.DataFrame) -> pl.DataFrame:
    """Append NCBI Entrez synonyms."""
    return (
        df
        .join(
            ncbi.rename({'synonyms': '_s', 'nameSynonyms': '_ns', 'symbolSynonyms': '_ss'}),
            on='id',
            how='left',
            maintain_order='left',
        )
        .with_columns(
            union(pl.col('symbolSynonyms'), pl.col('_ss')).alias('symbolSynonyms'),
            union(pl.col('nameSynonyms'), pl.col('_ns')).alias('nameSynonyms'),
            union(pl.col('synonyms'), pl.col('_s'), pl.col('_ns'), pl.col('_ss')).alias('synonyms'),
        )
        .drop('_s', '_ss', '_ns')
    )


def remove_duplicated_synonyms(df: pl.DataFrame) -> pl.DataFrame:
    """Keep the first of each repeated synonym."""
    return df.with_columns(pl.col(DEDUPLICATED).list.unique(maintain_order=True))


def add_tss(df: pl.DataFrame) -> pl.DataFrame:
    """The canonical transcript's start on the forward strand, end on the reverse, else null."""
    strand = pl.col('canonicalTranscript').struct.field('strand')
    return (
        df
        .rechunk()
        .with_columns(
            pl
            .when(strand == 1)
            .then(pl.col('canonicalTranscript').struct.field('start'))
            .when(strand == -1)
            .then(pl.col('canonicalTranscript').struct.field('end'))
            .alias('tss')
        )
        .drop('canonicalTranscript')
    )


def assemble(
    ensembl: pl.DataFrame,
    hgnc: pl.DataFrame,
    go: pl.DataFrame,
    hallmarks: pl.DataFrame,
    uniprot: pl.DataFrame,
    protein_class: pl.DataFrame,
    hpa: pl.DataFrame,
    constraint: pl.DataFrame,
    ncbi: pl.DataFrame,
    reactome: pl.DataFrame,
) -> pl.DataFrame:
    """The target dataset, one row per Ensembl gene, sorted by id."""
    genes = (
        merge_hgnc_ensembl(hgnc, ensembl)
        .join(go, left_on='id', right_on='ensemblId', how='left', maintain_order='left')
        .join(hallmarks, on='approvedSymbol', how='left', maintain_order='left')
    )
    by_gene = (
        uniprot_by_gene(hgnc, add_protein_classification(uniprot, protein_class))
        .rename({'proteinIds': 'pid'})
        .join(hpa, on='id', how='left', maintain_order='left')
        .with_columns(union(pl.col('subcellularLocations'), pl.col('locations')).alias('subcellularLocations'))
        .drop('locations')
    )
    interim = (
        genes
        .join(by_gene, on='id', how='left', maintain_order='left')
        .with_columns(
            union(pl.col('proteinIds'), pl.col('pid')).alias('proteinIds'),
            union(pl.col('hgncId'), pl.col('dbXrefs'), pl.col('signalP')).alias('dbXrefs'),
            union(pl.col('symbolSynonyms'), pl.col('hgncSymbolSynonyms')).alias('symbolSynonyms'),
            union(pl.col('nameSynonyms'), pl.col('hgncNameSynonyms')).alias('nameSynonyms'),
            union(pl.col('hgncObsoleteSymbols')).alias('obsoleteSymbols'),
            union(pl.col('hgncObsoleteNames')).alias('obsoleteNames'),
        )
        .with_columns(union(pl.col('synonyms'), pl.col('symbolSynonyms'), pl.col('nameSynonyms')).alias('synonyms'))
    )
    return (
        interim
        .join(constraint, on='id', how='left', maintain_order='left')
        .pipe(filter_and_sort_protein_ids)
        .pipe(remove_redundant_xrefs)
        .pipe(add_ncbi_synonyms, ncbi)
        .join(reactome, on='id', how='left', maintain_order='left')
        .pipe(remove_duplicated_synonyms)
        .pipe(add_tss)
        .select(list(TARGET))
        .sort('id')
    )
