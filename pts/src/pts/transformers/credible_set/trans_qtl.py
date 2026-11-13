"""The TRANS_QTL rule: QTL credible sets whose every variant is far from the gene of their study.

A QTL measures the expression, splicing or protein level of one gene. A credible set of a QTL
study is trans when every one of its variants is on another chromosome than that gene, or more
than a threshold from its transcription start site. `trans_qtls` locates the variants from their
identifiers (`_variant_location`) and the genes from the target index (`_qtl_gene_location`), and
keeps the sets of which every variant `_is_trans`.
"""

from __future__ import annotations

import polars as pl

from pts.transformers.credible_set.identifier import variant_chromosome, variant_position
from pts.transformers.credible_set.source import qtl_genes

_GENE_LOCATION_COLUMNS = ('studyId', 'gene_chromosome', 'gene_tss')
"""The columns of `_qtl_gene_location`: the QTL study, and the chromosome and TSS of its gene."""


_VARIANT_LOCATION_COLUMNS = ('id', 'variant_chromosome', 'variant_position')
"""The columns of `_variant_location`: the credible set, and the chromosome and position of a variant."""


def trans_qtls(
    credible_set: pl.LazyFrame,
    credible_set_variant: pl.LazyFrame,
    study: pl.LazyFrame,
    target: pl.LazyFrame,
    threshold: int,
) -> pl.LazyFrame:
    """The ids of the QTL sets whose every variant `_is_trans` to the gene of their study.

    GWAS sets and sets of a study without a `geneId` are never trans.

    Examples:
        >>> credible_set = pl.LazyFrame({'id': ['far', 'near'], 'studyId': ['s', 's']})
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['far', 'near', 'near'],
        ...     'variantId': ['2_10_A_T', '2_10_A_T', '1_9_A_T'],
        ... })
        >>> study = pl.LazyFrame({'studyId': ['s'], 'studyType': ['eqtl'], 'geneId': ['g']})
        >>> target = pl.LazyFrame({'id': ['g'], 'tss': [10], 'genomicLocation': [{'chromosome': '1'}]})
        >>> trans_qtls(credible_set, credible_set_variant, study, target, 5_000_000).collect()['id'].to_list()
        ['far']
    """
    variant_is_trans = _is_trans(
        pl.col('variant_chromosome'),
        pl.col('variant_position'),
        pl.col('gene_chromosome'),
        pl.col('gene_tss'),
        threshold,
    )
    return (
        _variant_location(credible_set_variant)
        .join(credible_set.select('id', 'studyId'), on='id')
        .join(_qtl_gene_location(study, target), on='studyId')
        .group_by('id')
        .agg(variant_is_trans.all().alias('all_variants_trans'))
        .filter('all_variants_trans')
        .select('id')
    )


def _is_trans(
    chromosome: pl.Expr, position: pl.Expr, gene_chromosome: pl.Expr, gene_tss: pl.Expr, threshold: int
) -> pl.Expr:
    """Whether a variant is trans to its gene: on another chromosome, or more than `threshold` from the TSS.

    A comparison with a missing value counts as not trans.

    Examples:
        >>> location = pl.DataFrame({
        ...     'chromosome': ['2', '1', '1', '1'],
        ...     'position': [100, 6_000_101, 5_000_100, 100],
        ...     'gene_chromosome': ['1', '1', '1', None],
        ...     'gene_tss': [100, 100, 100, None],
        ... })
        >>> variant = pl.col('chromosome'), pl.col('position')
        >>> gene = pl.col('gene_chromosome'), pl.col('gene_tss')
        >>> trans = _is_trans(*variant, *gene, 5_000_000)
        >>> location.select(trans).to_series().to_list()
        [True, True, False, False]
    """
    other_chromosome = chromosome != gene_chromosome
    far_from_tss = (position - gene_tss).abs() > threshold
    return (other_chromosome | far_from_tss).fill_null(False)


def _variant_location(credible_set_variant: pl.LazyFrame) -> pl.LazyFrame:
    """The `_VARIANT_LOCATION_COLUMNS` of each credible set variant, from its `variantId`.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({'credibleSetId': ['a'], 'variantId': ['2_10_A_T']})
        >>> _variant_location(credible_set_variant).collect().rows()
        [('a', '2', 10)]
    """
    variant_id = pl.col('variantId')
    return credible_set_variant.select(
        pl.col('credibleSetId').alias('id'),
        variant_chromosome(variant_id).alias('variant_chromosome'),
        variant_position(variant_id).alias('variant_position'),
    )


def _qtl_gene_location(study: pl.LazyFrame, target: pl.LazyFrame) -> pl.LazyFrame:
    """The `_GENE_LOCATION_COLUMNS` of each QTL study, null for a gene absent from the target index.

    Examples:
        >>> study = pl.LazyFrame({'studyId': ['q', 'g'], 'studyType': ['eqtl', 'gwas'], 'geneId': ['ENSG1', None]})
        >>> target = pl.LazyFrame({'id': ['ENSG1'], 'tss': [100], 'genomicLocation': [{'chromosome': '1'}]})
        >>> _qtl_gene_location(study, target).collect().rows()
        [('q', '1', 100)]
    """
    gene = target.select(
        pl.col('id').alias('geneId'),
        pl.col('genomicLocation').struct.field('chromosome').alias('gene_chromosome'),
        pl.col('tss').alias('gene_tss'),
    )
    return qtl_genes(study).join(gene, on='geneId', how='left').select(_GENE_LOCATION_COLUMNS)
