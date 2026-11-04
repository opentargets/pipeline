"""The TRANS_QTL rule: QTL credible sets whose every variant is far from the gene of their study."""

from __future__ import annotations

import polars as pl

from pts.transformers.credible_set.study_lookup import qtl_genes
from pts.transformers.credible_set.variant_id import variant_chromosome, variant_position


def qtl_gene_locations(study: pl.LazyFrame, target: pl.LazyFrame) -> pl.LazyFrame:
    """The chromosome and TSS of the gene of each QTL study, null for a gene absent from the target index.

    Examples:
        >>> study = pl.LazyFrame({'studyId': ['q', 'g'], 'studyType': ['eqtl', 'gwas'], 'geneId': ['ENSG1', None]})
        >>> target = pl.LazyFrame({'id': ['ENSG1'], 'tss': [100], 'genomicLocation': [{'chromosome': '1'}]})
        >>> qtl_gene_locations(study, target).collect().rows()
        [('q', '1', 100)]
    """
    gene = target.select(
        pl.col('id').alias('geneId'),
        pl.col('genomicLocation').struct.field('chromosome').alias('_gene_chromosome'),
        pl.col('tss').alias('_gene_tss'),
    )
    return qtl_genes(study).join(gene, on='geneId', how='left').select('studyId', '_gene_chromosome', '_gene_tss')


def is_trans(threshold: int) -> pl.Expr:
    """Whether a variant is trans to its gene: on another chromosome, or more than `threshold` from the TSS.

    It reads `_variant_chromosome`, `_variant_position`, `_gene_chromosome` and `_gene_tss`. A
    comparison with a missing value counts as not trans.

    Examples:
        >>> credible_set_variant = pl.DataFrame({
        ...     '_variant_chromosome': ['2', '1', '1', '1'],
        ...     '_variant_position': [100, 6_000_101, 5_000_100, 100],
        ...     '_gene_chromosome': ['1', '1', '1', None],
        ...     '_gene_tss': [100, 100, 100, None],
        ... })
        >>> credible_set_variant.select(is_trans(5_000_000))['_trans'].to_list()
        [True, True, False, False]
    """
    other_chromosome = pl.col('_variant_chromosome') != pl.col('_gene_chromosome')
    far_from_tss = (pl.col('_variant_position') - pl.col('_gene_tss')).abs() > threshold
    return (other_chromosome | far_from_tss).fill_null(False).alias('_trans')


def trans_qtls(
    credible_set: pl.LazyFrame,
    credible_set_variant: pl.LazyFrame,
    study: pl.LazyFrame,
    target: pl.LazyFrame,
    threshold: int,
) -> pl.LazyFrame:
    """The ids of the QTL sets whose every variant `is_trans` to the gene of their study.

    GWAS sets and sets of a study without a `geneId` are never trans. This is the `isTransQtl`
    rule of the `main` branch.

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
    variant_id = pl.col('variantId')
    variant_location = credible_set_variant.select(
        pl.col('credibleSetId').alias('id'),
        variant_chromosome(variant_id).alias('_variant_chromosome'),
        variant_position(variant_id).alias('_variant_position'),
    )
    return (
        variant_location
        .join(credible_set.select('id', 'studyId'), on='id')
        .join(qtl_gene_locations(study, target), on='studyId')
        .group_by('id')
        .agg(is_trans(threshold).all())
        .filter('_trans')
        .select('id')
    )
