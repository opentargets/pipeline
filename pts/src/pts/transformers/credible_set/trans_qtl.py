"""The TRANS_QTL rule: QTL credible sets whose every variant is far from the gene of their study."""

from __future__ import annotations

import polars as pl

from pts.transformers.credible_set.study_lookup import qtl_genes
from pts.transformers.credible_set.variant_id import variant_chromosome, variant_position


def qtl_gene_locations(studies: pl.LazyFrame, targets: pl.LazyFrame) -> pl.LazyFrame:
    """The chromosome and TSS of the gene of each QTL study, null for a gene absent from the target index.

    Examples:
        >>> studies = pl.LazyFrame({'studyId': ['q', 'g'], 'studyType': ['eqtl', 'gwas'], 'geneId': ['ENSG1', None]})
        >>> targets = pl.LazyFrame({'id': ['ENSG1'], 'tss': [100], 'genomicLocation': [{'chromosome': '1'}]})
        >>> qtl_gene_locations(studies, targets).collect().rows()
        [('q', '1', 100)]
    """
    genes = targets.select(
        pl.col('id').alias('geneId'),
        pl.col('genomicLocation').struct.field('chromosome').alias('_gene_chromosome'),
        pl.col('tss').alias('_gene_tss'),
    )
    return qtl_genes(studies).join(genes, on='geneId', how='left').select('studyId', '_gene_chromosome', '_gene_tss')


def is_trans(threshold: int) -> pl.Expr:
    """Whether a variant is trans to its gene: on another chromosome, or more than `threshold` from the TSS.

    It reads `_variant_chromosome`, `_variant_position`, `_gene_chromosome` and `_gene_tss`. A
    comparison with a missing value counts as not trans.

    Examples:
        >>> variants = pl.DataFrame({
        ...     '_variant_chromosome': ['2', '1', '1', '1'],
        ...     '_variant_position': [100, 6_000_101, 5_000_100, 100],
        ...     '_gene_chromosome': ['1', '1', '1', None],
        ...     '_gene_tss': [100, 100, 100, None],
        ... })
        >>> variants.select(is_trans(5_000_000))['_trans'].to_list()
        [True, True, False, False]
    """
    other_chromosome = pl.col('_variant_chromosome') != pl.col('_gene_chromosome')
    far_from_tss = (pl.col('_variant_position') - pl.col('_gene_tss')).abs() > threshold
    return (other_chromosome | far_from_tss).fill_null(False).alias('_trans')


def trans_qtls(
    sets: pl.LazyFrame, variants: pl.LazyFrame, studies: pl.LazyFrame, targets: pl.LazyFrame, threshold: int
) -> pl.LazyFrame:
    """The ids of the QTL sets whose every variant `is_trans` to the gene of their study.

    GWAS sets and sets of a study without a `geneId` are never trans. This is the `isTransQtl`
    rule of the `main` branch.

    Examples:
        >>> sets = pl.LazyFrame({'id': ['far', 'near'], 'studyId': ['s', 's']})
        >>> variants = pl.LazyFrame({
        ...     'credibleSetId': ['far', 'near', 'near'],
        ...     'variantId': ['2_10_A_T', '2_10_A_T', '1_9_A_T'],
        ... })
        >>> studies = pl.LazyFrame({'studyId': ['s'], 'studyType': ['eqtl'], 'geneId': ['g']})
        >>> targets = pl.LazyFrame({'id': ['g'], 'tss': [10], 'genomicLocation': [{'chromosome': '1'}]})
        >>> trans_qtls(sets, variants, studies, targets, 5_000_000).collect()['id'].to_list()
        ['far']
    """
    variant_id = pl.col('variantId')
    variant_locations = variants.select(
        pl.col('credibleSetId').alias('id'),
        variant_chromosome(variant_id).alias('_variant_chromosome'),
        variant_position(variant_id).alias('_variant_position'),
    )
    return (
        variant_locations
        .join(sets.select('id', 'studyId'), on='id')
        .join(qtl_gene_locations(studies, targets), on='studyId')
        .group_by('id')
        .agg(is_trans(threshold).all())
        .filter('_trans')
        .select('id')
    )
