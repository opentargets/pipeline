"""Collect the variant rows that the variant step annotates."""

from collections.abc import Sequence

import polars as pl
from loguru import logger

from pts.transformers.utils.dataset import scan_dataset


def as_vcf_df(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Extract lead and locus variants into the eight VCF columns.

    Rows without a lead variant are excluded, even when a locus is present. The
    source rsID accompanies every variant extracted from the row. Invalid positions
    and alleles are discarded; missing alleles are represented by dots.
    """
    logger.info('Extracting variants for VCF inputs')
    schema = frame.collect_schema()
    if 'variantId' not in schema:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias('variantId'))
    if 'variantRsId' not in schema:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias('variantRsId'))
    locus = pl.lit([], dtype=pl.List(pl.String))
    locus_dtype = schema.get('locus')
    if isinstance(locus_dtype, pl.List) and isinstance(locus_dtype.inner, pl.Struct):
        locus = pl.col('locus').list.eval(pl.element().struct.field('variantId')).fill_null([])
    variants = (
        frame
        .filter(pl.col('variantId').is_not_null())
        .select(
            pl.concat_list(pl.col('variantId'), locus).list.unique(maintain_order=True).alias('variantId'),
            'variantRsId',
        )
        .explode('variantId')
    )
    parts = pl.col('variantId').str.split('_')
    return (
        variants
        .select(
            parts.list.get(0, null_on_oob=True).fill_null('.').alias('CHROM'),
            parts.list.get(1, null_on_oob=True).cast(pl.Int32, strict=False).alias('POS'),
            pl.col('variantRsId').fill_null('.').alias('ID'),
            parts.list.get(2, null_on_oob=True).fill_null('.').alias('REF'),
            parts.list.get(3, null_on_oob=True).fill_null('.').alias('ALT'),
            pl.lit('.').alias('QUAL'),
            pl.lit('.').alias('FILTER'),
            pl.lit('.').alias('INFO'),
        )
        .unique()
        .filter(
            (pl.col('CHROM') != '.')
            & pl.col('POS').is_not_null()
            & pl.col('REF').str.contains(r'^[GCTA.]*$')
            & pl.col('ALT').str.contains(r'^[GCTA.]*$')
        )
        .sort('CHROM', 'POS')
    )


def collect_variants(
    uniprot_variants: str,
    eva: str,
    pharmacogenomics: str,
    credible_set_variant: str,
    additional_variants: Sequence[str] = (),
) -> pl.LazyFrame:
    """Deduplicated variants from the four evidence sources and any additional datasets, sorted by coordinate.

    The kept rsID is the smallest real one, with '.' for a variant that has none. The credible
    set variants are the validated CredibleSetVariant rows, which hold every lead and member; a
    variant has one row per credible set it is in, so their distinct identifiers are taken before
    they are split into VCF columns. Each of `additional_variants` is a parquet dataset with a
    `variantId` column, read the same way.

    Raises:
        ValueError: If the credible set variants have no `variantId` column, as a CredibleSet
            dataset given in their place does not.
    """
    logger.info('Reading variant evidence')
    credible_set_variants = scan_dataset(credible_set_variant)
    if 'variantId' not in credible_set_variants.collect_schema():
        raise ValueError(f'{credible_set_variant} has no variantId column: expected the credible set variants')
    frames = [
        as_vcf_df(scan_dataset(uniprot_variants, format='ndjson', infer_schema_length=None)),
        as_vcf_df(scan_dataset(eva, format='ndjson', infer_schema_length=None)),
        as_vcf_df(scan_dataset(pharmacogenomics)),
        as_vcf_df(credible_set_variants.select('variantId').unique()),
        *[as_vcf_df(scan_dataset(path)) for path in additional_variants],
    ]
    return (
        pl
        .concat(frames)
        .sort('CHROM', 'POS', 'REF', 'ALT', pl.col('ID') == '.', 'ID')
        .unique(subset=['CHROM', 'POS', 'REF', 'ALT'], keep='first', maintain_order=True)
        .select(
            pl.col('CHROM').alias('chromosome'),
            pl.col('POS').alias('position'),
            pl.col('ID').alias('rsId'),
            pl.col('REF').alias('referenceAllele'),
            pl.col('ALT').alias('alternateAllele'),
        )
    )
