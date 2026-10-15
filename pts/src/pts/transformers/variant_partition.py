"""Prepare partitioned VCF inputs for variant effect prediction."""

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError
from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.utils.dataset import scan_dataset


class VariantPartitionSource(BaseModel):
    """Variant evidence paths in their respective source formats."""

    model_config = ConfigDict(extra='forbid')

    uniprot_variants: str
    eva: str
    pharmacogenomics: str
    credible_set: str


class VariantPartitionDestination(BaseModel):
    """Directory of tab-separated VCF parts."""

    model_config = ConfigDict(extra='forbid')

    output: str


class VariantPartitionSettings(BaseModel):
    """Maximum number of variants per VEP input file."""

    model_config = ConfigDict(extra='forbid')

    partition_size: int = Field(default=2000, gt=0)


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


def variant_partition(source: dict[str, str], destination: dict[str, str], settings: dict, config: Config) -> None:
    """Write deduplicated, sorted variants into bounded VEP input files.

    Headers retain CHROM for the orchestration task that prepends the VCF hash.
    File boundaries have a strict row limit; execution uses the streaming engine.
    """
    sources = VariantPartitionSource.model_validate(source)
    destinations = VariantPartitionDestination.model_validate(destination)
    options = VariantPartitionSettings.model_validate(settings)
    try:
        StorageHandle(destinations.output).stat()
    except NotFoundError:
        pass
    else:
        raise ValueError(f'{destinations.output!r} already exists; remove it before writing VCF parts')
    logger.info('Reading variant evidence for partitioning')
    frames = [
        as_vcf_df(scan_dataset(sources.uniprot_variants, format='ndjson', infer_schema_length=None)),
        as_vcf_df(scan_dataset(sources.eva, format='ndjson', infer_schema_length=None)),
        as_vcf_df(scan_dataset(sources.pharmacogenomics)),
        as_vcf_df(scan_dataset(sources.credible_set)),
    ]
    variants = (
        pl
        .concat(frames)
        .unique(subset=['CHROM', 'POS', 'REF', 'ALT'], keep='first', maintain_order=True)
        .sort('CHROM', 'POS')
    )
    logger.info('Writing VCF parts to {} (at most {} rows per file)', destinations.output, options.partition_size)
    variants.sink_csv(
        pl.PartitionBy(destinations.output, max_rows_per_file=options.partition_size, approximate_bytes_per_file=None),
        separator='\t',
        quote_style='never',
        mkdir=True,
        engine='streaming',
    )
