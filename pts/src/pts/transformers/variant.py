"""Variant index generation from VEP and supplementary annotations."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.variant import variant_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.variant_utils.annotations import (
    add_annotation,
    annotate_with_amino_acid_consequences,
    hash_long_variant_ids,
)
from pts.transformers.variant_utils.vep import parse_vep, read_vep


class VariantSource(BaseModel):
    """VEP output and supplementary annotation paths."""

    model_config = ConfigDict(extra='forbid')

    vep: str
    variant_annotations: list[str] = Field(default_factory=list)
    amino_acid_annotations: list[str] = Field(default_factory=list)


class VariantDestination(BaseModel):
    """Variant index output path."""

    model_config = ConfigDict(extra='forbid')

    output: str


class VariantSettings(BaseModel):
    """Variant identifier settings."""

    model_config = ConfigDict(extra='forbid')

    hash_threshold: int = Field(default=300, gt=0)


def read_variant_annotations(path: str, hash_threshold: int = 300) -> pl.LazyFrame:
    """Scan recursively partitioned supplementary variant annotations."""
    logger.info('Reading supplementary variant annotations from {}', path)
    handle = StorageHandle(path)
    parts = sorted(handle.glob('**/*.parquet')) if handle.stat().is_dir else [path]
    if not parts:
        raise ValueError(f'No Parquet annotation parts found in {path!r}')
    frame = pl.scan_parquet(
        parts,
        schema=variant_schema,
        hive_partitioning=False,
        missing_columns='insert',
        cast_options=pl.ScanCastOptions(missing_struct_fields='insert'),
    )
    arrays = [pl.col(name).fill_null([]) for name, dtype in variant_schema.items() if isinstance(dtype, pl.List)]
    return frame.with_columns(arrays).pipe(hash_long_variant_ids, hash_threshold)


def variant(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and write the variant index with optional supplementary annotations."""
    sources = VariantSource.model_validate(source)
    destinations = VariantDestination.model_validate(destination)
    options = VariantSettings.model_validate(settings)

    logger.info('Parsing VEP annotations from {}', sources.vep)
    frame = read_vep(sources.vep).pipe(parse_vep, options.hash_threshold)
    arrays = [pl.col(name).fill_null([]) for name, dtype in variant_schema.items() if isinstance(dtype, pl.List)]
    frame = frame.with_columns(arrays)
    for path in sources.variant_annotations:
        frame = frame.pipe(add_annotation, read_variant_annotations(path, options.hash_threshold)).with_columns(arrays)
    for path in sources.amino_acid_annotations:
        logger.info('Reading amino-acid annotations from {}', path)
        annotations = scan_dataset(path).select('uniprotAccession', 'aminoAcidChange', 'variantEffect')
        frame = frame.pipe(annotate_with_amino_acid_consequences, annotations).with_columns(arrays)

    logger.info('Writing variant index to {}', destinations.output)
    write_dataset(
        frame.select(variant_schema.names()).sort('chromosome', 'position', nulls_last=False), destinations.output
    )
