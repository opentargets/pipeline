"""Variant index generation from VEP output and supplementary annotations (PySpark).

Ported from Gentropy's `variant_index` step. VEP output is parsed into the variant index,
supplementary variant annotations (gnomAD, LoF curation) are merged by variant, and
amino-acid level predictions (FoldX) are added through each variant's first transcript with
an amino-acid change.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session
from pts.pyspark.variant_utils.index import (
    VARIANT_SCHEMA,
    add_annotation,
    annotate_with_amino_acid_consequences,
    fill_null_arrays,
    read_variant_annotations,
)
from pts.pyspark.variant_utils.vep import parse_vep, read_vep


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
    """Identifier hashing and output layout."""

    model_config = ConfigDict(extra='forbid')

    hash_threshold: int = Field(default=300, gt=0)
    """Identifiers at least this long are hashed."""
    output_partitions: int = Field(default=25, gt=0)
    """Number of output parts, range-partitioned by genomic position."""


def variant(
    source: dict[str, Any],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate and write the variant index."""
    sources = VariantSource.model_validate(source)
    destinations = VariantDestination.model_validate(destination)
    options = VariantSettings.model_validate(settings)
    spark = Session(app_name='variant', properties=properties).spark

    logger.info('Parsing VEP annotations from {}', sources.vep)
    index = parse_vep(read_vep(spark, sources.vep), options.hash_threshold)
    for path in sources.variant_annotations:
        logger.info('Adding variant annotations from {}', path)
        annotations = read_variant_annotations(spark, path, options.hash_threshold)
        index = fill_null_arrays(add_annotation(index, annotations))
    for path in sources.amino_acid_annotations:
        logger.info('Adding amino-acid annotations from {}', path)
        annotations = spark.read.option('recursiveFileLookup', 'true').parquet(path)
        index = annotate_with_amino_acid_consequences(index, annotations)

    logger.info('Writing variant index to {}', destinations.output)
    (
        index
        .select(*VARIANT_SCHEMA.fieldNames())
        .repartitionByRange(options.output_partitions, f.col('chromosome'), f.col('position'))
        .sortWithinPartitions('chromosome', 'position')
        .write.mode('overwrite')
        .parquet(destinations.output)
    )
