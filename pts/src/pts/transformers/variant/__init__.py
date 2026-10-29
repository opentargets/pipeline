"""Variant index generation from VEP output and supplementary annotations.

Ported from Gentropy's `variant_index` step. VEP output is parsed into the variant index,
supplementary variant annotations (gnomAD, LoF curation) are merged by variant, and
amino-acid level predictions (FoldX) are added through each variant's first transcript with
an amino-acid change.

VEP parts are parsed into local Parquet by a pool of processes, each with few polars threads.
The index is then annotated one chromosome at a time, reading only that chromosome's
annotations, in batches of variants annotated in parallel; each chromosome is written as one
part of the output.
"""

from __future__ import annotations

import multiprocessing
import os
from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.variant import variant_schema
from pts.transformers.utils.dataset import check_destinations_absent, scan_dataset
from pts.transformers.variant.gpn_star import GpnStarCutoffs, add_gpn_star, gpn_star_files, read_gpn_star
from pts.transformers.variant.index import (
    add_annotation,
    amino_acid_keys,
    annotate_with_amino_acid_consequences,
    annotation_parts_by_chromosome,
    read_variant_annotations,
    require_rows,
)
from pts.transformers.variant.vep import list_vep_parts, parse_vep, read_vep


class VariantSource(BaseModel):
    """VEP output and supplementary annotation paths."""

    model_config = ConfigDict(extra='forbid')

    vep: str
    variant_annotations: list[str] = Field(default_factory=list)
    amino_acid_annotations: list[str] = Field(default_factory=list)
    gpn_star: str | None = None


class VariantDestination(BaseModel):
    """Variant index output path."""

    model_config = ConfigDict(extra='forbid')

    output: str


class VariantSettings(BaseModel):
    """Identifier hashing and parallelism."""

    model_config = ConfigDict(extra='forbid')

    hash_threshold: int = Field(default=300, gt=0)
    """Identifiers at least this long are hashed."""
    parse_processes: int | None = Field(default=None, gt=0)
    """Processes parsing VEP parts at the same time; one per CPU by default."""
    parse_threads: int = Field(default=1, gt=0)
    """Polars threads in each parsing process."""
    batch_size: int = Field(default=5_000, gt=0)
    """Variants per batch when annotating a chromosome."""
    gpn_star: GpnStarCutoffs | None = None
    """Cut-offs normalising GPN-Star scores; required with a GPN-Star source."""


@contextmanager
def _environment(name: str, value: str) -> Generator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            del os.environ[name]
        else:
            os.environ[name] = previous


def parse_part(part: str, path: Path, hash_threshold: int) -> list[str]:
    """Parse one VEP part into a Parquet file, returning the chromosomes it holds."""
    frame = parse_vep(read_vep(part), hash_threshold).collect(engine='streaming')
    frame.write_parquet(path)
    return frame['chromosome'].unique().to_list()


def parse_parts(
    parts: list[str],
    directory: Path,
    hash_threshold: int,
    processes: int | None = None,
    threads: int = 1,
) -> dict[str, list[Path]]:
    """Parse each VEP part into its own Parquet file, grouped by the chromosomes it holds."""
    directory.mkdir(parents=True)
    paths = [directory / f'{number:08d}.parquet' for number in range(len(parts))]
    by_chromosome: dict[str, list[Path]] = defaultdict(list)
    with (
        _environment('POLARS_MAX_THREADS', str(threads)),
        ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context('spawn')) as pool,
    ):
        parsed = pool.map(parse_part, parts, paths, [hash_threshold] * len(parts))
        for number, (path, chromosomes) in enumerate(zip(paths, parsed, strict=True)):
            if number % 500 == 0:
                logger.info('Parsed VEP part {} of {}', number + 1, len(parts))
            for chromosome in chromosomes:
                by_chromosome[chromosome].append(path)
    if not by_chromosome:
        raise ValueError('No VEP records found')
    return by_chromosome


def annotate(
    frame: pl.DataFrame,
    variant_annotations: list[pl.DataFrame],
    amino_acid_annotations: list[pl.DataFrame],
    gpn_star: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Merge GPN-Star scores, then variant annotations, then amino-acid annotations, into index rows."""
    if gpn_star is not None:
        frame = add_gpn_star(frame, gpn_star)
    for dataset in variant_annotations:
        frame = add_annotation(frame, dataset)
    for dataset in amino_acid_annotations:
        frame = annotate_with_amino_acid_consequences(frame, dataset)
    return frame


def variant(
    source: dict[str, Any],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and write the variant index."""
    sources = VariantSource.model_validate(source)
    destinations = VariantDestination.model_validate(destination)
    options = VariantSettings.model_validate(settings)
    check_destinations_absent([destinations.output])
    if sources.gpn_star is not None and options.gpn_star is None:
        raise ValueError('a gpn_star source needs gpn_star cut-offs in the settings')
    gpn_star_by_chromosome = gpn_star_files(sources.gpn_star) if sources.gpn_star is not None else {}

    with TemporaryDirectory(dir=config.work_path) as scratch:
        parts = list_vep_parts(sources.vep)
        logger.info('Parsing {} VEP parts from {}', len(parts), sources.vep)
        by_chromosome = parse_parts(
            parts, Path(scratch) / 'vep', options.hash_threshold, options.parse_processes, options.parse_threads
        )
        index = pl.scan_parquet(sorted({path for paths in by_chromosome.values() for path in paths}))

        annotation_parts = []
        for path in sources.variant_annotations:
            logger.info('Locating variant annotations in {}', path)
            annotation_parts.append(annotation_parts_by_chromosome(path))
        amino_acid_annotations = []
        if sources.amino_acid_annotations:
            keys = amino_acid_keys(index).collect(engine='streaming')
            for path in sources.amino_acid_annotations:
                logger.info('Reading amino-acid annotations from {}', path)
                dataset = require_rows(scan_dataset(path), path).select(
                    'uniprotAccession', 'aminoAcidChange', 'variantEffect'
                )
                amino_acid_annotations.append(
                    dataset.join(keys.lazy(), on=['uniprotAccession', 'aminoAcidChange'], how='semi').collect(
                        engine='streaming'
                    )
                )

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            for number, chromosome in enumerate(sorted(by_chromosome)):
                on_chromosome = pl.col('chromosome') == chromosome
                frame = pl.scan_parquet(by_chromosome[chromosome]).filter(on_chromosome).collect()
                loci = frame.select('chromosome', 'position').unique().lazy()
                variant_annotations = [
                    read_variant_annotations(located[chromosome], options.hash_threshold, loci)
                    .filter(on_chromosome)
                    .collect(engine='streaming')
                    if chromosome in located
                    else pl.DataFrame(schema=variant_schema)
                    for located in annotation_parts
                ]
                gpn_star = (
                    read_gpn_star(gpn_star_by_chromosome[chromosome], frame['position'].unique(), options.gpn_star)
                    if chromosome in gpn_star_by_chromosome and options.gpn_star is not None
                    else None
                )
                logger.info('Annotating {} variants on chromosome {}', frame.height, chromosome)
                batches = pool.map(
                    lambda batch, merged=variant_annotations, scores=gpn_star: annotate(
                        batch, merged, amino_acid_annotations, scores
                    ),
                    frame.iter_slices(options.batch_size),
                )
                pl.concat(batches).sort('position', maintain_order=True).select(variant_schema.names()).write_parquet(
                    f'{destinations.output}/part-{number:05d}.parquet', compression='zstd', mkdir=True
                )
    logger.info('Wrote variant index to {}', destinations.output)
