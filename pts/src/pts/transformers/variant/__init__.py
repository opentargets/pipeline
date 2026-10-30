"""Variant index generation with fastVEP and supplementary annotations.

The variant rows are collected from the evidence and credible sets, then annotated one chromosome
at a time by `chromosome.annotate_chromosomes`: fastVEP, then joins on the mirrored Ensembl data
(transcript fields, colocated variants, GERP, AlphaMissense, LOFTEE and SIFT) fill in the VEP
records that `parse_vep` turns into index rows. The inputs these joins share are prepared once,
before the chromosomes run.

The index is then annotated one chromosome at a time, reading only that chromosome's
annotations, in batches of variants annotated in parallel: GPN-Star scores, supplementary
variant annotations (gnomAD, LoF curation) merged by variant, and amino-acid level predictions
(FoldX) through each variant's first transcript with an amino-acid change. Each chromosome is
written as one part of the output.
"""

from __future__ import annotations

import gzip
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.variant import variant_schema
from pts.transformers.utils.dataset import check_destinations_absent, scan_dataset
from pts.transformers.variant.alphamissense import partition_alphamissense
from pts.transformers.variant.chromosome import Shared, annotate_chromosomes
from pts.transformers.variant.colocated import extract_all_vars
from pts.transformers.variant.fastvep import build_cache, prepare_gff3
from pts.transformers.variant.gpn_star import GpnStarCutoffs, add_gpn_star, gpn_star_files, read_gpn_star
from pts.transformers.variant.index import (
    add_annotation,
    amino_acid_keys,
    annotate_with_amino_acid_consequences,
    annotation_parts_by_chromosome,
    read_variant_annotations,
    require_rows,
)
from pts.transformers.variant.rows import collect_variants
from pts.transformers.variant.sift import sift_matrices
from pts.transformers.variant.transcripts import canonical_transcript_ids

COPY_BYTES = 64 << 20


class VariantSource(BaseModel):
    """Variant rows, the reference data fastVEP and its joins read, and supplementary annotation paths."""

    model_config = ConfigDict(extra='forbid')

    uniprot_variants: str
    eva: str
    pharmacogenomics: str
    credible_set: str
    fasta: str
    """The gzip GRCh38 primary assembly FASTA."""
    gff3: str
    gerp: str
    """The GERP conservation bigWig."""
    vep_cache: str
    """The indexed VEP cache tarball, read for its `all_vars.gz`."""
    sift_predictions: str
    """The variation dump's `protein_function_predictions.txt.gz`."""
    sift_translation_md5: str
    sift_attrib: str
    peptides: str
    """The `pep.all.fa.gz` FASTA, whose md5s key the SIFT matrices."""
    alphamissense: str
    ancestor: str
    """The bgzip LOFTEE human ancestor FASTA, with its `.fai` and `.gzi` next to it."""
    core: str
    """Directory of the Ensembl core database dump."""
    variant_annotations: list[str] = Field(default_factory=list)
    amino_acid_annotations: list[str] = Field(default_factory=list)
    gpn_star: str | None = None


class VariantDestination(BaseModel):
    """Variant index output path."""

    model_config = ConfigDict(extra='forbid')

    output: str


class VariantSettings(BaseModel):
    """Identifier hashing, fastVEP and parallelism."""

    model_config = ConfigDict(extra='forbid')

    hash_threshold: int = Field(default=300, gt=0)
    """Identifiers at least this long are hashed."""
    fastvep_binary: str = 'fastvep'
    fastvep_threads: int = Field(default=8, gt=0)
    """Threads of each fastVEP run, and polars threads of each chromosome's process."""
    parallel_chromosomes: int = Field(default=4, gt=0)
    """Chromosomes annotated at the same time."""
    batch_size: int = Field(default=5_000, gt=0)
    """Variants per batch when annotating a chromosome."""
    gpn_star: GpnStarCutoffs | None = None
    """Cut-offs normalising GPN-Star scores; required with a GPN-Star source."""


def _local(path: str, directory: Path) -> str:
    """`path` itself when it is local, else a copy of it in `directory`."""
    if '://' not in path:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.rsplit('/', 1)[-1]
    with StorageHandle(path).open('rb') as source, target.open('wb') as out:
        shutil.copyfileobj(source, out, COPY_BYTES)
    return str(target)


def _reference(sources: VariantSource, binary: str, directory: Path) -> tuple[str, Path, Path]:
    """The local gzip FASTA, the FASTA uncompressed, and the prepared GFF3 with its fastVEP cache."""
    directory.mkdir(parents=True)
    fasta = _local(sources.fasta, directory)
    plain = directory / Path(fasta).name.removesuffix('.gz')
    with gzip.open(fasta, 'rb') as source, plain.open('wb') as out:
        shutil.copyfileobj(source, out, COPY_BYTES)
    gff3 = prepare_gff3(sources.gff3, directory)
    build_cache(gff3, plain, binary)
    return fasta, plain, gff3


def _sift(sources: VariantSource, path: Path) -> Path:
    """The SIFT matrices of the canonical transcripts, written to `path`."""
    matrices = sift_matrices(
        sources.sift_predictions,
        sources.sift_translation_md5,
        sources.sift_attrib,
        sources.peptides,
        canonical_transcript_ids(sources.core),
    )
    matrices.write_parquet(path)
    return path


def _ancestor(path: str, directory: Path) -> str:
    """The human ancestor FASTA and its indexes, local."""
    for index in (f'{path}.fai', f'{path}.gzi'):
        _local(index, directory)
    return _local(path, directory)


def _rows(sources: VariantSource, directory: Path) -> tuple[dict[str, Path], dict[str, int]]:
    """Each chromosome's rows written to `directory/<chromosome>.parquet`, and their number."""
    rows = collect_variants(
        sources.uniprot_variants, sources.eva, sources.pharmacogenomics, sources.credible_set
    ).collect(engine='streaming')
    if rows.is_empty():
        raise ValueError('No variant rows found')
    directory.mkdir(parents=True)
    paths, sizes = {}, {}
    for (chromosome,), frame in rows.partition_by('chromosome', as_dict=True, maintain_order=True).items():
        paths[str(chromosome)] = directory / f'{chromosome}.parquet'
        sizes[str(chromosome)] = frame.height
        frame.write_parquet(paths[str(chromosome)])
    logger.info('Collected {} variant rows on {} chromosomes', rows.height, len(paths))
    return paths, sizes


def _index_by_chromosome(sources: VariantSource, options: VariantSettings, scratch: Path) -> dict[str, Path]:
    """The parsed index of every chromosome, from its rows, fastVEP and the joins on reference data."""
    rows, sizes = _rows(sources, scratch / 'rows')
    with ThreadPoolExecutor() as pool:
        reference = pool.submit(_reference, sources, options.fastvep_binary, scratch / 'reference')
        all_vars = pool.submit(extract_all_vars, sources.vep_cache, scratch / 'all_vars')
        alphamissense = pool.submit(partition_alphamissense, sources.alphamissense, scratch / 'alphamissense', rows)
        sift = pool.submit(_sift, sources, scratch / 'sift.parquet')
        gerp = pool.submit(_local, sources.gerp, scratch / 'gerp')
        ancestor = pool.submit(_ancestor, sources.ancestor, scratch / 'ancestor')
        fasta, plain_fasta, gff3 = reference.result()
        shared = Shared(
            fasta=fasta,
            plain_fasta=plain_fasta,
            gff3=gff3,
            gerp=gerp.result(),
            ancestor=ancestor.result(),
            core=sources.core,
            sift=sift.result(),
            binary=options.fastvep_binary,
            threads=options.fastvep_threads,
            hash_threshold=options.hash_threshold,
            directory=scratch / 'chromosomes',
        )
        extracted, partitions = all_vars.result(), alphamissense.result()
    for chromosome in sorted(set(rows) - set(extracted)):
        logger.warning('No colocated variants for chromosome {}: the VEP cache has no all_vars', chromosome)
    return annotate_chromosomes(rows, sizes, extracted, partitions, shared, options.parallel_chromosomes)


STANDARD = {*(str(number) for number in range(1, 23)), 'X', 'Y'}


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
        by_chromosome = _index_by_chromosome(sources, options, Path(scratch))
        index = pl.scan_parquet([by_chromosome[chromosome] for chromosome in sorted(by_chromosome)])

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
                frame = pl.read_parquet(by_chromosome[chromosome])
                loci = frame.select('chromosome', 'position').unique().lazy()
                variant_annotations = [
                    read_variant_annotations(located[chromosome], options.hash_threshold, loci)
                    .filter(on_chromosome)
                    .collect(engine='streaming')
                    if chromosome in located
                    else pl.DataFrame(schema=variant_schema)
                    for located in annotation_parts
                ]
                if options.gpn_star is not None and chromosome not in gpn_star_by_chromosome and chromosome in STANDARD:
                    logger.warning('No GPN-Star scores for chromosome {}', chromosome)
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
