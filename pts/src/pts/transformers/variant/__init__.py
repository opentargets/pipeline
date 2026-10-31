"""Variant index generation with fastVEP and supplementary annotations.

The variant rows are collected from the evidence and credible sets, then annotated by
`chromosome.annotate_chromosomes`: fastVEP on each chromosome, then, on chunks of each
chromosome's rows in parallel, joins on the mirrored Ensembl data (transcript fields, colocated
variants, GERP, AlphaMissense, LOFTEE and SIFT) fill in the VEP records that `parse_vep` turns into
index rows. GPN-Star scores, supplementary variant annotations (gnomAD, LoF curation) merged by
variant, and amino-acid level predictions (FoldX) through each variant's first transcript with an
amino-acid change are merged into each chunk. The inputs these share are prepared once, before the
chromosomes run, the remote ones read into local scratch, and each chunk is written as one part of
the output once every chunk has succeeded.
"""

from __future__ import annotations

import gzip
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

from pts.transformers.utils.dataset import check_destinations_absent, scan_dataset
from pts.transformers.variant.alleles import contig_spans, write_fai
from pts.transformers.variant.alphamissense import partition_alphamissense
from pts.transformers.variant.chromosome import AMINO_ACID_COLUMNS, Pools, Shared, annotate_chromosomes
from pts.transformers.variant.colocated import extract_all_vars
from pts.transformers.variant.fastvep import build_cache, prepare_gff3
from pts.transformers.variant.gpn_star import GpnStarCutoffs, gpn_star_files
from pts.transformers.variant.index import annotation_parts_by_chromosome, require_rows
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
    """The gzip LOFTEE human ancestor FASTA."""
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
    """Threads of each fastVEP run."""
    parallel_chromosomes: int = Field(default=4, gt=0)
    """Chromosomes run through fastVEP at the same time."""
    parallel_preparations: int = Field(default=4, gt=0)
    """Chromosomes whose shared chunk inputs are prepared at the same time."""
    chromosome_threads: int = Field(default=8, gt=0)
    """Polars threads of each fastVEP and preparation process."""
    chunk_rows: int = Field(default=50_000, gt=0)
    """Rows of each chunk a chromosome is annotated in after fastVEP."""
    parallel_chunks: int = Field(default=16, gt=0)
    """Chunks annotated at the same time."""
    chunk_threads: int = Field(default=2, gt=0)
    """Polars threads of each chunk process."""
    preparation_niceness: int = Field(default=5, ge=0, le=19)
    """Niceness of each preparation process; fastVEP keeps the step's own."""
    chunk_niceness: int = Field(default=15, ge=0, le=19)
    """Niceness of each chunk process."""
    batch_size: int = Field(default=5_000, gt=0)
    """Variants per batch when merging the annotations into a chunk."""
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


def amino_acid_projection(path: str, target: Path) -> Path:
    """The columns of an amino-acid annotation source that the chunks merge, written to `target`.

    Raises:
        ValueError: if the source has no rows.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    scan_dataset(path).select(AMINO_ACID_COLUMNS).sink_parquet(target)
    require_rows(pl.scan_parquet(target), path)
    return target


def _reference(
    sources: VariantSource, binary: str, directory: Path
) -> tuple[Path, dict[str, tuple[int, int]], Path]:
    """The FASTA uncompressed with its `.fai`, its `contig_spans`, and the prepared GFF3 with its fastVEP cache."""
    directory.mkdir(parents=True)
    fasta = _local(sources.fasta, directory)
    plain = directory / Path(fasta).name.removesuffix('.gz')
    with gzip.open(fasta, 'rb') as source, plain.open('wb') as out:
        shutil.copyfileobj(source, out, COPY_BYTES)
    if fasta != sources.fasta:
        Path(fasta).unlink()
    spans = contig_spans(plain)
    write_fai(plain, spans)
    gff3 = prepare_gff3(sources.gff3, directory)
    build_cache(gff3, plain, binary)
    return plain, spans, gff3


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


def _rows(sources: VariantSource, directory: Path) -> tuple[dict[str, Path], dict[str, int]]:
    """Each chromosome's rows, sorted by position, written to `directory/<chromosome>.parquet`, and their number."""
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
        frame.sort('position', maintain_order=True).write_parquet(paths[str(chromosome)])
    logger.info('Collected {} variant rows on {} chromosomes', rows.height, len(paths))
    return paths, sizes


def _parts(sources: VariantSource, options: VariantSettings, scratch: Path) -> list[Path]:
    """The annotated index of every chromosome, in parts, from its rows, fastVEP and the joins on reference data."""
    gpn_star = gpn_star_files(sources.gpn_star) if sources.gpn_star is not None else {}
    located = []
    for path in sources.variant_annotations:
        logger.info('Locating variant annotations in {}', path)
        located.append(annotation_parts_by_chromosome(path))
    amino_acids = tuple(
        amino_acid_projection(path, scratch / 'amino_acids' / f'{number:03d}.parquet')
        for number, path in enumerate(sources.amino_acid_annotations)
    )
    rows, sizes = _rows(sources, scratch / 'rows')
    for chromosome in sorted(rows):
        if options.gpn_star is not None and chromosome not in gpn_star and chromosome in STANDARD:
            logger.warning('No GPN-Star scores for chromosome {}', chromosome)
    with ThreadPoolExecutor() as pool:
        reference = pool.submit(_reference, sources, options.fastvep_binary, scratch / 'reference')
        all_vars = pool.submit(extract_all_vars, sources.vep_cache, scratch / 'all_vars')
        alphamissense = pool.submit(partition_alphamissense, sources.alphamissense, scratch / 'alphamissense', rows)
        sift = pool.submit(_sift, sources, scratch / 'sift.parquet')
        gerp = pool.submit(_local, sources.gerp, scratch / 'gerp')
        ancestor = pool.submit(_local, sources.ancestor, scratch / 'ancestor')
        plain_fasta, contigs, gff3 = reference.result()
        shared = Shared(
            plain_fasta=plain_fasta,
            contigs=contigs,
            gff3=gff3,
            gerp=gerp.result(),
            ancestor=ancestor.result(),
            core=sources.core,
            sift=sift.result(),
            binary=options.fastvep_binary,
            threads=options.fastvep_threads,
            hash_threshold=options.hash_threshold,
            directory=scratch / 'chromosomes',
            chunk_rows=options.chunk_rows,
            batch_size=options.batch_size,
            variant_annotations=tuple(located),
            amino_acid_annotations=amino_acids,
            gpn_star=gpn_star if options.gpn_star is not None else {},
            gpn_star_cutoffs=options.gpn_star,
        )
        extracted, partitions = all_vars.result(), alphamissense.result()
    for chromosome in sorted(set(rows) - set(extracted)):
        logger.warning('No colocated variants for chromosome {}: the VEP cache has no all_vars', chromosome)
    return annotate_chromosomes(
        rows,
        sizes,
        extracted,
        partitions,
        shared,
        Pools(
            fastvep=options.parallel_chromosomes,
            preparations=options.parallel_preparations,
            chunks=options.parallel_chunks,
            chromosome_threads=options.chromosome_threads,
            chunk_threads=options.chunk_threads,
            preparation_niceness=options.preparation_niceness,
            chunk_niceness=options.chunk_niceness,
        ),
    )


STANDARD = {*(str(number) for number in range(1, 23)), 'X', 'Y'}


def _publish(part: Path, destination: str) -> None:
    """Move a part into the output, a local directory or an object store prefix."""
    target = f'{destination}/{part.name}'
    if '://' not in destination:
        Path(destination).mkdir(parents=True, exist_ok=True)
        shutil.move(part, target)
        return
    with part.open('rb') as source, StorageHandle(target).open('wb') as out:
        shutil.copyfileobj(source, out, COPY_BYTES)
    part.unlink()


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
    with TemporaryDirectory(dir=config.work_path) as scratch:
        parts = _parts(sources, options, Path(scratch))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda part: _publish(part, destinations.output), parts))
    logger.info('Wrote variant index to {}', destinations.output)
