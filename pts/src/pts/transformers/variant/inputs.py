"""The variant step's inputs made local and prepared once, before the chromosomes run.

`chromosome_rows` collects and splits the variant rows by chromosome. `shared_inputs` copies the
remote inputs into scratch and prepares what every chromosome reads: the uncompressed reference and
its fastVEP cache, the colocated variants, the AlphaMissense partitions, the SIFT matrices and the
supplementary annotation sources. `prepared_chromosomes` gives each chromosome its handle.
"""

from __future__ import annotations

import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.variant.annotators.alphamissense import partition_alphamissense
from pts.transformers.variant.annotators.gpn_star import gpn_star_files
from pts.transformers.variant.annotators.sift import sift_matrices
from pts.transformers.variant.chromosome import FastvepReference
from pts.transformers.variant.index.annotations import annotation_parts_by_chromosome, require_rows
from pts.transformers.variant.prepared import (
    AMINO_ACID_COLUMNS,
    ChunkSettings,
    PreparationSources,
    PreparedChromosome,
)
from pts.transformers.variant.rows import collect_variants
from pts.transformers.variant.settings import VariantSettings, VariantSource
from pts.transformers.variant.vep.alleles import contig_spans, write_fai
from pts.transformers.variant.vep.colocated import extract_all_vars
from pts.transformers.variant.vep.fastvep import build_cache, prepare_gff3
from pts.transformers.variant.vep.transcripts import canonical_transcript_ids

COPY_BYTES = 64 << 20
STANDARD = {*(str(number) for number in range(1, 23)), 'X', 'Y'}


def _local(path: str, directory: Path) -> str:
    """`path` itself when it is _local, else a copy of it in `directory`."""
    if '://' not in path:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.rsplit('/', 1)[-1]
    with StorageHandle(path).open('rb') as source, target.open('wb') as out:
        shutil.copyfileobj(source, out, COPY_BYTES)
    return str(target)


def _local_directory(path: str, directory: Path) -> str:
    """`path` itself when it is _local, else a copy of the files directly in it, in `directory`."""
    if '://' not in path:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    for file in StorageHandle(path.rstrip('/')).glob('*'):
        if not file.endswith('/'):
            _local(file, directory)
    return str(directory)


def amino_acid_projection(path: str, target: Path) -> Path:
    """The columns of an amino-acid annotation source that the chunks merge, written to `target`.

    Raises:
        ValueError: if the source has no rows.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    scan_dataset(path).select(AMINO_ACID_COLUMNS).sink_parquet(target)
    require_rows(pl.scan_parquet(target), path)
    return target


def _reference(sources: VariantSource, binary: str, directory: Path) -> tuple[Path, dict[str, tuple[int, int]], Path]:
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


def _sift(sources: VariantSource, core: str, path: Path) -> Path:
    """The SIFT matrices of the canonical transcripts of the core dump `core`, written to `path`."""
    matrices = sift_matrices(
        sources.sift_predictions,
        sources.sift_translation_md5,
        sources.sift_attrib,
        sources.peptides,
        canonical_transcript_ids(core),
    )
    matrices.write_parquet(path)
    return path


def chromosome_rows(sources: VariantSource, directory: Path) -> tuple[dict[str, Path], dict[str, int]]:
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


@dataclass(frozen=True)
class SharedInputs:
    """What every chromosome reads, local and prepared."""

    reference: FastvepReference
    preparation: PreparationSources
    all_vars: dict[str, Path]
    """Each chromosome's `all_vars.gz`, for the chromosomes the VEP cache has one for."""
    alphamissense: dict[str, str]
    """Each chromosome's `partition_alphamissense` part."""
    gerp: str
    sift: Path


def shared_inputs(
    sources: VariantSource, options: VariantSettings, rows: dict[str, Path], scratch: Path
) -> SharedInputs:
    """The inputs every chromosome reads, copied into and prepared under `scratch`."""
    gpn_star = gpn_star_files(sources.gpn_star) if sources.gpn_star is not None else {}
    located = []
    for path in sources.variant_annotations:
        logger.info('Locating variant annotations in {}', path)
        located.append(annotation_parts_by_chromosome(path))
    amino_acids = tuple(
        amino_acid_projection(path, scratch / 'amino_acids' / f'{number:03d}.parquet')
        for number, path in enumerate(sources.amino_acid_annotations)
    )
    for chromosome in sorted(rows):
        if options.gpn_star is not None and chromosome not in gpn_star and chromosome in STANDARD:
            logger.warning('No GPN-Star scores for chromosome {}', chromosome)
    with ThreadPoolExecutor() as pool:
        reference = pool.submit(_reference, sources, options.fastvep_binary, scratch / 'reference')
        all_vars = pool.submit(extract_all_vars, sources.vep_cache, scratch / 'all_vars')
        alphamissense = pool.submit(partition_alphamissense, sources.alphamissense, scratch / 'alphamissense', rows)
        core = pool.submit(_local_directory, sources.core, scratch / 'core')
        sift = pool.submit(lambda: _sift(sources, core.result(), scratch / 'sift.parquet'))
        gerp = pool.submit(_local, sources.gerp, scratch / 'gerp')
        ancestor = pool.submit(_local, sources.ancestor, scratch / 'ancestor')
        plain_fasta, contigs, gff3 = reference.result()
        preparation = PreparationSources(
            ancestor=ancestor.result(),
            core=core.result(),
            variant_annotations=tuple(located),
            amino_acid_annotations=amino_acids,
            gpn_star=gpn_star if options.gpn_star is not None else {},
            gpn_star_cutoffs=options.gpn_star,
        )
        extracted, partitions = all_vars.result(), alphamissense.result()
        gerp_path, sift_path = gerp.result(), sift.result()
    for chromosome in sorted(set(rows) - set(extracted)):
        logger.warning('No colocated variants for chromosome {}: the VEP cache has no all_vars', chromosome)
    return SharedInputs(
        reference=FastvepReference(
            plain_fasta=plain_fasta,
            contigs=contigs,
            gff3=gff3,
            binary=options.fastvep_binary,
            threads=options.fastvep_threads,
        ),
        preparation=preparation,
        all_vars=extracted,
        alphamissense=partitions,
        gerp=gerp_path,
        sift=sift_path,
    )


def prepared_chromosomes(
    rows: dict[str, Path], shared: SharedInputs, settings: ChunkSettings, directory: Path
) -> dict[str, PreparedChromosome]:
    """Each chromosome's handle, its scratch directory under `directory`."""
    return {
        chromosome: PreparedChromosome(
            chromosome=chromosome,
            directory=directory / chromosome,
            rows=path,
            plain_fasta=shared.reference.plain_fasta,
            span=shared.reference.contigs.get(chromosome),
            all_vars=shared.all_vars.get(chromosome),
            alphamissense=shared.alphamissense[chromosome],
            gerp=shared.gerp,
            sift=shared.sift,
            settings=settings,
            annotation_sources=len(shared.preparation.variant_annotations),
            amino_acid_sources=len(shared.preparation.amino_acid_annotations),
        )
        for chromosome, path in rows.items()
    }
