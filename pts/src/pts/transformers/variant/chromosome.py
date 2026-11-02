"""The variant index of each chromosome, from fastVEP, the joins that complete its consequences and its annotations.

Each chromosome is run through fastVEP in a pool of processes, the largest first, each fastVEP
with a few threads; its records are split by row into chunks. Alongside, a second pool prepares
what every chunk of a chromosome shares: its transcript fields, LOFTEE models, ancestor contig,
colocated variants, supplementary annotations and GPN-Star scores, written in position order, and
the amino-acid annotations of its transcripts' proteins.
Once a chromosome's records and preparations are ready, a third pool annotates its chunks, each
into one part of the index: `PreparedChromosome.chunk` gives a chunk's inputs, and
`annotate_inputs` annotates them.
Workers read their inputs from paths and write into the scratch directory.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Generator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from pts.schemas.variant import variant_schema
from pts.transformers.variant.annotators.gerp import KentBigWig
from pts.transformers.variant.annotators.gpn_star import add_gpn_star
from pts.transformers.variant.annotators.loftee import add_loftee
from pts.transformers.variant.annotators.sift import add_sift
from pts.transformers.variant.index.annotations import (
    add_annotation,
    amino_acid_keys,
    annotate_with_amino_acid_consequences,
)
from pts.transformers.variant.index.rows import index_rows
from pts.transformers.variant.predictions import add_gerp_and_alphamissense
from pts.transformers.variant.prepared import (
    ChunkSettings,
    PreparationSources,
    PreparedChromosome,
    PreparedChunk,
    write_complete,
)
from pts.transformers.variant.vep.colocated import add_colocated
from pts.transformers.variant.vep.fastvep import run_fastvep, write_vcf
from pts.transformers.variant.vep.records import CHUNK_BYTES, chunk_frames, iter_records, unannotated_frames


@dataclass(frozen=True)
class FastvepReference:
    """The reference fastVEP runs every chromosome against, prepared once before the pools."""

    plain_fasta: Path
    """The uncompressed reference, which fastVEP and the chunks read."""
    contigs: dict[str, tuple[int, int]]
    """`contig_spans` of `plain_fasta`."""
    gff3: Path
    """The GFF3 whose fastVEP cache `build_cache` built from `plain_fasta`."""
    binary: str
    threads: int
    """Threads of each fastVEP run."""


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


def _submit(pool: ProcessPoolExecutor, threads: int, function: Callable[..., object], *args: object) -> Future:
    """Submit to a spawn pool whose processes run polars with `threads` threads.

    A pool starts its processes as work is submitted, and a spawned process reads
    `POLARS_MAX_THREADS` from the environment it starts with.
    """
    with _environment('POLARS_MAX_THREADS', str(threads)):
        return pool.submit(function, *args)


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


def split_records(output: Path, rows: int, prepared: PreparedChromosome, chunk_bytes: int = CHUNK_BYTES) -> None:
    """Write the records of a fastVEP output by chunk, each under the chunk of the row it echoes as `id`.

    The JSON is read in slices of about `chunk_bytes`.

    Raises:
        ValueError: if the record ids are not the row ordinals 0..rows - 1.
    """
    seen = np.zeros(rows, dtype=bool)
    count = 0
    for number, frame in enumerate(iter_records(output, chunk_bytes)):
        ids = frame['row'].to_numpy()
        count += ids.size
        if ids.size and (ids.min() < 0 or ids.max() >= rows or seen[ids].any() or np.unique(ids).size < ids.size):
            raise ValueError(f'{output}: fastVEP record ids are not the ordinals 0..{rows - 1} of the rows')
        seen[ids] = True
        chunks = frame.with_columns((pl.col('row') // prepared.settings.chunk_rows).alias('_chunk'))
        for (chunk,), part in chunks.partition_by('_chunk', as_dict=True).items():
            write_complete(part.drop('_chunk'), prepared.record_slice(int(chunk), number))
    if count != rows:
        raise ValueError(f'{output}: {count} fastVEP records for {rows} rows')


def run_chromosome(prepared: PreparedChromosome, reference: FastvepReference) -> tuple[int, bool]:
    """Run fastVEP on one chromosome's rows and split its records into chunks of `chunk_rows` rows.

    A chromosome the reference lacks is not run through fastVEP: its chunks are intergenic,
    without HGVS.

    Returns:
        The number of chunks, and whether fastVEP ran.

    Raises:
        RuntimeError: if fastVEP fails or its output is truncated.
        ValueError: if fastVEP's record ids are not the row ordinals.
    """
    chromosome = prepared.chromosome
    frame = pl.read_parquet(prepared.rows)
    sequence = prepared.sequence()
    chunks = -(-frame.height // prepared.settings.chunk_rows)
    if not sequence:
        logger.warning(
            'No reference sequence for chromosome {}: {} variants left without transcript consequences',
            chromosome,
            frame.height,
        )
        return chunks, False
    prepared.fastvep_directory().mkdir(parents=True)
    vcf = prepared.fastvep_vcf()
    write_vcf(frame, vcf, sequence)
    logger.info('Running fastVEP on {} variants of chromosome {}', frame.height, chromosome)
    output = run_fastvep(
        vcf, reference.gff3, reference.plain_fasta, prepared.fastvep_json(), reference.threads, reference.binary
    )
    vcf.unlink()
    try:
        split_records(output, frame.height, prepared)
    finally:
        output.unlink()
    return chunks, True


def prepare_chromosome(prepared: PreparedChromosome, sources: PreparationSources) -> None:
    """Write what every chunk of the chromosome reads, once, by `PreparedChromosome.write`."""
    prepared.write(sources)


def _merged(index: pl.DataFrame, inputs: PreparedChunk, batch_size: int) -> pl.DataFrame:
    """The index rows of a chunk with the prepared annotations of its loci merged in, by `annotate`."""
    loci = index.select('chromosome', 'position').unique()
    variant_annotations = inputs.annotations(loci)
    gpn_star = inputs.gpn_star(loci['position'])
    amino_acid_annotations = inputs.amino_acids(amino_acid_keys(index))
    batches = [
        annotate(batch, variant_annotations, amino_acid_annotations, gpn_star)
        for batch in index.iter_slices(batch_size)
    ]
    return pl.concat(batches).sort('position', maintain_order=True).select(variant_schema.names())


def annotate_inputs(inputs: PreparedChunk, settings: ChunkSettings) -> pl.DataFrame:
    """The annotated variant index of one chunk of a chromosome's rows, from its inputs.

    Raises:
        ValueError: if the inputs have records without transcripts, or transcripts without records.
    """
    chromosome = inputs.chromosome
    records = inputs.records()
    if records is None and inputs.transcripts is None:
        variants, consequences = unannotated_frames(inputs.rows, chromosome)
    elif records is None or inputs.transcripts is None:
        raise ValueError(f'chromosome {chromosome}: a chunk needs both its records and transcripts, or neither')
    else:
        variants, consequences = chunk_frames(records, inputs.rows, inputs.transcripts, inputs.sequence, chromosome)
    del records
    models, ancestor = inputs.models(), inputs.ancestor()
    consequences = add_gerp_and_alphamissense(consequences, inputs.gerp, inputs.alphamissense, chromosome)
    with KentBigWig(inputs.gerp) as gerp:
        consequences = add_loftee(consequences, models, inputs.sequence, ancestor, gerp, chromosome)
    del models, ancestor
    consequences = add_sift(consequences, inputs.sift_matrices(consequences['transcript_id']))
    variants = add_colocated(variants, inputs.colocated())
    index = index_rows(variants, consequences, settings.hash_threshold)
    return _merged(index, inputs, settings.batch_size)


def annotate_chunk(prepared: PreparedChromosome, chunk: int, annotated: bool, part: Path) -> Path:
    """The annotated variant index of one chunk of a chromosome's rows, written to `part`.

    Args:
        prepared: The chromosome, prepared.
        chunk: The chunk: `chunk_rows` rows from row `chunk * chunk_rows`.
        annotated: Whether `run_chromosome` ran fastVEP on the chromosome.
        part: Where to write the chunk's index.

    Returns:
        `part`, which is only written once the whole chunk is annotated. The chunk's fastVEP
        records are deleted once it is annotated, before `part` is written.
    """
    inputs = prepared.chunk(chunk, annotated)
    logger.info('Annotating {} variants of chromosome {}, chunk {}', inputs.rows.height, prepared.chromosome, chunk)
    index = annotate_inputs(inputs, prepared.settings)
    if annotated:
        prepared.release(chunk)
    return write_complete(index, part)


@dataclass(frozen=True)
class Pools:
    """The processes of each pool, and the polars threads and niceness of each process.

    The fastVEP pool, and the fastVEP runs it starts, keep the niceness of the step.
    """

    fastvep: int
    """Chromosomes run through fastVEP at a time."""
    preparations: int
    """Chromosomes prepared at a time."""
    chunks: int
    """Chunks annotated at a time."""
    chromosome_threads: int
    """Polars threads of each fastVEP and preparation process."""
    chunk_threads: int
    """Polars threads of each chunk process."""
    preparation_niceness: int = 0
    """Niceness of each preparation process."""
    chunk_niceness: int = 0
    """Niceness of each chunk process."""


def lower_priority(niceness: int) -> None:
    """Raise the process's niceness to `niceness`; a process already nicer is left as it is."""
    os.nice(max(0, niceness - os.nice(0)))


def annotate_chromosomes(
    prepared: dict[str, PreparedChromosome],
    sizes: dict[str, int],
    reference: FastvepReference,
    sources: PreparationSources,
    pools: Pools,
    parts: Path,
) -> list[Path]:
    """The annotated index of every chromosome, in chunks, written under `parts`.

    fastVEP and the preparations each run in their own pool, the chromosomes with the most rows
    first. A chromosome's chunks go to the third pool once its records and its preparation are
    ready. Part `part-<c>-<k>.parquet` holds chunk k of the c-th chromosome in name order.

    Returns:
        The paths of the parts, in name order.

    Raises:
        RuntimeError: as soon as a chromosome or a chunk fails; units not started are cancelled.
    """
    order = sorted(prepared, key=lambda chromosome: sizes[chromosome], reverse=True)
    numbers = {chromosome: number for number, chromosome in enumerate(sorted(prepared))}
    context = multiprocessing.get_context('spawn')
    written: list[Path] = []
    with (
        ProcessPoolExecutor(pools.fastvep, mp_context=context) as fastvep_pool,
        ProcessPoolExecutor(
            pools.preparations,
            mp_context=context,
            initializer=lower_priority,
            initargs=(pools.preparation_niceness,),
        ) as preparation_pool,
        ProcessPoolExecutor(
            pools.chunks, mp_context=context, initializer=lower_priority, initargs=(pools.chunk_niceness,)
        ) as chunk_pool,
    ):
        units: dict[Future, tuple[str, str]] = {}
        for chromosome in order:
            run = _submit(fastvep_pool, pools.chromosome_threads, run_chromosome, prepared[chromosome], reference)
            units[run] = ('fastvep', chromosome)
        for chromosome in order:
            prepare = _submit(
                preparation_pool, pools.chromosome_threads, prepare_chromosome, prepared[chromosome], sources
            )
            units[prepare] = ('prepare', chromosome)
        chunks: dict[str, tuple[int, bool]] = {}
        ready: set[str] = set()
        pending = set(units)
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    kind, chromosome = units[future]
                    result = future.result()
                    if kind == 'chunk':
                        written.append(result)
                        continue
                    if kind == 'fastvep':
                        chunks[chromosome] = result
                    else:
                        ready.add(chromosome)
                    if chromosome not in chunks or chromosome not in ready:
                        continue
                    count, annotated = chunks[chromosome]
                    for chunk in range(count):
                        part = parts / f'part-{numbers[chromosome]:05d}-{chunk:05d}.parquet'
                        submitted = _submit(
                            chunk_pool,
                            pools.chunk_threads,
                            annotate_chunk,
                            prepared[chromosome],
                            chunk,
                            annotated,
                            part,
                        )
                        units[submitted] = ('chunk', chromosome)
                        pending.add(submitted)
                    logger.info('Annotating chromosome {} in {} chunks', chromosome, count)
        except BaseException:
            for pool in (fastvep_pool, preparation_pool, chunk_pool):
                pool.shutdown(cancel_futures=True)
            raise
    logger.info('Annotated {} chromosomes in {} parts', len(prepared), len(written))
    return sorted(written)
