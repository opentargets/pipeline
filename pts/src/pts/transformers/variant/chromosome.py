"""The variant index of each chromosome, from fastVEP, the joins that complete its records and its annotations.

Each chromosome is run through fastVEP in a pool of processes, the largest first, each fastVEP
with a few threads; its records are split by row into chunks. Alongside, a second pool prepares
what every chunk of a chromosome shares: its transcript fields, LOFTEE models, ancestor contig,
colocated variants, supplementary annotations and GPN-Star scores, written in position order, and
the amino-acid annotations of its transcripts' proteins.
Once a chromosome's records and preparations are ready, a third pool annotates its chunks, each
into one part of the index, reading only the prepared rows in the chunk's position range.
Workers read their inputs from paths and write into the scratch directory.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import shutil
from collections.abc import Callable, Generator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from pts.schemas.variant import variant_schema
from pts.schemas.vep import vep_schema
from pts.transformers.variant.alleles import read_chromosome, read_contig
from pts.transformers.variant.colocated import KEY, colocated_variants
from pts.transformers.variant.fastvep import run_fastvep, write_vcf
from pts.transformers.variant.gerp import KentBigWig
from pts.transformers.variant.gpn_star import GpnStarCutoffs, add_gpn_star, read_gpn_star
from pts.transformers.variant.index import (
    add_annotation,
    amino_acid_keys,
    annotate_with_amino_acid_consequences,
    read_variant_annotations,
)
from pts.transformers.variant.loftee import add_loftee, loftee_models
from pts.transformers.variant.predictions import add_gerp_and_alphamissense
from pts.transformers.variant.records import CHUNK_BYTES, iter_records, records_from_fastvep, unannotated_records
from pts.transformers.variant.sift import add_sift
from pts.transformers.variant.transcripts import transcript_fields
from pts.transformers.variant.vep import parse_vep, uniprot_accessions


@dataclass(frozen=True)
class Shared:
    """The inputs every chromosome reads, prepared once before the pools."""

    plain_fasta: Path
    """The uncompressed reference, which fastVEP and the chunks read."""
    contigs: dict[str, tuple[int, int]]
    """`contig_spans` of `plain_fasta`."""
    gff3: Path
    """The GFF3 whose fastVEP cache `build_cache` built from `plain_fasta`."""
    gerp: str
    """The local GERP bigWig."""
    ancestor: str
    """The local gzip human ancestor FASTA."""
    core: str
    """Directory of the local Ensembl core database dump."""
    sift: Path
    """`sift_matrices`, as Parquet."""
    binary: str
    threads: int
    """Threads of each fastVEP run."""
    hash_threshold: int
    directory: Path
    """Scratch, with a directory per chromosome and one for the parts."""
    chunk_rows: int = 50_000
    """Rows of each chunk of a chromosome."""
    batch_size: int = 5_000
    """Variants per batch when merging the annotations into a chunk."""
    row_group_rows: int = 10_000
    """Rows of each row group of the prepared frames, the unit a chunk's position range reads."""
    variant_annotations: tuple[dict[str, list[str]], ...] = ()
    """The Parquet parts of each supplementary variant annotation source, by chromosome."""
    amino_acid_annotations: tuple[Path, ...] = ()
    """The local `amino_acid_projection` of each amino-acid annotation source (FoldX)."""
    gpn_star: dict[str, str] = field(default_factory=dict)
    """The GPN-Star file of each chromosome that has one."""
    gpn_star_cutoffs: GpnStarCutoffs | None = None


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


def _sequence(fasta: str, chromosome: str) -> bytes:
    """The chromosome's contig of a gzip FASTA, empty for a contig the FASTA lacks."""
    try:
        return read_chromosome(fasta, chromosome)
    except ValueError:
        return b''


def _reference(shared: Shared, chromosome: str) -> bytes:
    """The chromosome's contig of the uncompressed reference, empty for a contig it lacks."""
    span = shared.contigs.get(chromosome)
    return b'' if span is None else read_contig(shared.plain_fasta, span)


def _records(shared: Shared, chromosome: str, chunk: int) -> Path:
    """The directory of one chunk's fastVEP records."""
    return shared.directory / chromosome / 'records' / f'{chunk:05d}'


def _prepared(shared: Shared, chromosome: str) -> Path:
    """The directory of what `prepare_chromosome` writes."""
    return shared.directory / chromosome / 'prepared'


def _write(frame: pl.DataFrame, path: Path, row_group_rows: int | None = None) -> Path:
    """Write Parquet through a partial file, so that a file at `path` is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f'{path.name}.partial')
    frame.write_parquet(partial, compression='zstd', row_group_size=row_group_rows)
    return partial.rename(path)


def _by_position(frame: pl.DataFrame, path: Path, shared: Shared) -> Path:
    """Write a prepared frame sorted by `position`, in row groups whose statistics a position range can skip."""
    return _write(frame.sort('position', maintain_order=True), path, shared.row_group_rows)


def in_range(path: Path, positions: pl.Series) -> pl.LazyFrame:
    """The rows of a Parquet file with `position` between the least and the greatest of `positions`."""
    return pl.scan_parquet(path).filter(pl.col('position').is_between(positions.min(), positions.max()))


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


def split_records(output: Path, rows: int, shared: Shared, chromosome: str, chunk_bytes: int = CHUNK_BYTES) -> None:
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
        chunks = frame.with_columns((pl.col('row') // shared.chunk_rows).alias('_chunk'))
        for (chunk,), part in chunks.partition_by('_chunk', as_dict=True).items():
            _write(part.drop('_chunk'), _records(shared, chromosome, int(chunk)) / f'{number:05d}.parquet')
    if count != rows:
        raise ValueError(f'{output}: {count} fastVEP records for {rows} rows')


def run_chromosome(chromosome: str, rows: Path, shared: Shared) -> tuple[int, bool]:
    """Run fastVEP on one chromosome's rows and split its records into chunks of `shared.chunk_rows` rows.

    A chromosome the reference lacks is not run through fastVEP: its chunks are intergenic,
    without HGVS.

    Returns:
        The number of chunks, and whether fastVEP ran.

    Raises:
        RuntimeError: if fastVEP fails or its output is truncated.
        ValueError: if fastVEP's record ids are not the row ordinals.
    """
    frame = pl.read_parquet(rows)
    sequence = _reference(shared, chromosome)
    chunks = -(-frame.height // shared.chunk_rows)
    if not sequence:
        logger.warning(
            'No reference sequence for chromosome {}: {} variants left without transcript consequences',
            chromosome,
            frame.height,
        )
        return chunks, False
    directory = shared.directory / chromosome / 'fastvep'
    directory.mkdir(parents=True)
    vcf = directory / 'rows.vcf'
    write_vcf(frame, vcf, sequence)
    logger.info('Running fastVEP on {} variants of chromosome {}', frame.height, chromosome)
    output = run_fastvep(
        vcf, shared.gff3, shared.plain_fasta, directory / 'fastvep.json', shared.threads, shared.binary
    )
    vcf.unlink()
    try:
        split_records(output, frame.height, shared, chromosome)
    finally:
        output.unlink()
    return chunks, True


AMINO_ACID_COLUMNS = ('uniprotAccession', 'aminoAcidChange', 'variantEffect')


def amino_acid_annotations(path: Path, transcripts: pl.DataFrame) -> pl.DataFrame:
    """The rows of an amino-acid annotation file on a UniProt accession of one of `transcripts`."""
    accessions = (
        transcripts
        .select(uniprot_accessions(pl.col('swissprot'), pl.col('trembl')).alias('uniprotAccession'))
        .explode('uniprotAccession')
        .drop_nulls()
        .unique()
    )
    return (
        pl
        .scan_parquet(path)
        .select(AMINO_ACID_COLUMNS)
        .join(accessions.lazy(), on='uniprotAccession', how='semi', maintain_order='left')
        .collect()
    )


def prepare_chromosome(chromosome: str, rows: Path, all_vars: Path | None, shared: Shared) -> None:
    """Write what every chunk of the chromosome reads, once, into its `prepared` directory.

    That is its transcript fields, LOFTEE models and ancestor contig, its rows' colocated
    variants, the supplementary annotations and GPN-Star scores at its rows' loci, the last
    three sorted by position, and the amino-acid annotations of its transcripts' proteins.

    Args:
        chromosome: The chromosome of the rows.
        rows: Parquet of the chromosome's rows.
        all_vars: The chromosome's `all_vars.gz`, None when the cache has none.
        shared: The inputs every chromosome reads.
    """
    frame = pl.read_parquet(rows)
    directory = _prepared(shared, chromosome)
    directory.mkdir(parents=True)
    transcripts = transcript_fields(shared.core, chromosome)
    _write(transcripts, directory / 'transcripts.parquet')
    (directory / 'loftee.pickle').write_bytes(pickle.dumps(loftee_models(shared.core, chromosome)))
    (directory / 'ancestor.bin').write_bytes(_sequence(shared.ancestor, chromosome))
    if all_vars is not None:
        _by_position(colocated_variants(all_vars, frame), directory / 'colocated.parquet', shared)
    loci = frame.select('chromosome', 'position').unique().lazy()
    for number, located in enumerate(shared.variant_annotations):
        annotations = (
            read_variant_annotations(located[chromosome], shared.hash_threshold, loci)
            .filter(pl.col('chromosome') == chromosome)
            .collect(engine='streaming')
            if chromosome in located
            else pl.DataFrame(schema=variant_schema)
        )
        _by_position(annotations, directory / f'annotations-{number:03d}.parquet', shared)
    if chromosome in shared.gpn_star and shared.gpn_star_cutoffs is not None:
        scores = read_gpn_star(shared.gpn_star[chromosome], frame['position'].unique(), shared.gpn_star_cutoffs)
        _by_position(scores, directory / 'gpn_star.parquet', shared)
    for number, path in enumerate(shared.amino_acid_annotations):
        _write(amino_acid_annotations(path, transcripts), directory / f'amino-acids-{number:03d}.parquet')
    logger.info('Prepared chromosome {}', chromosome)


def with_colocated(records: pl.DataFrame, rows: pl.DataFrame, colocated: pl.DataFrame | None) -> pl.DataFrame:
    """The records with the colocated variants of their row, `[]` for none or without `colocated`.

    Records are matched to the `colocated_variants` rows on the VCF fields of their `input` line.
    """
    if colocated is None:
        return records
    fields = pl.col('input').str.split('\t')
    return (
        records
        .drop('colocated_variants')
        .with_columns(
            fields.list.get(0).alias('chromosome'),
            fields.list.get(1).cast(rows.schema['position']).alias('position'),
            fields.list.get(3).alias('referenceAllele'),
            fields.list.get(4).alias('alternateAllele'),
        )
        .join(colocated, on=KEY, how='left', maintain_order='left')
        .with_columns(pl.col('colocated_variants').fill_null(pl.lit([], dtype=vep_schema['colocated_variants'])))
        .select(records.columns)
    )


def _merged(index: pl.DataFrame, shared: Shared, chromosome: str) -> pl.DataFrame:
    """The index rows of a chunk with the prepared annotations of its loci merged in, by `annotate`."""
    directory = _prepared(shared, chromosome)
    loci = index.select('chromosome', 'position').unique()
    variant_annotations = [
        in_range(directory / f'annotations-{number:03d}.parquet', loci['position'])
        .join(loci.lazy(), on=['chromosome', 'position'], how='semi')
        .collect()
        for number in range(len(shared.variant_annotations))
    ]
    scores = directory / 'gpn_star.parquet'
    gpn_star = (
        in_range(scores, loci['position']).filter(pl.col('position').is_in(loci['position'].implode())).collect()
        if scores.exists()
        else None
    )
    keys = amino_acid_keys(index.lazy()).collect().lazy()
    amino_acid_annotations = [
        pl
        .scan_parquet(directory / f'amino-acids-{number:03d}.parquet')
        .join(keys, on=['uniprotAccession', 'aminoAcidChange'], how='semi')
        .collect()
        for number in range(len(shared.amino_acid_annotations))
    ]
    batches = [
        annotate(batch, variant_annotations, amino_acid_annotations, gpn_star)
        for batch in index.iter_slices(shared.batch_size)
    ]
    return pl.concat(batches).sort('position', maintain_order=True).select(variant_schema.names())


def annotate_chunk(
    chromosome: str, chunk: int, rows: Path, annotated: bool, alphamissense: str, shared: Shared, part: Path
) -> Path:
    """The annotated variant index of one chunk of a chromosome's rows, written to `part`.

    Args:
        chromosome: The chromosome of the rows.
        chunk: The chunk: `shared.chunk_rows` rows from row `chunk * shared.chunk_rows`.
        rows: Parquet of the chromosome's rows, sorted by position.
        annotated: Whether `run_chromosome` ran fastVEP on the chromosome.
        alphamissense: The chromosome's `partition_alphamissense` part.
        shared: The inputs every chromosome reads.
        part: Where to write the chunk's index.

    Returns:
        `part`, which is only written once the whole chunk is annotated. The chunk's fastVEP
        records are deleted once read.
    """
    offset = chunk * shared.chunk_rows
    frame = pl.scan_parquet(rows).slice(offset, shared.chunk_rows).collect()
    sequence = _reference(shared, chromosome)
    directory = _prepared(shared, chromosome)
    if annotated:
        fastvep = (
            pl
            .scan_parquet(_records(shared, chromosome, chunk) / '*.parquet')
            .with_columns(pl.col('row') - offset)
            .collect()
        )
        shutil.rmtree(_records(shared, chromosome, chunk))
        transcripts = pl.read_parquet(directory / 'transcripts.parquet')
        records = records_from_fastvep(fastvep, frame, transcripts, sequence, chromosome).collect()
        del fastvep
    else:
        records = unannotated_records(frame, chromosome).collect()
    located = directory / 'colocated.parquet'
    colocated = (
        in_range(located, frame['position']).join(frame.lazy().select(KEY), on=KEY, how='semi').collect()
        if located.exists()
        else None
    )
    records = with_colocated(records, frame, colocated)
    filled = add_gerp_and_alphamissense(records.lazy(), shared.gerp, alphamissense, chromosome)
    models = pickle.loads((directory / 'loftee.pickle').read_bytes())
    ancestor = (directory / 'ancestor.bin').read_bytes()
    with KentBigWig(shared.gerp) as gerp:
        filled = add_loftee(filled, models, sequence, ancestor, gerp, chromosome).collect()
    transcript_ids = filled.select(
        pl.col('transcript_consequences').explode().struct.field('transcript_id').drop_nulls().unique()
    )
    matrices = pl.scan_parquet(shared.sift).join(transcript_ids.lazy(), on='transcript_id', how='semi').collect()
    index = parse_vep(add_sift(filled.lazy(), matrices), shared.hash_threshold).collect()
    logger.info('Annotating {} variants of chromosome {}, chunk {}', index.height, chromosome, chunk)
    return _write(_merged(index, shared, chromosome), part)


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
    rows: dict[str, Path],
    sizes: dict[str, int],
    all_vars: dict[str, Path],
    alphamissense: dict[str, str],
    shared: Shared,
    pools: Pools,
) -> list[Path]:
    """The annotated index of every chromosome, in chunks, written under `shared.directory / 'parts'`.

    fastVEP and the preparations each run in their own pool, the chromosomes with the most rows
    first. A chromosome's chunks go to the third pool once its records and its preparation are
    ready. Part `part-<c>-<k>.parquet` holds chunk k of the c-th chromosome in name order.

    Returns:
        The paths of the parts, in name order.

    Raises:
        RuntimeError: as soon as a chromosome or a chunk fails; units not started are cancelled.
    """
    order = sorted(rows, key=lambda chromosome: sizes[chromosome], reverse=True)
    numbers = {chromosome: number for number, chromosome in enumerate(sorted(rows))}
    context = multiprocessing.get_context('spawn')
    parts: list[Path] = []
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
            run = _submit(fastvep_pool, pools.chromosome_threads, run_chromosome, chromosome, rows[chromosome], shared)
            units[run] = ('fastvep', chromosome)
        for chromosome in order:
            prepare = _submit(
                preparation_pool,
                pools.chromosome_threads,
                prepare_chromosome,
                chromosome,
                rows[chromosome],
                all_vars.get(chromosome),
                shared,
            )
            units[prepare] = ('prepare', chromosome)
        chunks: dict[str, tuple[int, bool]] = {}
        prepared: set[str] = set()
        pending = set(units)
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    kind, chromosome = units[future]
                    result = future.result()
                    if kind == 'chunk':
                        parts.append(result)
                        continue
                    if kind == 'fastvep':
                        chunks[chromosome] = result
                    else:
                        prepared.add(chromosome)
                    if chromosome not in chunks or chromosome not in prepared:
                        continue
                    count, annotated = chunks[chromosome]
                    for chunk in range(count):
                        part = shared.directory / 'parts' / f'part-{numbers[chromosome]:05d}-{chunk:05d}.parquet'
                        submitted = _submit(
                            chunk_pool,
                            pools.chunk_threads,
                            annotate_chunk,
                            chromosome,
                            chunk,
                            rows[chromosome],
                            annotated,
                            alphamissense[chromosome],
                            shared,
                            part,
                        )
                        units[submitted] = ('chunk', chromosome)
                        pending.add(submitted)
                    logger.info('Annotating chromosome {} in {} chunks', chromosome, count)
        except BaseException:
            for pool in (fastvep_pool, preparation_pool, chunk_pool):
                pool.shutdown(cancel_futures=True)
            raise
    logger.info('Annotated {} chromosomes in {} parts', len(rows), len(parts))
    return sorted(parts)
