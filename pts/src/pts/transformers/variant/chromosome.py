"""The variant index of each chromosome, from fastVEP and the joins that complete its records.

Chromosomes run in a pool of processes, the largest first, each fastVEP with a few threads.
Every worker reads its inputs from paths and writes the parsed index of its chromosome to its
own scratch directory.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.vep import vep_schema
from pts.transformers.variant.alleles import read_chromosome
from pts.transformers.variant.colocated import KEY, colocated_variants
from pts.transformers.variant.fastvep import run_fastvep, write_vcf
from pts.transformers.variant.gerp import KentBigWig
from pts.transformers.variant.loftee import add_loftee, loftee_models
from pts.transformers.variant.predictions import add_gerp_and_alphamissense
from pts.transformers.variant.records import fastvep_records, unannotated_records
from pts.transformers.variant.sift import add_sift
from pts.transformers.variant.transcripts import transcript_fields
from pts.transformers.variant.vep import parse_vep


@dataclass(frozen=True)
class Shared:
    """The inputs every chromosome reads, prepared once before the pool."""

    fasta: str
    """The gzip reference FASTA."""
    plain_fasta: Path
    """The same reference uncompressed, which fastVEP reads."""
    gff3: Path
    """The GFF3 whose fastVEP cache `build_cache` built from `plain_fasta`."""
    gerp: str
    """The local GERP bigWig."""
    ancestor: str
    """The local bgzip human ancestor FASTA, with its `.fai` and `.gzi`."""
    core: str
    """Directory of the Ensembl core database dump."""
    sift: Path
    """`sift_matrices`, as Parquet."""
    binary: str
    threads: int
    """fastVEP threads, and polars threads, of each worker."""
    hash_threshold: int
    directory: Path
    """Scratch, with a directory per chromosome."""


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
    """The chromosome's reference, empty for a contig the FASTA lacks."""
    try:
        return read_chromosome(fasta, chromosome)
    except ValueError:
        return b''


def _fastvep_records(frame: pl.DataFrame, sequence: bytes, chromosome: str, shared: Shared) -> pl.DataFrame:
    """The VEP records of the chromosome's rows, from fastVEP."""
    directory = shared.directory / chromosome
    vcf = directory / 'rows.vcf'
    write_vcf(frame, vcf, sequence)
    logger.info('Running fastVEP on {} variants of chromosome {}', frame.height, chromosome)
    output = run_fastvep(
        vcf, shared.gff3, shared.plain_fasta, directory / 'fastvep.json', shared.threads, shared.binary
    )
    vcf.unlink()
    try:
        transcripts = transcript_fields(shared.core, chromosome)
        return fastvep_records(output, frame, transcripts, sequence, chromosome).collect()
    finally:
        output.unlink()


def with_colocated(records: pl.DataFrame, rows: pl.DataFrame, all_vars: Path | None) -> pl.DataFrame:
    """The records with the colocated variants of their row, `[]` for none or without an `all_vars`.

    Records are matched to rows on the VCF fields of their `input` line.
    """
    if all_vars is None:
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
        .join(colocated_variants(all_vars, rows), on=KEY, how='left', maintain_order='left')
        .with_columns(pl.col('colocated_variants').fill_null(pl.lit([], dtype=vep_schema['colocated_variants'])))
        .select(records.columns)
    )


def annotate_chromosome(chromosome: str, rows: Path, all_vars: Path | None, alphamissense: str, shared: Shared) -> Path:
    """The variant index of one chromosome's rows, parsed from fastVEP and written to Parquet.

    Args:
        chromosome: The chromosome of the rows.
        rows: Parquet of the chromosome's rows, as `collect_variants` orders them.
        all_vars: The chromosome's `all_vars.gz`, None when the cache has none.
        alphamissense: The chromosome's `partition_alphamissense` part.
        shared: The inputs every chromosome reads.

    Returns:
        The path of the chromosome's index, in its directory under `shared.directory`.

    A chromosome the reference FASTA lacks is not run through fastVEP: its rows are intergenic,
    without HGVS.

    Raises:
        RuntimeError: if fastVEP fails or its output is truncated; no index is written.
    """
    frame = pl.read_parquet(rows)
    sequence = _sequence(shared.fasta, chromosome)
    directory = shared.directory / chromosome
    directory.mkdir(parents=True)
    if sequence:
        records = _fastvep_records(frame, sequence, chromosome, shared)
    else:
        logger.warning(
            'No reference sequence for chromosome {}: {} variants left without transcript consequences',
            chromosome,
            frame.height,
        )
        records = unannotated_records(frame, chromosome).collect()
    records = with_colocated(records, frame, all_vars)
    filled = add_gerp_and_alphamissense(records.lazy(), shared.gerp, alphamissense, chromosome)
    with KentBigWig(shared.gerp) as gerp:
        models = loftee_models(shared.core, chromosome)
        filled = add_loftee(filled, models, sequence, shared.ancestor, gerp, chromosome)
    filled = add_sift(filled, pl.read_parquet(shared.sift))
    index = directory / 'index.parquet'
    parse_vep(filled, shared.hash_threshold).collect().write_parquet(index)
    logger.info('Parsed {} variants of chromosome {}', frame.height, chromosome)
    return index


def annotate_chromosomes(
    rows: dict[str, Path],
    sizes: dict[str, int],
    all_vars: dict[str, Path],
    alphamissense: dict[str, str],
    shared: Shared,
    processes: int,
) -> dict[str, Path]:
    """`annotate_chromosome` of every chromosome, `processes` at a time, the most rows first.

    Raises:
        RuntimeError: as soon as a chromosome fails; chromosomes not started are cancelled.
    """
    order = sorted(rows, key=lambda chromosome: sizes[chromosome], reverse=True)
    indexes: dict[str, Path] = {}
    with (
        _environment('POLARS_MAX_THREADS', str(shared.threads)),
        ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context('spawn')) as pool,
    ):
        futures = {
            pool.submit(
                annotate_chromosome,
                chromosome,
                rows[chromosome],
                all_vars.get(chromosome),
                alphamissense[chromosome],
                shared,
            ): chromosome
            for chromosome in order
        }
        try:
            for future in as_completed(futures):
                indexes[futures[future]] = future.result()
                logger.info('Annotated {} of {} chromosomes', len(indexes), len(futures))
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    return indexes
