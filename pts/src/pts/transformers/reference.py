"""Stream FASTA into a Polars Parquet dataset with one row per reference base."""

import re
from collections.abc import Iterable, Iterator
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError
from pydantic import BaseModel, ConfigDict, Field


class ReferenceSettings(BaseModel):
    """Bound expansion in memory and the number of rows per Parquet part."""

    model_config = ConfigDict(extra='forbid')
    batch_size: int = Field(default=5_000_000, gt=0)


def fasta_batches(handle: Iterable[str], batch_size: int) -> Iterator[pl.DataFrame]:
    """Yield uppercase bases with 1-based positions, retaining all FASTA contigs.

    Headers use their first whitespace-delimited token. Strip only a leading
    ``chr`` and canonicalize mitochondrial ``M`` to ``MT``. Positions continue
    across sequence lines and batches, and restart at each header.
    """
    chromosome = None
    position = 1
    chunks: list[str] = []
    size = 0
    seen: set[str] = set()

    def frame() -> pl.DataFrame:
        return pl.DataFrame({'ref': [''.join(chunks)]}).select(
            pl.lit(chromosome).alias('chr'),
            pl.int_range(position, position + size, dtype=pl.Int64).alias('pos'),
            pl.col('ref').str.split('').explode(),
        )

    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if chromosome is not None and position == 1 and size == 0:
                raise ValueError(f'FASTA contig {chromosome!r} has no bases')
            if chunks:
                yield frame()
            tokens = line[1:].split()
            if not tokens or not tokens[0].removeprefix('chr'):
                raise ValueError('FASTA header has no contig name')
            chromosome = tokens[0].removeprefix('chr')
            chromosome = 'MT' if chromosome == 'M' else chromosome
            if chromosome in seen:
                raise ValueError(f'Duplicate FASTA contig after normalization: {chromosome!r}')
            seen.add(chromosome)
            chunks, size, position = [], 0, 1
            continue
        if chromosome is None:
            raise ValueError('FASTA sequence encountered before its header')
        sequence = line.upper()
        if re.fullmatch('[ACGTRYSWKMBDHVN]+', sequence) is None:
            raise ValueError(f'Invalid FASTA sequence in contig {chromosome!r}')
        while sequence:
            length = min(batch_size - size, len(sequence))
            chunks.append(sequence[:length])
            size += length
            sequence = sequence[length:]
            if size == batch_size:
                yield frame()
                position += size
                chunks, size = [], 0
    if chunks:
        yield frame()
    if chromosome is not None and position == 1 and size == 0:
        raise ValueError(f'FASTA contig {chromosome!r} has no bases')


def reference(source: str, destination: str, settings: dict[str, Any], config: Config) -> None:
    """Expand FASTA incrementally without loading a genome or base table into RAM.

    Write flat, Zstd-compressed Parquet parts so ``scan_dataset`` can read the
    result locally or on GCS. Each part covers consecutive positions of one
    contig; Parquet statistics support chromosome/position predicate pruning.
    Refuse occupied destinations, as the shared dataset writer does. Parts are
    written individually because the FASTA parser produces bounded batches.
    """
    options = ReferenceSettings.model_validate(settings)
    try:
        StorageHandle(destination).stat()
    except NotFoundError:
        pass
    else:
        raise ValueError(f'{destination!r} already exists; reference never overwrites')

    total = 0
    parts = 0
    with StorageHandle(source).open('rt') as handle:
        for parts, batch in enumerate(fasta_batches(handle, options.batch_size), start=1):
            path = f'{destination.rstrip("/")}/part-{parts:06d}.parquet'
            with StorageHandle(path).open('wb') as output:
                batch.write_parquet(output, compression='zstd', statistics=True)
            total += batch.height
    if not total:
        raise ValueError('FASTA contains no reference bases')
    logger.info('Wrote {} reference bases across {} Parquet parts to {}', total, parts, destination)
