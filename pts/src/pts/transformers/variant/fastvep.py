"""Run fastVEP on one chromosome."""

import gzip
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.variant.alleles import vep_allele_form

FLAGS = [
    '--pick-allele-gene',
    '--pick-order',
    'mane_select,canonical',
    '--distance',
    '500000',
    '--symbol',
    '--hgvs',
    '--canonical',
    '--output-format',
    'json',
    '--no-progress',
]

VCF_HEADER = '##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
_TRANSCRIPT_ID = re.compile(r'(?:^|;)ID=(transcript:[^;]+)')
_PARENT = re.compile(r'(?:^|;)Parent=([^;]+)')
_TAG = re.compile(r'(?:^|;)tag=([^;]+)')
_BIOTYPE = re.compile(r'(?:^|;)biotype=([^;]+)')
DROPPED_BIOTYPES = ('artifact',)
"""Transcript biotypes left out of the GFF3 fastVEP reads and of the core-side transcripts."""


def cache_path(gff3: Path) -> Path:
    """The fastVEP transcript cache of a GFF3, next to it."""
    return gff3.with_name(f'{gff3.name}.fastvep-cache')


def _partial(path: Path) -> Path:
    return path.with_name(f'{path.name}.partial')


def prepare_gff3(gff3: str, directory: Path) -> Path:
    """Copy a GFF3 into `directory` without its `readthrough_transcript` and `artifact` transcripts.

    Each such transcript is dropped with every line whose parent it is (exons, CDS, UTRs); genes are kept.
    The source may be gzip-compressed; the copy is plain text, named as the source without `.gz`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / Path(gff3).name.removesuffix('.gz')
    copy = _partial(target)
    with StorageHandle(gff3).open('rb') as raw, copy.open('wb') as out:
        if gff3.endswith('.gz'):
            with gzip.open(raw, 'rb') as source:
                shutil.copyfileobj(source, out)
        else:
            shutil.copyfileobj(raw, out)
    dropped: set[str] = set()
    with copy.open() as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            attributes = line.rstrip('\n').rsplit('\t', 1)[-1]
            transcript = _TRANSCRIPT_ID.search(attributes)
            if transcript is None:
                continue
            tag, biotype = _TAG.search(attributes), _BIOTYPE.search(attributes)
            if (tag and 'readthrough_transcript' in tag[1].split(',')) or (biotype and biotype[1] in DROPPED_BIOTYPES):
                dropped.add(transcript[1])
    with copy.open() as handle, target.open('w') as out:
        for line in handle:
            if not line.startswith('#'):
                attributes = line.rstrip('\n').rsplit('\t', 1)[-1]
                transcript, parent = _TRANSCRIPT_ID.search(attributes), _PARENT.search(attributes)
                if (transcript and transcript[1] in dropped) or (parent and dropped & set(parent[1].split(','))):
                    continue
            out.write(line)
    copy.unlink()
    return target


def write_vcf(rows: pl.DataFrame, path: Path, sequence: bytes) -> None:
    """Write rows as a VCF 4.2 in VEP's input allele form, sorted by chromosome, position and alleles.

    `ID` is the row's 0-based ordinal in `rows`. An allele left empty by the trim is padded with
    the reference base before it, from the chromosome `sequence` (byte i being position i + 1);
    without that base, as for a contig the FASTA lacks, the row is written as given.
    """
    frame = vep_allele_form(
        rows.lazy().select('chromosome', 'position', 'referenceAllele', 'alternateAllele').with_row_index('id')
    ).collect()
    indel = (pl.col('vepReference') == '-') | (pl.col('vepAlternate') == '-')
    padded = indel & (pl.col('vepStart') > 1) & (pl.col('vepStart') - 1 <= len(sequence))
    bases = np.frombuffer(sequence or b'N', dtype=np.uint8)
    pad_index = frame.select(pl.when(padded).then(pl.col('vepStart') - 2).otherwise(0)).to_series().to_numpy()
    pads = pl.Series('_pad', list(bases[pad_index].tobytes().decode('ascii')), dtype=pl.String)
    frame = frame.with_columns(pads).with_columns(
        pl
        .when(padded)
        .then(pl.col('vepStart') - 1)
        .when(~indel)
        .then(pl.col('vepStart'))
        .otherwise(pl.col('position'))
        .cast(pl.Int32)
        .alias('position'),
        pl
        .when(padded)
        .then(pl.col('_pad') + pl.col('vepReference').str.replace('-', '', literal=True))
        .when(~indel)
        .then(pl.col('vepReference'))
        .otherwise(pl.col('referenceAllele'))
        .alias('referenceAllele'),
        pl
        .when(padded)
        .then(pl.col('_pad') + pl.col('vepAlternate').str.replace('-', '', literal=True))
        .when(~indel)
        .then(pl.col('vepAlternate'))
        .otherwise(pl.col('alternateAllele'))
        .alias('alternateAllele'),
    )
    lines = frame.sort('chromosome', 'position', 'referenceAllele', 'alternateAllele').select(
        pl.concat_str(
            'chromosome',
            'position',
            'id',
            'referenceAllele',
            'alternateAllele',
            pl.lit('.\t.\t.'),
            separator='\t',
        )
    )
    with path.open('w') as out:
        out.write(VCF_HEADER)
        for (line,) in lines.iter_rows():
            out.write(f'{line}\n')


def _run(command: list[str], output: Path, env: dict[str, str] | None = None) -> Path:
    """Run fastVEP writing to a partial file next to `output`, removed if it exits with an error."""
    partial = _partial(output)
    partial.unlink(missing_ok=True)
    result = subprocess.run(
        [*command, '-o', str(partial)], check=False, capture_output=True, text=True, errors='replace', env=env
    )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'{command[0]} {command[1]} exited with {result.returncode}: {result.stderr[-2000:]}')
    return partial


def build_cache(gff3: Path, fasta: Path, binary: str = 'fastvep') -> None:
    """Build the fastVEP transcript cache of a GFF3 at `cache_path(gff3)`.

    The cache holds the transcript sequences from `fasta`, the FASTA the annotation runs use:
    they are added by annotating an empty VCF.

    Raises:
        RuntimeError: if fastVEP fails; no cache is written.
    """
    cache = cache_path(gff3)
    partial = _run([binary, 'cache', '--gff3', str(gff3), '--no-progress'], cache)
    empty = gff3.with_name(f'{gff3.name}.empty.vcf')
    empty.write_text(VCF_HEADER)
    try:
        command = [binary, 'annotate', '--input', str(empty), '--fasta', str(fasta), '--transcript-cache', str(partial)]
        _run([*command, *FLAGS], empty.with_suffix('.json')).unlink()
    except RuntimeError:
        partial.unlink(missing_ok=True)
        raise
    finally:
        empty.unlink()
    partial.rename(cache)


def _last_byte(path: Path) -> bytes:
    """The last non-whitespace byte of a file, empty if there is none."""
    with path.open('rb') as handle:
        end = handle.seek(0, os.SEEK_END)
        while end > 0:
            start = max(0, end - 4096)
            handle.seek(start)
            chunk = handle.read(end - start).rstrip()
            if chunk:
                return chunk[-1:]
            end = start
    return b''


def run_fastvep(vcf: Path, gff3: Path, fasta: Path, output: Path, threads: int, binary: str = 'fastvep') -> Path:
    """Annotate a VCF with fastVEP into a JSON array at `output`, using the cache `build_cache` built.

    Raises:
        FileNotFoundError: if the GFF3 has no cache.
        RuntimeError: if fastVEP fails or its JSON array is not closed; `output` is not written.
    """
    cache = cache_path(gff3)
    if not cache.exists():
        raise FileNotFoundError(f'{cache} not found: run build_cache first')
    command = [
        binary,
        'annotate',
        '--input',
        str(vcf),
        '--fasta',
        str(fasta),
        '--transcript-cache',
        str(cache),
        *FLAGS,
    ]
    partial = _run(command, output, env={**os.environ, 'RAYON_NUM_THREADS': str(threads)})
    if _last_byte(partial) != b']':
        partial.unlink()
        raise RuntimeError(f'fastVEP output for {vcf} is truncated: the JSON array is not closed')
    return partial.rename(output)
