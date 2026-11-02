"""Reference sequence, VEP's input allele form and genomic HGVS."""

import gzip
import mmap
from pathlib import Path

import numpy as np
import polars as pl
from otter.storage.synchronous.handle import StorageHandle

_COMPLEMENT = bytes.maketrans(b'ACGTN', b'TGCAN')


def read_chromosome(fasta: str, chromosome: str) -> bytes:
    """One contig of a gzip FASTA, in upper case.

    The file is streamed and reading stops at the end of the contig.

    Raises:
        ValueError: if the FASTA has no contig of that name.
    """
    lines: list[bytes] | None = None
    with StorageHandle(fasta).open('rb') as raw, gzip.open(raw, 'rb') as handle:
        for line in handle:
            if line.startswith(b'>'):
                if lines is not None:
                    break
                if line[1:].split(maxsplit=1)[0].decode() == chromosome:
                    lines = []
            elif lines is not None:
                lines.append(line.rstrip())
    if lines is None:
        raise ValueError(f'contig {chromosome} not found in {fasta}')
    return b''.join(lines).upper()


def contig_spans(fasta: Path) -> dict[str, tuple[int, int]]:
    """The byte span of each contig's sequence lines in an uncompressed FASTA, the first contig of a name kept."""
    spans: dict[str, tuple[int, int]] = {}
    if fasta.stat().st_size == 0:
        return spans
    with fasta.open('rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
        start = 0 if buffer[:1] == b'>' else buffer.find(b'\n>') + 1
        while 0 <= start < len(buffer) and buffer[start : start + 1] == b'>':
            header_end = buffer.find(b'\n', start)
            header_end = len(buffer) if header_end == -1 else header_end
            following = buffer.find(b'\n>', header_end)
            end = len(buffer) if following == -1 else following + 1
            name = buffer[start + 1 : header_end].split(maxsplit=1)
            if name:
                spans.setdefault(name[0].decode(), (min(header_end + 1, end), end))
            start = end
    return spans


def read_contig(fasta: Path, span: tuple[int, int]) -> bytes:
    """The sequence of one `contig_spans` span of an uncompressed FASTA, in upper case."""
    start, end = span
    with fasta.open('rb') as handle:
        handle.seek(start)
        return handle.read(end - start).translate(None, b' \t\r\n').upper()


def _fai_row(buffer: mmap.mmap, name: str, span: tuple[int, int]) -> str | None:
    """The faidx row of one contig; None if its lines are uneven."""
    start, end = span
    sequence = np.frombuffer(buffer, dtype=np.uint8, count=end - start, offset=start)
    if sequence.size == 0:
        return f'{name}\t0\t{start}\t0\t0'
    ends = np.flatnonzero(sequence == ord('\n')) + 1
    if ends.size == 0 or ends[-1] != sequence.size:
        ends = np.append(ends, sequence.size)
    widths = np.diff(ends, prepend=0)
    newline = sequence[ends - 1] == ord('\n')
    carriage = newline & (widths > 1) & (sequence[np.maximum(ends - 2, 0)] == ord('\r'))
    bases = widths - newline - carriage
    width, line_bases, last = int(widths[0]), int(bases[0]), int(bases[-1])
    if (widths[:-1] != width).any() or (bases[:-1] != line_bases).any() or not 0 < last <= line_bases:
        return None
    return f'{name}\t{line_bases * (ends.size - 1) + last}\t{start}\t{line_bases}\t{width}'


def write_fai(fasta: Path, spans: dict[str, tuple[int, int]]) -> Path:
    """Write the samtools faidx index of an uncompressed FASTA next to it, at `<fasta>.fai`.

    `spans` are the FASTA's `contig_spans`; each gets one row of name, length, offset, line
    bases and line width, in file order. The index is written to a temporary file, then renamed.

    Raises:
        ValueError: if a contig's lines, other than its last, differ in length, or its last
            line is longer than the others; no index is written.
    """
    rows = []
    if spans:
        with fasta.open('rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
            rows = [_fai_row(buffer, name, span) for name, span in spans.items()]
    for name, row in zip(spans, rows, strict=True):
        if row is None:
            raise ValueError(f'contig {name} of {fasta} has uneven line lengths: faidx cannot index it')
    fai = fasta.with_name(f'{fasta.name}.fai')
    partial = fai.with_name(f'{fai.name}.partial')
    partial.write_text(''.join(f'{row}\n' for row in rows))
    return partial.replace(fai)


def shared_prefix_length(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    """The number of leading characters two strings have in common."""
    length = pl.min_horizontal(left.str.len_chars(), right.str.len_chars())
    half = pl.len() // 2
    return (
        pl
        .concat_list(left.str.split('').list.head(length), right.str.split('').list.head(length))
        .list.eval((pl.element().head(half) == pl.element().tail(half)).cast(pl.UInt32).cum_min().sum())
        .list.first()
        .cast(pl.Int64)
    )


def vep_allele_form(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Add the coordinates and alleles VEP derives from a VCF row.

    The shared prefix, then the shared suffix, of the reference and alternate alleles is
    trimmed, except when both alleles have the same length greater than one. The start is
    the position plus the prefix length, the end is the start plus the trimmed reference
    length minus one (an insertion ends before it starts), and an empty allele is `-`.

    Adds `vepStart`, `vepEnd`, `vepReference` and `vepAlternate`.
    """
    ref, alt = pl.col('referenceAllele'), pl.col('alternateAllele')
    untrimmed = (ref.str.len_chars() == alt.str.len_chars()) & (ref.str.len_chars() > 1)
    ref_rest, alt_rest = ref.str.slice(pl.col('_prefix')), alt.str.slice(pl.col('_prefix'))
    return (
        frame
        .with_columns(pl.when(untrimmed).then(0).otherwise(shared_prefix_length(ref, alt)).alias('_prefix'))
        .with_columns(
            pl
            .when(untrimmed)
            .then(0)
            .otherwise(shared_prefix_length(ref_rest.str.reverse(), alt_rest.str.reverse()))
            .alias('_suffix'),
        )
        .with_columns(
            ref_rest.str.head(ref_rest.str.len_chars() - pl.col('_suffix')).alias('_ref'),
            alt_rest.str.head(alt_rest.str.len_chars() - pl.col('_suffix')).alias('_alt'),
            (pl.col('position').cast(pl.Int64) + pl.col('_prefix')).alias('vepStart'),
        )
        .with_columns(
            (pl.col('vepStart') + pl.col('_ref').str.len_chars().cast(pl.Int64) - 1).alias('vepEnd'),
            pl.when(pl.col('_ref') == '').then(pl.lit('-')).otherwise('_ref').alias('vepReference'),
            pl.when(pl.col('_alt') == '').then(pl.lit('-')).otherwise('_alt').alias('vepAlternate'),
        )
        .drop('_prefix', '_suffix', '_ref', '_alt')
    )


def _trim(position: int, ref: str, alt: str) -> tuple[int, str, str]:
    while ref and alt and ref[0] == alt[0]:
        ref, alt, position = ref[1:], alt[1:], position + 1
    while ref and alt and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    return position, ref, alt


def hgvs_genomic(sequence: bytes, chromosome: str, position: int, reference: str, alternate: str) -> str | None:
    """The genomic HGVS of a VCF row as VEP writes it, built from the chromosome sequence.

    `sequence` is the whole chromosome, byte i being position i + 1. Deletions and insertions
    are shifted to the 3' end and an insertion of the preceding bases is a `dup`. A substitution, `delins`
    or `inv` takes its reference bases from the sequence, not from the VCF row, and has no
    HGVS when they equal the alternate allele. MT uses the `m.` prefix.
    """
    kind = 'm' if chromosome == 'MT' else 'g'
    prefix = f'{chromosome}:{kind}.'
    start, ref, alt = _trim(position, reference, alternate)
    if ref and alt:
        end = start + len(ref) - 1
        ref = sequence[start - 1 : end].decode()
        if ref == alt:
            return None
        if len(ref) == 1 and len(alt) == 1:
            return f'{prefix}{start}{ref}>{alt}'
        span = f'{start}_{end}' if end > start else f'{start}'
        if len(ref) == len(alt) and alt.encode() == ref.encode().translate(_COMPLEMENT)[::-1]:
            return f'{prefix}{span}inv'
        return f'{prefix}{span}delins{alt}'
    if ref:
        end = start + len(ref) - 1
        while end < len(sequence) and sequence[end] == sequence[start - 1]:
            start, end = start + 1, end + 1
        return f'{prefix}{start}_{end}del' if end > start else f'{prefix}{start}del'
    inserted, left = alt.encode(), start - 1
    while left < len(sequence) and sequence[left] == inserted[0]:
        inserted, left = inserted[1:] + inserted[:1], left + 1
    size = len(inserted)
    text = inserted.decode()
    if sequence[left - size : left] == inserted:
        first = left - size + 1
        return f'{prefix}{first}_{left}dup' if left > first else f'{prefix}{left}dup'
    return f'{prefix}{left}_{left + 1}ins{text}'
