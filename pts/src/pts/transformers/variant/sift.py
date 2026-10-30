"""SIFT on transcript consequences, from Ensembl's protein function prediction matrices.

Each matrix is the gzip of `VEP` followed by, for every peptide position and every alternate
residue in `AMINO_ACIDS` order, a little-endian `uint16`: `0xFFFF` is no prediction, the top two
bits index `PREDICTIONS` and the low ten bits are the score times 1000. VEP finds a translation's
matrix by the md5 of its peptide and reads it only for a single-residue change to a standard
amino acid.
"""

from __future__ import annotations

import gzip
import hashlib
import re
from collections.abc import Collection

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PREDICTIONS = ('tolerated', 'deleterious', 'tolerated_low_confidence', 'deleterious_low_confidence')
SEVERITY = ('deleterious', 'deleterious_low_confidence', 'tolerated_low_confidence', 'tolerated')
"""SIFT predictions from the most to the least severe."""
CANDIDATE = rf'^[A-Z]/[{AMINO_ACIDS}]$'
MATRICES = pl.Schema({'transcript_id': pl.String, 'matrix': pl.Binary})

CHUNK_BYTES = 64 << 20
_HEADER = b'VEP'
_NONE = 0xFFFF
_ESCAPE = re.compile(rb'\\(.)', re.DOTALL)
_UNESCAPED = {b'0': b'\x00', b'n': b'\n', b't': b'\t', b'\\': b'\\'}
_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
_SCORES = pl.Schema({'sift_score': pl.Float64, 'sift_prediction': pl.String})


def unescape(blob: bytes) -> bytes:
    r"""A blob of the variation dump as stored: `\0`, `\n`, `\t` and `\\` stand for NUL, LF, TAB and `\`.

    Raises:
        ValueError: on a backslash before any other byte.
    """

    def replace(match: re.Match[bytes]) -> bytes:
        escaped = match.group(1)
        if escaped not in _UNESCAPED:
            raise ValueError(f'unknown escape \\{escaped.decode("latin-1")} in a prediction matrix')
        return _UNESCAPED[escaped]

    return _ESCAPE.sub(replace, blob)


def _analysis(attrib: str, name: str) -> bytes:
    """The `attrib_id` of an analysis, as the dump writes it."""
    with StorageHandle(attrib).open('rb') as raw, gzip.open(raw, 'rt') as handle:
        for line in handle:
            attrib_id, _, value = line.rstrip('\n').split('\t')
            if value == name:
                return attrib_id.encode()
    raise ValueError(f'no {name!r} attribute in {attrib}')


def _peptide_md5s(peptides: str, transcript_ids: Collection[str] | None) -> dict[str, str]:
    """The md5 of each translation's peptide, by unversioned transcript id."""
    sequences: dict[str, list[bytes]] = {}
    current: list[bytes] | None = None
    with StorageHandle(peptides).open('rb') as raw, gzip.open(raw, 'rb') as handle:
        for line in handle:
            if line.startswith(b'>'):
                fields = [field for field in line.split() if field.startswith(b'transcript:')]
                if not fields:
                    raise ValueError(f'{peptides}: no transcript in {line.decode().strip()}')
                transcript_id = fields[0].removeprefix(b'transcript:').split(b'.')[0].decode()
                wanted = transcript_ids is None or transcript_id in transcript_ids
                current = sequences.setdefault(transcript_id, []) if wanted else None
            elif current is not None:
                current.append(line.strip())
    return {transcript_id: hashlib.md5(b''.join(lines)).hexdigest() for transcript_id, lines in sequences.items()}


def _md5_ids(translation_md5: str, md5s: Collection[str]) -> dict[bytes, str]:
    """The dump's `translation_md5_id` of each wanted md5, as the dump writes it."""
    ids: dict[bytes, str] = {}
    with StorageHandle(translation_md5).open('rb') as raw, gzip.open(raw, 'rt') as handle:
        for line in handle:
            md5_id, md5 = line.rstrip('\n').split('\t')
            if md5 in md5s:
                ids[md5_id.encode()] = md5
    return ids


def _matrices(predictions: str, analysis: bytes, ids: dict[bytes, str]) -> dict[str, bytes]:
    """The gzip matrix of each wanted md5, streamed from the dump."""
    found: dict[str, bytes] = {}

    def keep(row: bytes) -> None:
        first = row.find(b'\t')
        second = row.find(b'\t', first + 1)
        if second < 0 or row[first + 1 : second] != analysis:
            return
        md5 = ids.get(row[:first])
        blob = row[second + 1 :]
        if md5 is not None and blob != b'\\N':
            found[md5] = unescape(blob)

    with StorageHandle(predictions).open('rb') as raw, gzip.open(raw, 'rb') as handle:
        tail = b''
        while chunk := handle.read(CHUNK_BYTES):
            rows = (tail + chunk).split(b'\n')
            tail = rows.pop()
            for row in rows:
                keep(row)
        if tail:
            keep(tail)
    return found


def sift_matrices(
    predictions: str,
    translation_md5: str,
    attrib: str,
    peptides: str,
    transcript_ids: Collection[str] | None = None,
) -> pl.DataFrame:
    """The SIFT matrix of each translated transcript, still gzip-encoded.

    Args:
        predictions: The variation dump's `protein_function_predictions.txt.gz`.
        translation_md5: The variation dump's `translation_md5.txt.gz`.
        attrib: The variation dump's `attrib.txt.gz`, which names the `sift` analysis.
        peptides: The `pep.all.fa.gz` FASTA of the same Ensembl release.
        transcript_ids: Unversioned transcript ids to keep; every transcript when None.

    Returns:
        (`transcript_id`, `matrix`), sorted by `transcript_id`; a transcript without a SIFT
        matrix has no row.
    """
    md5s = _peptide_md5s(peptides, transcript_ids)
    ids = _md5_ids(translation_md5, set(md5s.values()))
    matrices = _matrices(predictions, _analysis(attrib, 'sift'), ids)
    transcript_ids = sorted(transcript_id for transcript_id, md5 in md5s.items() if md5 in matrices)
    return pl.DataFrame(
        {
            'transcript_id': transcript_ids,
            'matrix': [matrices[md5s[transcript_id]] for transcript_id in transcript_ids],
        },
        schema=MATRICES,
    )


def sift_scores(records: pl.DataFrame, matrices: pl.DataFrame) -> pl.DataFrame:
    """The SIFT score and prediction of each transcript consequence.

    Only a change of one residue to a standard amino acid (`amino_acids` such as `A/V`) is
    looked up, at `protein_end`; each matrix is decompressed once.

    Args:
        records: One row per transcript consequence, with `transcript_id`, `amino_acids` and
            `protein_end`.
        matrices: `sift_matrices`.

    Returns:
        `sift_score` and `sift_prediction`, row for row; both null without a prediction.

    Raises:
        ValueError: on a matrix that does not start with `VEP`.
    """
    scores: list[float | None] = [None] * records.height
    predictions: list[str | None] = [None] * records.height
    candidates = (
        records
        .select('transcript_id', 'amino_acids', 'protein_end')
        .with_row_index('_index')
        .filter(pl.col('amino_acids').str.contains(CANDIDATE) & (pl.col('protein_end') > 0))
        .group_by('transcript_id')
        .agg('_index', pl.col('amino_acids').str.slice(2, 1).alias('residue'), 'protein_end')
        .join(matrices, on='transcript_id', how='inner')
    )
    for transcript_id, rows, residues, positions, matrix in candidates.iter_rows():
        decoded = gzip.decompress(matrix)
        if not decoded.startswith(_HEADER):
            raise ValueError(f'SIFT matrix of {transcript_id} does not start with {_HEADER!r}')
        for row, residue, position in zip(rows, residues, positions, strict=True):
            offset = len(_HEADER) + ((position - 1) * len(AMINO_ACIDS) + _INDEX[residue]) * 2
            if offset + 2 > len(decoded):
                continue
            value = int.from_bytes(decoded[offset : offset + 2], 'little')
            if value != _NONE:
                predictions[row] = PREDICTIONS[value >> 14]
                scores[row] = (value & 0x3FF) / 1000
    return pl.DataFrame({'sift_score': scores, 'sift_prediction': predictions}, schema=_SCORES)


def add_sift(records: pl.LazyFrame, matrices: pl.DataFrame) -> pl.LazyFrame:
    """Fill `sift_score` and `sift_prediction` of every transcript consequence of the records.

    Args:
        records: VEP-shaped records.
        matrices: `sift_matrices`.

    Returns:
        The records, row for row; see `sift_scores`.
    """
    frame = records.collect()
    consequences = (
        frame
        .select(pl.col('transcript_consequences').alias('transcript'))
        .with_row_index('row')
        .filter(pl.col('transcript').list.len() > 0)
        .explode('transcript')
        .rechunk()
        .with_columns(pl.col('transcript').struct.field('transcript_id', 'amino_acids', 'protein_end'))
    )
    scores = sift_scores(consequences, matrices)
    filled = (
        consequences
        .with_columns(scores.select(pl.col('sift_score').alias('_score'), pl.col('sift_prediction').alias('_pred')))
        .rechunk()
        .select(
            'row',
            pl.col('transcript').struct.with_fields(
                pl.col('_score').alias('sift_score'), pl.col('_pred').alias('sift_prediction')
            ),
        )
        .group_by('row', maintain_order=True)
        .agg(pl.col('transcript').alias('filled'))
    )
    return (
        frame
        .with_row_index('row')
        .join(filled, on='row', how='left', maintain_order='left')
        .with_columns(pl.coalesce('filled', 'transcript_consequences').alias('transcript_consequences'))
        .select(frame.columns)
        .lazy()
    )
