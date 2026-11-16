"""Frames shared by the variant step tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pyBigWig  # ty: ignore[unresolved-import] -- compiled module without stubs

from pts.transformers.variant.index.rows import index_rows
from pts.transformers.variant.vep.alleles import vep_allele_form
from pts.transformers.variant.vep.frames import CONSEQUENCES_SCHEMA, EFFECTS_SCHEMA, ID_LISTS
from pts.transformers.variant.vep.frames import VARIANTS_SCHEMA as BASE_VARIANTS_SCHEMA
from pts.transformers.variant.vep.records import rank_consequences

VARIANTS_SCHEMA = BASE_VARIANTS_SCHEMA | ID_LISTS
"""The columns of the variants frame, with the colocated id lists `add_colocated` adds."""
MISSENSE = {
    'transcript_id': 'ENST_1',
    'gene_id': 'ENSG_1',
    'biotype': 'protein_coding',
    'canonical': 1,
    'consequence_terms': ['missense_variant'],
    'impact': 'MODERATE',
    'amino_acids': 'A/T',
    'codons': 'Gcc/Acc',
    'protein_end': 1,
}
"""A fastVEP canonical missense consequence on `ENST_1`, the transcript of chromosome 1 of `CORE_TABLES`."""


def consequences(**columns: Sequence[object]) -> pl.DataFrame:
    """A consequences frame of the given columns, one list of values per column.

    Columns of `CONSEQUENCES_SCHEMA` that are not given are null; the given ones are cast to
    their dtype. The effect columns of `EFFECTS_SCHEMA` are added after them, in `EFFECTS_SCHEMA`
    order, only when given.

    Raises:
        ValueError: on a column that is in neither schema.
    """
    schema = CONSEQUENCES_SCHEMA | EFFECTS_SCHEMA
    unknown = sorted(set(columns) - set(schema))
    if unknown:
        raise ValueError(f'not a consequences column: {unknown}')
    height = max((len(values) for values in columns.values()), default=0)
    given = pl.DataFrame({name: pl.Series(name, values, dtype=schema[name]) for name, values in columns.items()})
    return given.select(
        *[
            pl.col(name) if name in columns else pl.lit(None, dtype).alias(name)
            for name, dtype in CONSEQUENCES_SCHEMA.items()
        ],
        *[pl.col(name) for name in EFFECTS_SCHEMA if name in columns],
    ).head(height)


def consequence_rows(*rows: tuple[int, int, str, str, str]) -> pl.DataFrame:
    """One consequence per (row, position, reference, alternate, transcript id), in VEP's coordinates."""
    row, position, reference, alternate, transcript = (list(column) for column in zip(*rows, strict=True))
    alleles = vep_allele_form(
        pl.DataFrame({'position': position, 'referenceAllele': reference, 'alternateAllele': alternate}).lazy()
    ).collect()
    return consequences(
        row=row,
        position=position,
        referenceAllele=reference,
        alternateAllele=alternate,
        transcript_id=transcript,
        **{name: alleles[name].to_list() for name in ('vepStart', 'vepEnd', 'vepReference', 'vepAlternate')},
    )


def with_effects(frame: pl.DataFrame) -> pl.DataFrame:
    """The consequences with every effect column of `EFFECTS_SCHEMA` they lack added as null."""
    return frame.with_columns(
        pl.lit(None, dtype).alias(name) for name, dtype in EFFECTS_SCHEMA.items() if name not in frame.columns
    )


def variants(**columns: Sequence[object]) -> pl.DataFrame:
    """A variants frame of the given columns, one list of values per column.

    Columns of `VARIANTS_SCHEMA` that are not given are null; the given ones are cast to their dtype.

    Raises:
        ValueError: on a column that is not in the schema.
    """
    unknown = sorted(set(columns) - set(VARIANTS_SCHEMA))
    if unknown:
        raise ValueError(f'not a variants column: {unknown}')
    height = max((len(values) for values in columns.values()), default=0)
    given = pl.DataFrame({
        name: pl.Series(name, values, dtype=VARIANTS_SCHEMA[name]) for name, values in columns.items()
    })
    return given.select(
        pl.col(name) if name in columns else pl.lit(None, dtype).alias(name) for name, dtype in VARIANTS_SCHEMA.items()
    ).head(height)


def consequence_frame(*transcripts: dict) -> pl.DataFrame:
    """Consequences of row 0 unless given, one per transcript's columns, ranked as `chunk_frames` ranks them.

    Columns no transcript gives are null; effect columns are only there when a transcript gives them.
    """
    names = [*dict.fromkeys(name for transcript in transcripts for name in transcript)]
    columns = {name: [transcript.get(name) for transcript in transcripts] for name in names}
    rows = columns.pop('row', [0] * len(transcripts))
    return rank_consequences(consequences(row=rows, **columns))


def index_of(consequence_rows: pl.DataFrame, hash_threshold: int = 300, **updates: object) -> pl.DataFrame:
    """The variant index of one variant over the consequences, with every effect column added.

    The variant is `1:123 A>G` of row 0 with `missense_variant` as its most severe consequence,
    unless `updates` change its columns.
    """
    variant = {
        'row': 0,
        'chromosome': '1',
        'position': 123,
        'referenceAllele': 'A',
        'alternateAllele': 'G',
        'most_severe_consequence': 'missense_variant',
        'reported_transcripts': True,
        'rsIds': [],
        'clinvarIds': [],
        'omimIds': [],
        **updates,
    }
    frame = variants(**{name: [value] for name, value in variant.items()})
    return index_rows(frame, with_effects(rank_consequences(consequence_rows)), hash_threshold)


def fastvep_json(
    path: Path, ids: list[int], consequences: Sequence[dict] = (), most_severe: str = 'intron_variant'
) -> Path:
    """A fastVEP JSON array, one record per id, as fastVEP writes it (a record opens on its own line).

    Every record reports `most_severe` and the transcript `consequences`.
    """
    records = [
        {'id': str(row), 'most_severe_consequence': most_severe, 'transcript_consequences': [*consequences]}
        for row in ids
    ]
    path.write_text('[\n' + ',\n'.join(json.dumps(record, indent=2) for record in records) + '\n]\n')
    return path


def write_bigwig(path: Path, chromosome: str, length: int, scores: dict[int, float]) -> str:
    """A GERP bigWig of one contig, with a score at each of the one-based `scores` positions."""
    bigwig = pyBigWig.open(str(path), 'w')
    bigwig.addHeader([(chromosome, length)])
    if scores:
        positions = sorted(scores)
        bigwig.addEntries(
            [chromosome] * len(positions),
            [position - 1 for position in positions],
            ends=positions,
            values=[scores[position] for position in positions],
        )
    bigwig.close()
    return str(path)
