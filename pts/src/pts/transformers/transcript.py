"""Transcript index dataset generation.

Produces a per-transcript index from the Ensembl core database: coordinates, exons and flags from the
`transcript`, `exon` and `transcript_attrib` tables, and UniProt and AlphaFold identifiers from each
translation's xrefs and protein features.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.outputs.transcript import Transcript
from pts.schemas.transcript import transcript as transcript_schema
from pts.transformers.utils import ensembl
from pts.transformers.utils.dataset import write_dataset

#: Attribute code, label and value of each static flag, in output order.
_STATIC_FLAGS = [
    ('MANE_Select', 'mane', 'select'),
    ('MANE_Plus_Clinical', 'mane', 'plus_clinical'),
    ('gencode_primary', 'gencode_primary', 'true'),
    ('is_canonical', 'ensembl_canonical', 'true'),
]


def _codes(attributes: pl.Expr) -> pl.Expr:
    return attributes.list.eval(pl.element().struct.field('code'))


def _non_empty(column: str) -> pl.Expr:
    """`column`, null where the list is empty."""
    return pl.when(pl.col(column).list.len() > 0).then(pl.col(column))


def build_flags(attributes: pl.Expr) -> pl.Expr:
    """Build the flags list from a transcript's attributes.

    Static flags are set by the presence of their code; the attribute value (e.g. the RefSeq accession
    of a MANE transcript) is not published. APPRIS values are spelled as GENCODE tags them
    (`principal1` -> `principal_1`).

    Args:
        attributes: list of `{code, value}` structs from `ensembl.transcripts`.

    Returns:
        List of `{label, value}` structs, the static flags first, then appris.
    """
    codes = _codes(attributes)
    static = pl.concat_list([
        pl.when(codes.list.contains(code)).then(pl.struct(label=pl.lit(label), value=pl.lit(value)))
        for code, label, value in _STATIC_FLAGS
    ]).list.drop_nulls()
    appris = attributes.list.eval(
        pl.struct(
            label=pl.lit('appris'),
            value=pl.element().struct.field('value').str.replace(r'^(principal|alternative)', '${1}_'),
        ).filter(pl.element().struct.field('code') == 'appris')
    )
    return pl.concat_list(static, appris)


def build_transcripts(transcripts: pl.DataFrame, xrefs: pl.DataFrame) -> pl.DataFrame:
    """Join transcripts with their translation's xrefs into the transcript index.

    Args:
        transcripts: output of `ensembl.transcripts`.
        xrefs: output of `ensembl.translation_xrefs`.

    Returns:
        DataFrame with the transcript index schema, sorted by gene and transcript.
    """
    return (
        transcripts
        .join(xrefs.drop('transcriptId'), on='translationId', how='left', maintain_order='left')
        .rechunk()
        .select(
            pl.col('geneId').alias('targetId'),
            'transcriptId',
            'biotype',
            pl.col('translationId').alias('proteinId'),
            _non_empty('uniprot_swissprot').alias('uniprotSwissprotIds'),
            _non_empty('uniprot_trembl').alias('uniprotTremblIds'),
            _non_empty('uniprot_isoform').alias('uniprotIsoformIds'),
            _non_empty('alphafold').alias('alphafoldIds'),
            _codes(pl.col('attributes')).list.contains('is_canonical').alias('isEnsemblCanonical'),
            'chromosome',
            'start',
            'end',
            'strand',
            pl
            .when(pl.col('strand') == 1)
            .then(pl.col('start'))
            .otherwise(pl.col('end'))
            .cast(pl.Int32)
            .alias('transcriptionStartSite'),
            build_flags(pl.col('attributes')).alias('flags'),
            _non_empty('exons').alias('exons'),
        )
        .select(transcript_schema.names())
        .sort('targetId', 'transcriptId', maintain_order=True)
    )


def transcript(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Generate the transcript index dataset from the Ensembl core database.

    Args:
        source: paths keyed by `ensembl` (the human core dump directory).
        destination: path of the transcript dataset.
        settings: unused.
        config: unused.
    """
    dump = ensembl.CoreDump(source['ensembl'])
    result = build_transcripts(ensembl.transcripts(dump), ensembl.translation_xrefs(dump))

    logger.info(f'writing {result.height} transcripts from Ensembl {dump.release} to {destination}')
    write_dataset(result, destination, schema=Transcript)
