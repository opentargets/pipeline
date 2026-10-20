"""Schemas for the transcript step."""

import polars as pl

from pts.schemas.ensembl_core import exon

__all__ = ['exon', 'flag', 'transcript']

flag = pl.Struct({'label': pl.String(), 'value': pl.String()})
"""One entry of `flags`."""

transcript = pl.Schema({
    'targetId': pl.String(),
    'transcriptId': pl.String(),
    'biotype': pl.String(),
    'proteinId': pl.String(),
    'uniprotSwissprotIds': pl.List(pl.String()),
    'uniprotTremblIds': pl.List(pl.String()),
    'uniprotIsoformIds': pl.List(pl.String()),
    'alphafoldIds': pl.List(pl.String()),
    'isEnsemblCanonical': pl.Boolean(),
    'chromosome': pl.String(),
    'start': pl.Int64(),
    'end': pl.Int64(),
    'strand': pl.Int32(),
    'transcriptionStartSite': pl.Int32(),
    'flags': pl.List(flag),
    'exons': pl.List(exon),
})
"""The published `output/transcript` dataset."""
