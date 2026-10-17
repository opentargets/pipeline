"""Reactome pathways per Ensembl gene id."""

from __future__ import annotations

import polars as pl

from pts.transformers.utils.dataset import scan_dataset

ENSEMBL2REACTOME_COLUMNS = ['ensemblId', 'reactomeId', 'url', 'eventName', 'eventCode', 'species']


def read_reactome_pathways(path: str) -> pl.DataFrame:
    """Ensembl2Reactome.txt, as text."""
    return scan_dataset(
        path,
        format='tsv',
        has_header=False,
        quote_char=None,
        infer_schema=False,
        new_columns=ENSEMBL2REACTOME_COLUMNS,
    ).collect()


def read_reactome_etl(path: str) -> pl.DataFrame:
    """The reactome step's output, the columns used here."""
    return scan_dataset(path).select('id', 'label', 'path').collect()


def build_reactome(pathways: pl.DataFrame, reactome: pl.DataFrame) -> pl.DataFrame:
    """Human pathways per gene, each with the label of its first path's top-level event.

    Returns:
        [id, pathways], `pathways` a sorted list of distinct {pathwayId, pathway, topLevelTerm}.
    """
    top_level_terms = (
        reactome
        .select('id', pl.col('path').list.first().list.first().alias('tlId'))
        .join(reactome.select(pl.col('id').alias('tlId'), pl.col('label').alias('topLevelTerm')), on='tlId')
        .select(pl.col('id').alias('reactomeId'), 'topLevelTerm')
    )
    return (
        pathways
        .filter((pl.col('species') == 'Homo sapiens') & pl.col('ensemblId').str.starts_with('ENSG'))
        .join(top_level_terms, on='reactomeId', how='inner')
        .select(
            pl.col('ensemblId').alias('id'),
            pl.col('reactomeId').alias('pathwayId'),
            pl.col('eventName').alias('pathway'),
            'topLevelTerm',
        )
        .unique()
        .sort(pl.all(), nulls_last=False)
        .group_by('id', maintain_order=True)
        .agg(pl.struct('pathwayId', 'pathway', 'topLevelTerm').alias('pathways'))
    )
