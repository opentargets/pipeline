"""WikiPathways pathway dataset generation.

Writes WikiPathways human pathways in the same schema as the Reactome dataset, so the two can
later be combined. WikiPathways has no pathway hierarchy, so every pathway is a root: the
ancestry columns are empty and `path` holds the pathway alone.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.reactome import reactome_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset

HUMAN = 'Homo sapiens'

#: A GMT line is `name%release%id%species<TAB>url<TAB>gene<TAB>gene...`. The name is free text,
#: so the header is matched from the right.
_HEADER = r'^(?<name>.*)%(?<release>[^%]+)%(?<id>[^%]+)%(?<species>[^%]+)$'


def _read_gmt(path: str) -> pl.DataFrame:
    """Read the pathway header and URL of each line of a GMT file.

    The gene columns that follow vary in number from line to line and are not part of this
    dataset, so they are dropped. `quote_char=None` keeps a double quote in a pathway name a
    literal character, as in the Reactome input.
    """
    return scan_dataset(
        path,
        format='tsv',
        has_header=False,
        quote_char=None,
        schema={'header': pl.String, 'url': pl.String},
        extra_columns='ignore',
        truncate_ragged_lines=True,
    ).collect()


def _parse_pathways(gmt: pl.DataFrame) -> pl.DataFrame:
    """Split GMT headers into id, name, release and species."""
    return gmt.select(pl.col('header').str.extract_groups(_HEADER)).unnest('header')


def _validate(pathways: pl.DataFrame, min_pathways: int) -> None:
    """Fail on input that is malformed, mixed or suspiciously small.

    Every check here catches a wrong or truncated download that would otherwise be written out
    silently: an unparseable header, two releases in one file, or far fewer pathways than a
    release normally holds.
    """
    unparsed = pathways.filter(pl.col('id').is_null()).height
    if unparsed:
        raise ValueError(f'{unparsed} GMT line(s) have a header that is not name%release%id%species')

    releases = pathways['release'].unique().sort().to_list()
    if len(releases) != 1:
        raise ValueError(f'expected one WikiPathways release in the GMT file, found {releases}')

    duplicates = pathways.filter(pl.col('id').is_duplicated())['id'].unique().sort().to_list()
    if duplicates:
        raise ValueError(f'duplicate WikiPathways ids in the GMT file: {duplicates}')

    human = pathways.filter(pl.col('species') == HUMAN).height
    if human < min_pathways:
        raise ValueError(f'found {human} {HUMAN} pathways, expected at least {min_pathways}')


def _build_documents(pathways: pl.DataFrame) -> pl.DataFrame:
    """Shape human pathways into the Reactome schema, each pathway its own root."""
    return (
        pathways
        .filter(pl.col('species') == HUMAN)
        .sort('id')
        .select(
            pl.col('id'),
            pl.col('name').alias('label'),
            pl.lit([], dtype=pl.List(pl.String)).alias('ancestors'),
            pl.lit([], dtype=pl.List(pl.String)).alias('descendants'),
            pl.lit([], dtype=pl.List(pl.String)).alias('children'),
            pl.lit([], dtype=pl.List(pl.String)).alias('parents'),
            pl.col('id').reshape((-1, 1, 1)).cast(pl.List(pl.List(pl.String))).alias('path'),
        )
        .cast(pl.Schema(reactome_schema))
    )


def wikipathways(
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate the WikiPathways pathway dataset."""
    logger.info(f'Reading WikiPathways GMT from {source}')
    pathways = _parse_pathways(_read_gmt(source))
    _validate(pathways, settings['min_pathways'])

    result = _build_documents(pathways)
    logger.info(f'Found {result.height} {HUMAN} pathways in release {pathways["release"][0]}')

    logger.info(f'Writing WikiPathways output to {destination}')
    write_dataset(result, destination)
