from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.ontology import schema


def so(source: Path, destination: Path) -> None:
    # load the ontology
    logger.debug('loading so')
    initial = pl.read_json(source)

    # prepare node data
    node_list = pl.DataFrame(
        initial['graphs'][0][0]['nodes'],
        schema=schema,
        strict=False,
    ).filter(
        pl.col('type') == 'CLASS',
    )

    # filter out non so terms and terms without labels, then select the id and label columns
    output = node_list.filter(
        pl.col('id').str.contains('SO_'),
        pl.col('lbl').is_not_null(),
    ).select(
        id=pl.col('id').str.split('/').list.last().str.replace('_', ':'),
        label=pl.col('lbl'),
    )

    # write the result locally
    output.write_parquet(destination)
    logger.info('transformation complete')
