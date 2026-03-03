from typing import Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.ontology import node


def so(source: str, destination: str, settings: dict[str, Any]) -> None:
    # load the ontology
    logger.debug('loading so')
    h = StorageHandle(source)
    f = h.open()
    initial = pl.read_json(f)

    # prepare node data
    node_list = pl.DataFrame(
        initial['graphs'][0][0]['nodes'],
        schema=node,
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
    output.write_parquet(destination, compression='gzip')
    logger.info('transformation complete')
