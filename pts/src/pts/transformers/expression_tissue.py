from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle


def expression_tissue(
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    # load the ontology
    logger.debug('loading expression tissue')
    h = StorageHandle(source)
    f = h.open()
    initial = pl.read_json(f)

    # unnest the tissues column and add a column for the tissue id
    n = initial.unnest('tissues')
    columns = n.columns
    tissue_list = [
        n.select(tissue=pl.col(column)).unnest('tissue').with_columns(tissue_id=pl.lit(column)) for column in columns
    ]
    output = pl.concat(tissue_list)

    # write the result locally
    output.write_parquet(destination, compression='gzip')
    logger.info('transformation complete')
