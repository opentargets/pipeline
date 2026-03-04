from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config


def disease_efo_webapp(
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    # load the ontology
    logger.debug('loading efo')
    initial = pl.read_parquet(source)

    logger.debug('starting transformation')

    final = initial.select(
        id=pl.col('id'),
        parentIds=pl.col('parents'),
        name=pl.col('name'),
    )

    # write the result locally
    final.write_ndjson(destination)
    logger.info('transformation complete')
