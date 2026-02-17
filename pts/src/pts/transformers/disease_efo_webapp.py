import polars as pl
from loguru import logger


def disease_efo_webapp(source: str, destination: str) -> None:
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
