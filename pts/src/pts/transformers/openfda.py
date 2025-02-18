from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.openfda import schema


def openfda(source: Path, destination: Path) -> None:
    df = pl.read_json(source, schema=schema)
    output = df.select('results').explode('results').unnest('results')

    # write the result locally
    output.write_parquet(destination)
    logger.info('transformation complete')
