import zipfile
from typing import Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.openfda import schema


def openfda(source: str, destination: str, settings: dict[str, Any]) -> None:
    h = StorageHandle(source)
    f = h.open('rb')

    with zipfile.ZipFile(f) as zip_file:
        filename = zip_file.namelist()[0]
        with zip_file.open(filename) as file:
            file_content = file.read()
            df = pl.read_json(file_content, schema=schema)
            output = df.select('results').explode('results').unnest('results')

            output.write_parquet(destination, compression='gzip')
            logger.info('transformation complete')
