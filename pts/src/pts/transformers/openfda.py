import zipfile
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.openfda import schema
from pts.transformers.utils.dataset import write_dataset


def openfda(
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    h = StorageHandle(source)
    f = h.open('rb')

    with zipfile.ZipFile(f) as zip_file:
        filename = zip_file.namelist()[0]
        with zip_file.open(filename) as file:
            file_content = file.read()
            df = pl.read_json(file_content, schema=schema)
            output = df.select('results').explode('results').unnest('results')

            write_dataset(output, destination)
            logger.info('transformation complete')
