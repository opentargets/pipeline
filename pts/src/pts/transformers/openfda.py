import zipfile
from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.openfda import schema


def openfda(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as zip_file:
        with zip_file.open(source.stem) as file:
            file_content = file.read()
            df = pl.read_json(file_content, schema=schema)
            output = df.select('results').explode('results').unnest('results')

            # write the result locally
            output.write_parquet(destination, compression='gzip')
            logger.info('transformation complete')
