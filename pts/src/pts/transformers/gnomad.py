import gzip
from pathlib import Path

from loguru import logger


def gnomad(source: Path, destination: Path) -> None:
    with gzip.open(source) as file:
        with gzip.open(destination, 'wb') as gzip_file:
            gzip_file.write(file.read())
    logger.debug('conversion completed')
