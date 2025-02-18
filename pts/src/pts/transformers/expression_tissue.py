from pathlib import Path

import polars as pl
from loguru import logger


def expression_tissue(source: Path, destination: Path) -> None:
    # load the ontology
    logger.debug('loading expression tissue')
    initial = pl.read_json(source)

    # unnest the tissues column and add a column for the tissue id
    n = initial.unnest('tissues')
    columns = n.columns
    tissue_list = [
        n.select(tissue=pl.col(column)).unnest('tissue').with_columns(tissue_id=pl.lit(column)) for column in columns
    ]
    output = pl.concat(tissue_list)

    # write the result locally
    output.write_parquet(destination)
    logger.info('transformation complete')
