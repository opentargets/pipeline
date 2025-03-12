from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.ensembl import schema


def ensembl(source: Path, destination: Path) -> None:
    df = (
        pl.read_json(source, schema=schema)
        .drop('id')
        .explode('genes')
        .unnest('genes')
        .select([
            'id',
            'biotype',
            'description',
            'end',
            'start',
            'strand',
            pl.col('seq_region_name').alias('chromosome'),
            pl.col('name').alias('approvedSymbol'),
            'transcripts',
            'SignalP',
            pl.col('Uniprot/SPTREMBL').alias('uniprot_trembl'),
            pl.col('Uniprot/SWISSPROT').alias('uniprot_swissprot'),
        ])
    )

    # write the result locally
    df.write_parquet(destination, compression='gzip')
    logger.info('transformation complete')
