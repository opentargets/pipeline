import subprocess
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.ensembl import schema_ndjson

# Only focusing on canonical chromosomes:
INCLUDED_CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']


def ensembl(
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    jq_query = """
        .genes[] | {
            id: .id,
            approvedSymbol: .name,
            biotype: .biotype,
            description: .description,
            chromosome: .seq_region_name,
            strand: .strand,
            start: .start,
            end: .end,
            SignalP: .SignalP,
            uniprot_trembl: ."Uniprot/SPTREMBL",
            uniprot_swissprot: ."Uniprot/SWISSPROT",
            uniprot_isoform: .Uniprot_isoform,
            alphafold: .alphafold,
            transcripts: [(.transcripts // [])[] | {
                id: .id,
                approvedSymbol: .name,
                biotype: .biotype,
                description: .description,
                chromosome: .seq_region_name,
                strand: .strand,
                start: .start,
                end: .end,
                SignalP: .SignalP,
                uniprot_trembl: ."Uniprot/SPTREMBL",
                uniprot_swissprot: ."Uniprot/SWISSPROT",
                uniprot_isoform: .Uniprot_isoform,
                alphafold: .alphafold,
                "exons": [(.exons // [])[] | {
                    id: .id,
                    start: .start,
                    end: .end,
                    strand: .strand,
                    chromosome: .seq_region_name,
                }],
                translations: [(.translations // [])[] | {
                    id: .id,
                }]
            }],
        }
    """.replace('\n', ' ').replace(' ', '')
    logger.info(f'transforming ensembl data from {source} and writing as parquet to {destination}')
    # we get a local handle to the ensembl file
    s = StorageHandle(source, config=config, force_local=True)
    t = StorageHandle(f'{s.absolute}.transformed.ndjson', force_local=True)

    # we will have to do it like this for now, until polars fixes https://github.com/pola-rs/polars/issues/17677
    logger.debug(f'Running jq on file: {s.absolute}')
    jq = subprocess.run(
        ['jq', '-c', jq_query, s.absolute],
        capture_output=True,
        text=True,
        close_fds=True,
    )
    if jq.returncode != 0:
        logger.error(f'jq error: {jq.stderr}')
        raise OSError(f'jq error: {jq.stderr}')
    logger.debug('jq transformation complete')

    logger.debug(f'writing transformed data into temporary file: {t.absolute}')
    with t.open('wt') as tmp_contents:
        tmp_contents.write(jq.stdout)
    logger.debug(f'transformed data written into {t.absolute}')

    logger.debug(f'transforming ndjson into parquet at {destination}')
    (
        pl
        .read_ndjson(t.absolute, schema=schema_ndjson)
        .filter(pl.col('chromosome').is_in(INCLUDED_CHROMOSOMES))
        .write_parquet(destination)
    )
    logger.info('transformation complete')
