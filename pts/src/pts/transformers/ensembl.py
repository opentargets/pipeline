import subprocess
import tempfile
from pathlib import Path

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.ensembl import schema_ndjson


def ensembl(source: str, destination: str) -> None:
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
    logger.info('transforming ensembl data into a ndjson')
    tmp_local_copy = Path(f'{tempfile.gettempdir()}/ensembl.jsonl')
    tmp_local_result = Path(f'{tempfile.gettempdir()}/ensembl_transformed.jsonl')

    s = StorageHandle(source)
    t = StorageHandle(tmp_local_copy, force_local=True)
    s.copy_to(t)

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

    logger.info(f'jq transformation complete into {tmp_local_result.absolute}')

    r = StorageHandle(tmp_local_result, force_local=True)
    with r.open('wt') as tmp_contents:
        tmp_contents.write(jq.stdout)

    logger.info('transforming ndjson into parquet')

    pl.read_ndjson(tmp_local_result, schema=schema_ndjson).write_parquet(destination)
    logger.info('transformation complete')
