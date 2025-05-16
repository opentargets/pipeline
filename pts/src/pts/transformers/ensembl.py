import subprocess
import tempfile
from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.ensembl import schema_ndjson


def ensembl(source: Path, destination: Path) -> None:
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
            "uniprot_trembl": ."Uniprot/SPTREMBL",
            "uniprot_swissprot": ."Uniprot/SWISSPROT",
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
                "uniprot_trembl": ."Uniprot/SPTREMBL",
                "uniprot_swissprot": ."Uniprot/SWISSPROT",
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
    tempfile_path = Path(f'{tempfile.gettempdir()}/ensembl.jsonl')

    logger.info('transforming ensembl data into a ndjson')

    # we will have to do it like this for now, until polars fixes https://github.com/pola-rs/polars/issues/17677
    jq = subprocess.run(
        ['jq', '-c', jq_query, str(source)],
        capture_output=True,
        text=True,
    )
    if jq.returncode != 0:
        logger.error(f'jq error: {jq.stderr}')
        raise OSError(f'jq error: {jq.stderr}')

    tempfile_path.write_text(jq.stdout)
    logger.info('transforming ndjson into parquet')

    # write the result locally
    pl.read_ndjson(tempfile_path, schema=schema_ndjson).write_parquet(destination)
    logger.info('transformation complete')
