import subprocess
import tempfile
from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.ensembl import schema_ndjson


def ensembl(source: Path, destination: Path) -> None:
    jq_query = '.genes[] | { id: .id, biotype: .biotype, description: .description, end: .end, start: .start, strand: .strand, chromosome: .seq_region_name, approvedSymbol: .name, transcripts: [.transcripts[] | { id: .id, biotype: .biotype, description: .description, end: .end, start: .start, strand: .strand, chromosome: .seq_region_name, SignalP: .SignalP, "Uniprot/uniprot_trembl": ."Uniprot/SPTREMBL", "uniprot_swissprot": ."Uniprot/SWISSPROT" }]}'  # noqa: E501
    tempfile_path = Path(f'{tempfile.gettempdir()}/ensembl.jsonl')

    logger.info('transforming ensembl data into a ndjson')

    # we will have to do it like this for now, until polars fixes https://github.com/pola-rs/polars/issues/17677
    jq = subprocess.run(
        ['jq', '-c', jq_query, str(source)],
        capture_output=True,
        text=True,
    )

    tempfile_path.write_text(jq.stdout)
    logger.info('transforming ndjson into parquet')

    # write the result locally
    pl.read_ndjson(tempfile_path, schema=schema_ndjson).write_parquet(destination)
    logger.info('transformation complete')
