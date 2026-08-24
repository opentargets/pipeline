"""Tests for the `csv_to_parquet` task."""

import asyncio
import gzip
from pathlib import Path
from threading import Event

import polars as pl
import pytest
from otter.config.model import Config
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext

from pts.tasks.csv_to_parquet import CsvToParquet, CsvToParquetSpec

ROWS = [
    ('ENSG00000141510', 'true', 'protein_coding', '1000', 'NA'),
    ('ENSG00000012048', 'false', 'protein_coding', '2000', '1'),
]
HEADER = ('gene_id', 'canonical', 'transcript_type', 'rank', 'decile')


def _tsv_lines() -> list[str]:
    return ['\t'.join(HEADER)] + ['\t'.join(r) for r in ROWS]


def _run(tmp_path: Path, source: str, **spec_kwargs) -> pl.DataFrame:
    """Run the task with `tmp_path` as the work path and read back the parquet."""
    config = Config(step='test', steps=['test'], work_path=tmp_path)
    context = TaskContext(config=config, scratchpad=Scratchpad())
    context.abort = Event()
    spec = CsvToParquetSpec(
        name='csv_to_parquet test fixture',
        source=source,
        destination='out.parquet',
        separator='\t',
        **spec_kwargs,
    )
    task = asyncio.run(CsvToParquet(spec, context).run())

    # the `report` decorator swallows exceptions, so surface the failure reason
    assert task.manifest.failure_reason is None, task.manifest.failure_reason

    return pl.read_parquet(tmp_path / 'out.parquet')


@pytest.fixture
def plain_tsv(tmp_path: Path) -> str:
    path = tmp_path / 'in.tsv'
    path.write_text('\n'.join(_tsv_lines()) + '\n')
    return 'in.tsv'


def test_infers_types_by_default(tmp_path, plain_tsv):
    """Without `infer_schema`, polars picks types — numeric columns become numeric."""
    df = _run(tmp_path, plain_tsv)
    assert df.height == 2
    assert df.schema['rank'] == pl.Int64
    assert df.schema['canonical'] == pl.Boolean


def test_infer_schema_false_keeps_every_column_as_string(tmp_path, plain_tsv):
    """With `infer_schema: false` every column stays Utf8.

    Downstream spark steps compare against the raw text (``canonical == 'true'``,
    ``rank != 'NA'``), so type inference here would silently break those predicates.
    """
    df = _run(tmp_path, plain_tsv, infer_schema=False)
    assert df.height == 2
    assert set(df.schema.values()) == {pl.String}
    assert df['canonical'].to_list() == ['true', 'false']
    assert df['decile'].to_list() == ['NA', '1']


def test_reads_a_multi_member_gzip_source(tmp_path):
    """A bgzip (bgzf) source is read in full, not truncated at the first member.

    bgzf files — as gnomAD ships its constraint metrics — are a concatenation of
    independent gzip members. A decompressor that stops at the first one yields a
    silently short file rather than an error.
    """
    lines = _tsv_lines()
    first = gzip.compress(('\n'.join(lines[:2]) + '\n').encode())
    second = gzip.compress((lines[2] + '\n').encode())
    (tmp_path / 'in.tsv.bgz').write_bytes(first + second)

    df = _run(tmp_path, 'in.tsv.bgz', infer_schema=False)

    assert df.height == 2
    assert df['gene_id'].to_list() == ['ENSG00000141510', 'ENSG00000012048']
