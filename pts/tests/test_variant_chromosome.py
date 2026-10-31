"""Tests for the per-chromosome and per-chunk units of the variant step."""

from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from pts.transformers.variant.chromosome import (
    Shared,
    _by_position,
    _submit,
    amino_acid_annotations,
    in_range,
    lower_priority,
    split_records,
)


def shared(directory: Path, chunk_rows: int = 2, row_group_rows: int = 10) -> Shared:
    """Shared inputs for the units that only read the scratch directory and the chunk settings."""
    unused = directory / 'unused'
    return Shared(
        plain_fasta=unused,
        contigs={},
        gff3=unused,
        gerp=str(unused),
        ancestor=str(unused),
        core=str(unused),
        sift=unused,
        binary='fastvep',
        threads=1,
        hash_threshold=300,
        directory=directory / 'chromosomes',
        chunk_rows=chunk_rows,
        row_group_rows=row_group_rows,
    )


def fastvep_json(path: Path, ids: list[int]) -> Path:
    """A fastVEP JSON array, one record per id, as fastVEP writes it (a record opens on its own line)."""
    records = [
        {'id': str(row), 'most_severe_consequence': 'intron_variant', 'transcript_consequences': []} for row in ids
    ]
    path.write_text('[\n' + ',\n'.join(json.dumps(record, indent=2) for record in records) + '\n]\n')
    return path


def test_split_records_writes_each_record_under_the_chunk_of_its_row(tmp_path: Path) -> None:
    units = shared(tmp_path)
    split_records(fastvep_json(tmp_path / 'out.json', [3, 0, 4, 1, 2]), 5, units, '22', chunk_bytes=64)
    chunks = sorted((tmp_path / 'chromosomes' / '22' / 'records').iterdir())
    assert [chunk.name for chunk in chunks] == ['00000', '00001', '00002']
    rows = [sorted(pl.read_parquet(chunk / '*.parquet')['row'].to_list()) for chunk in chunks]
    assert rows == [[0, 1], [2, 3], [4]]
    assert len(list(chunks[0].iterdir())) == 2


def test_split_records_fails_on_a_missing_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='2 fastVEP records for 3 rows'):
        split_records(fastvep_json(tmp_path / 'out.json', [0, 2]), 3, shared(tmp_path), '22')


def test_split_records_fails_on_an_id_repeated_across_slices(tmp_path: Path) -> None:
    path = fastvep_json(tmp_path / 'out.json', [0, 1, 0])
    with pytest.raises(ValueError, match=r'ids are not the ordinals 0\.\.2 '):
        split_records(path, 3, shared(tmp_path), '22', chunk_bytes=64)


def test_split_records_fails_on_an_id_repeated_within_a_slice(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r'ids are not the ordinals 0\.\.1 '):
        split_records(fastvep_json(tmp_path / 'out.json', [1, 1]), 2, shared(tmp_path), '22')


@pytest.mark.parametrize('ids', [[0, 2], [-1, 0]])
def test_split_records_fails_on_an_id_out_of_range(tmp_path: Path, ids: list[int]) -> None:
    with pytest.raises(ValueError, match=r'ids are not the ordinals 0\.\.1 '):
        split_records(fastvep_json(tmp_path / 'out.json', ids), 2, shared(tmp_path), '22')


def test_prepared_frames_are_read_by_position_range(tmp_path: Path) -> None:
    """Prepared frames are sorted into row groups by position; a chunk reads only its range."""
    positions = list(range(100, 0, -1))
    frame = pl.DataFrame({'position': positions, 'value': [str(p) for p in positions]}).with_columns(
        pl.col('position').cast(pl.Int32)
    )
    path = _by_position(frame, tmp_path / 'prepared.parquet', shared(tmp_path, row_group_rows=10))

    metadata = pq.ParquetFile(path).metadata
    bounds = [
        (metadata.row_group(i).column(0).statistics.min, metadata.row_group(i).column(0).statistics.max)
        for i in range(metadata.num_row_groups)
    ]
    assert bounds == [(start, start + 9) for start in range(1, 101, 10)]
    chunk = pl.Series([25, 18, 33])
    scan = in_range(path, chunk)
    assert 'SELECTION: col("position").is_between([18, 33])' in scan.explain()
    assert [low for low, high in bounds if low <= 33 and high >= 18] == [11, 21, 31]
    assert scan.collect()['position'].to_list() == list(range(18, 34))


def _polars_threads() -> int:
    pl.DataFrame({'a': [1]}).sum()
    return pl.thread_pool_size()


def test_each_pool_runs_polars_with_its_own_threads() -> None:
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(1, mp_context=context) as eight, ProcessPoolExecutor(2, mp_context=context) as two:
        wide = [_submit(eight, 8, _polars_threads) for _ in range(2)]
        narrow = [_submit(two, 2, _polars_threads) for _ in range(3)]
        assert [future.result() for future in wide + narrow] == [8, 8, 2, 2, 2]


def _niceness() -> int:
    return os.nice(0)


def test_each_pool_runs_at_its_own_niceness() -> None:
    """Like the step's pools: fastVEP without an initializer, preparations and chunks lowered."""
    context = multiprocessing.get_context('spawn')
    step = os.nice(0)
    with (
        ProcessPoolExecutor(1, mp_context=context) as fastvep,
        ProcessPoolExecutor(1, mp_context=context, initializer=lower_priority, initargs=(5,)) as preparations,
        ProcessPoolExecutor(2, mp_context=context, initializer=lower_priority, initargs=(15,)) as chunks,
    ):
        found = [pool.submit(_niceness).result() for pool in (fastvep, preparations, chunks, chunks)]
    assert found == [step, max(step, 5), max(step, 15), max(step, 15)]
    assert os.nice(0) == step


def test_amino_acid_annotations_keep_the_accessions_of_the_chromosomes_transcripts(tmp_path: Path) -> None:
    """Swiss-Prot and TrEMBL accessions, without their version, select the rows; every change of one is kept."""
    path = tmp_path / 'foldx.parquet'
    pl.DataFrame({
        'uniprotAccession': ['P1', 'Q9', 'P2', 'P1', 'P3'],
        'aminoAcidChange': ['A1T', 'A1T', 'G2C', 'M3V', 'A1T'],
        'variantEffect': [[0.5], [0.1], [0.2], [0.3], [0.4]],
    }).write_parquet(path)
    transcripts = pl.DataFrame(
        {'transcript_id': ['ENST1', 'ENST2'], 'swissprot': [['P1.2'], []], 'trembl': [[], ['P2']]},
        schema={'transcript_id': pl.String, 'swissprot': pl.List(pl.String), 'trembl': pl.List(pl.String)},
    )
    found = amino_acid_annotations(path, transcripts)
    assert found.columns == ['uniprotAccession', 'aminoAcidChange', 'variantEffect']
    assert found.rows() == [('P1', 'A1T', [0.5]), ('P2', 'G2C', [0.2]), ('P1', 'M3V', [0.3])]
