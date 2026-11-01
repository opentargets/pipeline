"""Tests for the prepared chromosome: what every chunk of one chromosome reads, written once."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.alleles import contig_spans
from pts.transformers.variant.prepared import ChunkSettings, PreparationSources, PreparedChromosome
from tests.ensembl_fixtures import cells, ddl, write_dump
from tests.test_variant import CONTIG_LENGTH, CORE_TABLES
from tests.test_variant_colocated import all_vars, cache_line, rows
from tests.test_variant_loftee import write_bgzip_fasta


def prepared(
    directory: Path, all_vars: Path | None = None, annotations: int = 0, amino_acids: int = 0
) -> PreparedChromosome:
    """Chromosome 1 of a two-contig reference, with its rows `1_123_A_G` and `1_200_C_T`."""
    fasta = directory / 'reference.fa'
    fasta.write_text('>1\n' + 'ACGT' * 15 + '\nAC\n>2\nGGGG\n')
    path = directory / 'rows.parquet'
    rows('1_123_A_G', '1_200_C_T').write_parquet(path)
    return PreparedChromosome(
        chromosome='1',
        directory=directory / 'chromosomes' / '1',
        rows=path,
        plain_fasta=fasta,
        span=contig_spans(fasta)['1'],
        all_vars=all_vars,
        alphamissense=str(directory / 'alphamissense.parquet'),
        gerp=str(directory / 'gerp.bw'),
        sift=directory / 'sift.parquet',
        settings=ChunkSettings(chunk_rows=2, row_group_rows=10),
        annotation_sources=annotations,
        amino_acid_sources=amino_acids,
    )


def sources(directory: Path) -> PreparationSources:
    """A core dump and ancestor holding chromosome 1, one annotation source at `1_123_A_G`, and one FoldX source."""
    core = directory / 'core'
    core.mkdir()
    write_dump(
        core,
        {name: [cells(*row) for row in table] for name, (_, table) in CORE_TABLES.items()},
        ddl({name: columns for name, (columns, _) in CORE_TABLES.items()}),
    )
    annotations = directory / 'annotations.parquet'
    pl.DataFrame(
        {'variantId': ['1_123_A_G'], 'chromosome': ['1'], 'position': [123]}, schema_overrides={'position': pl.Int32}
    ).write_parquet(annotations)
    foldx = directory / 'foldx.parquet'
    pl.DataFrame({
        'uniprotAccession': ['P1', 'P9'],
        'aminoAcidChange': ['A1T', 'A1T'],
        'variantEffect': [[0.5], [0.1]],
    }).write_parquet(foldx)
    return PreparationSources(
        gerp=str(directory / 'gerp.bw'),
        ancestor=write_bgzip_fasta(directory / 'ancestor.fa.gz', {'1': (CONTIG_LENGTH, 1, 'a' * CONTIG_LENGTH)}),
        core=str(core),
        sift=directory / 'sift.parquet',
        variant_annotations=({'1': [str(annotations)]},),
        amino_acid_annotations=(foldx,),
    )


def test_prepared_chromosome_survives_pickling(tmp_path: Path) -> None:
    handle = prepared(tmp_path)
    copy = pickle.loads(pickle.dumps(handle))
    assert copy == handle
    assert copy.sequence() == handle.sequence() == b'ACGT' * 15 + b'AC'


def test_sequence_of_a_contig_the_reference_lacks_is_empty(tmp_path: Path) -> None:
    handle = prepared(tmp_path)
    assert replace(handle, span=None).sequence() == b''


def test_records_are_one_directory_per_chunk(tmp_path: Path) -> None:
    handle = prepared(tmp_path)
    assert handle.records(3) == tmp_path / 'chromosomes' / '1' / 'records' / '00003'


def test_writing_over_an_existing_preparation_raises(tmp_path: Path) -> None:
    handle = prepared(tmp_path)
    inputs = replace(sources(tmp_path), variant_annotations=(), amino_acid_annotations=())
    handle.write(inputs)
    with pytest.raises(FileExistsError):
        handle.write(inputs)


def test_write_produces_the_prepared_files(tmp_path: Path) -> None:
    located = all_vars(tmp_path / 'all_vars.gz', cache_line('rs123', 123, 123, 'A/G'))
    handle = prepared(tmp_path, all_vars=located, annotations=1, amino_acids=1)
    handle.write(sources(tmp_path))
    directory = tmp_path / 'chromosomes' / '1' / 'prepared'
    assert sorted(path.name for path in directory.iterdir()) == [
        'amino-acids-000.parquet',
        'ancestor.bin',
        'annotations-000.parquet',
        'colocated.parquet',
        'loftee.pickle',
        'transcripts.parquet',
    ]
    assert pl.read_parquet(directory / 'colocated.parquet')['position'].to_list() == [123]
    assert pl.read_parquet(directory / 'annotations-000.parquet')['variantId'].to_list() == ['1_123_A_G']
    assert pl.read_parquet(directory / 'amino-acids-000.parquet')['uniprotAccession'].to_list() == ['P1']
    assert pl.read_parquet(directory / 'transcripts.parquet')['transcript_id'].to_list() == ['ENST_1']
