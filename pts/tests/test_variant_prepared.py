"""Tests for the prepared chromosome: what every chunk of one chromosome reads, written once."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from pts.transformers.variant.alleles import contig_spans
from pts.transformers.variant.chromosome import annotate_inputs, split_records
from pts.transformers.variant.colocated import KEY
from pts.transformers.variant.gpn_star import GpnStarCutoffs
from pts.transformers.variant.prepared import ChunkSettings, PreparationSources, PreparedChromosome, in_range
from pts.transformers.variant.sift import MATRICES
from tests.ensembl_fixtures import cells, ddl, write_dump
from tests.test_variant import CONTIG_LENGTH, CORE_TABLES, GPN_STAR_CUTOFFS, _write_gpn_star
from tests.test_variant_colocated import all_vars, cache_line, rows
from tests.test_variant_loftee import write_bgzip_fasta
from tests.variant_fixtures import MISSENSE, fastvep_json, write_bigwig


def prepared(
    directory: Path,
    all_vars: Path | None = None,
    annotations: int = 0,
    amino_acids: int = 0,
    variant_ids: tuple[str, ...] = ('1_123_A_G', '1_200_C_T'),
) -> PreparedChromosome:
    """Chromosome 1 of a two-contig reference with its rows `variant_ids`.

    It has a GERP bigWig, but no SIFT matrix and no AlphaMissense row.
    """
    fasta = directory / 'reference.fa'
    fasta.write_text('>1\n' + 'ACGT' * 15 + '\nAC\n>2\nGGGG\n')
    path = directory / 'rows.parquet'
    rows(*variant_ids).write_parquet(path)
    write_bigwig(directory / 'gerp.bw', '1', CONTIG_LENGTH, {123: 2.5})
    pl.DataFrame(schema=MATRICES).write_parquet(directory / 'sift.parquet')
    (directory / 'alphamissense.tsv').write_text('# AlphaMissense\n')
    return PreparedChromosome(
        chromosome='1',
        directory=directory / 'chromosomes' / '1',
        rows=path,
        plain_fasta=fasta,
        span=contig_spans(fasta)['1'],
        all_vars=all_vars,
        alphamissense=str(directory / 'alphamissense.tsv'),
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
        ancestor=write_bgzip_fasta(directory / 'ancestor.fa.gz', {'1': (CONTIG_LENGTH, 1, 'a' * CONTIG_LENGTH)}),
        core=str(core),
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
    assert handle.record_slice(3, 12) == handle.records(3) / '00012.parquet'


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


@pytest.mark.parametrize(('annotations', 'amino_acids'), [(0, 1), (2, 1), (1, 0), (1, 2)])
def test_write_refuses_sources_that_disagree_with_the_handle(
    tmp_path: Path, annotations: int, amino_acids: int
) -> None:
    handle = prepared(tmp_path, annotations=annotations, amino_acids=amino_acids)
    with pytest.raises(ValueError, match='sources'):
        handle.write(sources(tmp_path))
    assert not handle.directory.exists()


def test_chunk_reads_what_the_inline_code_read(tmp_path: Path) -> None:
    located = all_vars(tmp_path / 'all_vars.gz', cache_line('rs123', 123, 123, 'A/G'))
    handle = prepared(tmp_path, all_vars=located, annotations=1, amino_acids=1)
    _write_gpn_star(tmp_path / 'gpn_star', '1', [(123, 'A', 'G', -3.0), (150, 'A', 'G', -5.0)])
    handle.write(
        replace(
            sources(tmp_path),
            gpn_star={'1': str(tmp_path / 'gpn_star' / 'llr_chr1.parquet')},
            gpn_star_cutoffs=GpnStarCutoffs(**GPN_STAR_CUTOFFS),
        )
    )
    split_records(fastvep_json(tmp_path / 'fastvep.json', [1, 0], [MISSENSE]), 2, handle)
    directory = handle.directory / 'prepared'
    frame = pl.scan_parquet(handle.rows).slice(0, 2).collect()
    loci = frame.select('chromosome', 'position').unique()
    keys = pl.DataFrame({'uniprotAccession': ['P1', 'P9'], 'aminoAcidChange': ['A1T', 'A1T']})
    expected = {
        'records': pl.scan_parquet(handle.records(0) / '*.parquet').with_columns(pl.col('row') - 0).collect(),
        'transcripts': pl.read_parquet(directory / 'transcripts.parquet'),
        'colocated': in_range(directory / 'colocated.parquet', frame['position'])
        .join(frame.lazy().select(KEY), on=KEY, how='semi')
        .collect(),
        'annotations': [
            in_range(directory / 'annotations-000.parquet', loci['position'])
            .join(loci.lazy(), on=['chromosome', 'position'], how='semi')
            .collect()
        ],
        'gpn_star': in_range(directory / 'gpn_star.parquet', loci['position'])
        .filter(pl.col('position').is_in(loci['position'].implode()))
        .collect(),
        'amino_acids': [
            pl
            .scan_parquet(directory / 'amino-acids-000.parquet')
            .join(keys.lazy(), on=['uniprotAccession', 'aminoAcidChange'], how='semi')
            .collect()
        ],
    }
    inputs = handle.chunk(0, annotated=True)
    assert inputs.chromosome == '1'
    assert inputs.rows.equals(frame)
    assert inputs.sequence == handle.sequence()
    records = inputs.records()
    assert records is not None and records.equals(expected['records'])
    assert inputs.transcripts is not None and inputs.transcripts.equals(expected['transcripts'])
    colocated = inputs.colocated()
    assert colocated is not None and colocated.equals(expected['colocated'])
    assert colocated['position'].to_list() == [123]
    assert set(inputs.models()) == {'ENST_1'}
    assert len(inputs.ancestor()) == CONTIG_LENGTH
    assert (inputs.gerp, inputs.alphamissense) == (handle.gerp, handle.alphamissense)
    [annotations] = inputs.annotations(loci)
    assert annotations.equals(expected['annotations'][0])
    assert annotations['variantId'].to_list() == ['1_123_A_G']
    scores = inputs.gpn_star(loci['position'])
    assert scores is not None and scores.equals(expected['gpn_star'])
    assert scores['position'].to_list() == [123]
    [amino_acids] = inputs.amino_acids(keys)
    assert amino_acids.equals(expected['amino_acids'][0])
    assert amino_acids['uniprotAccession'].to_list() == ['P1']


def test_unannotated_chunk_does_not_read_loftee_or_ancestor(tmp_path: Path) -> None:
    handle = prepared(tmp_path, annotations=1, amino_acids=1)
    handle.write(sources(tmp_path))
    for name in ('loftee.pickle', 'ancestor.bin'):
        (handle.directory / 'prepared' / name).unlink()
    inputs = handle.chunk(0, annotated=False)
    assert (inputs.records(), inputs.transcripts, inputs.models(), inputs.ancestor()) == (None, None, {}, b'')
    index = annotate_inputs(inputs, handle.settings)
    assert index['variantId'].to_list() == ['1_123_A_G', '1_200_C_T']
    assert index['transcriptConsequences'].to_list() == [[], []]


def test_chunk_without_optional_inputs_reads_as_no_data(tmp_path: Path) -> None:
    """No `all_vars`, no GPN-Star file and no supplementary source."""
    handle = prepared(tmp_path)
    handle.write(replace(sources(tmp_path), variant_annotations=(), amino_acid_annotations=()))
    inputs = handle.chunk(0, annotated=False)
    loci = inputs.rows.select('chromosome', 'position').unique()
    keys = pl.DataFrame({'uniprotAccession': ['P1'], 'aminoAcidChange': ['A1T']})
    assert inputs.colocated() is None
    assert inputs.gpn_star(loci['position']) is None
    assert inputs.annotations(loci) == []
    assert inputs.amino_acids(keys) == []
    index = annotate_inputs(inputs, handle.settings)
    assert index['rsIds'].to_list() == [[], []]
    assert not {x['method'] for effects in index['variantEffect'] for x in effects} & {'GPN-Star', 'FoldX'}


def test_last_chunk_rebases_its_records(tmp_path: Path) -> None:
    """Five rows in chunks of two: the last chunk holds one row, its record rebased to row 0."""
    variant_ids = ('1_101_A_G', '1_102_A_G', '1_103_A_G', '1_104_A_G', '1_105_A_G')
    handle = prepared(tmp_path, variant_ids=variant_ids)
    handle.write(replace(sources(tmp_path), variant_annotations=(), amino_acid_annotations=()))
    split_records(fastvep_json(tmp_path / 'fastvep.json', [4, 3, 2, 1, 0], [MISSENSE]), 5, handle)
    middle, last = handle.chunk(1, annotated=True), handle.chunk(2, annotated=True)
    assert middle.rows['position'].to_list() == [103, 104]
    middle_records, last_records = middle.records(), last.records()
    assert middle_records is not None and sorted(middle_records['row'].to_list()) == [0, 1]
    assert last.rows['position'].to_list() == [105]
    assert last_records is not None and last_records['row'].to_list() == [0]
    index = annotate_inputs(last, handle.settings)
    assert index['variantId'].to_list() == ['1_105_A_G']
    assert index['transcriptConsequences'].list.len().to_list() == [1]


def test_release_deletes_only_its_chunks_records(tmp_path: Path) -> None:
    handle = prepared(tmp_path)
    split_records(fastvep_json(tmp_path / 'fastvep.json', [0, 1]), 2, replace(handle, settings=ChunkSettings(1)))
    handle.release(1)
    assert not handle.records(1).exists()
    assert handle.records(0).exists()


def test_prepared_annotations_layout_is_position_sorted_row_groups(tmp_path: Path) -> None:
    """Pins the layout of the written file, not the range a reader prunes to: that is invisible to equality.

    Six annotation rows over three positions, given unsorted, in row groups of two: one position each.
    """
    handle = replace(
        prepared(tmp_path, annotations=1, variant_ids=('1_101_A_G', '1_102_A_G', '1_103_A_G')),
        settings=ChunkSettings(chunk_rows=2, row_group_rows=2),
    )
    variant_ids = ['1_103_A_G', '1_101_A_G', '1_102_A_C', '1_103_A_T', '1_101_A_C', '1_102_A_G']
    annotations = tmp_path / 'unsorted.parquet'
    pl.DataFrame(
        {
            'variantId': variant_ids,
            'chromosome': ['1'] * len(variant_ids),
            'position': [int(variant_id.split('_')[1]) for variant_id in variant_ids],
        },
        schema_overrides={'position': pl.Int32},
    ).write_parquet(annotations)
    handle.write(
        replace(sources(tmp_path), variant_annotations=({'1': [str(annotations)]},), amino_acid_annotations=())
    )
    metadata = pq.ParquetFile(handle.directory / 'prepared' / 'annotations-000.parquet').metadata
    column = metadata.schema.names.index('position')
    bounds = [
        (metadata.row_group(i).column(column).statistics.min, metadata.row_group(i).column(column).statistics.max)
        for i in range(metadata.num_row_groups)
    ]
    assert bounds == [(101, 101), (102, 102), (103, 103)]
    loci = pl.DataFrame({'chromosome': ['1'], 'position': [102]}, schema_overrides={'position': pl.Int32})
    [found] = handle.chunk(0, annotated=False).annotations(loci)
    assert found['variantId'].to_list() == ['1_102_A_C', '1_102_A_G']


def test_missing_colocated_file_raises_when_the_chromosome_has_all_vars(tmp_path: Path) -> None:
    located = all_vars(tmp_path / 'all_vars.gz', cache_line('rs123', 123, 123, 'A/G'))
    handle = prepared(tmp_path, all_vars=located)
    handle.write(replace(sources(tmp_path), variant_annotations=(), amino_acid_annotations=()))
    (handle.directory / 'prepared' / 'colocated.parquet').unlink()
    with pytest.raises(FileNotFoundError):
        handle.chunk(0, annotated=False).colocated()
