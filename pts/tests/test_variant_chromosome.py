"""Tests for the per-chromosome and per-chunk units of the variant step."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.chromosome import _submit, annotate_inputs, lower_priority, split_records
from pts.transformers.variant.colocated import colocated_variants
from pts.transformers.variant.loftee import TranscriptModel
from pts.transformers.variant.prepared import ChunkSettings, PreparedChromosome, amino_acid_annotations
from pts.transformers.variant.records import iter_records
from pts.transformers.variant.sift import MATRICES
from pts.transformers.variant.transcripts import transcript_fields
from tests.ensembl_fixtures import cells, ddl, write_dump
from tests.test_variant import CONTIG_LENGTH, CORE_TABLES
from tests.test_variant_colocated import all_vars, cache_line
from tests.test_variant_colocated import rows as colocated_rows
from tests.variant_fixtures import fastvep_json, write_bigwig

MISSENSE = {
    'transcript_id': 'ENST_1',
    'gene_id': 'ENSG_1',
    'biotype': 'protein_coding',
    'canonical': 1,
    'consequence_terms': ['missense_variant'],
    'impact': 'MODERATE',
    'amino_acids': 'A/T',
    'codons': 'Gcc/Acc',
    'protein_end': 1,
}
"""The canonical consequence on `ENST_1` of chromosome 1 of `CORE_TABLES`."""


def prepared(directory: Path, chunk_rows: int = 2) -> PreparedChromosome:
    """A handle on chromosome 22 for the units that only read its scratch directory and chunk settings.

    Its input paths are never read.
    """
    return PreparedChromosome(
        chromosome='22',
        directory=directory / 'chromosomes' / '22',
        rows=directory / 'rows.parquet',
        plain_fasta=directory / 'reference.fa',
        span=None,
        all_vars=None,
        alphamissense=str(directory / 'alphamissense.parquet'),
        gerp=str(directory / 'gerp.bw'),
        sift=directory / 'sift.parquet',
        settings=ChunkSettings(chunk_rows=chunk_rows),
        annotation_sources=0,
        amino_acid_sources=0,
    )


def test_split_records_writes_each_record_under_the_chunk_of_its_row(tmp_path: Path) -> None:
    split_records(fastvep_json(tmp_path / 'out.json', [3, 0, 4, 1, 2]), 5, prepared(tmp_path), chunk_bytes=64)
    chunks = sorted((tmp_path / 'chromosomes' / '22' / 'records').iterdir())
    assert [chunk.name for chunk in chunks] == ['00000', '00001', '00002']
    rows = [sorted(pl.read_parquet(chunk / '*.parquet')['row'].to_list()) for chunk in chunks]
    assert rows == [[0, 1], [2, 3], [4]]
    assert len(list(chunks[0].iterdir())) == 2


def test_split_records_fails_on_a_missing_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='2 fastVEP records for 3 rows'):
        split_records(fastvep_json(tmp_path / 'out.json', [0, 2]), 3, prepared(tmp_path))


def test_split_records_fails_on_an_id_repeated_across_slices(tmp_path: Path) -> None:
    path = fastvep_json(tmp_path / 'out.json', [0, 1, 0])
    with pytest.raises(ValueError, match=r'ids are not the ordinals 0\.\.2 '):
        split_records(path, 3, prepared(tmp_path), chunk_bytes=64)


def test_split_records_fails_on_an_id_repeated_within_a_slice(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r'ids are not the ordinals 0\.\.1 '):
        split_records(fastvep_json(tmp_path / 'out.json', [1, 1]), 2, prepared(tmp_path))


@pytest.mark.parametrize('ids', [[0, 2], [-1, 0]])
def test_split_records_fails_on_an_id_out_of_range(tmp_path: Path, ids: list[int]) -> None:
    with pytest.raises(ValueError, match=r'ids are not the ordinals 0\.\.1 '):
        split_records(fastvep_json(tmp_path / 'out.json', ids), 2, prepared(tmp_path))


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


@dataclass(frozen=True)
class MemoryInputs:
    """A chunk's inputs held in memory, with no supplementary annotation, GPN-Star score or SIFT matrix."""

    chromosome: str
    rows: pl.DataFrame
    sequence: bytes
    records: pl.DataFrame | None
    transcripts: pl.DataFrame | None
    colocated: pl.DataFrame | None
    gerp: str
    alphamissense: str
    models: dict[str, TranscriptModel] = field(default_factory=dict)
    ancestor: bytes = b''

    def sift_matrices(self, transcript_ids: pl.Series) -> pl.DataFrame:
        return pl.DataFrame(schema=MATRICES)

    def annotations(self, loci: pl.DataFrame) -> list[pl.DataFrame]:
        return []

    def gpn_star(self, positions: pl.Series) -> pl.DataFrame | None:
        return None

    def amino_acids(self, keys: pl.DataFrame) -> list[pl.DataFrame]:
        return []


def test_annotate_inputs_on_memory_inputs(tmp_path: Path) -> None:
    """Two canonical missense consequences; the first variant also has a colocated id, GERP and AlphaMissense."""
    core = tmp_path / 'core'
    core.mkdir()
    write_dump(
        core,
        {name: [cells(*row) for row in table] for name, (_, table) in CORE_TABLES.items()},
        ddl({name: columns for name, (columns, _) in CORE_TABLES.items()}),
    )
    alphamissense = tmp_path / 'alphamissense.tsv'
    alphamissense.write_text(
        '# AlphaMissense\n'
        '#CHROM\tPOS\tREF\tALT\tgenome\tuniprot_id\ttranscript_id\tprotein_variant\tam_pathogenicity\tam_class\n'
        'chr1\t123\tA\tG\thg38\tP1\tENST_1.1\tA1T\t0.9\tlikely_pathogenic\n'
    )
    frame = colocated_rows('1_123_A_G', '1_200_A_T')
    records = fastvep_json(tmp_path / 'fastvep.json', [0, 1], [MISSENSE], most_severe='missense_variant')
    inputs = MemoryInputs(
        chromosome='1',
        rows=frame,
        sequence=b'A' * CONTIG_LENGTH,
        records=pl.concat(iter_records(records)),
        transcripts=transcript_fields(str(core), '1'),
        colocated=colocated_variants(all_vars(tmp_path / 'all_vars.gz', cache_line('rs123', 123, 123, 'A/G')), frame),
        gerp=write_bigwig(tmp_path / 'gerp.bw', '1', CONTIG_LENGTH, {123: 2.5}),
        alphamissense=str(alphamissense),
    )
    index = annotate_inputs(inputs, ChunkSettings())
    assert index['variantId'].to_list() == ['1_123_A_G', '1_200_A_T']
    assert index['rsIds'].to_list() == [['rs123'], []]
    assert index['transcriptConsequences'].list.len().to_list() == [1, 1]
    assert [sorted({x['method'] for x in effects}) for effects in index['variantEffect']] == [
        ['AlphaMissense', 'GERP', 'VEP'],
        ['VEP'],
    ]
