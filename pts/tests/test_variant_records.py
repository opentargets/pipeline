"""Tests for reading fastVEP output into the variants and consequences frames."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pts.transformers.variant.alleles import read_chromosome
from pts.transformers.variant.colocated import add_colocated
from pts.transformers.variant.records import (
    CONSEQUENCES_SCHEMA,
    FASTVEP_SCHEMA,
    ChunkFrames,
    chunk_frames,
    iter_records,
    unannotated_frames,
)
from pts.transformers.variant.vep import index_rows
from tests.variant_fixtures import with_effects

FIXTURES = Path(__file__).parent / 'test_data' / 'fastvep'
SAMPLE = Path(__file__).parent / 'test_data' / 'fastvep_sample.json'


def read_records(path: Path, chunk_bytes: int = 1 << 20) -> pl.DataFrame:
    """Every record of a fastVEP JSON array in one frame."""
    empty = pl.DataFrame(schema=FASTVEP_SCHEMA).select(
        pl.col('id').cast(pl.Int64).alias('row'), 'most_severe_consequence', 'transcript_consequences'
    )
    return pl.concat([empty, *iter_records(path, chunk_bytes)])


def fastvep_frames(
    path: Path, rows: pl.DataFrame, transcripts: pl.DataFrame, sequence: bytes, chromosome: str
) -> ChunkFrames:
    return chunk_frames(read_records(path), rows, transcripts, sequence, chromosome)


@pytest.fixture(scope='module')
def sequence() -> bytes:
    return read_chromosome(str(FIXTURES / 'chr22_slice.fa.gz'), '22')


@pytest.fixture(scope='module')
def rows() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / 'chr22_slice_rows.parquet')


@pytest.fixture(scope='module')
def transcripts() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / 'chr22_slice_transcripts.parquet')


@pytest.fixture(scope='module')
def frames(rows: pl.DataFrame, transcripts: pl.DataFrame, sequence: bytes) -> ChunkFrames:
    return fastvep_frames(SAMPLE, rows, transcripts, sequence, '22')


@pytest.fixture(scope='module')
def index(frames: ChunkFrames) -> pl.DataFrame:
    return index_of(frames)


def index_of(frames: ChunkFrames) -> pl.DataFrame:
    """The index rows of the frames, without colocated variants or effects."""
    return index_rows(add_colocated(frames.variants, None), with_effects(frames.consequences))


def variant(frames: ChunkFrames, position: int) -> dict:
    """The variants row of the fixture variant at `position`."""
    found = frames.variants.filter(pl.col('position') == position)
    assert found.height == 1
    return found.row(0, named=True)


def transcripts_of(frames: ChunkFrames, position: int) -> list[dict]:
    """The canonical consequences of the fixture variant at `position`, in fastVEP's order."""
    return frames.consequences.filter(pl.col('position') == position).to_dicts()


def transcript(frames: ChunkFrames, position: int, transcript_id: str) -> dict:
    return next(tc for tc in transcripts_of(frames, position) if tc['transcript_id'] == transcript_id)


def order(index: pl.DataFrame, variant_id: str) -> list[str]:
    found = index.filter(pl.col('variantId') == variant_id)['transcriptConsequences'][0]
    return [tc['transcriptId'] for tc in found]


def fastvep_json(path: Path, records: list[dict]) -> Path:
    path.write_text('[\n' + ',\n'.join(json.dumps(item, indent=2) for item in records) + '\n]\n')
    return path


def synthetic(terms: list[list[str]], most_severe: str) -> dict:
    return {
        'id': '0',
        'most_severe_consequence': most_severe,
        'transcript_consequences': [
            {'transcript_id': f'ENST0{i}', 'gene_id': f'ENSG0{i}', 'canonical': 1, 'consequence_terms': t}
            for i, t in enumerate(terms)
        ],
    }


ONE_ROW = pl.DataFrame(
    {'chromosome': ['22'], 'position': [5], 'rsId': ['rs1'], 'referenceAllele': ['A'], 'alternateAllele': ['G']},
    schema_overrides={'position': pl.Int32},
)
NO_TRANSCRIPTS = pl.read_parquet(FIXTURES / 'chr22_slice_transcripts.parquet').head(0)


def test_reads_json_in_slices() -> None:
    whole = read_records(SAMPLE)
    assert whole.height == 50
    assert whole['row'].to_list() == list(range(50))
    assert_frame_equal(read_records(SAMPLE, chunk_bytes=1024), whole)


def test_frames_conform_to_their_schemas(frames: ChunkFrames, rows: pl.DataFrame) -> None:
    variants, consequences = frames
    assert variants.height == rows.height
    assert variants['row'].to_list() == list(range(rows.height))
    assert variants.columns[:5] == ['row', 'chromosome', 'position', 'referenceAllele', 'alternateAllele']
    assert consequences.schema == CONSEQUENCES_SCHEMA
    assert consequences['rank'].min() == 1


def test_variants_frame_keeps_original_and_vep_alleles() -> None:
    row = pl.DataFrame(
        {'chromosome': ['22'], 'position': [100], 'rsId': [None], 'referenceAllele': ['AT'], 'alternateAllele': ['A']},
        schema_overrides={'position': pl.Int32},
    )
    [found] = unannotated_frames(row, '22').variants.to_dicts()
    assert (found['referenceAllele'], found['alternateAllele']) == ('AT', 'A')
    assert (found['vepReference'], found['vepAlternate'], found['vepStart']) == ('T', '-', 101)


def test_most_severe_uses_vep_ranks(tmp_path: Path, frames: ChunkFrames) -> None:
    fifth = fastvep_json(
        tmp_path / 'fifth.json',
        [
            synthetic(
                [['splice_region_variant', 'intron_variant'], ['splice_donor_5th_base_variant', 'intron_variant']],
                'splice_region_variant',
            )
        ],
    )
    acceptor = fastvep_json(
        tmp_path / 'acceptor.json',
        [synthetic([['splice_donor_variant'], ['splice_acceptor_variant']], 'splice_donor_variant')],
    )
    sequence = b'NNNNANNNNN'
    for path, expected in ((fifth, 'splice_donor_5th_base_variant'), (acceptor, 'splice_acceptor_variant')):
        found = fastvep_frames(path, ONE_ROW, NO_TRANSCRIPTS, sequence, '22')
        assert found.variants['most_severe_consequence'].to_list() == [expected]
    # OTVAR_22_20048859_…: acceptor, donor and 5th base in one record.
    assert variant(frames, 20048859)['most_severe_consequence'] == 'splice_acceptor_variant'
    # 22_19766007_G_A: 5th base.
    assert variant(frames, 19766007)['most_severe_consequence'] == 'splice_donor_5th_base_variant'


def test_most_severe_ranks_over_all_transcripts(tmp_path: Path) -> None:
    item = synthetic([['intron_variant'], ['stop_gained']], 'intron_variant')
    path = fastvep_json(tmp_path / 'noncanonical.json', [item])
    core = pl.concat(
        [NO_TRANSCRIPTS, pl.DataFrame({'transcript_id': ['ENST00', 'ENST01'], 'canonical': [1, 0]})], how='diagonal'
    )
    found = fastvep_frames(path, ONE_ROW, core, b'NNNNANNNNN', '22')
    assert found.variants['most_severe_consequence'].to_list() == ['stop_gained']
    assert found.consequences['transcript_id'].to_list() == ['ENST00']


def test_synonymous_amino_acids_are_single(frames: ChunkFrames) -> None:
    assert transcript(frames, 19723858, 'ENST00000366425')['amino_acids'] == 'P'
    assert {tc['amino_acids'] for tc in transcripts_of(frames, 19760885)} >= {'S'}
    # 22_19723939_C_A, stop gained: a change keeps both halves.
    assert '/' in transcript(frames, 19723939, 'ENST00000366425')['amino_acids']


def test_amino_acids_as_fastvep_writes_them(tmp_path: Path) -> None:
    item = synthetic([['synonymous_variant'], ['missense_variant']], 'missense_variant')
    item['transcript_consequences'][0]['amino_acids'] = 'L/L'
    item['transcript_consequences'][1]['amino_acids'] = 'L/P'
    path = fastvep_json(tmp_path / 'amino.json', [item])
    core = pl.concat(
        [NO_TRANSCRIPTS, pl.DataFrame({'transcript_id': ['ENST00', 'ENST01'], 'canonical': [1, 1]})], how='diagonal'
    )
    found = fastvep_frames(path, ONE_ROW, core, b'NNNNANNNNN', '22')
    assert found.consequences['amino_acids'].to_list() == ['L/L', 'L/P']


def test_cds_end_from_fastvep(frames: ChunkFrames) -> None:
    assert transcript(frames, 19723939, 'ENST00000366425')['cds_end'] == 96
    assert transcript(frames, 19723858, 'ENST00000366425')['cds_end'] is not None


def test_intergenic_placeholder_transcript_dropped(frames: ChunkFrames, index: pl.DataFrame) -> None:
    found = variant(frames, 18521430)
    assert transcripts_of(frames, 18521430) == []
    assert found['reported_transcripts'] is False
    assert found['hgvsg'] == '22:g.18521430A>G'
    assert found['most_severe_consequence'] == 'intergenic_variant'
    assert index.filter(pl.col('variantId') == '22_18521430_A_G')['hgvsId'].to_list() == ['22:g.18521430A>G']
    assert variant(frames, 19723858)['reported_transcripts'] is True


def test_transcript_ablation(frames: ChunkFrames) -> None:
    found = variant(frames, 19721911)
    ablated = transcript(frames, 19721911, 'ENST00000366425')
    assert ablated['consequence_terms'] == ['transcript_ablation']
    assert ablated['impact'] == 'HIGH'
    assert (ablated['amino_acids'], ablated['codons'], ablated['protein_end']) == (None, None, None)
    assert ablated['tssdistance'] == -1251
    assert found['most_severe_consequence'] == 'transcript_ablation'
    # Transcripts the deletion only overlaps keep fastVEP's calls.
    others = [tc for tc in transcripts_of(frames, 19721911) if tc['transcript_id'] != 'ENST00000366425']
    assert all('transcript_ablation' not in tc['consequence_terms'] for tc in others)


def test_mature_mirna_variant(frames: ChunkFrames) -> None:
    assert transcript(frames, 19963756, 'ENST00000585066')['consequence_terms'] == ['mature_miRNA_variant']
    assert transcript(frames, 19963756, 'ENST00000361682')['consequence_terms'] == ['missense_variant']
    # 22_19941808_G_T is upstream of the miRNA, not in it.
    assert transcript(frames, 19941808, 'ENST00000585066')['consequence_terms'] == ['upstream_gene_variant']


def test_ties_ordered_by_transcript_id(index: pl.DataFrame) -> None:
    assert order(index, '22_19941808_G_T')[:2] == ['ENST00000361682', 'ENST00000400521']
    assert order(index, '22_19969950_A_C')[:2] == ['ENST00000263207', 'ENST00000361682']


def test_rank_orders_by_score_then_distance_then_id(tmp_path: Path) -> None:
    item = synthetic([['missense_variant'], ['missense_variant'], ['intron_variant']], 'missense_variant')
    for entry, transcript_id in zip(item['transcript_consequences'], ['ENST3', 'ENST1', 'ENST2'], strict=True):
        entry['transcript_id'] = transcript_id
    item['transcript_consequences'][1]['distance'] = 5
    path = fastvep_json(tmp_path / 'rank.json', [item])
    core = pl.concat(
        [NO_TRANSCRIPTS, pl.DataFrame({'transcript_id': ['ENST3', 'ENST1', 'ENST2'], 'canonical': [1, 1, 1]})],
        how='diagonal',
    )
    found = fastvep_frames(path, ONE_ROW, core, b'NNNNANNNNN', '22').consequences
    # Rows stay in fastVEP's order; `rank` does not reorder them.
    assert found['transcript_id'].to_list() == ['ENST3', 'ENST1', 'ENST2']
    assert found['rank'].to_list() == [1, 2, 3]


def test_transcript_fields(frames: ChunkFrames) -> None:
    found = transcript(frames, 19723858, 'ENST00000366425')
    assert (found['appris'], found['mane_select']) == ('P2', 'NM_000407.5')
    assert found['swissprot'] == ['P13224']
    assert found['tssdistance'] == -319
    # 22_19760999_C_CCCG: an insertion ends before it starts.
    assert transcript(frames, 19760999, 'ENST00000649276')['tssdistance'] == -221
    assert transcript(frames, 19760999, 'ENST00000816392')['tssdistance'] == 11894
    assert transcript(frames, 19760999, 'ENST00000816392')['distance'] == 11893
    assert variant(frames, 19723858)['hgvsg'] == '22:g.19723858G>A'
    assert variant(frames, 19760999)['hgvsg'] == '22:g.19761013_19761015dup'


def test_unusable_alleles_have_no_hgvsg(tmp_path: Path) -> None:
    path = fastvep_json(tmp_path / 'dots.json', [synthetic([['intergenic_variant']], 'intergenic_variant')])
    path.write_text(path.read_text().replace('"ENST00"', '"-"'))
    dots = ONE_ROW.with_columns(pl.lit('.').alias('alternateAllele'))
    for frame, sequence in ((dots, b'NNNNANNNNN'), (ONE_ROW, b'')):
        found = fastvep_frames(path, frame, NO_TRANSCRIPTS, sequence, '22')
        assert found.variants['hgvsg'].to_list() == [None]
        assert found.variants['reported_transcripts'].to_list() == [False]


def test_only_non_canonical_transcripts_keep_severity_and_null_hgvs(tmp_path: Path) -> None:
    """The most severe consequence is ranked over every transcript, canonical or not; the HGVS needs a canonical one."""
    item = synthetic([['intron_variant'], ['missense_variant']], 'intron_variant')
    path = fastvep_json(tmp_path / 'non_canonical.json', [item])
    core = pl.concat(
        [NO_TRANSCRIPTS, pl.DataFrame({'transcript_id': ['ENST00', 'ENST01'], 'canonical': [0, 0]})], how='diagonal'
    )
    found = fastvep_frames(path, ONE_ROW, core, b'NNNNANNNNN', '22')
    assert found.consequences.height == 0
    [row] = index_of(found).to_dicts()
    assert row['mostSevereConsequenceId'] == 'SO_0001583'
    assert [x['assessment'] for x in row['variantEffect']] == ['missense_variant']
    assert row['transcriptConsequences'] == []
    assert row['hgvsId'] is None


def test_canonical_from_the_core(tmp_path: Path) -> None:
    item = synthetic([['intron_variant'], ['intron_variant']], 'intron_variant')
    item['transcript_consequences'][0]['canonical'] = None
    path = fastvep_json(tmp_path / 'canonical.json', [item])
    core = pl.concat(
        [NO_TRANSCRIPTS, pl.DataFrame({'transcript_id': ['ENST00', 'ENST01'], 'canonical': [1, 0]})], how='diagonal'
    )
    found = fastvep_frames(path, ONE_ROW, core, b'NNNNANNNNN', '22')
    assert found.consequences['transcript_id'].to_list() == ['ENST00']
    assert found.variants['reported_transcripts'].to_list() == [True]


def test_unannotated_frames_are_intergenic() -> None:
    found = unannotated_frames(ONE_ROW, 'Y')
    assert found.consequences.schema == CONSEQUENCES_SCHEMA
    assert found.consequences.height == 0
    [row] = found.variants.to_dicts()
    assert row['hgvsg'] is None
    assert (row['most_severe_consequence'], row['reported_transcripts']) == ('intergenic_variant', False)
    assert index_of(found)['hgvsId'].to_list() == [None]
