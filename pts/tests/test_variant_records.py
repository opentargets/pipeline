"""Tests for reading fastVEP output into VEP-shaped records."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pts.schemas.vep import vep_schema
from pts.transformers.variant.alleles import read_chromosome
from pts.transformers.variant.records import fastvep_records, read_records
from pts.transformers.variant.vep import parse_vep

FIXTURES = Path(__file__).parent / 'test_data' / 'fastvep'
SAMPLE = Path(__file__).parent / 'test_data' / 'fastvep_sample.json'


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
def records(rows: pl.DataFrame, transcripts: pl.DataFrame, sequence: bytes) -> pl.DataFrame:
    return fastvep_records(SAMPLE, rows, transcripts, sequence, '22').collect()


@pytest.fixture(scope='module')
def index(records: pl.DataFrame) -> pl.DataFrame:
    return parse_vep(records.lazy()).collect()


def record(records: pl.DataFrame, position: int) -> dict:
    """The record of the fixture variant at `position`."""
    found = records.filter(pl.col('input').str.split('\t').list.get(1) == str(position))
    assert found.height == 1
    return found.row(0, named=True)


def transcript(found: dict, transcript_id: str) -> dict:
    return next(tc for tc in found['transcript_consequences'] if tc['transcript_id'] == transcript_id)


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


def test_records_conform_to_vep_schema(records: pl.DataFrame, rows: pl.DataFrame) -> None:
    assert records.schema == vep_schema
    assert records.height == rows.height
    first = rows.row(0, named=True)
    assert records['input'][0] == '22\t18521430\t{}\tA\tG\t.\t.\t.'.format(first['rsId'])
    assert records['colocated_variants'].to_list() == [[]] * rows.height
    fields = records.select(pl.col('transcript_consequences').explode().struct.unnest())
    assert (
        fields.select('conservation', 'alphamissense', 'lof', 'lof_filter').null_count().row(0) == (fields.height,) * 4
    )


def test_most_severe_uses_vep_ranks(tmp_path: Path, records: pl.DataFrame) -> None:
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
        found = fastvep_records(path, ONE_ROW, NO_TRANSCRIPTS, sequence, '22').collect()
        assert found['most_severe_consequence'].to_list() == [expected]
    # OTVAR_22_20048859_…: acceptor, donor and 5th base in one record.
    assert record(records, 20048859)['most_severe_consequence'] == 'splice_acceptor_variant'
    # 22_19766007_G_A: 5th base.
    assert record(records, 19766007)['most_severe_consequence'] == 'splice_donor_5th_base_variant'


def test_synonymous_amino_acids_are_single(records: pl.DataFrame) -> None:
    assert transcript(record(records, 19723858), 'ENST00000366425')['amino_acids'] == 'P'
    assert {tc['amino_acids'] for tc in record(records, 19760885)['transcript_consequences']} >= {'S'}
    # 22_19723939_C_A, stop gained: a change keeps both halves.
    assert '/' in transcript(record(records, 19723939), 'ENST00000366425')['amino_acids']


def test_intergenic_placeholder_transcript_dropped(records: pl.DataFrame, index: pl.DataFrame) -> None:
    found = record(records, 18521430)
    assert found['transcript_consequences'] == []
    assert found['intergenic_consequences'] == [{'hgvsg': '22:g.18521430A>G'}]
    assert found['most_severe_consequence'] == 'intergenic_variant'
    assert index.filter(pl.col('variantId') == '22_18521430_A_G')['hgvsId'].to_list() == ['22:g.18521430A>G']
    assert record(records, 19723858)['intergenic_consequences'] is None


def test_transcript_ablation(records: pl.DataFrame) -> None:
    found = record(records, 19721911)
    ablated = transcript(found, 'ENST00000366425')
    assert ablated['consequence_terms'] == ['transcript_ablation']
    assert ablated['impact'] == 'HIGH'
    assert (ablated['amino_acids'], ablated['codons'], ablated['protein_end']) == (None, None, None)
    assert ablated['tssdistance'] == -1251
    assert found['most_severe_consequence'] == 'transcript_ablation'
    # Transcripts the deletion only overlaps keep fastVEP's calls.
    others = [tc for tc in found['transcript_consequences'] if tc['transcript_id'] != 'ENST00000366425']
    assert all('transcript_ablation' not in tc['consequence_terms'] for tc in others)


def test_mature_mirna_variant(records: pl.DataFrame) -> None:
    found = record(records, 19963756)
    assert transcript(found, 'ENST00000585066')['consequence_terms'] == ['mature_miRNA_variant']
    assert transcript(found, 'ENST00000361682')['consequence_terms'] == ['missense_variant']
    # 22_19941808_G_T is upstream of the miRNA, not in it.
    assert transcript(record(records, 19941808), 'ENST00000585066')['consequence_terms'] == ['upstream_gene_variant']


def test_ties_ordered_by_transcript_id(index: pl.DataFrame) -> None:
    assert order(index, '22_19941808_G_T')[:2] == ['ENST00000361682', 'ENST00000400521']
    assert order(index, '22_19969950_A_C')[:2] == ['ENST00000263207', 'ENST00000361682']


def test_transcript_fields(records: pl.DataFrame) -> None:
    found = transcript(record(records, 19723858), 'ENST00000366425')
    assert (found['appris'], found['mane_select'], found['canonical']) == ('P2', 'NM_000407.5', 1)
    assert found['swissprot'] == ['P13224']
    assert found['hgvsg'] == '22:g.19723858G>A'
    assert found['tssdistance'] == -319
    # 22_19760999_C_CCCG: an insertion ends before it starts.
    insertion = record(records, 19760999)
    assert transcript(insertion, 'ENST00000649276')['tssdistance'] == -221
    assert transcript(insertion, 'ENST00000816392')['tssdistance'] == 11894
    assert transcript(insertion, 'ENST00000816392')['distance'] == 11893
    assert transcript(insertion, 'ENST00000649276')['hgvsg'] == '22:g.19761013_19761015dup'


def test_unusable_alleles_have_no_hgvsg(tmp_path: Path) -> None:
    path = fastvep_json(tmp_path / 'dots.json', [synthetic([['intergenic_variant']], 'intergenic_variant')])
    path.write_text(path.read_text().replace('"ENST00"', '"-"'))
    dots = ONE_ROW.with_columns(pl.lit('.').alias('alternateAllele'))
    for frame, sequence in ((dots, b'NNNNANNNNN'), (ONE_ROW, b'')):
        found = fastvep_records(path, frame, NO_TRANSCRIPTS, sequence, '22').collect()
        assert found['intergenic_consequences'].to_list() == [[{'hgvsg': None}]]


def test_records_must_match_rows(tmp_path: Path) -> None:
    path = fastvep_json(tmp_path / 'one.json', [synthetic([['intron_variant']], 'intron_variant')])
    with pytest.raises(ValueError, match='1 fastVEP records for 2 rows'):
        fastvep_records(path, pl.concat([ONE_ROW, ONE_ROW]), NO_TRANSCRIPTS, b'', '22')
