"""VEP parsing, fixed schemas and nested annotation semantics."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.variant import variant_schema
from pts.schemas.vep import vep_schema
from pts.transformers.variant_utils.vep import (
    CONSEQUENCE_SCORES,
    SEQUENCE_ONTOLOGY,
    amino_acid_change,
    generate_dbxrefs,
    most_severe_transcript,
    parse_vep,
    read_vep,
    uniprot_accessions,
    variant_effect,
)


@pytest.fixture
def doctest_namespace():
    """Avoid starting Spark for Polars tests."""
    return {}


SAMPLE = Path(__file__).parent / 'test_data' / 'vep_sample.jsonl'


def parsed(**updates: object) -> dict:
    """Parse a sparse input with the same schema used by the JSON reader."""
    record = {'input': '1\t123\t.\tA\tG', 'most_severe_consequence': 'intergenic_variant'}
    record.update(updates)
    return parse_vep(pl.DataFrame([record], schema=vep_schema, strict=False).lazy()).collect().row(0, named=True)


def transcript(**updates: object) -> dict:
    """Canonical protein coding transcript with sensible default annotations."""
    record = {
        'canonical': 1,
        'gene_id': 'ENSG1',
        'transcript_id': 'ENST1',
        'consequence_terms': ['missense_variant'],
        'biotype': 'protein_coding',
        'gene_symbol': 'GENE1',
        'impact': 'MODERATE',
    }
    record.update(updates)
    return record


def test_sample_preserves_count_and_full_schema() -> None:
    frame = read_vep(str(SAMPLE))
    assert isinstance(frame, pl.LazyFrame)
    result = parse_vep(frame).collect()
    assert result.height == frame.collect().height == 4
    assert result.schema == variant_schema
    for row in result.to_dicts():
        transcripts = row['transcriptConsequences']
        assert [t['transcriptIndex'] for t in transcripts] == list(range(1, len(transcripts) + 1))
        assert len({t['targetId'] for t in transcripts}) == len(transcripts)
        assert all(t['isEnsemblCanonical'] for t in transcripts)
        assert all(row[name] is not None for name, dtype in variant_schema.items() if isinstance(dtype, pl.List))


def test_recursive_reader_gzip_and_schema_pinning(tmp_path: Path) -> None:
    nested = tmp_path / 'chunk' / 'nested'
    nested.mkdir(parents=True)
    record = {'input': '1\t123\t.\tA\tG', 'extra': 'discarded'}
    (tmp_path / 'part.jsonl').write_text(json.dumps(record) + '\n')
    with gzip.open(nested / 'part.json.gz', 'wt') as stream:
        stream.write(json.dumps(record) + '\n')
    (tmp_path / '_SUCCESS').touch()
    result = read_vep(str(tmp_path)).collect()
    assert result.height == 2
    assert result.schema == vep_schema
    assert result['transcript_consequences'].null_count() == 2


def test_reader_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='No VEP JSON'):
        read_vep(str(tmp_path))


def test_reader_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / 'part.json'
    path.write_text('')
    with pytest.raises(ValueError, match='empty'):
        read_vep(str(path))


@pytest.mark.parametrize(
    ('amino', 'end', 'expected'), [('A/B', 1, 'A1B'), ('A/B', None, None), (None, 1, None), ('A', 1, None)]
)
def test_amino_acid_change(amino: str | None, end: int | None, expected: str | None) -> None:
    frame = pl.DataFrame({'amino': [amino], 'end': [end]}, schema={'amino': pl.String, 'end': pl.Int64})
    assert frame.select(amino_acid_change(pl.col('amino'), pl.col('end'))).item() == expected


def test_uniprot_accessions_drop_versions_nulls_and_duplicates() -> None:
    frame = pl.DataFrame(
        {'sp': [['P1.2', None, 'P1.3'], None], 'tr': [['P2.1', 'P1.1'], None]},
        schema={'sp': pl.List(pl.String), 'tr': pl.List(pl.String)},
    )
    assert frame.select(uniprot_accessions(pl.col('sp'), pl.col('tr')).alias('ids'))['ids'].to_list() == [
        ['P1', 'P2'],
        [],
    ]


def test_generate_dbxrefs_filters_nulls_and_preserves_duplicates() -> None:
    frame = pl.DataFrame({'ids': [['rs1', None, 'rs1'], None]}, schema={'ids': pl.List(pl.String)})
    assert frame.select(generate_dbxrefs(pl.col('ids'), 'ensembl_variation'))['ids'].to_list() == [
        [{'id': 'rs1', 'source': 'ensembl_variation'}] * 2,
        [],
    ]


def test_effect_selects_highest_score_with_missing_assessment_and_flag() -> None:
    dtype = pl.List(pl.Struct({'score': pl.Float64, 'assessment': pl.String, 'flag': pl.String, 'gene_id': pl.String}))
    frame = pl.DataFrame(
        {
            'transcripts': [
                [
                    {'score': None, 'assessment': 'ignored', 'gene_id': 'g0'},
                    {'score': 0.1, 'assessment': 'low', 'gene_id': 'g1'},
                    {'score': 0.6, 'assessment': None, 'flag': None, 'gene_id': 'g2'},
                    {'score': 0.6, 'assessment': 'tied', 'gene_id': 'g3'},
                ],
                [{'score': None, 'assessment': 'deleterious', 'gene_id': 'g'}],
            ]
        },
        schema={'transcripts': dtype},
        strict=False,
    )
    selected = most_severe_transcript(pl.col('transcripts'), 'score')
    result = frame.select(variant_effect(selected, 'method', 'score', 'assessment', 'flag').alias('effect'))[
        'effect'
    ].to_list()
    assert result[0]['targetId'] == 'g2'
    assert result[0]['assessment'] is None
    assert result[0]['assessmentFlag'] is None
    assert result[0]['score'] == pytest.approx(0.6)
    assert result[1] is None


@pytest.mark.parametrize('transcripts', [None, []])
def test_no_transcripts_has_vep_prediction_and_empty_arrays(transcripts: list | None) -> None:
    result = parsed(transcript_consequences=transcripts)
    assert result['transcriptConsequences'] == []
    assert result['rsIds'] == []
    assert result['dbXrefs'] == []
    assert result['alleleFrequencies'] == []
    assert result['variantEffect'] == [
        {
            'method': 'VEP',
            'assessment': 'intergenic_variant',
            'score': pytest.approx(0.02),
            'assessmentFlag': None,
            'targetId': None,
            'normalisedScore': pytest.approx(0.02),
        }
    ]
    assert result['variantDescription'] == 'Intergenic variant no gene in window'


def test_canonical_filter_sort_and_stable_ties() -> None:
    records = [
        transcript(gene_id='g0', canonical=0, consequence_terms=['stop_gained']),
        transcript(gene_id='g1', distance=100),
        transcript(gene_id='g2', distance=50),
        transcript(gene_id='g3', distance=50),
        transcript(gene_id='g4', consequence_terms=['stop_gained'], distance=1000),
        transcript(gene_id='g5', consequence_terms=['unknown']),
    ]
    result = parsed(transcript_consequences=records)['transcriptConsequences']
    assert [t['targetId'] for t in result] == ['g4', 'g2', 'g3', 'g1']
    assert [t['transcriptIndex'] for t in result] == [1, 2, 3, 4]


def test_cross_references_hgvs_amino_acid_and_transcript_annotations() -> None:
    result = parsed(
        most_severe_consequence='missense_variant',
        transcript_consequences=[
            transcript(
                amino_acids='A/B',
                protein_end=12,
                hgvsg='1:g.123A>G',
                tssdistance=234,
                swissprot=['P1.2'],
                trembl=['P2.1'],
                lof='HC',
            )
        ],
        colocated_variants=[
            {'id': 'rs1', 'var_synonyms': {'OMIM': ['610266.0003'], 'ClinVar': ['RCV123', 'VCV123']}},
            {'id': None},
        ],
    )
    assert result['variantId'] == '1_123_A_G'
    assert result['rsIds'] == ['rs1', None]
    assert result['hgvsId'] == '1:g.123A>G'
    assert result['dbXrefs'] == [
        {'id': 'rs1', 'source': 'ensembl_variation'},
        {'id': '610266#0003', 'source': 'omim'},
        {'id': 'VCV123', 'source': 'clinvar'},
        {'id': '1_123_A_G', 'source': 'protvar'},
    ]
    t = result['transcriptConsequences'][0]
    assert t['aminoAcidChange'] == 'A12B'
    assert t['distanceFromFootprint'] == 0
    assert t['distanceFromTss'] == 234
    assert t['uniprotAccessions'] == ['P1', 'P2']
    assert t['siftPrediction'] is None and t['polyphenPrediction'] is None
    assert (
        result['variantDescription']
        == 'Missense variant overlapping with GENE1, causing amino-acid change: A12B with moderate impact.'
        ' A high-confidence loss-of-function variant by loftee.'
    )


def test_all_predictors_and_first_alpha_missense() -> None:
    result = parsed(
        transcript_consequences=[
            transcript(
                cadd_phred=10,
                polyphen_score=0.2,
                polyphen_prediction='benign',
                sift_score=0.1,
                sift_prediction='deleterious',
                lof='HC',
                lof_filter='flag',
                conservation=3,
                alphamissense={'am_class': 'likely_pathogenic', 'am_pathogenicity': 0.8},
            ),
            transcript(
                gene_id='g2',
                cadd_phred=20,
                polyphen_score=0.9,
                polyphen_prediction='damaging',
                sift_score=0.9,
                sift_prediction='tolerated',
                lof='LC',
                conservation=5,
                alphamissense={'am_class': 'benign', 'am_pathogenicity': 0.9},
            ),
        ]
    )
    effects = {e['method']: e for e in result['variantEffect']}
    assert list(effects) == ['CADD', 'PolyPhen', 'SIFT', 'LOFTEE', 'GERP', 'AlphaMissense', 'VEP']
    assert effects['CADD']['score'] == 20
    assert effects['SIFT']['score'] == pytest.approx(0.9)
    assert effects['PolyPhen']['assessment'] == 'damaging'
    assert effects['LOFTEE']['assessment'] == 'LC'
    assert effects['GERP']['score'] == 5
    assert effects['AlphaMissense']['targetId'] == 'ENSG1'
    assert effects['AlphaMissense']['score'] == pytest.approx(0.8)


def test_hgvs_intergenic_fallback() -> None:
    assert parsed(intergenic_consequences=[{'hgvsg': '1:g.123A>G'}])['hgvsId'] == '1:g.123A>G'


def test_null_first_transcript_hgvs_does_not_fall_back() -> None:
    assert (
        parsed(transcript_consequences=[transcript()], intergenic_consequences=[{'hgvsg': 'fallback'}])['hgvsId']
        is None
    )


def test_non_coding_description_and_distance_formatting() -> None:
    result = parsed(
        most_severe_consequence='non_coding_transcript_variant',
        transcript_consequences=[
            transcript(
                gene_id='lnc',
                gene_symbol=None,
                biotype='lncRNA',
                consequence_terms=['non_coding_transcript_variant'],
                distance=1234,
            ),
            transcript(gene_id='pc', gene_symbol='PC', consequence_terms=['upstream_gene_variant'], distance=2345),
        ],
    )
    assert (
        result['variantDescription'] == 'Non coding transcript variant 1,234 basepair away from lnc, a lncRNA gene.'
        ' The closest protein-coding gene is PC (2,345 basepair away).'
    )


def test_invalid_position_is_dropped_and_empty_frame_supported() -> None:
    frame = pl.DataFrame([{'input': '1\tnot_an_int\t.\tA\tG'}], schema=vep_schema, strict=False)
    result = parse_vep(frame.lazy()).collect()
    assert result.height == 0 and result.schema == variant_schema
    assert parse_vep(pl.DataFrame(schema=vep_schema).lazy()).collect().schema == variant_schema


def test_hashed_variant_keeps_unhashed_protvar_xref() -> None:
    frame = pl.DataFrame(
        [{'input': '1\t123\t.\tA\t' + 'G' * 300, 'transcript_consequences': [transcript()]}],
        schema=vep_schema,
        strict=False,
    )
    result = parse_vep(frame.lazy(), hash_threshold=30).collect().row(0, named=True)
    assert result['variantId'].startswith('OTVAR_1_123_')
    assert result['dbXrefs'][-1]['id'] == '1_123_A_' + 'G' * 300


def test_consequence_mapping_preserves_all_41_terms() -> None:
    assert len(SEQUENCE_ONTOLOGY) == len(CONSEQUENCE_SCORES) == 41
    assert SEQUENCE_ONTOLOGY['missense_variant'] == 'SO_0001583'
    assert CONSEQUENCE_SCORES['missense_variant'] == 0.68
    assert CONSEQUENCE_SCORES['sequence_variant'] == 0.0
