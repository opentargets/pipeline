"""Variant index: VEP parsing, annotation merging and the end-to-end step."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import stat
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pyBigWig  # ty: ignore[unresolved-import] -- compiled module without stubs
import pytest
import yaml
from loguru import logger
from pydantic import ValidationError

from pts.schemas.variant import variant_schema
from pts.schemas.vep import vep_schema
from pts.tasks.transform import TransformSpec
from pts.transformers.utils.variant import hash_long_variant_ids
from pts.transformers.variant import VariantDestination, VariantSettings, VariantSource, variant
from pts.transformers.variant.consequences import CONSEQUENCE_SCORES, SEQUENCE_ONTOLOGY
from pts.transformers.variant.gpn_star import GpnStarCutoffs
from pts.transformers.variant.index import (
    add_annotation,
    amino_acid_keys,
    annotate_with_amino_acid_consequences,
    annotation_parts_by_chromosome,
    normalise_score,
    read_variant_annotations,
)
from pts.transformers.variant.rows import collect_variants
from pts.transformers.variant.vep import amino_acid_change, generate_dbxrefs, parse_vep, uniprot_accessions
from tests.ensembl_fixtures import cells, ddl, write_dump
from tests.test_variant_colocated import cache_line
from tests.test_variant_loftee import write_bgzip_fasta

SAMPLE = Path(__file__).parent / 'test_data' / 'vep_sample.jsonl'
EFFECTS = variant_schema['variantEffect']


def vep_records(*records: dict) -> pl.LazyFrame:
    """Raw VEP records read through the same schema as the JSON reader."""
    lines = '\n'.join(json.dumps(record) for record in records).encode()
    return pl.read_ndjson(io.BytesIO(lines), schema=vep_schema).lazy()


def parsed(hash_threshold: int = 300, **updates: object) -> dict:
    record = {'input': '1\t123\t.\tA\tG', 'most_severe_consequence': 'intergenic_variant', **updates}
    [row] = parse_vep(vep_records(record), hash_threshold).collect().to_dicts()
    return row


def transcript(**updates: object) -> dict:
    """Canonical protein coding transcript with sensible default annotations."""
    return {
        'canonical': 1,
        'gene_id': 'ENSG1',
        'transcript_id': 'ENST1',
        'consequence_terms': ['missense_variant'],
        'biotype': 'protein_coding',
        'gene_symbol': 'GENE1',
        'impact': 'MODERATE',
        **updates,
    }


def effect(method: str = 'VEP', score: float | None = 1.0, assessment: str | None = None) -> dict:
    return {
        'method': method,
        'assessment': assessment,
        'score': score,
        'assessmentFlag': None,
        'targetId': None,
        'normalisedScore': None,
    }


def index_rows(rows: list[dict]) -> pl.DataFrame:
    """Variant index rows; fields not given are null, arrays not given are empty."""
    arrays = {name for name, dtype in variant_schema.items() if isinstance(dtype, pl.List)}
    full = [{name: row.get(name, [] if name in arrays else None) for name in variant_schema.names()} for row in rows]
    return pl.DataFrame(full, schema=variant_schema)


# --- VEP reader ---------------------------------------------------------------------------------


def test_sample_preserves_count_and_full_schema():
    result = parse_vep(pl.scan_ndjson(SAMPLE, schema=vep_schema)).collect()
    assert result.schema == variant_schema
    assert result.height == 4
    for row in result.to_dicts():
        transcripts = row['transcriptConsequences']
        assert [x['transcriptIndex'] for x in transcripts] == list(range(1, len(transcripts) + 1))
        assert len({x['targetId'] for x in transcripts}) == len(transcripts)
        assert all(x['isEnsemblCanonical'] for x in transcripts)
        assert all(row[name] is not None for name, dtype in variant_schema.items() if isinstance(dtype, pl.List))


# --- column helpers -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('amino', 'end', 'expected'), [('A/B', 1, 'A1B'), ('A/B', None, None), (None, 1, None), ('A', 1, None)]
)
def test_amino_acid_change(amino, end, expected):
    frame = pl.DataFrame({'amino': [amino], 'end': [end]}, schema={'amino': pl.String, 'end': pl.Int64})
    assert frame.select(amino_acid_change(pl.col('amino'), pl.col('end'))).item() == expected


def test_uniprot_accessions_drop_versions_nulls_and_duplicates():
    frame = pl.DataFrame(
        {'sp': [['P1.2', None, 'P1.3'], None], 'tr': [['P2.1', 'P1.1'], None]},
        schema={'sp': pl.List(pl.String), 'tr': pl.List(pl.String)},
    )
    assert frame.select(uniprot_accessions(pl.col('sp'), pl.col('tr'))).to_series().to_list() == [['P1', 'P2'], []]


def test_generate_dbxrefs_filters_nulls_and_preserves_duplicates():
    frame = pl.DataFrame({'ids': [['rs1', None, 'rs1'], None]}, schema={'ids': pl.List(pl.String)})
    result = frame.select(generate_dbxrefs(pl.col('ids'), 'ensembl_variation')).to_series().to_list()
    assert result == [[{'id': 'rs1', 'source': 'ensembl_variation'}] * 2, []]


def test_consequence_mapping_preserves_all_41_terms():
    assert len(SEQUENCE_ONTOLOGY) == len(CONSEQUENCE_SCORES) == 41
    assert SEQUENCE_ONTOLOGY['missense_variant'] == 'SO_0001583'
    assert CONSEQUENCE_SCORES['sequence_variant'] == 0.0


def test_incomplete_terminal_codon_uses_the_sequence_ontology_term():
    """Unlike Gentropy's misspelt label, VEP's term is mapped and scored."""
    assert 'incomplete_labelinal_codon_variant' not in SEQUENCE_ONTOLOGY.keys() | CONSEQUENCE_SCORES.keys()
    row = parsed(most_severe_consequence='incomplete_terminal_codon_variant')
    assert row['mostSevereConsequenceId'] == 'SO_0001626'
    assert row['variantEffect'][0]['score'] == pytest.approx(0.54)


# --- parsing ------------------------------------------------------------------------------------


@pytest.mark.parametrize('transcripts', [None, []])
def test_no_transcripts_has_vep_prediction_and_empty_arrays(transcripts):
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


def test_canonical_filter_sort_and_stable_ties():
    records = [
        transcript(gene_id='g0', canonical=0, consequence_terms=['stop_gained']),
        transcript(gene_id='g1', distance=100),
        transcript(gene_id='g2', distance=50),
        transcript(gene_id='g3', distance=50),
        transcript(gene_id='g4', consequence_terms=['stop_gained'], distance=1000),
        transcript(gene_id='g5', consequence_terms=['unknown']),
    ]
    result = parsed(transcript_consequences=records)['transcriptConsequences']
    assert [x['targetId'] for x in result] == ['g4', 'g2', 'g3', 'g1']
    assert [x['transcriptIndex'] for x in result] == [1, 2, 3, 4]


def test_cross_references_hgvs_amino_acid_and_transcript_annotations():
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
            {'id': 'rs1', 'var_synonyms': {'OMIM': [610266.0003], 'ClinVar': ['RCV123', 'VCV123']}},
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
    first = result['transcriptConsequences'][0]
    assert first['aminoAcidChange'] == 'A12B'
    assert first['distanceFromFootprint'] == 0
    assert first['distanceFromTss'] == 234
    assert first['uniprotAccessions'] == ['P1', 'P2']
    assert result['variantDescription'] == (
        'Missense variant overlapping with GENE1, causing amino-acid change: A12B with moderate impact.'
        ' A high-confidence loss-of-function variant by loftee.'
    )


def test_omim_ids_keep_their_four_digit_suffix():
    """VEP writes OMIM IDs as JSON numbers, sometimes without the suffix's trailing zero."""
    record = (
        json
        .dumps({
            'input': '1\t123\t.\tA\tG',
            'most_severe_consequence': 'intergenic_variant',
            'colocated_variants': [{'id': 'rs1', 'var_synonyms': {'OMIM': ['ENTRY_A', 'ENTRY_B', 'ENTRY_C']}}],
        })
        .replace('"ENTRY_A"', '600234.0010')
        .replace('"ENTRY_B"', '171834.0100')
        .replace('"ENTRY_C"', '159530.001')
    )
    [row] = parse_vep(pl.read_ndjson(io.BytesIO(record.encode()), schema=vep_schema).lazy()).collect().to_dicts()
    assert [x['id'] for x in row['dbXrefs'] if x['source'] == 'omim'] == ['600234#0010', '171834#0100', '159530#0010']


def test_all_predictors_without_sift_or_polyphen():
    result = parsed(
        transcript_consequences=[
            transcript(
                lof='HC',
                lof_filter='flag',
                conservation=3,
                alphamissense={'am_class': 'likely_pathogenic', 'am_pathogenicity': 0.8},
            ),
            transcript(
                gene_id='g2',
                lof='LC',
                conservation=5,
                alphamissense={'am_class': 'benign', 'am_pathogenicity': 0.9},
            ),
        ],
    )
    effects = {x['method']: x for x in result['variantEffect']}
    assert list(effects) == ['LOFTEE', 'GERP', 'AlphaMissense', 'VEP']
    assert effects['LOFTEE']['assessment'] == 'LC'
    assert effects['GERP']['score'] == 5
    assert effects['AlphaMissense']['targetId'] == 'ENSG1'
    assert effects['AlphaMissense']['score'] == pytest.approx(0.8)


def test_predictor_comes_from_the_first_transcript_with_the_highest_score():
    """Ties keep the index's transcript order, null scores never win, and no score means no effect."""
    result = parsed(
        transcript_consequences=[
            transcript(gene_id='none', transcript_id='ENST0'),
            transcript(gene_id='first', transcript_id='ENST1', conservation=5),
            transcript(gene_id='second', transcript_id='ENST2', conservation=5),
            transcript(gene_id='lower', transcript_id='ENST3', conservation=1),
        ]
    )
    effects = {x['method']: x for x in result['variantEffect']}
    assert effects['GERP']['targetId'] == 'first'
    assert 'LOFTEE' not in effects


def test_loftee_ties_keep_the_first_transcript():
    """Spark ranks LOFTEE by its string value, so `LC` beats `HC`; ties keep the index's order."""
    result = parsed(
        transcript_consequences=[
            transcript(gene_id='high_confidence', transcript_id='ENST0', lof='HC'),
            transcript(gene_id='second', transcript_id='ENST2', lof='LC', lof_filter='B'),
            transcript(gene_id='first', transcript_id='ENST1', lof='LC', lof_filter='A'),
        ]
    )
    [loftee] = [x for x in result['variantEffect'] if x['method'] == 'LOFTEE']
    assert (loftee['targetId'], loftee['assessment'], loftee['assessmentFlag']) == ('first', 'LC', 'A')


RANKED = [
    transcript(gene_id='severe', transcript_id='ENST9', consequence_terms=['stop_gained'], distance=0),
    transcript(gene_id='near', transcript_id='ENST8', consequence_terms=['upstream_gene_variant'], distance=10),
    transcript(gene_id='near_id', transcript_id='ENST9', consequence_terms=['upstream_gene_variant'], distance=10),
    transcript(gene_id='far', transcript_id='ENST1', consequence_terms=['upstream_gene_variant'], distance=900),
]
"""Canonical transcripts in the index's order: consequence score, then distance, then id."""


@pytest.mark.parametrize('order', [[0, 1, 2, 3], [3, 2, 1, 0], [2, 3, 0, 1], [1, 0, 3, 2]])
def test_tied_effects_target_the_top_ranked_transcript(order):
    """GERP, AlphaMissense and LOFTEE ties go to the transcript ranked first in `transcriptConsequences`."""
    alphamissense = {'am_class': 'benign', 'am_pathogenicity': 0.1}
    tied = [{**RANKED[i], 'conservation': 2.5, 'alphamissense': alphamissense, 'lof': 'LC'} for i in order]
    result = parsed(transcript_consequences=tied)
    effects = {x['method']: x['targetId'] for x in result['variantEffect']}
    assert [x['targetId'] for x in result['transcriptConsequences']] == ['severe', 'near', 'near_id', 'far']
    assert effects == {'GERP': 'severe', 'AlphaMissense': 'severe', 'LOFTEE': 'severe', 'VEP': None}


@pytest.mark.parametrize('order', [[0, 1, 2], [2, 1, 0], [1, 2, 0]])
def test_tied_effects_ignore_transcripts_without_the_score(order):
    """A top-ranked transcript without a GERP score does not win; the next tied one in the index's order does."""
    tied = [
        {**RANKED[1], 'conservation': 4.0},
        {**RANKED[2], 'conservation': 4.0},
        {**RANKED[3], 'conservation': 4.0},
    ]
    result = parsed(transcript_consequences=[RANKED[0], *(tied[i] for i in order)])
    assert {x['method']: x['targetId'] for x in result['variantEffect']}['GERP'] == 'near'


def test_hgvs_intergenic_fallback():
    assert parsed(intergenic_consequences=[{'hgvsg': '1:g.123A>G'}])['hgvsId'] == '1:g.123A>G'


def test_null_first_transcript_hgvs_does_not_fall_back():
    result = parsed(transcript_consequences=[transcript()], intergenic_consequences=[{'hgvsg': 'fallback'}])
    assert result['hgvsId'] is None


def test_non_coding_description_and_distance_formatting():
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
    assert result['variantDescription'] == (
        'Non coding transcript variant 1,234 basepair away from lnc, a lncRNA gene.'
        ' The closest protein-coding gene is PC (2,345 basepair away).'
    )


def test_invalid_position_is_dropped():
    assert parse_vep(vep_records({'input': '1\tnot_an_int\t.\tA\tG'})).collect().height == 0


def test_hashed_variant_keeps_unhashed_protvar_xref():
    alternate = 'G' * 300
    result = parsed(hash_threshold=30, input=f'1\t123\t.\tA\t{alternate}', transcript_consequences=[transcript()])
    assert result['variantId'].startswith('OTVAR_1_123_')
    assert result['dbXrefs'][-1]['id'] == f'1_123_A_{alternate}'


# --- variant index operations -------------------------------------------------------------------


def test_identifier_hashing():
    frame = pl.DataFrame(
        {'variantId': ['short', 'abcdefghij', 'missing', None], 'chromosome': ['1', '2', None, None]},
        schema={'variantId': pl.String, 'chromosome': pl.String},
    ).with_columns(position=pl.Series([1, 2, 3, None], dtype=pl.Int32))
    hashed = frame.select(hash_long_variant_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), 10))
    assert hashed.to_series().to_list() == [
        'short',
        'OTVAR_2_2_' + hashlib.md5(b'abcdefghij').hexdigest(),
        'OTVAR_' + hashlib.md5(b'missing').hexdigest(),
        None,
    ]


def test_identifier_hashing_takes_both_coordinates_or_neither():
    with pytest.raises(ValueError, match='both chromosome and position'):
        hash_long_variant_ids(pl.col('variantId'), pl.col('chromosome'))


def _rescale(score: float, low: float, high: float, minimum: float, maximum: float) -> float:
    """A single precision score rescaled in double precision."""
    return float((np.float64(np.float32(score)) - low) / (high - low) * (maximum - minimum) + minimum)


@pytest.mark.parametrize(
    ('method', 'score', 'assessment', 'expected'),
    [
        ('LOFTEE', None, 'HC', 1.0),
        ('LOFTEE', None, 'LC', 0.85),
        ('LOFTEE', 1.0, 'other', None),
        ('AlphaMissense', 0.0, None, -1.0),
        ('AlphaMissense', 0.3, None, _rescale(0.3, 0.06, 0.77, -0.25, 0.25)),
        ('AlphaMissense', 0.77, None, _rescale(0.77, 0.06, 0.77, -0.25, 0.25)),
        ('VEP', 0.4, None, float(np.float32(0.4))),
        ('GERP', -5.0, None, 0.0),
        ('GERP', 2.0, None, 0.75),
        ('GERP', -2.31, None, _rescale(-2.31, -3, 0, 0.0, 0.25)),
        ('GERP', 7.0, None, 1.0),
        ('GERP', None, None, None),
    ],
)
def test_normalisation(method, score, assessment, expected):
    frame = pl.DataFrame(
        {'score': [score], 'assessment': [assessment]}, schema={'score': pl.Float32, 'assessment': pl.String}
    )
    assert frame.select(normalise_score(method, pl.col('score'), pl.col('assessment'))).item() == expected


@pytest.mark.parametrize('method', ['FoldX', 'PolyPhen', 'unknown'])
def test_normalisation_rejects_methods_vep_does_not_produce(method):
    with pytest.raises(ValueError, match='No normalisation'):
        normalise_score(method, pl.col('score'), pl.col('assessment'))


def test_read_variant_annotations_inserts_missing_columns_and_keeps_indexed_loci(tmp_path):
    directory = tmp_path / 'gnomad' / 'chromosome=1'
    directory.mkdir(parents=True)
    long_id = '1_10_A_' + 'T' * 300
    pl.DataFrame(
        {'variantId': ['1_10_A_T', long_id, '1_99_A_T'], 'chromosome': ['1', '1', '1']},
    ).with_columns(position=pl.Series([10, 10, 99], dtype=pl.Int32)).write_parquet(directory / 'part.parquet')
    loci = pl.LazyFrame({'chromosome': ['1']}).with_columns(position=pl.lit(10, dtype=pl.Int32))
    [parts] = annotation_parts_by_chromosome(str(tmp_path / 'gnomad')).values()
    result = read_variant_annotations(parts, 300, loci).collect().sort('variantId')
    assert result.schema == variant_schema
    assert result['variantId'].to_list() == ['1_10_A_T', 'OTVAR_1_10_' + hashlib.md5(long_id.encode()).hexdigest()]
    assert result['variantEffect'].to_list() == [[], []]
    assert result['variantDescription'].to_list() == [None, None]


def test_annotation_parts_require_parquet_parts(tmp_path):
    with pytest.raises(ValueError, match=r'no \*\*/\*\.parquet files found'):
        annotation_parts_by_chromosome(str(tmp_path))


def test_annotation_parts_are_grouped_by_the_chromosomes_they_hold(tmp_path):
    for name, chromosomes in [('a', ['1', '1']), ('b', ['1', '2']), ('c', [None, '3'])]:
        pl.DataFrame({'chromosome': chromosomes}, schema={'chromosome': pl.String}).write_parquet(
            tmp_path / f'{name}.parquet'
        )
    located = annotation_parts_by_chromosome(str(tmp_path))
    assert {chromosome: [Path(part).name for part in parts] for chromosome, parts in located.items()} == {
        '1': ['a.parquet', 'b.parquet'],
        '2': ['b.parquet'],
        '3': ['c.parquet'],
    }


def test_annotation_parts_tolerate_parts_with_different_columns(tmp_path):
    pl.DataFrame({'chromosome': ['1']}).write_parquet(tmp_path / 'a.parquet')
    pl.DataFrame({'chromosome': ['2'], 'extra': [1]}).write_parquet(tmp_path / 'b.parquet')
    pl.DataFrame({'extra': [1], 'chromosome': ['1']}).write_parquet(tmp_path / 'c.parquet')
    located = annotation_parts_by_chromosome(str(tmp_path))
    assert {chromosome: [Path(part).name for part in parts] for chromosome, parts in located.items()} == {
        '1': ['a.parquet', 'c.parquet'],
        '2': ['b.parquet'],
    }


def test_add_annotation_removes_repeats_only_where_an_annotation_joined():
    """Spark's union removes repeats in the index's array; with nothing joined it keeps them."""
    repeated = [effect('VEP'), effect('VEP')]
    index = index_rows([
        {'variantId': 'joined', 'variantEffect': repeated},
        {'variantId': 'alone', 'variantEffect': repeated},
    ])
    rows = {row['variantId']: row for row in add_annotation(index, index_rows([{'variantId': 'joined'}])).to_dicts()}
    assert len(rows['joined']['variantEffect']) == 1
    assert len(rows['alone']['variantEffect']) == 2
    assert [row['variantId'] for row in add_annotation(index, index_rows([])).to_dicts()] == ['joined', 'alone']


def test_add_annotation_unions_arrays_and_prefers_index_values():
    index = index_rows([
        {
            'variantId': 'v1',
            'chromosome': '1',
            'rsIds': ['rs1'],
            'variantEffect': [effect('VEP')],
            'variantDescription': 'Index.',
        },
        {'variantId': 'v2', 'chromosome': '1'},
    ])
    annotations = index_rows([
        {
            'variantId': 'v1',
            'chromosome': 'X',
            'rsIds': ['rs1', 'rs2'],
            'variantEffect': [effect('LossOfFunctionCuration', None, 'lof')],
            'variantDescription': 'Curated.',
            'alleleFrequencies': [{'populationName': 'nfe', 'alleleFrequency': 0.1}],
        },
        {'variantId': 'unrelated', 'rsIds': ['rs9']},
    ])
    rows = {row['variantId']: row for row in add_annotation(index, annotations).to_dicts()}
    assert set(rows) == {'v1', 'v2'}
    assert rows['v1']['rsIds'] == ['rs1', 'rs2']
    assert [x['method'] for x in rows['v1']['variantEffect']] == ['VEP', 'LossOfFunctionCuration']
    assert rows['v1']['chromosome'] == '1'
    assert rows['v1']['variantDescription'] == 'Index. Curated.'
    assert rows['v1']['alleleFrequencies'] == [{'populationName': 'nfe', 'alleleFrequency': 0.1}]
    assert rows['v2']['variantDescription'] == ''


def test_add_annotation_sorts_struct_arrays_descending_as_spark():
    """Spark's `safe_array_union` sorts both sides descending, null fields last, before the union."""
    index = index_rows([{'variantId': 'v1', 'variantEffect': [effect('A'), effect('C', None), effect('C', 2.0)]}])
    annotations = index_rows([{'variantId': 'v1', 'variantEffect': [effect('B')]}])
    [row] = add_annotation(index, annotations).to_dicts()
    assert [(x['method'], x['score']) for x in row['variantEffect']] == [
        ('C', 2.0),
        ('C', None),
        ('A', 1.0),
        ('B', 1.0),
    ]


def _consequence(change: str | None, accessions: list[str] | None) -> dict:
    inner = variant_schema['transcriptConsequences'].inner  # ty: ignore[unresolved-attribute]
    return {**dict.fromkeys(inner.to_schema()), 'aminoAcidChange': change, 'uniprotAccessions': accessions}


def test_amino_acid_annotation_keeps_variants_without_accessions_and_ignores_unrelated():
    index = index_rows([
        {
            'variantId': 'annotated',
            'variantEffect': [effect('VEP')],
            'transcriptConsequences': [_consequence('A1T', ['P1', 'P2'])],
        },
        {
            'variantId': 'no_accession',
            'variantEffect': [effect('VEP')],
            'transcriptConsequences': [_consequence('A1T', [])],
        },
        {
            'variantId': 'null_accession',
            'variantEffect': [effect('VEP')],
            'transcriptConsequences': [_consequence('A1T', None)],
        },
    ])
    annotations = pl.DataFrame(
        {
            'uniprotAccession': ['P2', 'P1', 'P9'],
            'aminoAcidChange': ['A1T', 'C9G', 'A1T'],
            'variantEffect': [[effect('FoldX', 1.0)], [effect('FoldX', 2.0)], [effect('FoldX', 3.0)]],
        },
        schema={'uniprotAccession': pl.String, 'aminoAcidChange': pl.String, 'variantEffect': EFFECTS},
    )
    rows = {
        row['variantId']: [x['method'] for x in row['variantEffect']]
        for row in annotate_with_amino_acid_consequences(index, annotations).to_dicts()
    }
    assert rows == {'annotated': ['VEP', 'FoldX'], 'no_accession': ['VEP'], 'null_accession': ['VEP']}


def test_amino_acid_keys_cover_every_transcript_change():
    index = index_rows([
        {'variantId': 'v1', 'transcriptConsequences': [_consequence('A1T', ['P1', 'P2']), _consequence('C2G', [])]},
        {'variantId': 'v2', 'transcriptConsequences': [_consequence(None, ['P3'])]},
    ])
    keys = amino_acid_keys(index.lazy()).collect().sort('uniprotAccession').rows()
    assert keys == [('P1', 'A1T'), ('P2', 'A1T')]


# --- step and configuration ---------------------------------------------------------------------

FASTVEP_FIXTURES = Path(__file__).parent / 'test_data' / 'fastvep'
FASTVEP_SAMPLE = Path(__file__).parent / 'test_data' / 'fastvep_sample.json'
SIFT_FIXTURES = Path(__file__).parent / 'test_data' / 'sift'
CONTIG_LENGTH = 1000

#: A stand-in for fastVEP. It replays `fastvep_sample.json` for a VCF of chromosome 22, exits with
#: an error for a VCF holding chromosome FAIL, and otherwise reports one canonical missense
#: consequence on `ENST_<chromosome>` per VCF line.
FAKE_FASTVEP = r"""#!PYTHON
import json, shutil, sys

args = sys.argv[1:]
out = args[args.index('-o') + 1]
if args[0] == 'cache':
    open(out, 'w').close()
    sys.exit(0)
lines = [line.split('\t') for line in open(args[args.index('--input') + 1]) if not line.startswith('#')]
chromosomes = {line[0] for line in lines}
if FAIL in chromosomes:
    sys.exit('fastVEP failed')
if chromosomes == {'22'}:
    shutil.copyfile(SAMPLE, out)
    sys.exit(0)
records = [
    {
        'id': line[2],
        'most_severe_consequence': 'missense_variant',
        'transcript_consequences': [
            {
                'transcript_id': f'ENST_{line[0]}',
                'gene_id': f'ENSG_{line[0]}',
                'biotype': 'protein_coding',
                'canonical': 1,
                'consequence_terms': ['missense_variant'],
                'impact': 'MODERATE',
                'amino_acids': 'A/T',
                'codons': 'Gcc/Acc',
                'protein_end': 1,
            }
        ],
    }
    for line in lines
]
with open(out, 'w') as handle:
    handle.write('[\n' + ',\n'.join(json.dumps(record, indent=2) for record in records) + '\n]\n')
"""

#: Ensembl core tables: one canonical protein coding transcript on each of chromosomes 1 and MT.
CORE_TABLES: dict[str, tuple[list[str], list[tuple[object, ...]]]] = {
    'coord_system': (['coord_system_id', 'species_id', 'name'], [(1, 1, 'chromosome')]),
    'seq_region': (
        ['seq_region_id', 'name', 'coord_system_id', 'length'],
        [(10, '1', 1, CONTIG_LENGTH), (20, 'MT', 1, CONTIG_LENGTH)],
    ),
    'gene': (
        [
            'gene_id', 'biotype', 'seq_region_id', 'seq_region_start', 'seq_region_end', 'seq_region_strand',
            'display_xref_id', 'description', 'canonical_transcript_id', 'stable_id',
        ],
        [
            (1, 'protein_coding', 10, 100, 900, 1, 501, None, 100, 'ENSG_1'),
            (2, 'protein_coding', 20, 100, 900, 1, 502, None, 200, 'ENSG_MT'),
        ],
    ),
    'transcript': (
        [
            'transcript_id', 'gene_id', 'seq_region_id', 'seq_region_start', 'seq_region_end',
            'seq_region_strand', 'biotype', 'canonical_translation_id', 'stable_id',
        ],
        [
            (100, 1, 10, 100, 900, 1, 'protein_coding', 1000, 'ENST_1'),
            (200, 2, 20, 100, 900, 1, 'protein_coding', 2000, 'ENST_MT'),
        ],
    ),
    'translation': (
        ['translation_id', 'transcript_id', 'seq_start', 'start_exon_id', 'seq_end', 'end_exon_id', 'stable_id'],
        [(1000, 100, 1, 1, 801, 1, 'ENSP_1'), (2000, 200, 1, 2, 801, 2, 'ENSP_MT')],
    ),
    'exon': (
        ['exon_id', 'seq_region_id', 'seq_region_start', 'seq_region_end', 'seq_region_strand', 'stable_id'],
        [(1, 10, 100, 900, 1, 'ENSE_1'), (2, 20, 100, 900, 1, 'ENSE_MT')],
    ),
    'exon_transcript': (['exon_id', 'transcript_id', 'rank'], [(1, 100, 1), (2, 200, 1)]),
    'attrib_type': (['attrib_type_id', 'code', 'name'], [(554, 'is_canonical', 'Ensembl Canonical')]),
    'transcript_attrib': (['transcript_id', 'attrib_type_id', 'value'], [(100, 554, '1'), (200, 554, '1')]),
    'external_db': (['external_db_id', 'db_name'], [(2200, 'Uniprot/SWISSPROT'), (1100, 'HGNC')]),
    'xref': (
        ['xref_id', 'external_db_id', 'dbprimary_acc', 'display_label'],
        [(501, 1100, 'HGNC:1', 'GENE1'), (502, 1100, 'HGNC:2', 'GENEMT'), (601, 2200, 'P1', 'P1')],
    ),
    'object_xref': (
        ['object_xref_id', 'ensembl_id', 'ensembl_object_type', 'xref_id'],
        [(1, 1000, 'Translation', 601)],
    ),
    'analysis': (['analysis_id', 'created', 'logic_name'], [(30, None, 'signalp')]),
    'protein_feature': (['protein_feature_id', 'translation_id', 'hit_name', 'analysis_id'], [(1, 1000, 'x', 30)]),
}  # fmt: skip


def _write_fasta(path: Path) -> None:
    """Contigs 1, 2 and MT of CONTIG_LENGTH bases around the chromosome 22 slice, as gzip members."""

    def contig(name: str, base: str) -> bytes:
        lines = [base * 60] * (CONTIG_LENGTH // 60) + [base * (CONTIG_LENGTH % 60)]
        return gzip.compress(f'>{name}\n'.encode() + '\n'.join(lines).encode() + b'\n')

    slice_ = (FASTVEP_FIXTURES / 'chr22_slice.fa.gz').read_bytes()
    path.write_bytes(contig('1', 'A') + contig('2', 'C') + slice_ + contig('MT', 'G'))


def _write_vep_cache(path: Path, directory: Path) -> None:
    """An indexed VEP cache with `all_vars.gz` for chromosomes 1 and 22, none for MT."""
    lines = {
        '1': cache_line('rs123', 123, 123, 'A/G'),
        '22': cache_line('rs2542835', 18521430, 18521430, 'A/G', chr='22'),
    }
    with tarfile.open(path, 'w:gz') as archive:
        for chromosome, line in lines.items():
            member = directory / f'{chromosome}.all_vars.gz'
            member.write_bytes(gzip.compress(f'{line}\n'.encode()))
            archive.add(member, arcname=f'homo_sapiens/116_GRCh38/{chromosome}/all_vars.gz')


@pytest.fixture(scope='module')
def step_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """The sources every run of the step shares: reference, GERP, cache, SIFT, AlphaMissense, ancestor, core."""
    directory = tmp_path_factory.mktemp('step_inputs')
    _write_fasta(directory / 'reference.fa.gz')
    gerp = pyBigWig.open(str(directory / 'gerp.bw'), 'w')
    gerp.addHeader([('1', CONTIG_LENGTH), ('2', CONTIG_LENGTH), ('22', 50_818_468)])
    gerp.addEntries(['1', '22'], [122, 18521429], ends=[123, 18521430], values=[2.5, 1.5])
    gerp.close()
    _write_vep_cache(directory / 'vep_cache.tar.gz', directory)
    (directory / 'alphamissense.tsv').write_text(
        '# AlphaMissense\n'
        '#CHROM\tPOS\tREF\tALT\tgenome\tuniprot_id\ttranscript_id\tprotein_variant\tam_pathogenicity\tam_class\n'
        'chr1\t123\tA\tG\thg38\tP1\tENST_1.1\tA1T\t0.9\tlikely_pathogenic\n'
    )
    ancestor = write_bgzip_fasta(directory / 'human_ancestor.fa.gz', {'1': (CONTIG_LENGTH, 1, 'a' * CONTIG_LENGTH)})
    core = directory / 'core'
    core.mkdir()
    write_dump(
        core,
        {name: [cells(*row) for row in rows] for name, (_, rows) in CORE_TABLES.items()},
        ddl({name: columns for name, (columns, _) in CORE_TABLES.items()}),
    )
    return {
        'fasta': str(directory / 'reference.fa.gz'),
        'gff3': str(FASTVEP_FIXTURES / 'chr22_slice.gff3.gz'),
        'gerp': str(directory / 'gerp.bw'),
        'vep_cache': str(directory / 'vep_cache.tar.gz'),
        'sift_predictions': str(SIFT_FIXTURES / 'protein_function_predictions.txt.gz'),
        'sift_translation_md5': str(SIFT_FIXTURES / 'translation_md5.txt.gz'),
        'sift_attrib': str(SIFT_FIXTURES / 'attrib.txt.gz'),
        'peptides': str(SIFT_FIXTURES / 'Homo_sapiens.GRCh38.pep.all.fa.gz'),
        'alphamissense': str(directory / 'alphamissense.tsv'),
        'ancestor': ancestor,
        'core': str(core),
    }


def _row_sources(directory: Path, variants: list[tuple[str, str | None]]) -> dict[str, str]:
    """The four row sources: every (variant id, rsID) in the credible sets, the first also in the others."""
    directory.mkdir(parents=True, exist_ok=True)
    first = json.dumps({'variantId': variants[0][0], 'variantRsId': variants[0][1]}) + '\n'
    sources = {}
    for name in ('uniprot_variants', 'eva'):
        sources[name] = str(directory / f'{name}.json.gz')
        with gzip.open(sources[name], 'wt') as stream:
            stream.write(first)
    sources['pharmacogenomics'] = str(directory / 'pharmacogenomics')
    sources['credible_set'] = str(directory / 'credible_set')
    for name, frame in [('pharmacogenomics', variants[:1]), ('credible_set', variants)]:
        Path(sources[name]).mkdir()
        pl.DataFrame(frame, schema={'variantId': pl.String, 'variantRsId': pl.String}, orient='row').write_parquet(
            Path(sources[name]) / 'part.parquet'
        )
    return sources


def _fake_fastvep(directory: Path, fail: str | None = None) -> str:
    script = directory / 'fastvep'
    script.write_text(
        FAKE_FASTVEP
        .replace('PYTHON', sys.executable)
        .replace('FAIL', repr(fail))
        .replace('SAMPLE', repr(str(FASTVEP_SAMPLE)))
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _run_step(
    tmp_path: Path,
    inputs: dict[str, str],
    variants: list[tuple[str, str | None]],
    name: str = 'output',
    fail: str | None = None,
    source: dict[str, object] | None = None,
    settings: dict[str, object] | None = None,
) -> Path:
    """Run the step on `variants` with a fake fastVEP, returning its output directory."""
    work = tmp_path / f'{name}_work'
    work.mkdir()
    output = tmp_path / name
    variant(
        source={**inputs, **_row_sources(tmp_path / f'{name}_rows', variants), **(source or {})},
        destination={'output': str(output)},
        settings={
            'fastvep_binary': _fake_fastvep(work, fail),
            'parallel_chromosomes': 2,
            'fastvep_threads': 1,
            **(settings or {}),
        },
        config=Mock(work_path=work),
    )
    return output


def _slice_variants() -> list[tuple[str, str | None]]:
    """The variants of the chromosome 22 fixture, in its order, with their rsIDs."""
    rows = pl.read_parquet(FASTVEP_FIXTURES / 'chr22_slice_rows.parquet')
    return [
        (f'{chromosome}_{position}_{reference}_{alternate}', None if rs_id == '.' else rs_id)
        for chromosome, position, rs_id, reference, alternate in rows.iter_rows()
    ]


@pytest.fixture(scope='module')
def end_to_end(tmp_path_factory: pytest.TempPathFactory, step_inputs: dict[str, str]) -> tuple[pl.DataFrame, Path]:
    """The index of chromosomes 1, 22 and MT, and the step's row sources."""
    tmp_path = tmp_path_factory.mktemp('end_to_end')
    variants = [('1_123_A_G', 'rs1'), *_slice_variants(), ('MT_123_G_A', None)]
    output = _run_step(tmp_path, step_inputs, variants)
    assert list((tmp_path / 'output_work').iterdir()) == [tmp_path / 'output_work' / 'fastvep']
    return pl.read_parquet(output / '*.parquet'), tmp_path / 'output_rows'


def test_variant_end_to_end_with_fake_fastvep(end_to_end):
    index, rows = end_to_end
    key = ['chromosome', 'position', 'referenceAllele', 'alternateAllele']
    collected = collect_variants(
        str(rows / 'uniprot_variants.json.gz'),
        str(rows / 'eva.json.gz'),
        str(rows / 'pharmacogenomics'),
        str(rows / 'credible_set'),
    ).collect()
    assert index.schema == variant_schema
    assert index.height == collected.height == 52
    assert index.select(key).sort(key).equals(collected.select(key).sort(key))
    by_id = {row['variantId']: row for row in index.to_dicts()}
    assert (
        by_id['22_19709333_G_A']['mostSevereConsequenceId'] == SEQUENCE_ONTOLOGY['non_coding_transcript_exon_variant']
    )
    assert by_id['22_18521430_A_G']['rsIds'] == ['rs2542835']
    assert sorted(index['chromosome'].unique()) == ['1', '22', 'MT']


def test_missing_source_data_is_null(end_to_end):
    """MT has no `all_vars`, GERP contig or AlphaMissense rows; chromosome 1 has all three."""
    index, _ = end_to_end
    by_id = {row['variantId']: row for row in index.to_dicts()}
    present, missing = by_id['1_123_A_G'], by_id['MT_123_G_A']
    assert present['rsIds'] == ['rs123']
    assert {x['method'] for x in present['variantEffect']} >= {'GERP', 'AlphaMissense'}
    assert missing['rsIds'] == []
    assert not {x['method'] for x in missing['variantEffect']} & {'GERP', 'AlphaMissense'}
    assert [x['transcriptId'] for x in missing['transcriptConsequences']] == ['ENST_MT']
    assert missing['hgvsId'] == 'MT:m.123G>A'


def test_failed_chromosome_writes_nothing(tmp_path, step_inputs):
    variants = [('1_123_A_G', None), *_slice_variants(), ('MT_123_G_A', None)]
    with pytest.raises(RuntimeError, match='fastVEP failed'):
        _run_step(tmp_path, step_inputs, variants, fail='22')
    assert not list(tmp_path.glob('output/part-*.parquet'))


def test_chromosome_without_reference_is_intergenic(tmp_path, step_inputs):
    """Y is not in the reference FASTA: fastVEP, which fails on Y here, is not run on it."""
    output = _run_step(tmp_path, step_inputs, [('1_123_A_G', None), ('Y_100_A_G', None)], fail='Y')
    result = pl.read_parquet(output / '*.parquet')
    assert result.schema == variant_schema
    found = result.filter(pl.col('chromosome') == 'Y').to_dicts()
    assert len(found) == 1
    assert found[0]['variantId'] == 'Y_100_A_G'
    assert found[0]['hgvsId'] is None
    assert found[0]['transcriptConsequences'] == []
    assert found[0]['mostSevereConsequenceId'] == SEQUENCE_ONTOLOGY['intergenic_variant']
    assert result.filter(pl.col('chromosome') == '1')['transcriptConsequences'].list.len().to_list() == [1]


def test_variant_step_writes_the_annotated_index(tmp_path, step_inputs):
    (tmp_path / 'gnomad').mkdir()
    pl.DataFrame(
        {'variantId': ['1_123_A_G'], 'chromosome': ['1'], 'rsIds': [['rs1']]},
    ).with_columns(position=pl.lit(123, dtype=pl.Int32)).write_parquet(tmp_path / 'gnomad' / 'part.parquet')
    (tmp_path / 'foldx').mkdir()
    pl.DataFrame(
        {'uniprotAccession': ['P1'], 'aminoAcidChange': ['A1T'], 'variantEffect': [[effect('FoldX', 1.0)]]},
        schema={'uniprotAccession': pl.String, 'aminoAcidChange': pl.String, 'variantEffect': EFFECTS},
    ).write_parquet(tmp_path / 'foldx' / 'part.parquet')
    output = _run_step(
        tmp_path,
        step_inputs,
        [('1_123_A_G', None), ('2_5_C_T', None)],
        source={'variant_annotations': [str(tmp_path / 'gnomad')], 'amino_acid_annotations': [str(tmp_path / 'foldx')]},
    )
    result = pl.read_parquet(output / '*.parquet')
    assert result.schema == variant_schema
    assert result['variantId'].to_list() == ['1_123_A_G', '2_5_C_T']
    first = result.row(0, named=True)
    assert sorted(first['rsIds']) == ['rs1', 'rs123']
    assert 'FoldX' in [x['method'] for x in first['variantEffect']]


GPN_STAR_CUTOFFS = {'top_10_percent': -2.0, 'top_1_percent': -4.0, 'top_0_1_percent': -6.0, 'minimum': -10.0}


def _write_gpn_star(directory: Path, chromosome: str, rows: list[tuple[int, str, str, float]]) -> None:
    directory.mkdir(exist_ok=True)
    pl.DataFrame(
        {
            'chrom': [chromosome] * len(rows),
            'pos': [row[0] for row in rows],
            'ref': [row[1] for row in rows],
            'alt': [row[2] for row in rows],
            'llr_calibrated': [row[3] for row in rows],
            'abs_llr_calibrated': [abs(row[3]) for row in rows],
        },
        schema={
            'chrom': pl.String,
            'pos': pl.Int64,
            'ref': pl.String,
            'alt': pl.String,
            'llr_calibrated': pl.Float32,
            'abs_llr_calibrated': pl.Float32,
        },
    ).write_parquet(directory / f'llr_chr{chromosome}.parquet')


def _run_two_chromosome_step(
    tmp_path: Path, inputs: dict[str, str], name: str, gpn_star: bool, **source: object
) -> pl.DataFrame:
    output = _run_step(
        tmp_path,
        inputs,
        [('1_123_A_G', None), ('2_5_C_T', None)],
        name=name,
        source={'gpn_star': str(tmp_path / 'gpn_star'), **source} if gpn_star else source,
        settings={'gpn_star': GPN_STAR_CUTOFFS} if gpn_star else None,
    )
    return pl.read_parquet(output / '*.parquet')


def test_variant_step_adds_gpn_star(tmp_path, step_inputs):
    _write_gpn_star(tmp_path / 'gpn_star', '1', [(123, 'A', 'G', -3.0), (124, 'A', 'G', -9.0)])
    result = _run_two_chromosome_step(tmp_path, step_inputs, 'output', gpn_star=True)
    entries = {
        row['variantId']: [x for x in row['variantEffect'] if x['method'] == 'GPN-Star'] for row in result.to_dicts()
    }
    assert [x['normalisedScore'] for x in entries['1_123_A_G']] == [0.25]
    assert entries['2_5_C_T'] == []


def test_variant_step_without_a_gpn_star_file_for_a_chromosome(tmp_path, step_inputs):
    plain = _run_two_chromosome_step(tmp_path, step_inputs, 'plain', gpn_star=False)
    _write_gpn_star(tmp_path / 'gpn_star', '1', [(123, 'A', 'G', -3.0)])
    scored = _run_two_chromosome_step(tmp_path, step_inputs, 'scored', gpn_star=True)
    assert scored.filter(pl.col('chromosome') == '2').equals(plain.filter(pl.col('chromosome') == '2'))
    assert not scored.filter(pl.col('chromosome') == '1').equals(plain.filter(pl.col('chromosome') == '1'))


def test_variant_step_warns_about_a_missing_standard_chromosome(tmp_path, step_inputs):
    _write_gpn_star(tmp_path / 'gpn_star', '1', [(123, 'A', 'G', -3.0)])
    messages: list[str] = []
    sink_id = logger.add(messages.append, level='WARNING', format='{message}')
    try:
        _run_two_chromosome_step(tmp_path, step_inputs, 'output', gpn_star=True)
    finally:
        logger.remove(sink_id)
    assert [m for m in messages if 'GPN-Star' in m and 'chromosome 2' in m]
    assert not [m for m in messages if 'chromosome 1 ' in m]


def test_variant_step_needs_cutoffs_with_a_gpn_star_source(tmp_path, step_inputs):
    with pytest.raises(ValueError, match='cut-offs'):
        variant(
            source={**step_inputs, **_row_sources(tmp_path / 'rows', [('1_123_A_G', None)]), 'gpn_star': 'gpn_star'},
            destination={'output': str(tmp_path / 'output')},
            settings={},
            config=Mock(work_path=tmp_path),
        )


def test_variant_step_rejects_an_empty_gpn_star_source(tmp_path, step_inputs):
    (tmp_path / 'gpn_star').mkdir()
    with pytest.raises(ValueError, match='llr_chr'):
        _run_two_chromosome_step(tmp_path, step_inputs, 'output', gpn_star=True)


def test_variant_step_refuses_an_existing_destination(tmp_path, step_inputs):
    (tmp_path / 'output').mkdir()
    with pytest.raises(ValueError, match='already exists'):
        _run_step(tmp_path, step_inputs, [('1_123_A_G', None)])


@pytest.mark.parametrize('source', ['variant_annotations', 'amino_acid_annotations'])
def test_variant_step_rejects_an_empty_annotation_source(tmp_path, step_inputs, source):
    """An empty gnomAD or FoldX input fails the step instead of silently annotating nothing."""
    empty = tmp_path / 'empty'
    empty.mkdir()
    pl.DataFrame(schema={'variantId': pl.String}).write_parquet(empty / 'part.parquet')
    with pytest.raises(ValueError, match='No annotations found'):
        _run_step(tmp_path, step_inputs, [('1_123_A_G', None)], source={source: [str(empty)]})


def test_variant_configuration_matches_models():
    config = yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text())
    for spec in config['steps']['variant']:
        TransformSpec.model_validate(spec)
    [task] = [spec for spec in config['steps']['variant'] if spec['transformer'] == 'variant']
    VariantSource.model_validate(task['source'])
    VariantDestination.model_validate(task['destination'])
    VariantSettings.model_validate(task['settings'])
    assert task['source']['gpn_star'] == 'input/variant/gpn_star'
    assert 'vep' not in task['source']
    GpnStarCutoffs.model_validate(task['settings']['gpn_star'])


def test_ppp_variant_step_differs_only_by_the_lof_curation():
    root = Path(__file__).parents[2]
    base = yaml.safe_load((root / 'pts' / 'config.yaml').read_text())['steps']['variant']
    override_path = root / 'orchestration/src/orchestration/dags/config/ppp/pts.override.yaml'
    override = yaml.safe_load(override_path.read_text())['steps']['variant']
    [task] = [spec for spec in override if spec.get('transformer') == 'variant']
    task['source']['variant_annotations'].remove('intermediate/variant/lof_curation')
    assert override == base


def test_gpn_star_cutoffs_are_in_step_with_the_revision():
    root = Path(__file__).parents[2]
    base_text = (root / 'pts' / 'config.yaml').read_text()
    override_text = (root / 'orchestration/src/orchestration/dags/config/ppp/pts.override.yaml').read_text()
    revision = yaml.safe_load((root / 'pis' / 'config.yaml').read_text())['scratchpad']['gpn_star_revision']

    def variant_task(text: str) -> dict:
        steps = yaml.safe_load(text)['steps']['variant']
        [task] = [spec for spec in steps if spec.get('transformer') == 'variant']
        return task

    base, override = variant_task(base_text), variant_task(override_text)
    assert override['source']['gpn_star'] == base['source']['gpn_star']
    assert override['settings']['gpn_star'] == base['settings']['gpn_star']
    for text in (base_text, override_text):
        assert re.findall(r'gpn-star-scores@([0-9a-f]{40})', text) == [revision]


@pytest.mark.parametrize(
    'settings',
    [
        {'hash_threshold': 0},
        {'fastvep_threads': 0},
        {'parallel_chromosomes': 0},
        {'chunk_rows': 0},
        {'parallel_chunks': 0},
        {'chunk_threads': 0},
        {'parallel_preparations': 0},
        {'chromosome_threads': 0},
        {'preparation_niceness': -1},
        {'preparation_niceness': 20},
        {'chunk_niceness': -1},
        {'chunk_niceness': 20},
        {'parse_processes': 2},
        {'unknown': 1},
    ],
)
def test_variant_settings_are_validated(settings):
    with pytest.raises(ValidationError):
        VariantSettings.model_validate(settings)


def test_chunked_step_matches_an_unchunked_run(tmp_path, step_inputs):
    """Chunks of a few rows, annotated in parallel, give the same index, row for row, as one chunk per chromosome."""
    variants = [('1_123_A_G', 'rs1'), ('2_5_C_T', None), *_slice_variants(), ('MT_123_G_A', None)]
    chr22 = [variant for variant, _ in variants if variant.startswith('22_')]
    (tmp_path / 'gnomad').mkdir()
    pl.DataFrame(
        {
            'variantId': ['1_123_A_G', chr22[3], chr22[20]],
            'chromosome': ['1', '22', '22'],
            'position': [int(variant.split('_')[1]) for variant in ['1_123_A_G', chr22[3], chr22[20]]],
            'rsIds': [['rs1'], ['rs9'], ['rs8']],
        },
        schema_overrides={'position': pl.Int32},
    ).write_parquet(tmp_path / 'gnomad' / 'part.parquet')
    (tmp_path / 'foldx').mkdir()
    pl.DataFrame(
        {'uniprotAccession': ['P1'], 'aminoAcidChange': ['A1T'], 'variantEffect': [[effect('FoldX', 1.0)]]},
        schema={'uniprotAccession': pl.String, 'aminoAcidChange': pl.String, 'variantEffect': EFFECTS},
    ).write_parquet(tmp_path / 'foldx' / 'part.parquet')
    _write_gpn_star(tmp_path / 'gpn_star', '1', [(123, 'A', 'G', -3.0)])
    _write_gpn_star(
        tmp_path / 'gpn_star',
        '22',
        [(int(variant.split('_')[1]), *variant.split('_')[2:], -5.0) for variant in chr22[::5]],
    )
    source = {
        'variant_annotations': [str(tmp_path / 'gnomad')],
        'amino_acid_annotations': [str(tmp_path / 'foldx')],
        'gpn_star': str(tmp_path / 'gpn_star'),
    }

    def run(name: str, **settings: object) -> tuple[pl.DataFrame, list[str]]:
        output = _run_step(
            tmp_path,
            step_inputs,
            variants,
            name=name,
            source=source,
            settings={'gpn_star': GPN_STAR_CUTOFFS, **settings},
        )
        parts = sorted(path.name for path in output.glob('*.parquet'))
        return pl.read_parquet([output / part for part in parts]), parts

    whole, whole_parts = run('whole', chunk_rows=1_000_000)
    chunked, chunked_parts = run('chunked', chunk_rows=7, parallel_chunks=3, batch_size=3)

    assert len(whole_parts) == 4
    assert len(chunked_parts) == 2 + 1 + -(-len(chr22) // 7)
    assert {x['method'] for row in whole['variantEffect'] for x in row} >= {'GPN-Star', 'FoldX', 'GERP'}
    assert whole.equals(chunked)
