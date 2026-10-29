"""Variant index: VEP parsing, annotation merging and the end-to-end step."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from pts.schemas.variant import variant_schema
from pts.schemas.vep import vep_schema
from pts.tasks.transform import TransformSpec
from pts.transformers.utils.variant import hash_variant_index_ids
from pts.transformers.variant import VariantDestination, VariantSettings, VariantSource, variant
from pts.transformers.variant_utils.consequences import CONSEQUENCE_SCORES, SEQUENCE_ONTOLOGY
from pts.transformers.variant_utils.index import (
    add_annotation,
    amino_acid_keys,
    annotate_with_amino_acid_consequences,
    annotation_parts_by_chromosome,
    normalise_score,
    read_variant_annotations,
)
from pts.transformers.variant_utils.vep import (
    amino_acid_change,
    generate_dbxrefs,
    list_vep_parts,
    parse_vep,
    read_vep,
    uniprot_accessions,
)

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
    result = parse_vep(read_vep(str(SAMPLE))).collect()
    assert result.schema == variant_schema
    assert result.height == 4
    for row in result.to_dicts():
        transcripts = row['transcriptConsequences']
        assert [x['transcriptIndex'] for x in transcripts] == list(range(1, len(transcripts) + 1))
        assert len({x['targetId'] for x in transcripts}) == len(transcripts)
        assert all(x['isEnsemblCanonical'] for x in transcripts)
        assert all(row[name] is not None for name, dtype in variant_schema.items() if isinstance(dtype, pl.List))


def test_parts_are_found_recursively_including_gzip_and_jsonl(tmp_path):
    nested = tmp_path / 'chunk' / 'nested'
    nested.mkdir(parents=True)
    record = {'input': '1\t123\t.\tA\tG', 'extra': 'discarded'}
    (tmp_path / 'part.jsonl').write_text(json.dumps(record) + '\n')
    with gzip.open(nested / 'part.json.gz', 'wt') as stream:
        stream.write(json.dumps(record) + '\n')
    (tmp_path / '_SUCCESS').touch()
    parts = list_vep_parts(str(tmp_path))
    assert [Path(part).name for part in parts] == ['part.json.gz', 'part.jsonl']
    frames = [read_vep(part).collect() for part in parts]
    assert [frame.height for frame in frames] == [1, 1]
    assert all(frame.schema == vep_schema for frame in frames)


def test_parts_skip_vep_warning_files(tmp_path):
    """VEP's `<part>.json_warnings.txt` sits next to the data and is not JSON."""
    (tmp_path / '0000062a.json').write_text(json.dumps({'input': '1\t123\t.\tA\tG'}) + '\n')
    (tmp_path / '0000062a.json_warnings.txt').write_text('WARNING: 14 : substr outside of string\n')
    assert [Path(part).name for part in list_vep_parts(str(tmp_path))] == ['0000062a.json']


def test_parts_reject_a_directory_without_vep_files(tmp_path):
    (tmp_path / 'part.json_warnings.txt').write_text('WARNING: only warnings\n')
    with pytest.raises(ValueError, match='No VEP JSON files'):
        list_vep_parts(str(tmp_path))


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


def test_omim_ids_keep_their_four_digit_suffix(tmp_path):
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
    (tmp_path / '00000001.json').write_text(record + '\n')
    [row] = parse_vep(read_vep(str(tmp_path / '00000001.json'))).collect().to_dicts()
    assert [x['id'] for x in row['dbXrefs'] if x['source'] == 'omim'] == ['600234#0010', '171834#0100', '159530#0010']


def test_all_predictors_without_sift_or_polyphen():
    result = parsed(
        transcript_consequences=[
            transcript(
                cadd_phred=10,
                lof='HC',
                lof_filter='flag',
                conservation=3,
                alphamissense={'am_class': 'likely_pathogenic', 'am_pathogenicity': 0.8},
            ),
            transcript(
                gene_id='g2',
                cadd_phred=20,
                lof='LC',
                conservation=5,
                alphamissense={'am_class': 'benign', 'am_pathogenicity': 0.9},
            ),
        ],
    )
    effects = {x['method']: x for x in result['variantEffect']}
    assert list(effects) == ['CADD', 'LOFTEE', 'GERP', 'AlphaMissense', 'VEP']
    assert effects['CADD']['score'] == 20
    assert effects['LOFTEE']['assessment'] == 'LC'
    assert effects['GERP']['score'] == 5
    assert effects['AlphaMissense']['targetId'] == 'ENSG1'
    assert effects['AlphaMissense']['score'] == pytest.approx(0.8)


def test_predictor_comes_from_the_first_transcript_with_the_highest_score():
    """Ties keep VEP's transcript order, null scores never win, and no score means no effect."""
    result = parsed(
        transcript_consequences=[
            transcript(gene_id='none'),
            transcript(gene_id='first', cadd_phred=5),
            transcript(gene_id='second', cadd_phred=5),
            transcript(gene_id='lower', cadd_phred=1),
        ]
    )
    effects = {x['method']: x for x in result['variantEffect']}
    assert effects['CADD']['targetId'] == 'first'
    assert 'GERP' not in effects


def test_loftee_ties_keep_the_first_transcript():
    """Spark ranks LOFTEE by its string value, so `LC` beats `HC`; ties keep VEP's order."""
    result = parsed(
        transcript_consequences=[
            transcript(gene_id='high_confidence', lof='HC'),
            transcript(gene_id='first', lof='LC', lof_filter='A'),
            transcript(gene_id='second', lof='LC', lof_filter='B'),
        ]
    )
    [loftee] = [x for x in result['variantEffect'] if x['method'] == 'LOFTEE']
    assert (loftee['targetId'], loftee['assessment'], loftee['assessmentFlag']) == ('first', 'LC', 'A')


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
    hashed = frame.select(hash_variant_index_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), 10))
    assert hashed.to_series().to_list() == [
        'short',
        'OTVAR_2_2_' + hashlib.md5(b'abcdefghij').hexdigest(),
        'OTVAR_' + hashlib.md5(b'missing').hexdigest(),
        None,
    ]


def _spark_rescale(score: float, low: float, high: float, minimum: float, maximum: float) -> float:
    """Spark's arithmetic: an integer bound is subtracted in single precision, then double from there."""
    single = np.float32(score)
    shifted = np.float64(single - np.float32(low)) if isinstance(low, int) else np.float64(single) - low
    return float(shifted / (high - low) * (maximum - minimum) + minimum)


@pytest.mark.parametrize(
    ('method', 'score', 'assessment', 'expected'),
    [
        ('LOFTEE', None, 'HC', 1.0),
        ('LOFTEE', None, 'LC', 0.85),
        ('LOFTEE', 1.0, 'other', None),
        ('AlphaMissense', 0.0, None, -1.0),
        ('AlphaMissense', 0.3, None, _spark_rescale(0.3, 0.06, 0.77, -0.25, 0.25)),
        ('AlphaMissense', 0.77, None, _spark_rescale(0.77, 0.06, 0.77, -0.25, 0.25)),
        ('CADD', 10.0, None, 0.0),
        ('CADD', 17.3, None, _spark_rescale(17.3, 10, 20, 0.0, 0.5)),
        ('CADD', 81.0, None, 1.0),
        ('VEP', 0.4, None, float(np.float32(0.4))),
        ('GERP', -5.0, None, 0.0),
        ('GERP', 2.0, None, 0.75),
        ('GERP', -2.31, None, _spark_rescale(-2.31, -3, 0, 0.0, 0.25)),
        ('GERP', 7.0, None, 1.0),
        ('GERP', None, None, None),
    ],
)
def test_normalisation_matches_spark_exactly(method, score, assessment, expected):
    frame = pl.DataFrame(
        {'score': [score], 'assessment': [assessment]}, schema={'score': pl.Float32, 'assessment': pl.String}
    )
    assert frame.select(normalise_score(method, pl.col('score'), pl.col('assessment'))).item() == expected


@pytest.mark.parametrize('method', ['FoldX', 'SIFT', 'unknown'])
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


def test_variant_step_writes_the_annotated_index(tmp_path):
    vep = tmp_path / 'vep'
    vep.mkdir()
    records = [
        {
            'input': '1\t123\t.\tA\tG',
            'most_severe_consequence': 'missense_variant',
            'transcript_consequences': [transcript(amino_acids='A/T', protein_end=1, swissprot=['P1.1'])],
        },
        {'input': '2\t5\t.\tC\tT', 'most_severe_consequence': 'intergenic_variant'},
    ]
    (vep / '00000001.json').write_text(json.dumps(records[0]) + '\n')
    (vep / '00000002.json').write_text(json.dumps(records[1]) + '\n')
    (vep / '00000001.json_warnings.txt').write_text('WARNING: ignored\n')
    (tmp_path / 'gnomad').mkdir()
    pl.DataFrame(
        {'variantId': ['1_123_A_G'], 'chromosome': ['1'], 'rsIds': [['rs1']]},
    ).with_columns(position=pl.lit(123, dtype=pl.Int32)).write_parquet(tmp_path / 'gnomad' / 'part.parquet')
    (tmp_path / 'foldx').mkdir()
    pl.DataFrame(
        {'uniprotAccession': ['P1'], 'aminoAcidChange': ['A1T'], 'variantEffect': [[effect('FoldX', 1.0)]]},
        schema={'uniprotAccession': pl.String, 'aminoAcidChange': pl.String, 'variantEffect': EFFECTS},
    ).write_parquet(tmp_path / 'foldx' / 'part.parquet')
    output = tmp_path / 'output'
    variant(
        source={
            'vep': str(vep),
            'variant_annotations': [str(tmp_path / 'gnomad')],
            'amino_acid_annotations': [str(tmp_path / 'foldx')],
        },
        destination={'output': str(output)},
        settings={'parse_processes': 2},
        config=Mock(work_path=tmp_path),
    )
    result = pl.read_parquet(output / '*.parquet')
    assert result.schema == variant_schema
    assert result['variantId'].to_list() == ['1_123_A_G', '2_5_C_T']
    first = result.row(0, named=True)
    assert first['rsIds'] == ['rs1']
    assert [x['method'] for x in first['variantEffect']] == ['VEP', 'FoldX']
    assert sorted(path.name for path in tmp_path.iterdir()) == ['foldx', 'gnomad', 'output', 'vep']


def test_variant_step_refuses_an_existing_destination(tmp_path):
    (tmp_path / 'output').mkdir()
    with pytest.raises(ValueError, match='already exists'):
        variant(
            source={'vep': str(tmp_path / 'vep')},
            destination={'output': str(tmp_path / 'output')},
            settings={},
            config=Mock(work_path=tmp_path),
        )


@pytest.mark.parametrize('source', ['variant_annotations', 'amino_acid_annotations'])
def test_variant_step_rejects_an_empty_annotation_source(tmp_path, source):
    """An empty gnomAD or FoldX input fails the step instead of silently annotating nothing."""
    vep = tmp_path / 'vep'
    vep.mkdir()
    (vep / '00000001.json').write_text(json.dumps({'input': '1\t123\t.\tA\tG'}) + '\n')
    empty = tmp_path / 'empty'
    empty.mkdir()
    pl.DataFrame(schema={'variantId': pl.String}).write_parquet(empty / 'part.parquet')
    with pytest.raises(ValueError, match='No annotations found'):
        variant(
            source={'vep': str(vep), source: [str(empty)]},
            destination={'output': str(tmp_path / 'output')},
            settings={'parse_processes': 1},
            config=Mock(work_path=tmp_path),
        )


def test_variant_configuration_matches_models():
    config = yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text())
    [task] = config['steps']['variant']
    TransformSpec.model_validate(task)
    assert task['transformer'] == 'variant'
    VariantSource.model_validate(task['source'])
    VariantDestination.model_validate(task['destination'])
    VariantSettings.model_validate(task['settings'])


@pytest.mark.parametrize(
    'settings', [{'hash_threshold': 0}, {'parse_processes': 0}, {'parse_threads': 0}, {'unknown': 1}]
)
def test_variant_settings_are_validated(settings):
    with pytest.raises(ValidationError):
        VariantSettings.model_validate(settings)
