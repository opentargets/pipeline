"""PySpark variant index: VEP parsing, annotation merging and the end-to-end job."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.variant import VariantDestination, VariantSettings, VariantSource, variant
from pts.pyspark.variant_utils.consequences import CONSEQUENCE_SCORES, SEQUENCE_ONTOLOGY
from pts.pyspark.variant_utils.index import (
    VARIANT_SCHEMA,
    add_annotation,
    annotate_with_amino_acid_consequences,
    hash_long_variant_ids,
    normalise_score,
    read_variant_annotations,
)
from pts.pyspark.variant_utils.vep import (
    VEP_SCHEMA,
    amino_acid_change,
    generate_dbxrefs,
    parse_vep,
    read_vep,
    uniprot_accessions,
)
from pts.tasks.pyspark import PysparkSpec

SAMPLE = Path(__file__).parent / 'test_data' / 'vep_sample.jsonl'


@pytest.fixture(autouse=True)
def legacy_casts(spark: SparkSession):
    """The pts Session runs with ANSI off, as Gentropy did; invalid casts give null."""
    spark.conf.set('spark.sql.ansi.enabled', 'false')


def same_types(left: t.StructType, right: t.StructType) -> bool:
    """Compare schemas by name and type; Spark marks computed columns non-nullable."""
    return [(x.name, x.dataType.simpleString()) for x in left.fields] == [
        (x.name, x.dataType.simpleString()) for x in right.fields
    ]


def vep_records(spark: SparkSession, *records: dict) -> DataFrame:
    """Raw VEP records read through the same schema as the JSON reader."""
    return spark.read.schema(VEP_SCHEMA).json(spark.sparkContext.parallelize([json.dumps(r) for r in records]))


def parsed(spark: SparkSession, hash_threshold: int = 300, **updates: object) -> dict:
    record = {'input': '1\t123\t.\tA\tG', 'most_severe_consequence': 'intergenic_variant', **updates}
    row = parse_vep(vep_records(spark, record), hash_threshold).first()
    assert row is not None
    return row.asDict(recursive=True)


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


def index_rows(spark: SparkSession, rows: list[dict]) -> DataFrame:
    """Variant index rows; fields not given are null."""
    full = [{field.name: row.get(field.name) for field in VARIANT_SCHEMA.fields} for row in rows]
    return spark.createDataFrame(full, VARIANT_SCHEMA)


# --- VEP reader ---------------------------------------------------------------------------------


def test_sample_preserves_count_and_full_schema(spark):
    result = parse_vep(read_vep(spark, str(SAMPLE)))
    assert same_types(result.schema, VARIANT_SCHEMA)
    rows = [row.asDict(recursive=True) for row in result.collect()]
    assert len(rows) == 4
    for row in rows:
        transcripts = row['transcriptConsequences']
        assert [x['transcriptIndex'] for x in transcripts] == list(range(1, len(transcripts) + 1))
        assert len({x['targetId'] for x in transcripts}) == len(transcripts)
        assert all(x['isEnsemblCanonical'] for x in transcripts)
        assert all(
            row[field.name] is not None for field in VARIANT_SCHEMA.fields if isinstance(field.dataType, t.ArrayType)
        )


def test_reader_is_recursive_and_reads_gzip_and_jsonl(spark, tmp_path):
    nested = tmp_path / 'chunk' / 'nested'
    nested.mkdir(parents=True)
    record = {'input': '1\t123\t.\tA\tG', 'extra': 'discarded'}
    (tmp_path / 'part.jsonl').write_text(json.dumps(record) + '\n')
    with gzip.open(nested / 'part.json.gz', 'wt') as stream:
        stream.write(json.dumps(record) + '\n')
    (tmp_path / '_SUCCESS').touch()
    result = read_vep(spark, str(tmp_path))
    assert result.count() == 2
    assert result.schema == VEP_SCHEMA


def test_reader_skips_vep_warning_files(spark, tmp_path):
    """VEP's `<part>.json_warnings.txt` sits next to the data and is not JSON."""
    (tmp_path / '0000062a.json').write_text(json.dumps({'input': '1\t123\t.\tA\tG'}) + '\n')
    (tmp_path / '0000062a.json_warnings.txt').write_text('WARNING: 14 : substr outside of string\n')
    assert read_vep(spark, str(tmp_path)).count() == 1


def test_reader_rejects_a_directory_without_vep_records(spark, tmp_path):
    (tmp_path / 'part.json_warnings.txt').write_text('WARNING: only warnings\n')
    with pytest.raises(ValueError, match='No VEP records'):
        read_vep(spark, str(tmp_path))


# --- column helpers -----------------------------------------------------------------------------


@pytest.mark.parametrize(('amino', 'end', 'expected'), [('A/B', 1, 'A1B'), ('A/B', None, None), (None, 1, None)])
def test_amino_acid_change(spark, amino, end, expected):
    frame = spark.createDataFrame([(amino, end)], 'amino string, end long')
    assert frame.select(amino_acid_change(f.col('amino'), f.col('end'))).first()[0] == expected


def test_uniprot_accessions_drop_versions_nulls_and_duplicates(spark):
    frame = spark.createDataFrame(
        [(['P1.2', None, 'P1.3'], ['P2.1', 'P1.1']), (None, None)], 'sp array<string>, tr array<string>'
    )
    result = frame.select(uniprot_accessions(f.col('sp'), f.col('tr'))).collect()
    assert [row[0] for row in result] == [['P1', 'P2'], []]


def test_generate_dbxrefs_filters_nulls_and_preserves_duplicates(spark):
    frame = spark.createDataFrame([(['rs1', None, 'rs1'],), (None,)], 'ids array<string>')
    result = [
        [x.asDict() for x in row[0]]
        for row in frame.select(generate_dbxrefs(f.col('ids'), 'ensembl_variation')).collect()
    ]
    assert result == [[{'id': 'rs1', 'source': 'ensembl_variation'}] * 2, []]


def test_consequence_mapping_preserves_all_41_terms():
    assert len(SEQUENCE_ONTOLOGY) == len(CONSEQUENCE_SCORES) == 41
    assert SEQUENCE_ONTOLOGY['missense_variant'] == 'SO_0001583'
    assert CONSEQUENCE_SCORES['sequence_variant'] == 0.0


def test_incomplete_terminal_codon_uses_the_sequence_ontology_term(spark):
    """Unlike Gentropy's misspelt label, VEP's term is mapped and scored."""
    assert 'incomplete_labelinal_codon_variant' not in SEQUENCE_ONTOLOGY.keys() | CONSEQUENCE_SCORES.keys()
    row = parsed(spark, most_severe_consequence='incomplete_terminal_codon_variant')
    assert row['mostSevereConsequenceId'] == 'SO_0001626'
    assert row['variantEffect'][0]['score'] == pytest.approx(0.54)


# --- parsing ------------------------------------------------------------------------------------


@pytest.mark.parametrize('transcripts', [None, []])
def test_no_transcripts_has_vep_prediction_and_empty_arrays(spark, transcripts):
    result = parsed(spark, transcript_consequences=transcripts)
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


def test_canonical_filter_sort_and_stable_ties(spark):
    records = [
        transcript(gene_id='g0', canonical=0, consequence_terms=['stop_gained']),
        transcript(gene_id='g1', distance=100),
        transcript(gene_id='g2', distance=50),
        transcript(gene_id='g3', distance=50),
        transcript(gene_id='g4', consequence_terms=['stop_gained'], distance=1000),
        transcript(gene_id='g5', consequence_terms=['unknown']),
    ]
    result = parsed(spark, transcript_consequences=records)['transcriptConsequences']
    assert [x['targetId'] for x in result] == ['g4', 'g2', 'g3', 'g1']
    assert [x['transcriptIndex'] for x in result] == [1, 2, 3, 4]


def test_cross_references_hgvs_amino_acid_and_transcript_annotations(spark):
    result = parsed(
        spark,
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
    assert 'siftPrediction' not in first and 'polyphenPrediction' not in first
    assert result['variantDescription'] == (
        'Missense variant overlapping with GENE1, causing amino-acid change: A12B with moderate impact.'
        ' A high-confidence loss-of-function variant by loftee.'
    )


def test_all_predictors_without_sift_or_polyphen(spark):
    result = parsed(
        spark,
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


def test_hgvs_intergenic_fallback(spark):
    assert parsed(spark, intergenic_consequences=[{'hgvsg': '1:g.123A>G'}])['hgvsId'] == '1:g.123A>G'


def test_null_first_transcript_hgvs_does_not_fall_back(spark):
    result = parsed(spark, transcript_consequences=[transcript()], intergenic_consequences=[{'hgvsg': 'fallback'}])
    assert result['hgvsId'] is None


def test_non_coding_description_and_distance_formatting(spark):
    result = parsed(
        spark,
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


def test_invalid_position_is_dropped(spark):
    assert parse_vep(vep_records(spark, {'input': '1\tnot_an_int\t.\tA\tG'})).count() == 0


def test_hashed_variant_keeps_unhashed_protvar_xref(spark):
    alternate = 'G' * 300
    result = parsed(
        spark, hash_threshold=30, input=f'1\t123\t.\tA\t{alternate}', transcript_consequences=[transcript()]
    )
    assert result['variantId'].startswith('OTVAR_1_123_')
    assert result['dbXrefs'][-1]['id'] == f'1_123_A_{alternate}'


# --- variant index operations -------------------------------------------------------------------


def test_identifier_hashing(spark):
    frame = spark.createDataFrame(
        [('short', '1', 1), ('abcdefghij', '2', 2), ('missing', None, 3), (None, None, None)],
        'variantId string, chromosome string, position int',
    )
    hashed = frame.select(hash_long_variant_ids(f.col('variantId'), f.col('chromosome'), f.col('position'), 10))
    assert [row[0] for row in hashed.collect()] == [
        'short',
        'OTVAR_2_2_' + hashlib.md5(b'abcdefghij').hexdigest(),
        'OTVAR_' + hashlib.md5(b'missing').hexdigest(),
        None,
    ]


@pytest.mark.parametrize(
    ('method', 'score', 'assessment', 'expected'),
    [
        ('LOFTEE', None, 'HC', 1.0),
        ('LOFTEE', None, 'LC', 0.85),
        ('LOFTEE', 1.0, 'other', None),
        ('AlphaMissense', 0.0, None, -1.0),
        ('AlphaMissense', 0.77, None, 0.25),
        ('CADD', 10.0, None, 0.0),
        ('CADD', 81.0, None, 1.0),
        ('Pangolin', -1.0, None, 1.0),
        ('LossOfFunctionCuration', None, 'likely_lof', 0.5),
        ('LossOfFunctionCuration', None, 'not_lof', -1.0),
        ('SpliceAI', 0.2, None, 0.2),
        ('VEP', 0.4, None, 0.4),
        ('GERP', -5.0, None, 0.0),
        ('GERP', 2.0, None, 0.75),
        ('FoldX', -2.0, None, 1.0),
        ('FoldX', 1.0, None, 0.5),
        ('SIFT', 0.0, 'deleterious', None),
        ('PolyPhen', 1.0, 'probably_damaging', None),
        ('unknown', 1.0, None, None),
    ],
)
def test_normalisation(spark, method, score, assessment, expected):
    frame = spark.createDataFrame([(method, score, assessment)], 'method string, score float, assessment string')
    result = frame.select(normalise_score(f.col('score'), f.col('method'), f.col('assessment'))).first()[0]
    assert result == (None if expected is None else pytest.approx(expected, abs=1e-6))


def test_read_variant_annotations_inserts_missing_columns_and_hashes(spark, tmp_path):
    directory = tmp_path / 'gnomad' / 'chromosome=1'
    spark.createDataFrame(
        [('1_10_A_T', '1', 10), ('long_id_without_coordinates', None, None)],
        'variantId string, chromosome string, position int',
    ).write.parquet(str(directory))
    rows = read_variant_annotations(spark, str(tmp_path / 'gnomad'), 300).orderBy('variantId').collect()
    assert same_types(read_variant_annotations(spark, str(tmp_path / 'gnomad'), 300).schema, VARIANT_SCHEMA)
    assert rows[0].variantId == '1_10_A_T'
    assert rows[0].variantEffect == [] and rows[0].variantDescription is None
    assert rows[1].variantId == 'OTVAR_' + hashlib.md5(b'long_id_without_coordinates').hexdigest()


def test_add_annotation_unions_arrays_and_prefers_index_values(spark):
    index = index_rows(
        spark,
        [
            {
                'variantId': 'v1',
                'chromosome': '1',
                'rsIds': ['rs1'],
                'variantEffect': [effect('VEP')],
                'variantDescription': 'Index.',
            },
            {'variantId': 'v2', 'chromosome': '1', 'rsIds': [], 'variantEffect': []},
        ],
    )
    annotations = index_rows(
        spark,
        [
            {
                'variantId': 'v1',
                'chromosome': 'X',
                'rsIds': ['rs1', 'rs2'],
                'variantEffect': [effect('LossOfFunctionCuration', None, 'lof')],
                'variantDescription': 'Curated.',
                'alleleFrequencies': [{'populationName': 'nfe', 'alleleFrequency': 0.1}],
            },
            {'variantId': 'unrelated', 'rsIds': ['rs9']},
        ],
    )
    rows = {row.variantId: row.asDict(recursive=True) for row in add_annotation(index, annotations).collect()}
    assert set(rows) == {'v1', 'v2'}
    assert sorted(rows['v1']['rsIds']) == ['rs1', 'rs2']
    assert {x['method'] for x in rows['v1']['variantEffect']} == {'VEP', 'LossOfFunctionCuration'}
    assert rows['v1']['chromosome'] == '1'
    assert rows['v1']['variantDescription'] == 'Index. Curated.'
    assert rows['v1']['alleleFrequencies'] == [{'populationName': 'nfe', 'alleleFrequency': 0.1}]


def test_amino_acid_annotation_keeps_variants_without_accessions_and_ignores_unrelated(spark):
    def consequence(change: str, accessions: list[str] | None) -> dict:
        transcripts = VARIANT_SCHEMA['transcriptConsequences'].dataType
        assert isinstance(transcripts, t.ArrayType) and isinstance(transcripts.elementType, t.StructType)
        names = transcripts.elementType.fieldNames()
        return {**dict.fromkeys(names), 'aminoAcidChange': change, 'uniprotAccessions': accessions}

    index = index_rows(
        spark,
        [
            {
                'variantId': 'annotated',
                'variantEffect': [effect('VEP')],
                'transcriptConsequences': [consequence('A1T', ['P1', 'P2'])],
            },
            {
                'variantId': 'no_accession',
                'variantEffect': [effect('VEP')],
                'transcriptConsequences': [consequence('A1T', [])],
            },
            {
                'variantId': 'null_accession',
                'variantEffect': [effect('VEP')],
                'transcriptConsequences': [consequence('A1T', None)],
            },
        ],
    )
    foldx_effect = VARIANT_SCHEMA['variantEffect'].dataType
    annotations = spark.createDataFrame(
        [
            ('P2', 'A1T', [effect('FoldX', 1.0)]),
            ('P1', 'C9G', [effect('FoldX', 2.0)]),
            ('P9', 'A1T', [effect('FoldX', 3.0)]),
        ],
        t.StructType([
            t.StructField('uniprotAccession', t.StringType()),
            t.StructField('aminoAcidChange', t.StringType()),
            t.StructField('variantEffect', foldx_effect),
        ]),
    )
    rows = {
        row.variantId: [x.method for x in row.variantEffect]
        for row in annotate_with_amino_acid_consequences(index, annotations).collect()
    }
    assert rows == {'annotated': ['VEP', 'FoldX'], 'no_accession': ['VEP'], 'null_accession': ['VEP']}


# --- job and configuration ----------------------------------------------------------------------


def test_variant_job_writes_the_annotated_index(spark, tmp_path):
    vep = tmp_path / 'vep'
    vep.mkdir()
    record = {
        'input': '1\t123\t.\tA\tG',
        'most_severe_consequence': 'missense_variant',
        'transcript_consequences': [transcript(amino_acids='A/T', protein_end=1, swissprot=['P1.1'])],
    }
    (vep / '00000001.json').write_text(json.dumps(record) + '\n')
    (vep / '00000001.json_warnings.txt').write_text('WARNING: ignored\n')
    spark.createDataFrame(
        [('1_123_A_G', '1', 123, ['rs1'])], 'variantId string, chromosome string, position int, rsIds array<string>'
    ).write.parquet(str(tmp_path / 'gnomad'))
    spark.createDataFrame(
        [('P1', 'A1T', [effect('FoldX', 1.0)])],
        t.StructType([
            t.StructField('uniprotAccession', t.StringType()),
            t.StructField('aminoAcidChange', t.StringType()),
            t.StructField('variantEffect', VARIANT_SCHEMA['variantEffect'].dataType),
        ]),
    ).write.parquet(str(tmp_path / 'foldx'))
    output = tmp_path / 'output'
    variant(
        source={
            'vep': str(vep),
            'variant_annotations': [str(tmp_path / 'gnomad')],
            'amino_acid_annotations': [str(tmp_path / 'foldx')],
        },
        destination={'output': str(output)},
        settings={'output_partitions': 2},
        properties={},
    )
    result = spark.read.parquet(str(output))
    assert same_types(result.schema, VARIANT_SCHEMA)
    row = result.first().asDict(recursive=True)
    assert row['variantId'] == '1_123_A_G'
    assert row['rsIds'] == ['rs1']
    assert {x['method'] for x in row['variantEffect']} == {'VEP', 'FoldX'}


def test_variant_configuration_matches_models():
    config = yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text())
    [task] = config['steps']['variant']
    PysparkSpec.model_validate(task)
    assert task['pyspark'] == 'variant'
    VariantSource.model_validate(task['source'])
    VariantDestination.model_validate(task['destination'])
    VariantSettings.model_validate(task['settings'])


@pytest.mark.parametrize('settings', [{'hash_threshold': 0}, {'output_partitions': 0}, {'unknown': 1}])
def test_variant_settings_are_validated(settings):
    with pytest.raises(ValidationError):
        VariantSettings.model_validate(settings)
