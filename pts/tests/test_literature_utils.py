"""Tests for the literature_utils package (moved from opentargets/ot-literature)."""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.utils import parse_spark_schema
from pts.pyspark.literature_utils.common.schemas import SchemaValidationError, compare_struct_schemas
from pts.pyspark.literature_utils.dataset.match_mapped import IdValidReason, MatchMapped
from pts.pyspark.literature_utils.datasource.epmc.publication import EPMCPublication
from pts.pyspark.literature_utils.datasource.epmc.publication_id_lut import PublicationIdLUT


def _mapped_df(spark, rows):
    """Build a DataFrame with the full MatchMapped schema; missing fields are null."""
    return spark.createDataFrame(rows, schema=MatchMapped.get_schema())


class TestSchemas:
    """Test the packaged schemas and the schema comparison helpers."""

    @pytest.mark.parametrize('name', ['publication', 'match', 'match_mapped', 'cooccurrence'])
    def test_packaged_schemas_parse(self, name):
        assert parse_spark_schema(f'literature_{name}.json').fields

    def test_identical_schemas_have_no_issues(self):
        schema = parse_spark_schema('literature_match_mapped.json')
        assert compare_struct_schemas(schema, schema) == {}

    def test_flags_unexpected_and_mistyped_columns(self):
        expected = t.StructType([t.StructField('a', t.StringType()), t.StructField('b', t.LongType())])
        observed = t.StructType([t.StructField('b', t.StringType()), t.StructField('c', t.StringType())])
        issues = compare_struct_schemas(observed, expected)
        assert issues['unexpected_columns'] == ['c']
        assert issues['columns_with_non_matching_type'] == ['For column "b" found string instead of long']

    def test_flags_missing_mandatory_column(self):
        expected = t.StructType([t.StructField('a', t.StringType(), nullable=False)])
        assert compare_struct_schemas(t.StructType([]), expected)['missing_mandatory_columns'] == ['a']

    def test_recurses_into_structs_in_arrays(self):
        def schema(inner_type):
            inner = t.StructType([t.StructField('x', inner_type)])
            return t.StructType([t.StructField('arr', t.ArrayType(inner))])

        issues = compare_struct_schemas(schema(t.StringType()), schema(t.LongType()))
        assert issues['columns_with_non_matching_type'] == ['For column "arr[][].x" found string instead of long']

    def test_dataset_rejects_unexpected_column(self, spark):
        df = spark.createDataFrame([Row(pmid='1', notInSchema='x')])
        with pytest.raises(SchemaValidationError, match='MatchMapped') as excinfo:
            MatchMapped(df)
        assert 'unexpected_columns: notInSchema' in str(excinfo.value)

    def test_dataset_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match='Invalid type for _df'):
            MatchMapped('not a dataframe')


class TestMatchMapped:
    """Test MatchMapped scoring, validation and cooccurrence generation."""

    @pytest.mark.parametrize(
        ('section', 'score'),
        [
            ('title', 10.0),
            ('abstract', 3.0),
            ('results', 5.0),
            ('fig', 5.0),
            ('discussion', 2.0),
            ('methods', 1.0),
            ('unknown', 1.0),
            # word boundaries: 'titles' does not match 'title'
            ('titles', 1.0),
            # the highest-scoring matching section wins
            ('title abstract', 10.0),
        ],
    )
    def test_section_to_score(self, spark, section, score):
        df = spark.createDataFrame([Row(section=section)])
        result = df.select(MatchMapped._section_to_score(f.col('section')).alias('s')).first()
        assert result['s'] == score

    def test_update_flag(self, spark):
        df = spark.createDataFrame(
            [(None, True), (['a'], True), (None, False)],
            schema='flags array<string>, cond boolean',
        )
        result = df.select(
            MatchMapped._update_flag(f.col('flags'), f.col('cond'), IdValidReason.ONLY_ID).alias('flags')
        ).collect()
        assert [r['flags'] for r in result] == [[IdValidReason.ONLY_ID.value], ['a', IdValidReason.ONLY_ID.value], []]

    def test_identify_valid_ids(self, spark):
        df = _mapped_df(
            spark,
            [
                # single id for the label
                {'pmid': '1', 'entityIds': [Row(entityId='E1', entitySource='syn')], 'mappedId': 'E1'},
                # two ids, all from a trusted source
                {
                    'pmid': '2',
                    'entityIds': [Row(entityId='E1', entitySource='name'), Row(entityId='E2', entitySource='name')],
                    'mappedId': 'E1',
                },
                # two ids, one from an untrusted source
                {
                    'pmid': '3',
                    'entityIds': [Row(entityId='E1', entitySource='name'), Row(entityId='E2', entitySource='syn')],
                    'mappedId': 'E1',
                },
            ],
        )
        result = {r['pmid']: r for r in MatchMapped._identify_valid_ids(df, ['name']).collect()}
        assert result['1']['validReasons'] == [IdValidReason.ONLY_ID.value]
        assert result['2']['validReasons'] == [IdValidReason.ID_FROM_TRUSTED_SOURCE.value]
        assert result['3']['validReasons'] == []
        assert [result[k]['isValid'] for k in ('1', '2', '3')] == [True, True, False]

    def test_disambiguate_uses_ids_validated_elsewhere_in_the_publication(self, spark):
        ambiguous = [Row(entityId='E1', entitySource='syn'), Row(entityId='E2', entitySource='syn')]
        df = _mapped_df(
            spark,
            [
                # an unambiguous mention validates E1 for pmid 1
                {'pmid': '1', 'label': 'gene one', 'entityIds': [Row(entityId='E1', entitySource='syn')],
                 'mappedId': 'E1', 'isMapped': True},
                # an ambiguous mention, exploded into one row per candidate id
                {'pmid': '1', 'label': 'G1', 'entityIds': ambiguous, 'mappedId': 'E1', 'isMapped': True},
                {'pmid': '1', 'label': 'G1', 'entityIds': ambiguous, 'mappedId': 'E2', 'isMapped': True},
                # the same ambiguous mention in a publication with no validating mention
                {'pmid': '2', 'label': 'G1', 'entityIds': ambiguous, 'mappedId': 'E1', 'isMapped': True},
                # unmapped rows are dropped
                {'pmid': '1', 'label': 'nothing', 'entityIds': [], 'mappedId': None, 'isMapped': False},
            ],
        )
        result = {
            (r['pmid'], r['label'], r['mappedId']): (r['isValid'], r['validReasons'])
            for r in MatchMapped(df).disambiguate(trusted_sources=['name']).df.collect()
        }
        assert result == {
            ('1', 'gene one', 'E1'): (True, [IdValidReason.ONLY_ID.value]),
            ('1', 'G1', 'E1'): (True, [IdValidReason.DISAMBIGUATED.value]),
            ('1', 'G1', 'E2'): (False, []),
            ('2', 'G1', 'E1'): (False, []),
        }

    def test_generate_target_disease_cooccurrences(self, spark):
        def mention(type_, label, mapped_id, text='s1', section='abstract'):
            return {'pmid': '1', 'text': text, 'section': section, 'type': type_, 'label': label,
                    'mappedId': mapped_id, 'startInSentence': 0, 'endInSentence': 1}

        df = _mapped_df(
            spark,
            [
                mention('GP', 'BRCA1', 'ENSG1'),
                mention('DS', 'cancer', 'EFO1'),
                mention('DS', 'unmapped', None),
                # different sentence: no pair with BRCA1
                mention('DS', 'asthma', 'EFO2', text='s2'),
                mention('CD', 'aspirin', 'CHEMBL1'),
            ],
        )
        rows = MatchMapped(df).generate_target_disease_cooccurrences().df.collect()
        result = {(r['label1'], r['label2']): (r['type'], r['isMapped'], r['evidenceScore']) for r in rows}
        assert result == {
            ('BRCA1', 'cancer'): ('GP-DS', True, 3.0),
            ('BRCA1', 'unmapped'): ('GP-DS', False, 3.0),
        }


class TestEPMCPublication:
    """Test the EPMC publication building blocks pts calls directly."""

    def test_annotate_fulltexts_with_pmid(self, spark):
        fulltext = spark.createDataFrame(
            [('PMC1', None), ('PMC2', '2'), ('PMC3', '999'), ('PMC4', '4')],
            schema='pmcid string, pmid string',
        )
        lut = spark.createDataFrame(
            [('1', 'PMC1'), ('2', 'PMC2'), ('3', 'PMC3')],
            schema='pmid_lut string, pmcid_lut string',
        )
        result = EPMCPublication._annotate_fulltexts_with_pmid(fulltext, lut)
        assert result.columns == ['pmcid', 'pmid']
        # PMC1 gains its pmid; PMC3 disagrees with the lut and PMC4 is not in it, so both are dropped
        assert sorted(tuple(r) for r in result.collect()) == [('PMC1', '1'), ('PMC2', '2')]

    def test_merge_abstracts_with_fulltexts(self, spark):
        abstract = spark.createDataFrame([('1', 'abstract'), ('2', 'abstract')], schema='pmid string, kind string')
        fulltext = spark.createDataFrame([('2', 'PMC2', 'fulltext')], schema='pmid string, pmcid string, kind string')
        result = EPMCPublication._merge_abstracts_with_fulltexts(abstract, fulltext)
        assert sorted((r['pmid'], r['pmcid'], r['kind']) for r in result.collect()) == [
            ('1', None, 'abstract'),
            ('2', 'PMC2', 'fulltext'),
        ]

    def test_get_most_recent_publications(self, spark):
        df = spark.createDataFrame(
            [
                ('1', None, '2026-01-01 00:00:00', 'abstract', 'old'),
                ('1', None, '2026-02-01 00:00:00', 'abstract', 'new'),
                ('2', 'PMC2', '2026-01-01 00:00:00', 'fulltext', 'only'),
            ],
            schema='pmid string, pmcid string, timestamp string, kind string, payload string',
        )
        result = EPMCPublication._get_most_recent_publications(df)
        assert sorted(result.columns) == ['payload', 'pmcid', 'pmid']
        assert sorted((r['pmid'], r['payload']) for r in result.collect()) == [('1', 'new'), ('2', 'only')]


class TestPublicationIdLUT:
    """Test the publication id lookup table parser."""

    def test_lut_parser(self, spark):
        df = spark.createDataFrame(
            [
                ('1', 'PMC1', 'doi1'),
                ('1', 'PMC1', 'doi1'),
                ('2', None, 'doi2'),
                (None, 'PMC3', 'doi3'),
                ('4', 'X4', 'doi4'),
            ],
            schema='PMID string, PMCID string, DOI string',
        )
        result = PublicationIdLUT._lut_parser(df)
        assert result.columns == ['pmid_lut', 'pmcid_lut']
        assert [tuple(r) for r in result.collect()] == [('1', 'PMC1')]
