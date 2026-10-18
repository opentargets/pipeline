"""Tests for the polars port of OnToma's lookup-table builders, checked against OnToma itself."""

# ruff: noqa: RUF001

import polars as pl
import pytest
from ontoma import DiseaseCuration, OpenTargetsDisease

from pts.schemas.ontoma_lut import RAW_ENTITY_LUT
from pts.transformers.ontoma_lut.build import curation_label_lut, disease_id_lut, disease_label_lut

DISEASE_SCHEMA = pl.Schema({
    'id': pl.String,
    'name': pl.String,
    'exactSynonyms': pl.List(pl.String),
    'narrowSynonyms': pl.List(pl.String),
    'broadSynonyms': pl.List(pl.String),
    'relatedSynonyms': pl.List(pl.String),
    'obsoleteTerms': pl.List(pl.String),
    'dbXRefs': pl.List(pl.String),
    'obsoleteXRefs': pl.List(pl.String),
})
DISEASE_DDL = (
    'id string, name string, exactSynonyms array<string>, narrowSynonyms array<string>, '
    'broadSynonyms array<string>, relatedSynonyms array<string>, obsoleteTerms array<string>, '
    'dbXRefs array<string>, obsoleteXRefs array<string>'
)

DISEASE = pl.DataFrame(
    [
        (
            'MONDO_0000001',
            'Cystic fibrosis',
            ['CF', ' Mucoviscidosis ', None, '', 'CF', '\tfibrosis\t'],
            None,
            [],
            ['CF', 'ΟΔΟΣ', ' ALS '],
            ['EFO_0000001', 'MONDO:0000002'],
            [
                'PMID:12345',
                'doi:10.1/x',
                'http://example.org/a',
                'orcid:1',
                'ORDO:558',
                'Orphanet_559',
                'efo:Orphanet_560',
                'MeSH:D003550',
                ' umls:C0010674 ',
                '\tICD10:E84',
                'É:1',
            ],
            None,
        ),
        (
            'EFO_0000002',
            'Papillon-Lefèvre syndrome',
            [
                'α-thalassemia',
                'Straße disease',
                'İstanbul fever',
                'ﬁbrosis',
                'ΟΔΟΣ syndrome',
                'Ñandú',
                'ΑΣ',
                'AbCdEfGhIj',
            ],
            ['  '],
            None,
            None,
            None,
            [
                'ORPHA:100',
                'icd10:Q82.8',
                'ßeta:1',
                'quant:1',
                'wikipedia:foo',
                'um-bbd_pathwayid:x',
                'TEST:\ua7d31',
                'ουσία:2',
            ],
            ['OMIM:245000', 'Orphanet_100', 'person:x', 'isbn:1'],
        ),
        ('MONDO_0000003', None, None, None, None, None, None, ['Orphanet_100', 'OMIM:245000'], None),
        ('MONDO_0000004', ' ', [], [], [], [], [], [], []),
        ('MONDO_0000010', None, None, None, None, None, None, ['DOID:10'], None),
        (None, 'no id', None, None, None, None, None, ['DOID:10'], None),
    ],
    schema=DISEASE_SCHEMA,
    orient='row',
)

CURATION = pl.DataFrame(
    [
        ('http://www.ebi.ac.uk/efo/EFO_0000002', 'Papillon Lefevre'),
        ('http://purl.obolibrary.org/obo/MONDO_0000002', ' obsolete label '),
        ('http://www.orpha.net/ORDO/Orphanet_100', 'two-id label'),
        ('http://identifiers.org/omim/OMIM_245000', 'remapped by rank'),
        ('http://www.ebi.ac.uk/efo/EFO_9999999', 'no match'),
        ('http://x/EFO_0000002', '#Q82.8 Papillon-Lefèvre'),
        ('http://x/MONDO_0000001', 'E84 #Chapter IV Endocrine'),
        ('http://x/MONDO_0000001', 'a#b#E84.0-A1 Cystic'),
        ('http://x/MONDO_0000001', 'x#lower e84 kept'),
        ('http://x/MONDO_0000001', '1)'),
        ('http://x/MONDO_0000001', '2)'),
        ('http://x/MONDO_0000001', '3)'),
        ('http://x/MONDO_0000001', '1)\n'),
        ('http://x/MONDO_0000001', '\tCF\t'),
        ('http://x/MONDO_0000001', ' MS '),
        ('http://x/MONDO_0000001', 'ß İ ﬁ α'),
        ('http://x/MONDO_0000001', '#'),
        ('notaurl', 'x'),
        (None, 'null tag'),
        ('http://x/MONDO_0000001', None),
        ('http://www.ebi.ac.uk/efo/EFO_0000002\n', 'newline tag'),
        ('http://x/MONDO_0000001/', 'trailing slash'),
        ('http://x/efo_0000002', 'lowercase tag'),
        ('http://x/é_1', 'unicode word'),
        ('http\r/EFO_0000002', 'carriage return'),
        ('http://x/ORPHANET_558', 'orphanet'),
        ('http://x/DOID_10', 'label one'),
    ],
    schema={'SEMANTIC_TAG': pl.String, 'PROPERTY_VALUE': pl.String},
    orient='row',
)


@pytest.fixture(scope='module')
def spark_disease(spark):
    return spark.createDataFrame(DISEASE.rows(), DISEASE_DDL)


def _rows(lut) -> list[tuple]:
    rows = (tuple(r) for r in lut.df.select(*RAW_ENTITY_LUT.names()).collect())
    return sorted(rows, key=lambda row: [(value is not None, value) for value in row])


def test_disease_label_lut_matches_ontoma(spark_disease) -> None:
    out = disease_label_lut(DISEASE)
    assert out.rows() == _rows(OpenTargetsDisease.as_label_lut(spark_disease))


def test_disease_id_lut_matches_ontoma(spark_disease) -> None:
    out = disease_id_lut(DISEASE)
    assert out.rows() == _rows(OpenTargetsDisease.as_id_lut(spark_disease))


def test_curation_label_lut_matches_ontoma(spark, spark_disease) -> None:
    out = curation_label_lut(CURATION, 'ot_curation', DISEASE)
    curation = spark.createDataFrame(CURATION.rows(), 'SEMANTIC_TAG string, PROPERTY_VALUE string')
    expected = _rows(DiseaseCuration.as_label_lut(curation, 'ot_curation', spark_disease, True))
    assert out.rows() == expected
    ids = {(label, entity_id) for entity_id, label, *_ in expected}
    assert ('remapped by rank', 'MONDO_0000003') in ids
    assert ('obsolete label', 'MONDO_0000001') in ids
    assert ('label one', 'MONDO_0000010') in ids
    assert not {label for _, label in ids} & {'two-id label', 'no match', 'unicode word', 'carriage return', '1)'}


@pytest.mark.parametrize('build', [disease_label_lut, disease_id_lut])
def test_schema_is_raw_entity_lut(build) -> None:
    assert build(DISEASE).schema == RAW_ENTITY_LUT
    assert build(DISEASE.clear()).schema == RAW_ENTITY_LUT


def test_curation_schema_is_raw_entity_lut() -> None:
    assert curation_label_lut(CURATION, 'eva_clinvar', DISEASE).schema == RAW_ENTITY_LUT
    assert curation_label_lut(CURATION.clear(), 'eva_clinvar', DISEASE).schema == RAW_ENTITY_LUT
