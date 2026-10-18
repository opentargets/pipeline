"""Frames shared by the OnToma lookup table tests; no spark and no ontoma."""

# ruff: noqa: RUF001

import polars as pl

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

LABEL_SOURCE = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))
TARGET_SCHEMA = pl.Schema({
    'id': pl.String,
    'approvedName': pl.String,
    'approvedSymbol': pl.String,
    'nameSynonyms': LABEL_SOURCE,
    'symbolSynonyms': LABEL_SOURCE,
    'proteinIds': pl.List(pl.Struct({'id': pl.String, 'source': pl.String})),
    'obsoleteNames': LABEL_SOURCE,
    'obsoleteSymbols': LABEL_SOURCE,
})
_LABEL_SOURCE_DDL = 'array<struct<label:string,source:string>>'
TARGET_DDL = (
    f'id string, approvedName string, approvedSymbol string, nameSynonyms {_LABEL_SOURCE_DDL}, '
    f'symbolSynonyms {_LABEL_SOURCE_DDL}, proteinIds array<struct<id:string,source:string>>, '
    f'obsoleteNames {_LABEL_SOURCE_DDL}, obsoleteSymbols {_LABEL_SOURCE_DDL}'
)


def _labels(*labels: str | None) -> list[dict | None]:
    return [{'label': label, 'source': 's'} for label in labels]


TARGET = pl.DataFrame(
    [
        (
            'ENSG01',
            ' tumor necrosis factor α ',
            'TNF',
            [*_labels('TNF-α', 'cachectin', None, '', ' '), None],
            _labels('TNFA', 'TNFSF2', 'Straße'),
            [{'id': 'P01375', 'source': 'uniprot_swissprot'}, {'id': None, 'source': 'x'}, None],
            _labels('tumour necrosis factor-β', 'Papillon-Lefèvre'),
            _labels('TNFA', 'ΑΣ'),
        ),
        ('ENSG02', None, None, None, None, None, None, None),
        ('ENSG03', '', ' ', [], [], [], [], []),
        ('ENSG04', 'shared name', 'TNF', None, _labels('TNF'), None, None, None),
        (None, 'no id', 'NOID', None, None, None, None, None),
    ],
    schema=TARGET_SCHEMA,
    orient='row',
)

DRUG_SCHEMA = pl.Schema({
    'id': pl.String,
    'name': pl.String,
    'synonyms': LABEL_SOURCE,
    'tradeNames': LABEL_SOURCE,
    'crossReferences': pl.List(pl.Struct({'source': pl.String, 'ids': pl.List(pl.String)})),
})
DRUG_DDL = (
    f'id string, name string, synonyms {_LABEL_SOURCE_DDL}, tradeNames {_LABEL_SOURCE_DDL}, '
    'crossReferences array<struct<source:string,ids:array<string>>>'
)


def _sourced(source: str | None, *labels: str | None) -> list[dict]:
    return [{'label': label, 'source': source} for label in labels]


DRUG = pl.DataFrame(
    [
        (
            'CHEMBL1',
            'IVACAFTOR',
            [
                *_sourced('ChEMBL', 'Ivacaftor component of symkevi', 'VX-770', ' β-ivacaftor ', None),
                *_sourced('AACT', 'kalydeco', 'VX-770', 'Ivacaftor COMPONENT OF  Trikafta ;- '),
                *_sourced('other', 'ignored synonym'),
                *_sourced(None, 'null source synonym'),
                *_sourced('ChEMBL', 'a component of b\r\n', 'a component of c\r', 'A COMPONENT OF e\u0085'),
                *_sourced('ChEMBL', 'a \u212aomponent of d', 'a component of \u017f x'),
                None,
            ],
            [
                *_sourced(
                    'ChEMBL', 'Kalydeco', 'a component of b component of  //Orkambi,, ', 'Ivacaftor component of'
                ),
                *_sourced('ChEMBL', 'x component of y\n', 'x component of \tz\x0b', 'x component of nbsp\u00a0'),
                *_sourced('ChEMBL', 'x component of a\u2028b', 'x Component Of   spaced    product  ', None),
            ],
            [
                {'source': 'DailyMed', 'ids': ['ivacaftor%20tablets', 'kalydeco']},
                {'source': 'USAN', 'ids': ['IVACAFTOR%20%20X']},
                {'source': 'EMA', 'ids': ['https://ema.europa.eu/en/medicines/human/EPAR/kalydeco', 'no-epar', None]},
                {'source': 'INN', 'ids': ['9123']},
                {'source': 'drugbank', 'ids': ['DB08820']},
                {'source': None, 'ids': ['null source']},
                None,
            ],
        ),
        ('CHEMBL2', 'CHEMBL2', [], [], None),
        ('CHEMBL3', 'chembl3', None, None, [{'source': 'DailyMed', 'ids': ['kept%20too']}]),
        ('CHEMBL4', 'ChEMBL4', _sourced('AACT', 'aact only'), None, None),
        ('CHEMBL5', 'CHEMBL5', None, _sourced('ChEMBL', 'Brand'), [{'source': 'EMA', 'ids': None}]),
        ('CHEMBL6', 'CHEMBL6', _sourced('other', 'other only'), [None], [{'source': 'USAN', 'ids': ['x']}]),
        ('CHEMBL7', None, _sourced('ChEMBL', 'nameless'), None, None),
        ('CHEMBL8', None, None, None, None),
        (
            'CHEMBL9',
            ' Ñandú ',
            None,
            None,
            [{'source': 'USAN', 'ids': ['lost']}, {'source': 'DailyMed', 'ids': None}, {'source': 'EMA', 'ids': []}],
        ),
        (None, 'no id', None, _sourced('ChEMBL', 'no id component of product'), None),
    ],
    schema=DRUG_SCHEMA,
    orient='row',
)
