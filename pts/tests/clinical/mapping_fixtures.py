"""Frames shared by the clinical report entity mapping test and its goldens; no spark, no ontoma and no mira."""

# ruff: noqa: RUF001

import polars as pl

from pts.schemas.drug_molecule import molecule_schema

DISEASE_SCHEMA = pl.Schema({
    'id': pl.String,
    'name': pl.String,
    'description': pl.String,
    'exactSynonyms': pl.List(pl.String),
    'narrowSynonyms': pl.List(pl.String),
    'broadSynonyms': pl.List(pl.String),
    'relatedSynonyms': pl.List(pl.String),
    'obsoleteTerms': pl.List(pl.String),
    'dbXRefs': pl.List(pl.String),
    'obsoleteXRefs': pl.List(pl.String),
})

#: Shaped like `output/disease`, with one column no lookup table reads.
DISEASE = pl.DataFrame(
    [
        ('MONDO_0009061', 'cystic fibrosis', 'd', ['mucoviscidosis', 'CF'], [], [], [], [], ['OMIM:219700'], []),
        ('MONDO_0011996', 'α-thalassemia', 'd', [], [], [], [], [], [], []),
        ('EFO_0000001', 'shared syndrome', 'd', [], [], [], [], [], [], []),
        ('EFO_0000002', 'shared syndrome', 'd', [], [], [], [], [], [], []),
    ],
    schema=DISEASE_SCHEMA,
    orient='row',
)


def _labels(source: str, *labels: str) -> list[dict]:
    return [{'label': label, 'source': source} for label in labels]


def _molecule(chembl_id: str, name: str, synonyms: list[dict], trade_names: list[dict]) -> dict:
    return {**dict.fromkeys(molecule_schema.names()), 'id': chembl_id, 'name': name} | {
        'synonyms': synonyms,
        'tradeNames': trade_names,
        'crossReferences': [],
        'childChemblIds': [],
        'drugType': 'Small molecule',
    }


#: Shaped like `intermediate/chembl_molecule`.
DRUG = pl.DataFrame(
    [
        _molecule('CHEMBL25', 'ASPIRIN', _labels('ChEMBL', 'acetylsalicylic acid'), _labels('ChEMBL', 'Ecotrin')),
        _molecule('CHEMBL2010601', 'IVACAFTOR', [], _labels('ChEMBL', 'Kalydeco')),
        _molecule('CHEMBL1201562', 'INTERFERON BETA-1A', _labels('ChEMBL', 'interferon β-1a'), []),
        _molecule('CHEMBL2', 'DRUG TWO', [], _labels('ChEMBL', 'Shared Brand')),
        _molecule('CHEMBL3', 'DRUG THREE', [], _labels('ChEMBL', 'Shared Brand')),
    ],
    schema=molecule_schema,
)

REPORT_SCHEMA = pl.Schema({
    'id': pl.String,
    'drugFromSource': pl.String,
    'diseaseFromSource': pl.String,
    'drugId': pl.String,
    'diseaseId': pl.String,
})

#: Reports to map: (id, drugFromSource, diseaseFromSource, drugId, diseaseId).
REPORTS = pl.DataFrame(
    [
        ('NCT01', 'Aspirin', 'Cystic fibrosis', None, None),
        ('NCT02', '  ASPIRIN  ', '  cystic   FIBROSIS ', None, None),
        ('NCT03', 'Interferon β-1a', 'α-thalassemia', None, None),
        ('NCT04', 'interferon beta-1a', 'a-thalassemia', None, None),
        ('NCT05', 'Shared Brand', 'shared syndrome', None, None),
        ('NCT06', 'unknownium', 'made-up disease', None, None),
        ('NCT07', 'Ivacaftor 150 mg tablet', 'CF', None, None),
        ('NCT08', 'aspirin + kalydeco', 'mucoviscidosis', None, None),
        ('NCT09', 'aspirin tabs', 'Cystic fibrosis', None, None),
        ('NCT10', None, 'Cystic fibrosis', None, None),
        ('NCT11', 'Study Drug X', 'Disease Y', None, None),
        ('NCT12', 'Aspirin', 'Cystic fibrosis', None, None),
        ('NCT13', 'Study Drug X', 'Cystic fibrosis', None, None),
        ('CHEMBL_IND1', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'Aspirin', 'made-up disease', 'CHEMBL_PRESET', None),
    ],
    schema=REPORT_SCHEMA,
    orient='row',
)

#: ChEMBL's clinical trial curation: (studyId, drugFromSource, drugId, diseaseFromSource, diseaseId).
CHEMBL_CURATION = pl.DataFrame(
    [
        ('nct11', ' study drug x ', 'CHEMBL_CURATED', 'disease y', 'EFO_CURATED_1'),
        ('NCT11', 'Study Drug X', 'CHEMBL_CURATED', 'Disease Y', 'EFO_CURATED_2'),
        ('NCT12', 'aspirin', 'CHEMBL_CURATED', 'cystic  fibrosis', 'EFO_CURATED_3'),
        ('NCT13', 'Other Drug', 'CHEMBL_OTHER', 'Cystic fibrosis', None),
    ],
    schema={
        'studyId': pl.String,
        'drugFromSource': pl.String,
        'drugId': pl.String,
        'diseaseFromSource': pl.String,
        'diseaseId': pl.String,
    },
    orient='row',
)

#: What the stubbed drug NER extracts from each label; any other label extracts nothing.
NER = {
    'Ivacaftor 150 mg tablet': ['ivacaftor'],
    'aspirin + kalydeco': ['Aspirin', 'kalydeco'],
    'Study Drug X': ['unmapped extraction'],
}

#: The NER cache before the run: a label in the reports, extracted differently from the stub, and a stale label.
NER_CACHE = pl.DataFrame(
    [('aspirin tabs', ['acetylsalicylic acid']), ('stale label', ['ivacaftor']), ('unknownium', [])],
    schema={'query_label': pl.String, 'extracted_drugs': pl.List(pl.String)},
    orient='row',
)


def ner_stub(labels: pl.Series, batch_size: int) -> pl.Series:
    """`extract_drug_entities` with `NER` in place of the models."""
    return pl.Series(labels.name, [NER.get(label, []) for label in labels], dtype=pl.List(pl.String))
