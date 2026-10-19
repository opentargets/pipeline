"""Goldens for the clinical report entity mapping: mira's `map_entities` on `tests.clinical.mapping_fixtures`.

Captured with mira 1.0.5 and OnToma 2.7.1 on a local Spark session, by a standalone script outside
the repository (`.superpowers/sdd/Mira-Into-Pts-Plan/task4/capture_goldens.py` in the
mira-into-pts worktree). The reports, `CHEMBL_CURATION` and the disease and drug frames were passed
as they are; mira's call to OnToma's drug NER was replaced by a stub returning `NER` for each
label, and the NER cache started as `NER_CACHE`. `MAPPED` is mira's output sorted on every column
(nulls last), `NER_CACHE_WRITTEN` the cache it wrote, sorted, and `NER_EXTRACTED_LABELS` the
labels the stub was asked for.
"""

# ruff: noqa: RUF001

MAPPED_COLUMNS = [
    'id',
    'drugFromSource',
    'diseaseFromSource',
    'drugId',
    'diseaseId',
]

MAPPED = [
    ('CHEMBL_IND1', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', 'EFO_PRESET'),
    ('CHEMBL_IND1', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', 'EFO_PRESET'),
    ('CHEMBL_IND1', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', 'EFO_PRESET'),
    ('CHEMBL_IND1', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', 'EFO_PRESET'),
    ('CHEMBL_IND2', 'Aspirin', 'made-up disease', 'CHEMBL_PRESET', None),
    ('NCT01', 'Aspirin', 'Cystic fibrosis', 'CHEMBL25', 'MONDO_0009061'),
    ('NCT02', '  ASPIRIN  ', '  cystic   FIBROSIS ', 'CHEMBL25', 'MONDO_0009061'),
    ('NCT03', 'Interferon β-1a', 'α-thalassemia', 'CHEMBL1201562', 'MONDO_0011996'),
    ('NCT04', 'interferon beta-1a', 'a-thalassemia', 'CHEMBL1201562', 'MONDO_0011996'),
    ('NCT05', 'Shared Brand', 'shared syndrome', 'CHEMBL2', 'EFO_0000001'),
    ('NCT05', 'Shared Brand', 'shared syndrome', 'CHEMBL2', 'EFO_0000002'),
    ('NCT05', 'Shared Brand', 'shared syndrome', 'CHEMBL3', 'EFO_0000001'),
    ('NCT05', 'Shared Brand', 'shared syndrome', 'CHEMBL3', 'EFO_0000002'),
    ('NCT06', 'unknownium', 'made-up disease', None, None),
    ('NCT07', 'Ivacaftor 150 mg tablet', 'CF', 'CHEMBL2010601', 'MONDO_0009061'),
    ('NCT08', 'aspirin + kalydeco', 'mucoviscidosis', 'CHEMBL2010601', 'MONDO_0009061'),
    ('NCT08', 'aspirin + kalydeco', 'mucoviscidosis', 'CHEMBL25', 'MONDO_0009061'),
    ('NCT09', 'aspirin tabs', 'Cystic fibrosis', 'CHEMBL25', 'MONDO_0009061'),
    ('NCT10', None, 'Cystic fibrosis', None, 'MONDO_0009061'),
    ('NCT11', 'Study Drug X', 'Disease Y', 'CHEMBL_CURATED', 'EFO_CURATED_1'),
    ('NCT11', 'Study Drug X', 'Disease Y', 'CHEMBL_CURATED', 'EFO_CURATED_2'),
    ('NCT12', 'Aspirin', 'Cystic fibrosis', 'CHEMBL_CURATED', 'EFO_CURATED_3'),
    ('NCT13', 'Study Drug X', 'Cystic fibrosis', None, 'MONDO_0009061'),
]

NER_CACHE_WRITTEN = [
    ('Ivacaftor 150 mg tablet', ['ivacaftor']),
    ('Study Drug X', ['unmapped extraction']),
    ('aspirin + kalydeco', ['Aspirin', 'kalydeco']),
    ('aspirin tabs', ['acetylsalicylic acid']),
    ('stale label', ['ivacaftor']),
    ('unknownium', []),
]

NER_EXTRACTED_LABELS = [
    'Ivacaftor 150 mg tablet',
    'Study Drug X',
    'aspirin + kalydeco',
]
