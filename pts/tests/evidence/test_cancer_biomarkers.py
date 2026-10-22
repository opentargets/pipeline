"""Tests for Cancer Biomarkers evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.cancer_biomarkers import (
    CANCER_BIOMARKERS,
    explode_alterations,
    generate,
    get_variant_id,
    initcap,
)

ASSOCIATIONS_HEADER = [
    'Alteration',
    'AlterationType',
    'Association',
    'Biomarker',
    'Drug',
    'DrugFullName',
    'EvidenceLevel',
    'Gene',
    'PrimaryTumorTypeFullName',
    'Source',
    'gDNA',
    'IndividualMutation',
]

URL = pl.Struct({'niceName': pl.String, 'url': pl.String})
GENE_EXPRESSION = pl.Struct({'id': pl.String, 'name': pl.String})
GENETIC_VARIATION = pl.Struct({'functionalConsequenceId': pl.String, 'id': pl.String, 'name': pl.String})
RAW_SCHEMA = {
    'datasourceId': pl.String,
    'datatypeId': pl.String,
    'drugFromSource': pl.String,
    'drugId': pl.String,
    'drugResponse': pl.String,
    'targetFromSourceId': pl.String,
    'diseaseFromSource': pl.String,
    'diseaseFromSourceMappedId': pl.String,
    'confidence': pl.String,
    'biomarkerName': pl.String,
    'literature': pl.List(pl.String),
    'urls': pl.List(URL),
    'biomarkers': pl.Struct({
        'geneExpression': pl.List(GENE_EXPRESSION),
        'geneticVariation': pl.List(GENETIC_VARIATION),
    }),
}

CT = 'https://clinicaltrials.gov/ct2/show/'
MELANOMA = 'EFO_0000389'
ANY_CANCER = 'EFO_0000311'
ALL = 'EFO_0000220'

# Every field is a string; '' is an empty, so null, tsv cell.
BASE = {
    'Alteration': 'MSI:.',
    'AlterationType': 'BIA',
    'Association': 'Unknown',
    'Biomarker': 'MSI high',
    'Drug': 'Pembrolizumab',
    'DrugFullName': 'Pembrolizumab (PD1 inhibitor)',
    'EvidenceLevel': 'FDA guidelines',
    'Gene': 'MLH1',
    'PrimaryTumorTypeFullName': 'Any cancer type',
    'Source': 'PMID:666',
    'gDNA': '',
    'IndividualMutation': '',
}

ASSOCIATIONS = [
    # Two rows of one evidence: their literature, urls and variants are collected.
    {
        **BASE,
        'Alteration': 'BRAF:V600E',
        'AlterationType': 'MUT',
        'Association': 'Responsive',
        'Biomarker': 'BRAF (V600E)',
        'Drug': 'Vemurafenib',
        'Gene': 'BRAF',
        'PrimaryTumorTypeFullName': 'Cutaneous melanoma',
        'Source': 'PMID:111;NCT0001',
        'gDNA': 'chr7:g.140453136A>T',
        'IndividualMutation': 'BRAF:V600E',
    },
    {
        **BASE,
        'Alteration': 'BRAF:V600E',
        'AlterationType': 'MUT',
        'Association': 'Responsive',
        'Biomarker': 'BRAF (V600E)',
        'Drug': 'Vemurafenib',
        'Gene': 'BRAF',
        'PrimaryTumorTypeFullName': 'Cutaneous melanoma',
        'Source': 'PMID:222 ;AACR 2012',
        'gDNA': 'chr7:g.140453136_140453137delCAinsTT',
        'IndividualMutation': 'BRAF:V600K',
    },
    # Bracketed drugs split on ',' and evidence levels on ','; one type serves both alterations.
    {
        **BASE,
        'Alteration': 'MET:Y1230C;Y1235D',
        'AlterationType': 'MUT',
        'Association': 'Resistant',
        'Biomarker': 'MET (Y1230C,Y1235D)',
        'Drug': '[Crizotinib,Capmatinib]',
        'EvidenceLevel': 'Pre-clinical,Case report',
        'Gene': 'MET',
        'Source': 'PMID:333',
    },
    # Gene names are overridden before upper-casing, so `kmt2d` and `MLL2` both become KMT2D.
    {
        **BASE,
        'Alteration': 'MLL:amp',
        'AlterationType': 'CNA',
        'Association': 'Not Responsive',
        'Biomarker': 'MLL amplification',
        'Drug': 'Venetoclax',
        'EvidenceLevel': 'Early trials',
        'Gene': 'MLL;kmt2d;MLL2',
        'PrimaryTumorTypeFullName': 'Acute lymphoblastic leukemia',
        'Source': 'PMID:444',
    },
    # Special alteration strings; an empty drug falls back to the drug's full name.
    {
        **BASE,
        'Alteration': 'NRAS:.12.,.13.,.59.,.61.,.117.,.146.;ARAF:.;ALK__EML4',
        'AlterationType': 'MUT;CNA;FUS',
        'Association': 'Responsive',
        'Biomarker': 'NRAS (12,13,59,61,117,146)',
        'Drug': '[]',
        'DrugFullName': 'MEK inhibitors',
        'EvidenceLevel': 'Late trials',
        'Gene': 'NRAS',
        'PrimaryTumorTypeFullName': 'Cutaneous melanoma',
        'Source': 'PMID:555',
    },
    # A fusion of a combined biomarker keeps its alteration; a drug name with two ids gives two rows.
    {
        **BASE,
        'Alteration': 'BCR__ABL1',
        'AlterationType': 'FUS',
        'Association': 'Resistant',
        'Biomarker': 'BCR-ABL1 + T315I',
        'Drug': 'Imatinib',
        'EvidenceLevel': 'Clinical trials',
        'Gene': 'ABL1',
        'PrimaryTumorTypeFullName': 'Acute lymphoblastic leukemia',
        'Source': 'NCT0002',
    },
    # Expression alterations; an unknown source and an unmapped tumour type give nulls.
    {
        **BASE,
        'Alteration': 'ERBB2:over;ESR1:under;PGR:norm',
        'AlterationType': 'EXPR',
        'Association': 'Increased Toxicity (Ototoxicity)',
        'Biomarker': 'ERBB2 overexpression',
        'Drug': 'Trastuzumab',
        'EvidenceLevel': 'NCCN guidelines',
        'Gene': 'ERBB2',
        'PrimaryTumorTypeFullName': 'Breast adenocarcinoma',
        'Source': 'doi: 10.1/x',
    },
    # A biomarker alteration has no functional consequence; an unknown association is kept.
    BASE,
    # A null drug falls back to the drug's full name, like an empty one.
    {**BASE, 'Biomarker': 'no drug', 'Drug': ''},
    # A null source, evidence level, alteration type or alteration explodes to no rows.
    {**BASE, 'Biomarker': 'no source', 'Source': ''},
    {**BASE, 'Biomarker': 'no confidence', 'EvidenceLevel': ''},
    {**BASE, 'Biomarker': 'no type', 'AlterationType': ''},
    {**BASE, 'Biomarker': 'no alteration', 'Alteration': ''},
]

SOURCES = [{'source': 'AACR 2012', 'url': 'http://aacr/2012', 'label': 'AACR Meeting 2012'}]
DISEASES = [
    {'name': 'Cutaneous_melanoma', 'label': 'Cutaneous melanoma', 'url': f'http://www.ebi.ac.uk/efo/{MELANOMA}'},
    {'name': 'Any_cancer_type', 'label': 'Any cancer type', 'url': f'http://www.ebi.ac.uk/efo/{ANY_CANCER}'},
    {'name': 'Acute_lymphoblastic_leukemia', 'label': 'ALL', 'url': f'http://www.ebi.ac.uk/efo/{ALL}'},
]
DRUGS = {
    'id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3', 'CHEMBL4'],
    'name': ['VEMURAFENIB', 'CRIZOTINIB', 'IMATINIB', 'IMATINIB'],
    'drugType': ['Small molecule'] * 4,
}


def _jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return str(path)


def _sources(src: Path, associations: list[dict] = ASSOCIATIONS) -> dict[str, str]:
    src.mkdir(parents=True, exist_ok=True)
    lines = ['\t'.join(ASSOCIATIONS_HEADER)] + ['\t'.join(row[c] for c in ASSOCIATIONS_HEADER) for row in associations]
    (src / 'associations.tsv').write_text('\n'.join(lines) + '\n')
    (src / 'drug_molecule').mkdir()
    pl.DataFrame(DRUGS).write_parquet(src / 'drug_molecule' / 'part-0.parquet')
    return {
        'associations': str(src / 'associations.tsv'),
        'source': _jsonl(src / 'sources.jsonl', SOURCES),
        'biomarker_disease': _jsonl(src / 'diseases.jsonl', DISEASES),
        'drugs': str(src / 'drug_molecule'),
    }


def _evidence(**fields) -> dict:
    return {
        'datasourceId': 'cancer_biomarkers',
        'datatypeId': 'affected_pathway',
        'drugId': None,
        'diseaseFromSourceMappedId': None,
        'literature': None,
        'urls': None,
        **fields,
    }


def _variant(consequence: str | None, variant_id: str | None, name: str) -> dict:
    return {'functionalConsequenceId': consequence, 'id': variant_id, 'name': name}


def _variants(*variants: dict) -> dict:
    return {'geneExpression': None, 'geneticVariation': list(variants)}


MET = _variants(_variant('SO_0001777', None, 'MET:Y1230C'), _variant('SO_0001777', None, 'Y1235D'))
KMT2 = _variants(_variant('SO_0001563', None, 'MLL:amp'))
NRAS = 'NRAS (12,13,59,61,117,146)'

EXPECTED = [
    _evidence(
        drugFromSource='Vemurafenib',
        drugId='CHEMBL1',
        drugResponse='GO_0042493',
        targetFromSourceId='BRAF',
        diseaseFromSource='Cutaneous melanoma',
        diseaseFromSourceMappedId=MELANOMA,
        confidence='FDA guidelines',
        biomarkerName='BRAF (V600E)',
        literature=['111', '222'],
        urls=[
            {'niceName': 'AACR Meeting 2012', 'url': 'http://aacr/2012'},
            {'niceName': 'Clinical Trials', 'url': f'{CT}NCT0001'},
        ],
        biomarkers=_variants(
            _variant('SO_0001777', '7_140453136_A_T', 'BRAF:V600E'),
            _variant('SO_0001777', '7_140453136_CA_TT', 'BRAF:V600K'),
        ),
    ),
    *(
        _evidence(
            drugFromSource=drug,
            drugId=drug_id,
            drugResponse='EFO_0020001',
            targetFromSourceId='MET',
            diseaseFromSource='Any cancer type',
            diseaseFromSourceMappedId=ANY_CANCER,
            confidence=confidence,
            biomarkerName='MET (Y1230C,Y1235D)',
            literature=['333'],
            biomarkers=MET,
        )
        for confidence in ['Pre-clinical', 'Case report']
        for drug, drug_id in [('Crizotinib', 'CHEMBL2'), ('Capmatinib', None)]
    ),
    *(
        _evidence(
            drugFromSource='Venetoclax',
            drugResponse='EFO_0020002',
            targetFromSourceId=target,
            diseaseFromSource='Acute lymphoblastic leukemia',
            diseaseFromSourceMappedId=ALL,
            confidence='Early trials',
            biomarkerName='MLL amplification',
            literature=['444'],
            biomarkers=KMT2,
        )
        for target in ['KMT2A', 'KMT2D']
    ),
    _evidence(
        drugFromSource='Mek Inhibitors',
        drugResponse='GO_0042493',
        targetFromSourceId='NRAS',
        diseaseFromSource='Cutaneous melanoma',
        diseaseFromSourceMappedId=MELANOMA,
        confidence='Late trials',
        biomarkerName=NRAS,
        literature=['555'],
        biomarkers=_variants(
            _variant('SO_0001563', None, 'ARAF'),
            _variant('SO_0001777', None, NRAS),
            _variant('SO_0001882', None, NRAS),
        ),
    ),
    *(
        _evidence(
            drugFromSource='Imatinib',
            drugId=drug_id,
            drugResponse='EFO_0020001',
            targetFromSourceId='ABL1',
            diseaseFromSource='Acute lymphoblastic leukemia',
            diseaseFromSourceMappedId=ALL,
            confidence='Clinical trials',
            biomarkerName='BCR-ABL1 + T315I',
            urls=[{'niceName': 'Clinical Trials', 'url': f'{CT}NCT0002'}],
            biomarkers=_variants(_variant('SO_0001882', None, 'BCR__ABL1')),
        )
        for drug_id in ['CHEMBL3', 'CHEMBL4']
    ),
    _evidence(
        drugFromSource='Trastuzumab',
        drugResponse='EFO_0006951',
        targetFromSourceId='ERBB2',
        diseaseFromSource='Breast adenocarcinoma',
        confidence='NCCN guidelines',
        biomarkerName='ERBB2 overexpression',
        biomarkers={
            'geneExpression': [
                {'id': 'GO_0010467', 'name': 'PGR:norm'},
                {'id': 'GO_0010628', 'name': 'ERBB2:over'},
                {'id': 'GO_0010629', 'name': 'ESR1:under'},
            ],
            'geneticVariation': None,
        },
    ),
    _evidence(
        drugFromSource='Pembrolizumab',
        drugResponse='Unknown',
        targetFromSourceId='MLH1',
        diseaseFromSource='Any cancer type',
        diseaseFromSourceMappedId=ANY_CANCER,
        confidence='FDA guidelines',
        biomarkerName='MSI high',
        literature=['666'],
        biomarkers=_variants(_variant(None, None, 'MSI')),
    ),
    _evidence(
        drugFromSource='Pembrolizumab (pd1 Inhibitor)',
        drugResponse='Unknown',
        targetFromSourceId='MLH1',
        diseaseFromSource='Any cancer type',
        diseaseFromSourceMappedId=ANY_CANCER,
        confidence='FDA guidelines',
        biomarkerName='no drug',
        literature=['666'],
        biomarkers=_variants(_variant(None, None, 'MSI')),
    ),
]

SORT_KEY = ['biomarkerName', 'targetFromSourceId', 'confidence', 'drugFromSource', 'drugId']


def _sorted(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: tuple(row[k] or '' for k in SORT_KEY))


def test_generate_builds_one_row_per_evidence_with_collected_lists(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path)).collect()

    assert raw.schema == pl.Schema(RAW_SCHEMA)
    assert _sorted(raw.to_dicts()) == _sorted([{c: row[c] for c in RAW_SCHEMA} for row in EXPECTED])


def test_generate_is_independent_of_input_row_order(tmp_path: Path) -> None:
    forward = generate(**_sources(tmp_path / 'forward')).collect()
    backward = generate(**_sources(tmp_path / 'backward', ASSOCIATIONS[::-1])).collect()

    assert _sorted(forward.to_dicts()) == _sorted(backward.to_dicts())


def test_generate_drops_rows_whose_split_lists_other_than_drug_are_null(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path)).collect()

    assert raw.filter(pl.col('biomarkerName').str.starts_with('no '))['biomarkerName'].to_list() == ['no drug']


def test_generate_maps_a_drug_full_name_fallback_to_its_drug_id(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path, [{**BASE, 'Drug': '', 'DrugFullName': 'Vemurafenib'}])).collect()

    assert raw.select('drugFromSource', 'drugId').rows() == [('Vemurafenib', 'CHEMBL1')]


def test_generate_gives_no_urls_for_pmid_and_ncit_sources(tmp_path: Path) -> None:
    sources = _sources(tmp_path, [{**BASE, 'Source': 'PMID:666;NCIT:C1;AACR 2012'}])
    _jsonl(
        Path(sources['source']),
        [*SOURCES, *({'source': s, 'url': f'http://{s}', 'label': s} for s in ['PMID:666', 'NCIT'])],
    )

    raw = generate(**sources).collect()

    assert raw['urls'].to_list() == [[{'niceName': 'AACR Meeting 2012', 'url': 'http://aacr/2012'}]]


def _zip_alterations_with_type(alterations: list[str], alteration_type: list[str]) -> list[tuple[str, str]]:
    """The former `zip_alterations_with_type` UDF."""
    alteration_types = alteration_type * len(alterations)
    return list(zip(alterations, alteration_types, strict=False))


ZIP_CASES = [
    (['MET:Y1230C', 'Y1235D'], ['MUT']),
    (['a', 'b', 'c'], ['MUT', 'CNA']),
    (['a', 'b'], ['MUT', 'CNA', 'FUS']),
    (['a'], ['MUT', 'CNA']),
    (['a', 'a'], ['MUT', 'MUT']),
    (['a', 'b', 'c', 'd', 'e'], ['MUT', 'CNA']),
    ([''], ['']),
    ([], ['MUT']),
    (['a'], []),
    ([], []),
    (None, ['MUT']),
    (['a'], None),
]


def _pairs(cases: list) -> list[list[tuple[str, str]]]:
    frame = pl.DataFrame(
        {'row': range(len(cases)), 'alterations': [c[0] for c in cases], 'alteration_types': [c[1] for c in cases]},
        schema={'row': pl.Int64, 'alterations': pl.List(pl.String), 'alteration_types': pl.List(pl.String)},
    )
    exploded = explode_alterations(frame)
    return [
        list(exploded.filter(pl.col('row') == i).select('alteration', 'alteration_type').iter_rows())
        for i in range(len(cases))
    ]


def test_explode_alterations_matches_udf() -> None:
    expected = [_zip_alterations_with_type(a or [], t or []) for a, t in ZIP_CASES]

    assert _pairs(ZIP_CASES) == expected
    assert _pairs(ZIP_CASES)[1] == [('a', 'MUT'), ('b', 'CNA'), ('c', 'MUT')]


def test_explode_alterations_null_yields_no_pairs() -> None:
    assert _pairs([(None, ['MUT']), (['a'], None)]) == [[], []]


@pytest.mark.parametrize(
    ('gdna', 'expected'),
    [
        ('chr14:g.105243048G>T', '14_105243048_G_T'),
        ('chrX:g.1234A>C', 'X_1234_A_C'),
        ('chr7:g.140453136_140453137delCAinsTT', '7_140453136_CA_TT'),
        ('chr7:g.A>T', None),
        (None, None),
    ],
)
def test_get_variant_id(gdna: str | None, expected: str | None) -> None:
    assert get_variant_id(gdna) == expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('FORETINIB', 'Foretinib'),
        ('MEK inhibitors', 'Mek Inhibitors'),
        ('anti-PD1 a/b', 'Anti-pd1 A/b'),
        ('5-FU', '5-fu'),
        ('JAK/TBK1/IKKε inhibitors', 'Jak/tbk1/ikkε Inhibitors'),
        ('FORETINIB\xa0', 'Foretinib\xa0'),
        (' a  b', ' A  B'),
        ('', ''),
        (None, None),
    ],
)
def test_initcap_capitalises_after_spaces_only(value: str | None, expected: str | None) -> None:
    assert pl.select(initcap(pl.lit(value, pl.String))).item() == expected


def test_score_is_one() -> None:
    df = pl.DataFrame({'dummy': ['x', None]})

    assert df.with_columns(CANCER_BIOMARKERS.score.alias('s'))['s'].to_list() == [1.0, 1.0]


def test_cancer_biomarkers_unique_fields_are_the_config_ones() -> None:
    assert CANCER_BIOMARKERS.datasource_id == 'cancer_biomarkers'
    assert CANCER_BIOMARKERS.unique_fields == [
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'biomarkerName',
        'confidence',
        'diseaseFromSource',
        'drugFromSource',
        'drugResponse',
    ]


