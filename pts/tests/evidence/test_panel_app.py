"""Tests for Genomics England PanelApp evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from loguru import logger

from pts.transformers.evidence.panel_app import (
    OMIM_RE,
    OTHER_RE,
    PANEL_APP,
    PHENOTYPE_AFTER_SPLIT,
    PHENOTYPE_BEFORE_SPLIT,
    extract_pmids,
    panel_app,
    parse_panel_app,
)
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol, normalise_term
from tests.evidence.fixtures import write_disease, write_literature, write_ontoma_lut, write_target

COLUMNS = [
    'Symbol',
    'Panel Id',
    'Panel Name',
    'Panel Version',
    'Panel Status',
    'List',
    'Mode of inheritance',
    'Phenotypes',
]
#: Arabic-Indic digits one to six, which Java's `\d` does not match.
ARABIC_DIGITS = ''.join(chr(0x0661 + i) for i in range(6))

#: Strings that trigger each of the phenotype regular expressions, and some that should not.
PHENOTYPES = [
    'A (HP:0006574;);B',
    'Abruzzo-Erickson;syndrome',
    'Deafness, autosomal recessive; 12',
    'Waardenburg syndrome, type; 3',
    'Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome; 3',
    '{Bladder cancer, somatic}, 109800',
    'Aarskog-Scott syndrome, 305400Mental retardation, X-linked syndromic 16, 305400',
    '{123456}Abc',
    ARABIC_DIGITS + 'Abc',
    '1234567Abc',
    'a \t  b',
    'x  ;  y',
    'a; b ;c',
    'Disease (no OMIM number)',
    'Disease (NO phenotype number in OMIM)',
    'Disease no OMIM number',
    'Disease NO OMIM NUMBER OR DISEASE',
    'Disease No OMIM phenotype',
    'Disease (PMID: 12345).',
    'Disease from PMID 12345',
    'Disease PMID:' + ARABIC_DIGITS[:2],
    'Marfan syndrome, 154700',
    'Disease OMIM:154700',
    'Disease (MIM 154700)',
    'Disease, OrphaNet: ORPHA558',
    'Disease, Orphanet_558',
    'Disease ORPHA:558.',
    'Disease HP:0001250',
    'Disease, MONDO_0007254',
    'Disease 154700, ORPHA:558',
    'Disease 1234567',
    'Disease #154700:.',
    'HP: test',
    'Disease ' + ARABIC_DIGITS,
    '',
    '()',
    None,
]


def _row(
    symbol: str = 'CFTR',
    panel_id: str = '1',
    name: str = 'Panel one',
    version: str = '1.0',
    status: str = 'PUBLIC',
    confidence: str = 'green',
    moi: str | None = 'BIALLELIC',
    phenotypes: str | None = 'cystic fibrosis',
) -> list[str | None]:
    return [symbol, panel_id, name, version, status, confidence, moi, phenotypes]


def _write_tsv(path: Path, rows: list[list[str | None]]) -> str:
    lines = ['\t'.join(COLUMNS), *('\t'.join('' if v is None else v for v in row) for row in rows)]
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def _panel(panel_id: int | None, genes: dict[str, list[str]]) -> dict:
    return {
        'id': panel_id,
        'name': 'p',
        'genes': [
            {'gene_data': {'gene_symbol': s, 'hgnc_id': 'HGNC:1'}, 'publications': pubs, 'confidence_level': '3'}
            for s, pubs in genes.items()
        ],
    }


def _write_panels(path: Path, panels: list[dict]) -> str:
    path.write_text(''.join(json.dumps(p) + '\n' for p in panels))
    return str(path)


def _files(tmp: Path, rows: list[list[str | None]], panels: list[dict] | None = None) -> tuple[str, str]:
    tmp.mkdir(parents=True, exist_ok=True)
    panels = panels if panels is not None else [_panel(999, {'X': ['1']})]
    return _write_tsv(tmp / 'panel_app.tsv', rows), _write_panels(tmp / 'panels.jsonl', panels)


def _parse(tmp: Path, rows: list[list[str | None]], panels: list[dict] | None = None) -> pl.DataFrame:
    return parse_panel_app(*_files(tmp, rows, panels)).collect()


def _polars(values: list[str | None]) -> pl.Series:
    return pl.Series('v', values, dtype=pl.String)


def test_only_green_amber_public_v1_panels(tmp_path: Path) -> None:
    rows = [
        _row(symbol='GREEN'),
        _row(symbol='AMBER', confidence='amber'),
        _row(symbol='RED', confidence='red'),
        _row(symbol='V0', version='0.999'),
        _row(symbol='V1', version='1'),
        _row(symbol='V1_10', version='1.10'),
        _row(symbol='RETIRED', status='RETIRED'),
        _row(symbol='LOWER', confidence='Green'),
    ]
    raw = _parse(tmp_path, rows)
    assert raw['targetFromSourceId'].to_list() == ['GREEN', 'AMBER', 'V1', 'V1_10']


@pytest.mark.parametrize(
    ('phenotypes', 'expected'),
    [
        ('A (HP:0006574;);B', ['A (HP:0006574)', 'B']),
        ('Abruzzo-Erickson;syndrome', ['Abruzzo-Erickson syndrome']),
        ('Deafness, autosomal recessive; 12', ['Deafness, autosomal recessive, 12']),
        ('Waardenburg syndrome, type; 3', ['Waardenburg syndrome, type 3']),
        (
            'Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome; 3',
            ['Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome, 3'],
        ),
        ('{Bladder cancer, somatic}, 109800', ['Bladder cancer, somatic, 109800']),
        ('{123456}Abc', ['123456', 'Abc']),
        ('Aarskog-Scott syndrome, 305400Mental retardation', ['Aarskog-Scott syndrome, 305400', 'Mental retardation']),
        (ARABIC_DIGITS + 'Abc', [ARABIC_DIGITS + 'Abc']),
        ('"a \t  b"', ['a b']),
        ('x  ;  y', ['x', 'y']),
        ('a;a; a', ['a']),
        ('a;', ['a', '']),
    ],
)
def test_regex_replacements_in_order(tmp_path: Path, phenotypes: str, expected: list[str]) -> None:
    raw = _parse(tmp_path, [_row(phenotypes=phenotypes)])
    assert raw['cohortPhenotypes'].unique().to_list() == [expected]


#: PHENOTYPES each before-split replacement changes, by index, captured from the removed pyspark implementation.
BEFORE_SPLIT_BASELINE = [
    {0: 'A (HP:0006574);B'},
    {1: 'Abruzzo-Erickson syndrome'},
    {2: 'Deafness, autosomal recessive, 12'},
    {3: 'Waardenburg syndrome, type 3'},
    {4: 'Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome, 3'},
    {5: 'Bladder cancer, somatic, 109800', 7: '123456Abc'},
    {6: 'Aarskog-Scott syndrome, 305400;Mental retardation, X-linked syndromic 16, 305400', 9: '1234567;Abc'},
    {10: 'a b', 11: 'x ; y'},
    {
        2: 'Deafness, autosomal recessive;12',
        3: 'Waardenburg syndrome, type;3',
        4: 'Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome;3',
        11: 'x ; y',
        12: 'a;b;c',
    },
]
#: As BEFORE_SPLIT_BASELINE, for the after-split removals.
AFTER_SPLIT_BASELINE = [
    {13: 'Disease'},
    {14: 'Disease'},
    {13: 'Disease ( number)', 15: 'Disease  number', 16: 'Disease  NUMBER OR DISEASE', 17: 'Disease '},
    {18: 'Disease', 19: 'Disease'},
]
#: (group 1, group 2, phenotype with the match removed) for the PHENOTYPES each ontology regex matches,
#: captured from the removed pyspark implementation; the rest give ('', '', phenotype).
OMIM_BASELINE = {
    0: ('', '000657', 'A (HP4;);B'),
    5: ('', '109800', '{Bladder cancer, somatic}'),
    6: ('', '305400', 'Aarskog-Scott syndromeMental retardation, X-linked syndromic 16'),
    7: ('', '123456', '{}Abc'),
    9: ('', '123456', '7Abc'),
    21: ('', '154700', 'Marfan syndrome'),
    22: ('OMIM', '154700', 'Disease'),
    23: ('MIM', '154700', 'Disease ()'),
    27: ('', '000125', 'Disease HP0'),
    28: ('', '000725', 'Disease, MONDO4'),
    29: ('', '154700', 'Disease, ORPHA:558'),
    30: ('', '123456', 'Disease7'),
    31: ('', '154700', 'Disease'),
}
OTHER_BASELINE = {
    0: ('HP', '0006574', 'A (;);B'),
    24: ('OrphaNet: ORPHA', '558', 'Disease'),
    25: ('Orphanet', '558', 'Disease'),
    26: ('ORPHA', '558', 'Disease'),
    27: ('HP', '0001250', 'Disease'),
    28: ('MONDO', '0007254', 'Disease'),
    29: ('ORPHA', '558', 'Disease 154700'),
}


@pytest.mark.parametrize('index', range(9))
def test_before_split_replacement_matches_spark_baseline(index: int) -> None:
    assert len(PHENOTYPE_BEFORE_SPLIT) == len(BEFORE_SPLIT_BASELINE)
    pattern, replacement = PHENOTYPE_BEFORE_SPLIT[index]
    expected = [BEFORE_SPLIT_BASELINE[index].get(i, v) for i, v in enumerate(PHENOTYPES)]
    assert _polars(PHENOTYPES).str.replace_all(pattern, replacement).to_list() == expected


@pytest.mark.parametrize('index', range(4))
def test_after_split_removal_matches_spark_baseline(index: int) -> None:
    assert len(PHENOTYPE_AFTER_SPLIT) == len(AFTER_SPLIT_BASELINE)
    expected = [AFTER_SPLIT_BASELINE[index].get(i, v) for i, v in enumerate(PHENOTYPES)]
    assert _polars(PHENOTYPES).str.replace_all(PHENOTYPE_AFTER_SPLIT[index], '').to_list() == expected


@pytest.mark.parametrize(('regex', 'baseline'), [(OMIM_RE, OMIM_BASELINE), (OTHER_RE, OTHER_BASELINE)])
def test_ontology_regexes_match_spark_baseline(regex: str, baseline: dict[int, tuple[str, str, str]]) -> None:
    got = pl.DataFrame({'v': _polars(PHENOTYPES)}).select(
        pl.when(pl.col('v').is_not_null()).then(pl.col('v').str.extract(regex, 1).fill_null('')),
        pl.when(pl.col('v').is_not_null()).then(pl.col('v').str.extract(regex, 2).fill_null('')).alias('b'),
        pl.col('v').str.replace_all(regex, '').alias('c'),
    )
    expected = [baseline.get(i, ('', '', v) if v is not None else (None, None, None)) for i, v in enumerate(PHENOTYPES)]
    assert got.rows() == expected


def test_null_phenotypes_rows_dropped(tmp_path: Path) -> None:
    raw = _parse(tmp_path, [_row(symbol='NULL', phenotypes=None), _row(symbol='KEPT')])
    assert raw['targetFromSourceId'].to_list() == ['KEPT']


@pytest.mark.parametrize(
    ('phenotype', 'label', 'disease_id'),
    [
        ('Marfan syndrome, 154700', 'Marfan syndrome', 'OMIM:154700'),
        ('Disease OMIM:154700', 'Disease', 'OMIM:154700'),
        ('Disease (MIM 154700)', 'Disease', 'OMIM:154700'),
        ('Disease #154700:.', 'Disease', 'OMIM:154700'),
        ('Disease, OrphaNet: ORPHA558', 'Disease', 'ORPHA:558'),
        ('Disease, Orphanet_558', 'Disease', 'Orphanet:558'),
        ('Disease ORPHA:558.', 'Disease', 'ORPHA:558'),
        ('Disease HP:0001250', 'Disease', 'HP:0001250'),
        ('Disease, MONDO_0007254', 'Disease', 'MONDO:0007254'),
        ('Disease 154700, ORPHA:558', 'Disease', 'OMIM:154700'),
        ('Disease 1234567', 'Disease7', 'OMIM:123456'),
        ('Disease ' + ARABIC_DIGITS, 'Disease ' + ARABIC_DIGITS, None),
        ('HP: test', 'HP: test', None),
        ('No ids here', 'No ids here', None),
        ('Disease (no OMIM number)', 'Disease', None),
        ('Disease (PMID: 12345).', 'Disease', None),
    ],
)
def test_ontology_and_omim_ids(tmp_path: Path, phenotype: str, label: str, disease_id: str | None) -> None:
    raw = _parse(tmp_path, [_row(phenotypes=phenotype)])
    assert raw.select('diseaseFromSource', 'diseaseFromSourceId').rows() == [(label, disease_id)]


def test_question_mark_phenotypes_dropped(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        [
            _row(symbol='A', phenotypes='?Disease;Kept'),
            _row(symbol='B', phenotypes=' ?Spaced, 154700'),
            _row(symbol='C', phenotypes='?'),
            _row(symbol='D', phenotypes='Not ? first'),
        ],
    )
    assert raw.select('targetFromSourceId', 'diseaseFromSource').rows() == [('A', 'Kept'), ('D', 'Not ? first')]


def test_panel_name_fallback_when_label_and_id_null(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        [
            _row(symbol='A', name='Panel A', phenotypes='Disease;()'),
            _row(symbol='B', name='Panel B', phenotypes='{}'),
            _row(symbol='C', name='Panel C', phenotypes='154700'),
        ],
    )
    assert raw.select('targetFromSourceId', 'diseaseFromSource', 'diseaseFromSourceId').sort(pl.all()).rows() == [
        ('A', 'Disease', None),
        ('A', 'Panel A', None),
        ('B', 'Panel B', None),
        ('C', None, 'OMIM:154700'),
    ]


PMID_CASES = [
    ('15643612', ['15643612']),
    ('28055140, 27333055, 23063529', ['28055140', '27333055', '23063529']),
    ('observed in the patient. PMID: 1908107 - publication describing function', ['1908107']),
    ('PubMed 123, 456 and 789', ['123', '456']),
    ('1234 something 5678', ['1234']),
    ('12\N{NO-BREAK SPACE}34', ['1234']),
    ('PMID\N{FULLWIDTH COLON}123', ['123']),
    ('\n12\r', ['12']),
    ('224,614,752,030,146,000,000,000', []),
    ('224,614,752,030,146,000,000,000 PMID 5', ['5']),
    ('0, 00, 123', ['00', '123']),
    ('123456789, 12345678', ['12345678']),
    ('5, 5, 6', ['5', '6']),
    ('PMID 1;PMID 2', ['1', '2']),
    ('no pmids', []),
    ('', []),
    (None, None),
]


@pytest.mark.parametrize(('text', 'expected'), PMID_CASES)
def test_pmid_extraction(text: str | None, expected: list[str] | None) -> None:
    assert pl.select(extract_pmids(pl.lit(text, dtype=pl.String))).to_series().to_list() == [expected]


def test_literature_per_panel_and_symbol_sorted_distinct(tmp_path: Path) -> None:
    panels = [
        _panel(1, {'CFTR': ['30, 20', 'PMID: 10', '20'], 'BRCA1': ['7']}),
        _panel(2, {'CFTR': ['99']}),
        _panel(None, {'CFTR': ['55']}),
    ]
    raw = _parse(tmp_path, [_row(), _row(symbol='BRCA1'), _row(symbol='NOPUB'), _row(panel_id='3')], panels)
    assert raw.select('studyId', 'targetFromSourceId', 'literature').sort('studyId', 'targetFromSourceId').rows() == [
        ('1', 'BRCA1', ['7']),
        ('1', 'CFTR', ['10', '20', '30']),
        ('1', 'NOPUB', None),
        ('3', 'CFTR', None),
    ]


def test_all_zero_pmids_and_omim_matches_are_logged_on_evidence_rows(tmp_path: Path) -> None:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), format='{message}')
    try:
        _parse(
            tmp_path,
            [
                _row(phenotypes='Disease, 154700;Other 1234567;Again 1234567;None'),
                _row(symbol='BRCA1'),
                _row(symbol='Q', phenotypes='?Gone 154700;?Gone 1234567'),
                _row(symbol='RED', confidence='red', phenotypes='Red 154700'),
            ],
            [_panel(1, {'CFTR': ['00, 000, 0, 1'], 'BRCA1': ['00'], 'NOROW': ['00']})],
        )
    finally:
        logger.remove(sink)
    text = '\n'.join(messages)
    assert '3 evidence strings with an OMIM id, 2 of them six digits of a longer number' in text
    assert '5 evidence strings with an all-zero PMID' in text


def test_panels_without_gene_symbol_raise(tmp_path: Path) -> None:
    panel = _panel(1, {'CFTR': ['1']})
    panel['genes'][0]['gene_data'] = {'symbol': 'CFTR'}
    with pytest.raises(ValueError, match='gene_symbol'):
        _parse(tmp_path, [_row()], [panel])


def test_panels_without_publications_raise(tmp_path: Path) -> None:
    panel = _panel(1, {'CFTR': ['1']})
    del panel['genes'][0]['publications']
    with pytest.raises(ValueError, match='publications'):
        _parse(tmp_path, [_row()], [panel])


def test_panels_without_id_raise(tmp_path: Path) -> None:
    panel = _panel(1, {'CFTR': ['1']})
    panel['panel_id'] = panel.pop('id')
    with pytest.raises(ValueError, match='id'):
        _parse(tmp_path, [_row()], [panel])


def test_non_integer_panel_id_raises(tmp_path: Path) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        _parse(tmp_path, [_row(panel_id='1a')])


def test_panel_id_beyond_int32(tmp_path: Path) -> None:
    raw = _parse(tmp_path, [_row(panel_id='3000000000')], [_panel(3_000_000_000, {'CFTR': ['1']})])
    assert raw.select('studyId', 'literature').rows() == [('3000000000', ['1'])]


def test_study_id_is_string(tmp_path: Path) -> None:
    raw = _parse(tmp_path, [_row(panel_id='007')], [_panel(7, {'CFTR': ['1']})])
    assert raw.schema['studyId'] == pl.String
    assert raw.select('studyId', 'literature').rows() == [('7', ['1'])]


@pytest.mark.parametrize(('confidence', 'score'), [('amber', 0.5), ('green', 1.0), ('red', None), (None, None)])
def test_score(confidence: str | None, score: float | None) -> None:
    df = pl.DataFrame({'confidence': [confidence]}, schema={'confidence': pl.String})
    assert df.select(PANEL_APP.score).item() == score


def test_unique_fields_follow_the_spark_config() -> None:
    assert PANEL_APP.datasource_id == 'genomics_england'
    assert PANEL_APP.unique_fields == [
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'diseaseFromSource',
        'studyId',
    ]


def test_nbsp_is_not_trimmed(tmp_path: Path) -> None:
    raw = _parse(tmp_path, [_row(phenotypes='\N{NO-BREAK SPACE}Disease\N{NO-BREAK SPACE} ; Other')])
    assert raw['diseaseFromSource'].to_list() == ['\N{NO-BREAK SPACE}Disease\N{NO-BREAK SPACE}', 'Other']


PARSED_COLUMNS = [
    'studyId',
    'targetFromSourceId',
    'studyOverview',
    'confidence',
    'cohortPhenotypes',
    'diseaseFromSource',
    'diseaseFromSourceId',
    'literature',
    'allelicRequirements',
    'datasourceId',
    'datatypeId',
]
PARSED_CONSTANTS = ('genomics_england', 'genetic_literature')


def _baseline(
    study: str,
    symbol: str,
    confidence: str,
    cohort: list[str],
    disease: str | None,
    disease_id: str | None,
    literature: list[str] | None = None,
    moi: tuple[str, ...] | None = ('BIALLELIC',),
    name: str = 'Panel one',
) -> tuple:
    """One parsed row in PARSED_COLUMNS order."""
    moi_list = list(moi) if moi is not None else None
    return (study, symbol, name, confidence, cohort, disease, disease_id, literature, moi_list, *PARSED_CONSTANTS)


def _polars_rows(raw: pl.DataFrame) -> list[str]:
    return sorted(repr(r) for r in raw.rows())


def test_escaped_quote_matches_spark_baseline(tmp_path: Path) -> None:
    rows = [
        _row(symbol='A', phenotypes='"say \\"hi\\" disease, 154700"'),
        _row(symbol='B', phenotypes='"back\\\\slash disease"'),
        _row(symbol='C', phenotypes='plain \\n text'),
        _row(symbol='D', name='"quoted name"', phenotypes='"unescaped" inner'),
    ]
    evidence, panels = _files(tmp_path, rows)
    raw = parse_panel_app(evidence, panels).collect()
    assert raw.sort('targetFromSourceId')['diseaseFromSource'].to_list()[:3] == [
        'say "hi" disease',
        'back\\slash disease',
        'plain \\n text',
    ]
    # expected rows captured from the removed pyspark implementation
    assert raw.columns == PARSED_COLUMNS
    assert _polars_rows(raw) == sorted(
        map(
            repr,
            [
                _baseline('1', 'A', 'green', ['say "hi" disease, 154700'], 'say "hi" disease', 'OMIM:154700'),
                _baseline('1', 'B', 'green', ['back\\slash disease'], 'back\\slash disease', None),
                _baseline('1', 'C', 'green', ['plain \\n text'], 'plain \\n text', None),
                _baseline('1', 'D', 'green', ['"unescaped" inner'], '"unescaped" inner', None, name='quoted name'),
            ],
        )
    )


def test_parse_matches_spark_baseline_on_a_mixed_file(tmp_path: Path) -> None:
    rows = [
        _row(symbol=f'G{i}', panel_id=str(1 + i % 3), confidence=['green', 'amber', 'red'][i % 3], phenotypes=p)
        for i, p in enumerate(PHENOTYPES)
    ]
    rows += [
        _row(symbol='G0', panel_id='1', phenotypes=PHENOTYPES[0]),
        _row(symbol='M', moi=None, phenotypes='Disease;Disease, 154700;Disease (154700)'),
        _row(symbol='V', version='0.5'),
        _row(symbol='Q', phenotypes='?x;y'),
    ]
    panels = [
        _panel(1, {'G0': ['1, 2', 'PMID: 3'], 'G3': ['00'], 'M': ['224,614,752,030,146,000,000,000']}),
        _panel(2, {'G1': ['5', '5'], 'G4': []}),
        _panel(3, {}),
        _panel(None, {'G2': ['8']}),
    ]
    evidence, panels_path = _files(tmp_path, rows, panels)
    raw = parse_panel_app(evidence, panels_path).collect()
    # expected rows captured from the removed pyspark implementation
    aarskog = ['Aarskog-Scott syndrome, 305400', 'Mental retardation, X-linked syndromic 16, 305400']
    m_cohort = ['Disease', 'Disease, 154700', 'Disease (154700)']
    expected = [
        _baseline('1', 'G0', 'green', ['A (HP:0006574)', 'B'], 'A', 'HP:0006574', ['1', '2', '3']),
        _baseline('1', 'G0', 'green', ['A (HP:0006574)', 'B'], 'B', None, ['1', '2', '3']),
        _baseline('1', 'G12', 'green', ['a', 'b', 'c'], 'a', None),
        _baseline('1', 'G12', 'green', ['a', 'b', 'c'], 'b', None),
        _baseline('1', 'G12', 'green', ['a', 'b', 'c'], 'c', None),
        _baseline('1', 'G15', 'green', ['Disease no OMIM number'], 'Disease  number', None),
        _baseline('1', 'G18', 'green', ['Disease (PMID: 12345).'], 'Disease', None),
        _baseline('1', 'G21', 'green', ['Marfan syndrome, 154700'], 'Marfan syndrome', 'OMIM:154700'),
        _baseline('1', 'G24', 'green', ['Disease, OrphaNet: ORPHA558'], 'Disease', 'ORPHA:558'),
        _baseline('1', 'G27', 'green', ['Disease HP:0001250'], 'Disease', 'HP:0001250'),
        _baseline('1', 'G3', 'green', ['Waardenburg syndrome, type 3'], 'Waardenburg syndrome, type 3', None, ['00']),
        _baseline('1', 'G30', 'green', ['Disease 1234567'], 'Disease7', 'OMIM:123456'),
        _baseline('1', 'G33', 'green', ['Disease ' + ARABIC_DIGITS], 'Disease ' + ARABIC_DIGITS, None),
        _baseline('1', 'G6', 'green', aarskog, 'Aarskog-Scott syndrome', 'OMIM:305400'),
        _baseline('1', 'G6', 'green', aarskog, 'Mental retardation, X-linked syndromic 16', 'OMIM:305400'),
        _baseline('1', 'G9', 'green', ['1234567', 'Abc'], '7', 'OMIM:123456'),
        _baseline('1', 'G9', 'green', ['1234567', 'Abc'], 'Abc', None),
        _baseline('1', 'M', 'green', m_cohort, 'Disease', 'OMIM:154700', moi=None),
        _baseline('1', 'M', 'green', m_cohort, 'Disease', None, moi=None),
        _baseline('1', 'Q', 'green', ['?x', 'y'], 'y', None),
        _baseline('2', 'G1', 'amber', ['Abruzzo-Erickson syndrome'], 'Abruzzo-Erickson syndrome', None, ['5']),
        _baseline('2', 'G10', 'amber', ['a '], 'a', None),
        _baseline('2', 'G13', 'amber', ['Disease (no OMIM number)'], 'Disease', None),
        _baseline('2', 'G16', 'amber', ['Disease NO OMIM NUMBER OR DISEASE'], 'Disease  NUMBER OR DISEASE', None),
        _baseline('2', 'G19', 'amber', ['Disease from PMID 12345'], 'Disease', None),
        _baseline('2', 'G22', 'amber', ['Disease OMIM:154700'], 'Disease', 'OMIM:154700'),
        _baseline('2', 'G25', 'amber', ['Disease, Orphanet_558'], 'Disease', 'Orphanet:558'),
        _baseline('2', 'G28', 'amber', ['Disease, MONDO_0007254'], 'Disease', 'MONDO:0007254'),
        _baseline('2', 'G31', 'amber', ['Disease #154700:.'], 'Disease', 'OMIM:154700'),
        _baseline(
            '2',
            'G4',
            'amber',
            ['Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome, 3'],
            'Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome, 3',
            None,
        ),
        _baseline('2', 'G7', 'amber', ['123456', 'Abc'], 'Abc', None),
        _baseline('2', 'G7', 'amber', ['123456', 'Abc'], None, 'OMIM:123456'),
    ]
    assert raw.columns == PARSED_COLUMNS
    assert _polars_rows(raw) == sorted(map(repr, expected))


def _sources(root: Path, rows: list[list[str | None]], panels: list[dict] | None = None) -> dict[str, str]:
    evidence, panels_path = _files(root, rows, panels)
    return {
        'evidence': evidence,
        'panels': panels_path,
        'ontoma_disease_label_lut': write_ontoma_lut(
            root / 'label_lut', 'label', normalise_term('cystic fibrosis'), 'MONDO_0009061'
        ),
        'ontoma_disease_id_lut': write_ontoma_lut(
            root / 'id_lut', 'id', normalise_symbol('OMIM:154700'), 'MONDO_0007947'
        ),
        'disease': write_disease(root / 'disease', ['MONDO_0009061', 'MONDO_0007947']),
        'target': write_target(root / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'CFTR', 'ENSG2': 'FBN1'}),
        'literature': write_literature(root / 'literature', {'111': '2020-05-01'}),
    }


def _run(tmp_path: Path, rows: list[list[str | None]], panels: list[dict] | None = None) -> tuple[pl.DataFrame, ...]:
    destination = {'evidence': str(tmp_path / 'out' / 'valid'), 'failed_evidence': str(tmp_path / 'out' / 'failed')}
    panel_app(_sources(tmp_path / 'src', rows, panels), destination, {}, None)
    return pl.read_parquet(destination['evidence']), pl.read_parquet(destination['failed_evidence'])


def test_maps_scores_and_dates(tmp_path: Path) -> None:
    valid, _ = _run(
        tmp_path,
        [_row(), _row(symbol='FBN1', confidence='amber', moi=None, phenotypes='Marfan syndrome, 154700')],
        [_panel(1, {'CFTR': ['111']})],
    )
    assert valid.sort('targetId').select(
        'targetId', 'diseaseId', 'score', 'studyId', 'literature', 'evidenceDate', 'allelicRequirements'
    ).rows() == [
        ('ENSG1', 'MONDO_0009061', 1.0, '1', ['111'], '2020-05-01', ['BIALLELIC']),
        ('ENSG2', 'MONDO_0007947', 0.5, '1', None, None, None),
    ]
    assert valid['datasourceId'].to_list() == ['genomics_england'] * 2
    assert valid['datatypeId'].to_list() == ['genetic_literature'] * 2
    assert valid['studyOverview'].to_list() == ['Panel one'] * 2
    assert valid['id'].n_unique() == 2


def test_unmapped_disease_lands_in_failed_evidence(tmp_path: Path) -> None:
    valid, failed = _run(tmp_path, [_row(phenotypes='nothing at all')])
    assert failed['diseaseFromSource'].to_list() == ['nothing at all']
    assert failed['diseaseFromSourceMappedId'].to_list() == [None]
    assert failed['targetId'].to_list() == ['ENSG1']
    assert valid.height == 0


def test_empty_input_writes_empty_outputs(tmp_path: Path) -> None:
    valid, failed = _run(tmp_path, [], [])
    assert valid.height == 0
    assert failed.height == 0
