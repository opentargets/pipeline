"""Tests for Genomics England PanelApp evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.pyspark.panel_app import PanelAppEvidenceGenerator
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
SPARK = PanelAppEvidenceGenerator
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


def _spark_replace(spark: SparkSession, values: list[str | None], pattern: str, replacement: str) -> list[str | None]:
    df = spark.createDataFrame([(v,) for v in values], 'v string')
    return [r[0] for r in df.select(f.regexp_replace('v', pattern, replacement)).collect()]


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


@pytest.mark.parametrize('index', range(9))
def test_before_split_replacement_matches_spark(spark: SparkSession, index: int) -> None:
    assert len(PHENOTYPE_BEFORE_SPLIT) == len(SPARK.PHENOTYPE_BEFORE_SPLIT_RE) == 9
    old_pattern, old_replacement = list(SPARK.PHENOTYPE_BEFORE_SPLIT_RE.items())[index]
    pattern, replacement = PHENOTYPE_BEFORE_SPLIT[index]
    expected = _spark_replace(spark, PHENOTYPES, old_pattern, old_replacement)
    assert _polars(PHENOTYPES).str.replace_all(pattern, replacement).to_list() == expected


@pytest.mark.parametrize('index', range(4))
def test_after_split_removal_matches_spark(spark: SparkSession, index: int) -> None:
    assert len(PHENOTYPE_AFTER_SPLIT) == len(SPARK.PHENOTYPE_AFTER_SPLIT_RE) == 4
    expected = _spark_replace(spark, PHENOTYPES, f'({SPARK.PHENOTYPE_AFTER_SPLIT_RE[index]})', '')
    assert _polars(PHENOTYPES).str.replace_all(PHENOTYPE_AFTER_SPLIT[index], '').to_list() == expected


@pytest.mark.parametrize(('new', 'old'), [(OMIM_RE, SPARK.OMIM_RE), (OTHER_RE, SPARK.OTHER_RE)])
def test_ontology_regexes_match_spark(spark: SparkSession, new: str, old: str) -> None:
    df = spark.createDataFrame([(v,) for v in PHENOTYPES], 'v string')
    expected = df.select(
        f.regexp_extract('v', old, 1), f.regexp_extract('v', old, 2), f.regexp_replace('v', f'({old})', '')
    ).collect()
    values = _polars(PHENOTYPES)
    got = pl.DataFrame({'v': values}).select(
        pl.when(pl.col('v').is_not_null()).then(pl.col('v').str.extract(new, 1).fill_null('')),
        pl.when(pl.col('v').is_not_null()).then(pl.col('v').str.extract(new, 2).fill_null('')).alias('b'),
        pl.col('v').str.replace_all(new, '').alias('c'),
    )
    assert got.rows() == [tuple(r) for r in expected]


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


def test_pmid_extraction_matches_spark(spark: SparkSession) -> None:
    texts = [text for text, _ in PMID_CASES]
    df = spark.createDataFrame([(t,) for t in texts], 'v string')
    expected = [r[0] for r in df.select(SPARK(None, None).extract_pubmed_ids(f.col('v'))).collect()]
    got = pl.DataFrame({'v': _polars(texts)}).select(extract_pmids(pl.col('v'))).to_series().to_list()
    assert got == expected


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


def test_all_zero_pmids_and_omim_matches_are_logged(tmp_path: Path) -> None:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), format='{message}')
    try:
        _parse(
            tmp_path,
            [_row(phenotypes='Disease, 154700;Other 1234567;None'), _row(symbol='BRCA1')],
            [_panel(1, {'CFTR': ['00, 000, 0, 1'], 'BRCA1': ['00']})],
        )
    finally:
        logger.remove(sink)
    text = ''.join(messages)
    assert '3 all-zero PMIDs' in text
    assert '2 phenotypes with an OMIM id, 1 of them cut from a longer number' in text


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


def _spark_parse(spark: SparkSession, evidence: str, panels: str) -> tuple[list[str], list[str]]:
    panelapp_df = spark.read.load(evidence, format='csv', sep='\t', header=True, inferSchema=True)
    out = SPARK(None, panelapp_df).generate_panelapp_evidence(spark.read.json(panels))
    rows = []
    for row in out.withColumn('studyId', f.col('studyId').cast('string')).collect():
        values = row.asDict()
        # collect_set order is unspecified; the port sorts
        if values['literature'] is not None:
            values['literature'] = sorted(values['literature'])
        rows.append(repr(tuple(values.values())))
    return out.columns, sorted(rows)


def _polars_rows(raw: pl.DataFrame) -> list[str]:
    return sorted(repr(r) for r in raw.rows())


def test_escaped_quote_reads_like_spark(spark: SparkSession, tmp_path: Path) -> None:
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
    columns, expected = _spark_parse(spark, evidence, panels)
    assert raw.columns == columns
    assert _polars_rows(raw) == expected


def test_parse_matches_spark_on_a_mixed_file(spark: SparkSession, tmp_path: Path) -> None:
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
    columns, expected = _spark_parse(spark, evidence, panels_path)
    assert raw.columns == columns
    assert _polars_rows(raw) == expected


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
