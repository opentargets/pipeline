"""Tests for the pharmacogenetics step."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest
from loguru import logger

from pts.transformers import pharmacogenetics as pgx
from pts.transformers.pharmacogenetics import (
    add_variant_id,
    annotate_phenotype,
    enrich_with_drugs,
    find_unparsed_texts,
    pharmacogenetics,
)
from pts.transformers.utils.disease_mapping.normalise import normalise_term
from tests.evidence.fixtures import write_ontoma_lut

PGX_SCHEMA = {
    'genotypeAnnotationText': pl.String,
    'phenotypeText': pl.String,
    'phenotypeFromSourceId': pl.String,
    'studyId': pl.String,
}
LUT_SCHEMA = {'genotypeAnnotationText': pl.String, 'phenotypeText': pl.List(pl.String)}

PGX_ROWS = [
    ('parsed', 'verbose', None, 's1'),
    ('parsed empty', 'verbose', None, 's2'),
    ('unparsed b', 'verbose', None, 's3'),
    ('unparsed a', 'verbose', None, 's4'),
    (None, 'verbose', None, 's5'),
]
LUT_ROWS = [('parsed', ['increased response']), ('parsed empty', [])]

DRUGS = pl.List(pl.Struct({'drugFromSource': pl.String}))
ENRICH_SCHEMA = {
    'studyId': pl.String,
    'targetFromSourceId': pl.String,
    'drugs': DRUGS,
}
MOLECULE_SCHEMA = {'id': pl.String, 'name': pl.String}
MOA_SCHEMA = {'chemblIds': pl.List(pl.String), 'targets': pl.List(pl.String)}


@pytest.fixture
def pgx_df() -> pl.DataFrame:
    return pl.DataFrame(PGX_ROWS, schema=PGX_SCHEMA, orient='row')


@pytest.fixture
def lut_df() -> pl.DataFrame:
    return pl.DataFrame(LUT_ROWS, schema=LUT_SCHEMA, orient='row')


def _rows(df: pl.DataFrame) -> list[tuple]:
    return sorted(df.rows(), key=repr)


class TestFindUnparsedTexts:
    def test_texts_missing_from_the_lookup_are_returned_sorted(self, pgx_df, lut_df) -> None:
        assert find_unparsed_texts(pgx_df, lut_df) == ['unparsed a', 'unparsed b']

    def test_empty_when_everything_is_parsed(self, lut_df) -> None:
        df = pl.DataFrame(
            [('parsed', None, None, 's1'), ('parsed empty', None, None, 's2')], schema=PGX_SCHEMA, orient='row'
        )
        assert find_unparsed_texts(df, lut_df) == []


class TestAnnotatePhenotype:
    def test_unparsed_rows_are_kept_with_a_null_phenotype(self, pgx_df, lut_df) -> None:
        rows = dict(annotate_phenotype(pgx_df, lut_df).select('studyId', 'phenotypeText').rows())
        assert rows == {
            's1': 'increased response',
            's2': None,
            's3': None,
            's4': None,
            's5': None,
        }

    def test_one_row_per_phenotype_and_identical_rows_once(self, lut_df) -> None:
        lut = pl.DataFrame([('multi', ['a', 'b', 'a'])], schema=LUT_SCHEMA, orient='row')
        df = pl.DataFrame(
            [('multi', 'verbose', 'X', 's1'), ('multi', 'other', None, 's1')], schema=PGX_SCHEMA, orient='row'
        )
        out = annotate_phenotype(df, lut)
        assert out.select('studyId', 'phenotypeText').rows() == [('s1', 'a'), ('s1', 'b')]
        assert 'phenotypeFromSourceId' not in out.columns

    def test_matches_spark_baseline(self, pgx_df, lut_df) -> None:
        out = annotate_phenotype(pgx_df, lut_df).select('genotypeAnnotationText', 'studyId', 'phenotypeText')
        # expected rows captured from the removed pyspark implementation
        assert _rows(out) == [
            ('parsed empty', 's2', None),
            ('parsed', 's1', 'increased response'),
            ('unparsed a', 's4', None),
            ('unparsed b', 's3', None),
            (None, 's5', None),
        ]


GENOTYPES = [
    ('1_100_C_C,T', 's1'),
    ('2_200_A_G,T', 's2'),
    ('3_300_G_G', 's3'),
    (None, 's4'),
    ('4_400_A_', 's5'),
    ('5_500_A', 's6'),
    ('6_600__G', 's7'),
    ('7_700_A_G,', 's8'),
    ('8_800_A_G_extra', 's9'),
    ('1_100_C_C,T', 's10'),
]


def _genotypes() -> pl.DataFrame:
    return pl.DataFrame(GENOTYPES, schema={'genotypeId': pl.String, 'studyId': pl.String}, orient='row')


def test_variant_id_from_genotype() -> None:
    out = add_variant_id(_genotypes())
    by_study = {
        s: sorted(v for sid, v in out.select('studyId', 'variantId').rows() if sid == s) for s in out['studyId']
    }
    assert by_study['s1'] == ['1_100_C_T']
    assert by_study['s2'] == ['2_200_A_G', '2_200_A_T']
    assert by_study['s3'] == [None]
    assert by_study['s4'] == [None]
    assert by_study['s10'] == ['1_100_C_T']


def test_empty_alt_gives_null_variant_id() -> None:
    out = dict(
        add_variant_id(_genotypes())
        .filter(pl.col('studyId').is_in(['s5', 's6', 's7']))
        .select('studyId', 'variantId')
        .rows()
    )
    assert out == {'s5': None, 's6': None, 's7': None}


def test_variant_id_trailing_comma_and_extra_tokens() -> None:
    out = add_variant_id(_genotypes())
    by_study = {s: sorted(v for sid, v in out.select('studyId', 'variantId').rows() if sid == s) for s in ['s8', 's9']}
    assert by_study == {'s8': ['7_700_A_', '7_700_A_G'], 's9': ['8_800_A_G']}


def test_variant_id_matches_spark_baseline() -> None:
    # expected rows captured from the removed pyspark implementation
    assert _rows(add_variant_id(_genotypes()).select('genotypeId', 'studyId', 'variantId')) == [
        ('1_100_C_C,T', 's1', '1_100_C_T'),
        ('1_100_C_C,T', 's10', '1_100_C_T'),
        ('2_200_A_G,T', 's2', '2_200_A_G'),
        ('2_200_A_G,T', 's2', '2_200_A_T'),
        ('3_300_G_G', 's3', None),
        ('4_400_A_', 's5', None),
        ('5_500_A', 's6', None),
        ('6_600__G', 's7', None),
        ('7_700_A_G,', 's8', '7_700_A_'),
        ('7_700_A_G,', 's8', '7_700_A_G'),
        ('8_800_A_G_extra', 's9', '8_800_A_G'),
        (None, 's4', None),
    ]


def _molecules(rows: list[tuple[str | None, str | None]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=MOLECULE_SCHEMA, orient='row')


def _moa(rows: list[tuple[list[str] | None, list[str | None] | None]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=MOA_SCHEMA, orient='row')


def _annotations(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=ENRICH_SCHEMA, orient='row')


def test_drug_name_lut_lowercases_and_takes_max_id() -> None:
    out = enrich_with_drugs(
        _annotations([('s1', 'ENSG1', [{'drugFromSource': 'Warfarin'}])]),
        _molecules([('CHEMBL10', 'WARFARIN'), ('CHEMBL9', 'warfarin'), (None, 'Warfarin'), ('CHEMBL1', None)]),
        _moa([]),
    )
    assert out['drugs'].to_list() == [[{'drugFromSource': 'warfarin', 'drugId': 'CHEMBL9'}]]


def test_is_direct_target_null_is_false() -> None:
    out = enrich_with_drugs(
        _annotations([
            ('no moa', 'ENSG1', [{'drugFromSource': 'a'}]),
            ('no target', None, [{'drugFromSource': 'b'}]),
            ('other target', 'ENSG1', [{'drugFromSource': 'b'}]),
            ('unknown drug', 'ENSG1', [{'drugFromSource': 'z'}]),
        ]),
        _molecules([('CHEMBL_A', 'a'), ('CHEMBL_B', 'b')]),
        _moa([(['CHEMBL_B'], ['ENSG2', None])]),
    )
    assert dict(out.select('studyId', 'isDirectTarget').rows()) == {
        'no moa': False,
        'no target': False,
        'other target': False,
        'unknown drug': False,
    }


def test_is_direct_target_with_a_null_among_the_targets() -> None:
    out = enrich_with_drugs(
        _annotations([('s1', 'ENSG1', [{'drugFromSource': 'b'}])]),
        _molecules([('CHEMBL_B', 'b')]),
        _moa([(['CHEMBL_B'], [None, 'ENSG1'])]),
    )
    assert out['isDirectTarget'].to_list() == [True]


def test_rows_without_drugs_are_dropped_and_counted() -> None:
    messages: list[str] = []
    sink_id = logger.add(messages.append, level='INFO')
    try:
        out = enrich_with_drugs(
            _annotations([
                ('null', 'ENSG1', None),
                ('empty', 'ENSG1', []),
                ('kept', 'ENSG1', [{'drugFromSource': 'a'}]),
            ]),
            _molecules([('CHEMBL_A', 'a')]),
            _moa([]),
        )
    finally:
        logger.remove(sink_id)
    assert out['studyId'].to_list() == ['kept']
    assert any('2 rows without drugs' in m for m in messages)


def test_direct_target_split_rows() -> None:
    out = enrich_with_drugs(
        _annotations([
            (
                's1',
                'ENSG1',
                [{'drugFromSource': 'a'}, {'drugFromSource': 'b'}, {'drugFromSource': 'c'}, {'drugFromSource': 'a'}],
            )
        ]),
        _molecules([('CHEMBL_A', 'a'), ('CHEMBL_B', 'b'), ('CHEMBL_C', 'c')]),
        _moa([(['CHEMBL_B', 'CHEMBL_X'], ['ENSG1']), (['CHEMBL_A'], []), (['CHEMBL_C'], None)]),
    )
    assert out.select('isDirectTarget', 'drugs').rows() == [
        (False, [{'drugFromSource': 'a', 'drugId': 'CHEMBL_A'}, {'drugFromSource': 'c', 'drugId': 'CHEMBL_C'}]),
        (True, [{'drugFromSource': 'b', 'drugId': 'CHEMBL_B'}]),
    ]


def test_case_variants_of_a_drug_are_kept_apart() -> None:
    out = enrich_with_drugs(
        _annotations([('s1', 'ENSG1', [{'drugFromSource': 'A'}, {'drugFromSource': 'a'}])]),
        _molecules([('CHEMBL_A', 'a')]),
        _moa([]),
    )
    assert out['drugs'].to_list() == [[{'drugFromSource': 'a', 'drugId': 'CHEMBL_A'}] * 2]


ENRICH_ROWS = [
    (
        's1',
        'ENSG1',
        [{'drugFromSource': 'a'}, {'drugFromSource': 'B'}, {'drugFromSource': 'a'}, {'drugFromSource': 'A'}],
    ),
    ('s2', None, [{'drugFromSource': 'b'}]),
    ('s3', 'ENSG2', [{'drugFromSource': 'unknown'}, {'drugFromSource': None}]),
    ('s4', 'ENSG1', None),
    ('s5', 'ENSG1', []),
    ('s1', 'ENSG1', [{'drugFromSource': 'c'}]),
]
MOLECULE_ROWS = [('CHEMBL_A', 'a'), ('CHEMBL_A2', 'A'), ('CHEMBL_B', 'b'), ('CHEMBL_C', 'C'), (None, 'c')]
MOA_ROWS = [
    (['CHEMBL_B'], ['ENSG1', None]),
    (['CHEMBL_A2', 'CHEMBL_C'], ['ENSG2']),
    (['CHEMBL_C'], ['ENSG1']),
    (None, ['ENSG1']),
]


def test_enrich_matches_spark_baseline() -> None:
    out = enrich_with_drugs(_annotations(ENRICH_ROWS), _molecules(MOLECULE_ROWS), _moa(MOA_ROWS)).select(
        'studyId', 'targetFromSourceId', 'isDirectTarget', 'drugs'
    )

    def drugs_sorted(rows: list[tuple]) -> list[tuple]:
        return sorted(((*r[:-1], sorted(r[-1], key=repr)) for r in rows), key=repr)

    # expected rows captured from the removed pyspark implementation
    assert drugs_sorted(out.rows()) == [
        ('s1', 'ENSG1', False, [{'drugFromSource': 'a', 'drugId': 'CHEMBL_A2'}] * 2),
        ('s1', 'ENSG1', True, [{'drugFromSource': 'b', 'drugId': 'CHEMBL_B'}]),
        ('s1', 'ENSG1', True, [{'drugFromSource': 'c', 'drugId': 'CHEMBL_C'}]),
        ('s2', None, False, [{'drugFromSource': 'b', 'drugId': 'CHEMBL_B'}]),
        (
            's3',
            'ENSG2',
            False,
            [{'drugFromSource': 'unknown', 'drugId': None}, {'drugFromSource': None, 'drugId': None}],
        ),
    ]



CLINPGX_RECORD = {
    'datasourceId': 'clinpgx',
    'datasourceVersion': '2026-07-05',
    'datatypeId': 'clinical_annotation',
    'studyId': '655384602',
    'evidenceLevel': '3',
    'literature': ['17588543'],
    'genotype': 'CT',
    'genotypeAnnotationText': 'parsed',
    'drugs': [{'drugFromSource': 'Risperidone'}],
    'pgxCategory': 'efficacy',
    'phenotypeText': 'Schizophrenia',
    'phenotypeFromSourceId': 'MONDO_0005090',
    'genotypeId': '1_163063901_C_C,T',
    'variantRsId': 'rs951439',
    'variantFunctionalConsequenceId': 'SO_0001631',
    'targetFromSourceId': 'ENSG00000117152',
    'variantAnnotation': [
        {
            'literature': '1',
            'effect': 'x',
            'id': 'v1',
            'entity': 'e',
            'effectType': 't',
            'effectDescription': 'd',
            'directionality': 'up',
            'comparisonAlleleOrGenotype': 'T',
            'baseAlleleOrGenotype': 'C',
        }
    ],
}


def _write_inputs(root: Path, records: list[dict], phenotypes: list[dict]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    clinpgx = root / 'clinpgx_annotation.json.gz'
    clinpgx.write_bytes(gzip.compress(''.join(json.dumps(r) + '\n' for r in records).encode()))
    (root / 'phenotypes.json').write_text(json.dumps(phenotypes))
    (root / 'chembl_molecule').mkdir()
    _molecules([('CHEMBL85', 'RISPERIDONE')]).write_parquet(root / 'chembl_molecule' / 'part-0.parquet')
    (root / 'moa').mkdir()
    _moa([(['CHEMBL85'], ['ENSG00000117152'])]).write_parquet(root / 'moa' / 'part-0.parquet')
    return {
        'clinpgx': str(clinpgx),
        'phenotypes': str(root / 'phenotypes.json'),
        'ontoma_disease_label_lut': write_ontoma_lut(
            root / 'label_lut', 'label', normalise_term('increased response'), 'EFO_0000001'
        ),
        'chembl_molecule': str(root / 'chembl_molecule'),
        'drug_mechanism_of_action': str(root / 'moa'),
    }


def _destination(root: Path) -> dict[str, str]:
    return {'associations': str(root / 'out'), 'unparsed_texts': str(root / 'unparsed_texts.json')}


def _read_output(path: str) -> pl.DataFrame:
    return pl.read_parquet(f'{path}/*.parquet')








def test_missing_column_in_a_non_empty_input_raises(tmp_path: Path) -> None:
    record = {k: v for k, v in CLINPGX_RECORD.items() if k != 'drugs'}
    source = _write_inputs(tmp_path / 'in', [record], [{'genotypeAnnotationText': 'parsed', 'phenotypeText': []}])
    with pytest.raises(ValueError, match="'drugs'"):
        pharmacogenetics(source, _destination(tmp_path), {}, None)


def test_missing_column_in_a_non_empty_phenotypes_lookup_raises(tmp_path: Path) -> None:
    source = _write_inputs(tmp_path / 'in', [CLINPGX_RECORD], [{'genotypeAnnotationText': 'parsed'}])
    with pytest.raises(ValueError, match="'phenotypeText'"):
        pharmacogenetics(source, _destination(tmp_path), {}, None)


class TestStep:
    """The step writes the unparsed texts and warns, without calling any external API."""

    def _run(self, monkeypatch, tmp_path, rows: list[tuple]) -> tuple[list[str], list[str]]:
        records = [
            {**dict(zip(PGX_SCHEMA, row, strict=True)), 'genotypeId': None, 'drugs': None, 'targetFromSourceId': None}
            for row in rows
        ]
        source = _write_inputs(
            tmp_path / 'in', records, [{'genotypeAnnotationText': t, 'phenotypeText': p} for t, p in LUT_ROWS]
        )
        monkeypatch.setattr(pgx, 'add_variant_id', lambda df: df.with_columns(variantId=pl.lit(None, pl.String)))
        monkeypatch.setattr(
            pgx, 'add_efo_mapping', lambda lf, **_: lf.with_columns(diseaseFromSourceMappedId=pl.lit(None, pl.String))
        )
        monkeypatch.setattr(pgx, 'enrich_with_drugs', lambda df, *_: df.with_columns(isDirectTarget=pl.lit(False)))

        destination = _destination(tmp_path)
        messages: list[str] = []
        sink_id = logger.add(messages.append, level='WARNING')
        try:
            pharmacogenetics(source, destination, {}, None)
        finally:
            logger.remove(sink_id)
        self.out = _read_output(destination['associations'])
        return json.loads(Path(destination['unparsed_texts']).read_text()), messages

    def test_unparsed_texts_are_written_and_warned(self, monkeypatch, tmp_path) -> None:
        unparsed, messages = self._run(monkeypatch, tmp_path, PGX_ROWS)
        assert unparsed == ['unparsed a', 'unparsed b']
        assert any('2 genotypeAnnotationText' in m for m in messages)
        assert self.out.height == 5
        assert self.out['phenotypeText'].null_count() == 4

    def test_an_empty_list_is_written_when_everything_is_parsed(self, monkeypatch, tmp_path) -> None:
        unparsed, messages = self._run(monkeypatch, tmp_path, [('parsed', None, None, 's1')])
        assert unparsed == []
        assert messages == []
