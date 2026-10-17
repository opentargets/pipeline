"""Tests for gene2phenotype evidence."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.schemas.gene2phenotype import G2P_INPUT_SCHEMA
from pts.transformers.evidence.gene2phenotype import (
    DIRECTION_ON_TARGET,
    DIRECTION_ON_TRAIT,
    GENE2PHENOTYPE,
    PANELS,
    PYTHON_WHITESPACE,
    gene2phenotype,
    parse_gene2phenotype,
)
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol, normalise_term
from tests.evidence.fixtures import write_disease, write_literature, write_ontoma_lut, write_target

COLUMNS = list(G2P_INPUT_SCHEMA)
PARSED_COLUMNS = [
    'literature',
    'targetFromSourceId',
    'studyId',
    'confidence',
    'allelicRequirements',
    'diseaseFromSourceId',
    'diseaseFromSource',
    'variantFunctionalConsequenceId',
    'datasourceId',
    'datatypeId',
]
HEADER = ','.join(f'"{c}"' for c in COLUMNS) + '\n'


def _line(**fields: str) -> str:
    """One panel line from `column_name=value` pairs, with underscores for spaces; the rest empty."""
    given = {k.replace('_', ' '): v for k, v in fields.items()}
    unknown = set(given) - set(COLUMNS)
    assert not unknown, unknown
    return ','.join(f'"{given[c]}"' if c in given else '' for c in COLUMNS) + '\n'


def _write(path: Path, *lines: str, header: str = HEADER) -> str:
    path.write_text(header + ''.join(lines))
    return str(path)


def _parse(path: Path, *lines: str, header: str = HEADER) -> pl.DataFrame:
    return parse_gene2phenotype([_write(path / 'panel.csv', *lines, header=header)]).collect()


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def test_columns_are_read_by_position(tmp: Path) -> None:
    header = ','.join(f'renamed{i}' for i in range(len(COLUMNS))) + '\n'
    line = _line(gene_symbol='CFTR', disease_name='cystic fibrosis', confidence='strong', panel='Eye')
    raw = _parse(tmp, line, header=header).row(0, named=True)
    assert raw['targetFromSourceId'] == 'CFTR'
    assert raw['diseaseFromSource'] == 'cystic fibrosis'
    assert raw['confidence'] == 'strong'
    assert raw['studyId'] == 'Eye'
    assert raw['datasourceId'] == 'gene2phenotype'
    assert raw['datatypeId'] == 'genetic_literature'


def test_panels_are_concatenated_in_order(tmp: Path) -> None:
    a = _write(tmp / 'a.csv', _line(gene_symbol='A'))
    b = _write(tmp / 'b.csv', _line(gene_symbol='B1'), _line(gene_symbol='B2'))
    assert parse_gene2phenotype([a, b]).collect()['targetFromSourceId'].to_list() == ['A', 'B1', 'B2']


def test_literature_splits_on_semicolon_and_trims_spaces_only(tmp: Path) -> None:
    raw = _parse(
        tmp,
        _line(publications='1; 2;'),
        _line(publications='\t3'),
        _line(),
    )
    assert raw['literature'].to_list() == [['1', '2', ''], ['\t3'], None]


@pytest.mark.parametrize(
    ('fields', 'expected'),
    [
        ({'disease_MONDO': 'MONDO:1', 'disease_mim': '123'}, 'MONDO:1'),
        ({'disease_mim': '123'}, 'OMIM:123'),
        ({'disease_mim': 'No disease mim'}, None),
        ({'disease_mim': 'Has No disease mim in it'}, None),
        ({}, None),
    ],
)
def test_omim_fallback_id(tmp: Path, fields: dict[str, str], expected: str | None) -> None:
    raw = _parse(tmp, _line(**fields))
    assert raw['diseaseFromSourceId'].to_list() == [expected]


def test_related_prefix_is_stripped_greedily(tmp: Path) -> None:
    names = ['A-related B-related disease', 'A-related disease', 'unrelated disease', 'x-related ']
    raw = _parse(tmp, *(_line(disease_name=n) for n in names), _line())
    assert raw['diseaseFromSource'].to_list() == ['disease', 'disease', 'unrelated disease', '', None]


@pytest.mark.parametrize(
    ('consequence', 'expected'),
    [
        ('uncertain; absent gene product', 'SO_0002317'),
        (' absent gene product', 'SO_0002317'),
        ('absent gene product;increased gene product level', 'SO_0002317'),
        ('increased gene product level;decreased gene product level', 'SO_0002316'),
        ('uncertain', 'SO_0002220'),
        ('cis-regulatory or promotor mutation;5_prime or 3_prime UTR mutation', 'SO_0001622'),
        ('unknown term', None),
        ('unknown term; ;', None),
        ('Absent Gene Product', None),
    ],
)
def test_consequence_keeps_highest_ranked_term(tmp: Path, consequence: str, expected: str | None) -> None:
    raw = _parse(tmp, _line(variant_consequence=consequence))
    assert raw['variantFunctionalConsequenceId'].to_list() == [expected]


def test_consequence_strips_python_whitespace(tmp: Path) -> None:
    raw = _parse(tmp, _line(variant_consequence='\N{NO-BREAK SPACE}absent gene product\x1f'))
    assert raw['variantFunctionalConsequenceId'].to_list() == ['SO_0002317']


def test_python_whitespace_is_every_character_python_strips() -> None:
    assert set(PYTHON_WHITESPACE) == {chr(c) for c in range(0x110000) if chr(c).isspace()}


def test_consequence_is_null_when_missing(tmp: Path) -> None:
    assert _parse(tmp, _line())['variantFunctionalConsequenceId'].to_list() == [None]


def test_allelic_requirements_null_not_empty(tmp: Path) -> None:
    raw = _parse(tmp, _line(allelic_requirement='monoallelic'), _line())
    assert raw['allelicRequirements'].to_list() == [['monoallelic'], None]


@pytest.mark.parametrize(
    ('confidence', 'score'),
    [
        ('definitive', 1.0),
        ('both RD and IF', 1.0),
        ('strong', 1.0),
        ('moderate', 0.5),
        ('limited', 0.01),
        ('Strong', None),
        ('disputed', None),
        (None, None),
    ],
)
def test_score_follows_the_confidence(confidence: str | None, score: float | None) -> None:
    df = pl.DataFrame({'confidence': [confidence]}, schema={'confidence': pl.String})
    assert df.select(GENE2PHENOTYPE.score).item() == score


def test_score_and_direction() -> None:
    df = pl.DataFrame(
        {
            'diseaseId': ['EFO_1', 'EFO_1', None, 'EFO_1'],
            'variantFunctionalConsequenceId': ['SO_0002315', 'SO_0002317', 'SO_0002317', None],
        },
        schema={'diseaseId': pl.String, 'variantFunctionalConsequenceId': pl.String},
    )
    out = df.select(trait=DIRECTION_ON_TRAIT, target=DIRECTION_ON_TARGET)
    assert out['trait'].to_list() == ['risk', 'risk', None, 'risk']
    assert out['target'].to_list() == ['GoF', 'LoF', 'LoF', None]


def test_unique_fields_follow_the_spark_config() -> None:
    assert GENE2PHENOTYPE.datasource_id == 'gene2phenotype'
    assert GENE2PHENOTYPE.unique_fields == [
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'diseaseFromSource',
        'allelicRequirements',
        'studyId',
    ]


def test_panel_keys() -> None:
    assert PANELS == [
        'cancer_panel',
        'cardiac_panel',
        'developmental_panel',
        'ear_panel',
        'eye_panel',
        'skeletal_panel',
        'skin_panel',
    ]


def test_escaped_quote_matches_spark_baseline(tmp: Path) -> None:
    lines = [
        _line(gene_symbol='A', disease_name='say \\"hi\\" disease', publications='1'),
        _line(gene_symbol='B', disease_name='back\\\\slash disease', variant_consequence='uncertain'),
        _line(gene_symbol='C', disease_name='plain \\n text', allelic_requirement='AD'),
        _line(gene_symbol='D', disease_name='"unescaped" inner'),
    ]
    path = _write(tmp / 'panel.csv', *lines)
    raw = parse_gene2phenotype([path]).collect()
    assert raw['diseaseFromSource'].to_list()[:3] == ['say "hi" disease', 'back\\slash disease', 'plain \\n text']
    # expected rows captured from the removed pyspark implementation
    assert raw.columns == PARSED_COLUMNS
    assert raw.rows() == [
        (['1'], 'A', None, None, None, None, 'say "hi" disease', None, 'gene2phenotype', 'genetic_literature'),
        (
            None,
            'B',
            None,
            None,
            None,
            None,
            'back\\slash disease',
            'SO_0002220',
            'gene2phenotype',
            'genetic_literature',
        ),
        (None, 'C', None, None, ['AD'], None, 'plain \\n text', None, 'gene2phenotype', 'genetic_literature'),
        (None, 'D', None, None, None, None, '""unescaped" inner"', None, 'gene2phenotype', 'genetic_literature'),
    ]


def test_parse_matches_spark_baseline_on_a_mixed_panel(tmp: Path) -> None:
    lines = [
        _line(
            g2p_id='G1',
            gene_symbol='CFTR',
            gene_mim='602421',
            hgnc_id='1884',
            disease_name='CFTR-related cystic fibrosis',
            disease_mim='219700',
            disease_MONDO='MONDO:0009061',
            allelic_requirement='biallelic',
            confidence='definitive',
            variant_consequence='uncertain; absent gene product',
            publications='1; 2;',
            panel='Developmental',
        ),
        _line(gene_symbol='X', disease_mim='No disease mim', confidence='limited', publications='\t3'),
        _line(gene_symbol='Y', disease_mim='5', variant_consequence='increased gene product level', panel='Eye'),
        _line(gene_symbol='Z', gene_mim='not a number', hgnc_id='HGNC:1'),
    ]
    path = _write(tmp / 'panel.csv', *lines)
    raw = parse_gene2phenotype([path]).collect()
    # expected rows captured from the removed pyspark implementation
    assert raw.columns == PARSED_COLUMNS
    assert raw.rows() == [
        (
            ['1', '2', ''],
            'CFTR',
            'Developmental',
            'definitive',
            ['biallelic'],
            'MONDO:0009061',
            'cystic fibrosis',
            'SO_0002317',
            'gene2phenotype',
            'genetic_literature',
        ),
        (['\t3'], 'X', None, 'limited', None, None, None, None, 'gene2phenotype', 'genetic_literature'),
        (None, 'Y', 'Eye', None, None, 'OMIM:5', None, 'SO_0002315', 'gene2phenotype', 'genetic_literature'),
        (None, 'Z', None, None, None, None, None, None, 'gene2phenotype', 'genetic_literature'),
    ]


def _sources(root: Path, panels: dict[str, list[str]]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    source = {key: _write(root / f'{key}.csv', *panels.get(key, [])) for key in PANELS}
    return source | {
        'ontoma_disease_label_lut': write_ontoma_lut(
            root / 'label_lut', 'label', normalise_term('cystic fibrosis'), 'MONDO_0009061'
        ),
        'ontoma_disease_id_lut': write_ontoma_lut(
            root / 'id_lut', 'id', normalise_symbol('MONDO:0007254'), 'MONDO_0007254'
        ),
        'disease': write_disease(root / 'disease', ['MONDO_0009061', 'MONDO_0007254']),
        'target': write_target(root / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'CFTR', 'ENSG2': 'BRCA1'}),
        'literature': write_literature(root / 'literature', {'111': '2020-05-01'}),
    }


def _destination(root: Path) -> dict[str, str]:
    return {'evidence': str(root / 'valid'), 'failed_evidence': str(root / 'failed')}


def test_maps_scores_and_splits_valid_from_failed(tmp: Path) -> None:
    source = _sources(
        tmp / 'src',
        {
            'eye_panel': [
                _line(
                    gene_symbol='CFTR',
                    disease_name='CFTR-related cystic fibrosis',
                    allelic_requirement='biallelic',
                    confidence='definitive',
                    variant_consequence='absent gene product',
                    publications='111',
                    panel='Eye',
                ),
                _line(
                    gene_symbol='BRCA1',
                    disease_name='something else',
                    disease_MONDO='MONDO:0007254',
                    confidence='moderate',
                    variant_consequence='increased gene product level',
                    panel='Eye',
                ),
                _line(gene_symbol='BRCA1', disease_name='unmappable', confidence='limited', panel='Eye'),
            ]
        },
    )
    destination = _destination(tmp / 'out')
    gene2phenotype(source, destination, {}, None)

    valid = pl.read_parquet(destination['evidence']).sort('targetId')
    assert valid.select(
        'targetId', 'diseaseId', 'score', 'directionOnTrait', 'directionOnTarget', 'evidenceDate'
    ).rows() == [
        ('ENSG1', 'MONDO_0009061', 1.0, 'risk', 'LoF', '2020-05-01'),
        ('ENSG2', 'MONDO_0007254', 0.5, 'risk', 'GoF', None),
    ]
    assert valid['datasourceId'].to_list() == ['gene2phenotype'] * 2
    assert valid['id'].n_unique() == 2


def test_unmapped_disease_lands_in_failed_evidence(tmp: Path) -> None:
    source = _sources(
        tmp / 'src',
        {'skin_panel': [_line(gene_symbol='CFTR', disease_name='nothing at all', confidence='strong', panel='Skin')]},
    )
    destination = _destination(tmp / 'out')
    gene2phenotype(source, destination, {}, None)

    failed = pl.read_parquet(destination['failed_evidence'])
    assert failed['diseaseFromSource'].to_list() == ['nothing at all']
    assert failed['diseaseFromSourceMappedId'].to_list() == [None]
    assert failed['targetId'].to_list() == ['ENSG1']
    assert pl.read_parquet(destination['evidence']).height == 0


def test_empty_panels_write_empty_outputs(tmp: Path) -> None:
    source = _sources(tmp / 'src', {})
    destination = _destination(tmp / 'out')
    gene2phenotype(source, destination, {}, None)

    assert pl.read_parquet(destination['evidence']).height == 0
    assert pl.read_parquet(destination['failed_evidence']).height == 0
