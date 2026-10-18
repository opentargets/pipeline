"""Tests for the gene burden step: assembly, disease mapping, post-processing and writing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.schemas.evidence import evidence_schema
from pts.schemas.gene_burden import GENE_BURDEN_PARSED, GENE_BURDEN_UNION
from pts.transformers.evidence.gene_burden import (
    DIRECTION_ON_TARGET,
    DIRECTION_ON_TRAIT,
    GENE_BURDEN,
    assemble,
    check_union,
    gene_burden,
    map_diseases,
    parse_sub_sources,
)
from pts.transformers.evidence.gene_burden.common import to_parsed
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol, normalise_term
from tests.evidence.fixtures import write_disease, write_literature, write_ontoma_lut, write_target
from tests.evidence.gene_burden.fixtures import (
    curated_line,
    cvdi_row,
    genebass_row,
    write_az,
    write_brava,
    write_curated,
    write_cvdi,
    write_finngen,
    write_genebass,
    write_genes_and_health,
)
from tests.evidence.gene_burden.logs import logged

# the removed evidence_postprocess_gene_burden settings
SPARK_SCORE = """
    CASE
      WHEN resourceScore IS NULL OR isnan(resourceScore) THEN NULL
      WHEN resourceScore <= 0 THEN 1.0
      ELSE LEAST(1.0, GREATEST(0.25,
        (1.0 - 0.25) * (log10(resourceScore) - log10(1e-7)) / (log10(1e-17) - log10(1e-7)) + 0.25
      ))
    END
"""
SPARK_DIRECTION_ON_TRAIT = """
    CASE WHEN oddsRatio > 1.0 THEN 'risk'
    WHEN oddsRatio < 1.0 THEN 'protect'
    WHEN beta > 0.0 THEN 'risk'
    WHEN beta < 0.0 THEN 'protect'
    END
"""


def _parsed(*rows: dict[str, Any]) -> pl.LazyFrame:
    base = {'datasourceId': 'gene_burden', 'targetFromSourceId': 'ENSG1', 'diseaseFromSource': 'Disease A'}
    frame = pl.DataFrame([base | row for row in rows], infer_schema_length=None)
    return to_parsed(frame).lazy()


def _label_lut(root: Path, label: str = 'Mapped disease', disease_id: str = 'EFO_0000002') -> str:
    return write_ontoma_lut(root / 'label_lut', 'label', normalise_term(label), disease_id)


def _id_lut(root: Path) -> str:
    return write_ontoma_lut(root / 'id_lut', 'id', normalise_symbol('30000'), 'EFO_0000003')


def _map(tmp: Path, union: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    return logged(lambda: map_diseases(union, _label_lut(tmp), _id_lut(tmp)))


# assembly


def test_union_is_distinct_in_order() -> None:
    first = _parsed({'statisticalMethod': 'M1', 'beta': -0.0}, {'statisticalMethod': 'M2'})
    second = _parsed({'statisticalMethod': 'M3'}, {'statisticalMethod': 'M1', 'beta': 0.0})
    union = assemble([first, second])
    assert union.schema == GENE_BURDEN_PARSED
    assert union['statisticalMethod'].to_list() == ['M1', 'M2', 'M3']
    assert math.copysign(1.0, union['beta'][0]) == 1.0


def test_mantissa_normalised_per_source_counted() -> None:
    union, log = logged(
        lambda: assemble([
            _parsed(
                {'projectId': 'AstraZeneca PheWAS Portal', 'pValueMantissa': 10.0, 'pValueExponent': -9},
                {'projectId': 'AstraZeneca PheWAS Portal', 'pValueMantissa': 9.999, 'pValueExponent': -9},
                {'projectId': 'AstraZeneca PheWAS Portal', 'pValueMantissa': float('nan'), 'pValueExponent': 3},
            ),
            _parsed(
                {'projectId': 'REGENERON', 'pValueMantissa': 10.0, 'pValueExponent': -12},
                {'projectId': 'REGENERON', 'pValueMantissa': 12.5, 'pValueExponent': -5},
                {'projectId': None, 'pValueMantissa': None, 'pValueExponent': None},
            ),
        ])
    )
    rows = union.select('pValueMantissa', 'pValueExponent').rows()
    assert rows[:2] == [(1.0, -8), (9.999, -9)]
    assert math.isnan(rows[2][0])
    assert rows[2][1] == 3
    assert rows[3:] == [(1.0, -11), (1.25, -4), (None, None)]
    assert 'AstraZeneca PheWAS Portal: 1' in log
    assert 'REGENERON: 2' in log


def test_normalised_rows_that_meet_are_one_counted() -> None:
    union, log = logged(
        lambda: assemble([
            _parsed(
                {'projectId': 'REGENERON', 'pValueMantissa': 10.0, 'pValueExponent': -9},
                {'projectId': 'REGENERON', 'pValueMantissa': 10.0, 'pValueExponent': -12},
                {'projectId': 'Genebass', 'pValueMantissa': 10.0, 'pValueExponent': -9},
                {'projectId': 'Genebass', 'pValueMantissa': 10.0, 'pValueExponent': -9},
            ),
            _parsed(
                {'projectId': 'REGENERON', 'pValueMantissa': 1.0, 'pValueExponent': -8},
                {'projectId': 'Genebass', 'pValueMantissa': 1.0, 'pValueExponent': -8},
                {'projectId': 'AZ', 'pValueMantissa': 1.0, 'pValueExponent': -8},
            ),
        ])
    )
    assert union.select('projectId', 'pValueMantissa', 'pValueExponent').rows() == [
        ('REGENERON', 1.0, -8),
        ('REGENERON', 1.0, -11),
        ('Genebass', 1.0, -8),
        ('AZ', 1.0, -8),
    ]
    assert '2 normalised rows merged with an identical row, by project: Genebass: 1, REGENERON: 1' in log


# disease mapping


def test_mapping_gives_the_union_schema(tmp_path: Path) -> None:
    mapped, _ = _map(tmp_path, assemble([_parsed({'diseaseFromSource': 'Mapped disease'})]))
    assert mapped.schema == GENE_BURDEN_UNION
    assert mapped['diseaseFromSourceMappedId'].to_list() == ['EFO_0000002']


def test_curated_id_used_when_unmapped(tmp_path: Path) -> None:
    union = assemble([
        _parsed(
            {'diseaseFromSource': 'Disease A', 'curatedDiseaseFromSourceMappedId': 'EFO_0000001'},
            {'diseaseFromSource': 'Disease B', 'curatedDiseaseFromSourceMappedId': None},
        )
    ])
    mapped, _ = _map(tmp_path, union)
    assert mapped['diseaseFromSourceMappedId'].to_list() == ['EFO_0000001', None]


def test_id_lut_maps_the_source_id(tmp_path: Path) -> None:
    mapped, _ = _map(tmp_path, assemble([_parsed({'diseaseFromSource': 'Disease A', 'diseaseFromSourceId': '30000'})]))
    assert mapped['diseaseFromSourceMappedId'].to_list() == ['EFO_0000003']


def test_ontoma_overrides_curated_counted(tmp_path: Path) -> None:
    def row(project: str, curated: str | None, method: str | None = None) -> dict[str, Any]:
        return {
            'projectId': project,
            'diseaseFromSource': 'Mapped disease',
            'curatedDiseaseFromSourceMappedId': curated,
            'statisticalMethod': method,
        }

    union = assemble([
        _parsed(
            row('Genes & Health', 'EFO_0000001'),
            row('Genes & Health', 'EFO_0000002', 'same'),
            row('REGENERON', 'EFO_0000009'),
            row('REGENERON', None, 'no curated id'),
        )
    ])
    mapped, log = _map(tmp_path, union)
    assert mapped['diseaseFromSourceMappedId'].to_list() == ['EFO_0000002'] * 4
    assert 'Genes & Health: 1' in log
    assert 'REGENERON: 1' in log


def test_a_frame_still_holding_the_curated_id_fails() -> None:
    with pytest.raises(ValueError, match='curatedDiseaseFromSourceMappedId'):
        check_union(_parsed({}).collect())


# post-processing settings


def test_datasource_settings_follow_the_spark_config() -> None:
    assert GENE_BURDEN.datasource_id == 'gene_burden'
    assert GENE_BURDEN.unique_fields == [
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'ancestryId',
        'diseaseFromSource',
        'statisticalMethod',
        'projectId',
    ]


RESOURCE_SCORES = [None, float('nan'), -1.0, 0.0, 1.0, 1e-7, 1.5e-9, 3.3e-12, 1e-17, 5e-320, 0.5]


def test_score_matches_spark(spark: SparkSession) -> None:
    frame = pl.DataFrame({'resourceScore': RESOURCE_SCORES}, schema={'resourceScore': pl.Float64})
    expected = (
        spark
        .createDataFrame([(v,) for v in RESOURCE_SCORES], 'resourceScore double')
        .select(f.expr(SPARK_SCORE).alias('score'))
        .collect()
    )
    assert frame.select(GENE_BURDEN.score.alias('score'))['score'].to_list() == [row.score for row in expected]


DIRECTION_ROWS = [
    (2.0, None, 'risk'),
    (0.5, 1.0, 'protect'),
    (float('nan'), -1.0, 'risk'),
    (1.0, -1.0, 'protect'),
    (1.0, None, None),
    (None, 0.25, 'risk'),
    (None, -0.25, 'protect'),
    (None, float('nan'), 'risk'),
    (None, 0.0, None),
    (None, None, None),
]


def test_direction_on_trait_matches_spark(spark: SparkSession) -> None:
    frame = pl.DataFrame(
        [r[:2] for r in DIRECTION_ROWS], schema={'oddsRatio': pl.Float64, 'beta': pl.Float64}, orient='row'
    )
    spark_rows = (
        spark
        .createDataFrame([r[:2] for r in DIRECTION_ROWS], 'oddsRatio double, beta double')
        .select(f.expr(SPARK_DIRECTION_ON_TRAIT).alias('direction'))
        .collect()
    )
    expected = [r[2] for r in DIRECTION_ROWS]
    assert [row.direction for row in spark_rows] == expected
    assert frame.select(DIRECTION_ON_TRAIT.alias('direction'))['direction'].to_list() == expected


def test_direction_on_target_is_lof_on_every_row() -> None:
    frame = pl.DataFrame({'diseaseId': ['EFO_1', None], 'targetId': [None, 'ENSG1']})
    assert frame.with_columns(DIRECTION_ON_TARGET.alias('direction'))['direction'].to_list() == ['LoF', 'LoF']


# the step


def _sources(
    root: Path,
    curated: list[str],
    genebass: list[dict[str, Any]] | None = None,
    cvdi: list[dict[Any, Any]] | None = None,
) -> dict[str, str]:
    root.mkdir(parents=True)
    az_binary, az_quantitative, az_genes, az_phenotypes = write_az(root, [], [])
    finngen, finngen_phenotypes = write_finngen(root, [])
    return {
        'az_binary': az_binary,
        'az_quantitative': az_quantitative,
        'az_genes': az_genes,
        'az_phenotypes': az_phenotypes,
        'finngen': finngen,
        'finngen_phenotypes': finngen_phenotypes,
        'genebass': write_genebass(root, genebass or []),
        'cvdi': write_cvdi(root, cvdi or []),
        'genes_and_health': write_genes_and_health(root, [], [], []),
        'brava': write_brava(root),
        'curated_studies': write_curated(root / 'curated_evidence.tsv', *curated),
        'ontoma_disease_label_lut': _label_lut(root),
        'ontoma_disease_id_lut': _id_lut(root),
        'disease': write_disease(root / 'disease', ['EFO_0000001', 'EFO_0000002', 'EFO_0000003']),
        'target': write_target(root / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'G1', 'ENSG2': 'G2'}),
        'literature': write_literature(root / 'literature', {'12345': '2020-05-01'}),
    }


def test_sub_sources_in_the_old_union_order(tmp_path: Path) -> None:
    cvdi = [cvdi_row() | {('LOF (MAF<0.1%)', 'Meta P-value'): 1e-9}]
    sources = _sources(tmp_path / 'src', [curated_line()], [genebass_row()], cvdi)
    frames = parse_sub_sources(sources, {'finngen_release': 'R12'})
    assert [frame.collect()['projectId'].unique().to_list() for frame in frames] == [
        [],
        ['REGENERON'],
        ['Genebass'],
        [],
        ['CVDI Human Disease Portal'],
        [],
        [],
    ]


def _run(tmp: Path, curated: list[str], genebass: list[dict[str, Any]] | None = None) -> tuple[pl.DataFrame, ...]:
    destination = {'evidence': str(tmp / 'out' / 'valid'), 'failed_evidence': str(tmp / 'out' / 'failed')}
    gene_burden(_sources(tmp / 'src', curated, genebass), destination, {'finngen_release': 'R12'}, None)
    return pl.read_parquet(destination['evidence']), pl.read_parquet(destination['failed_evidence'])


def test_maps_scores_and_splits_valid_from_failed(tmp_path: Path) -> None:
    valid, failed = _run(
        tmp_path,
        [
            curated_line(statisticalMethod='curated', oddsRatio='0.5'),
            curated_line(statisticalMethod='mapped', diseaseFromSource='Mapped disease', beta='0.25'),
            curated_line(statisticalMethod='unmapped', diseaseFromSource='nothing', diseaseFromSourceMappedId=''),
        ],
        [genebass_row(Pvalue_Burden=2e-9)],
    )
    rows = valid.sort('statisticalMethod').select(
        'statisticalMethod', 'targetId', 'diseaseId', 'directionOnTrait', 'directionOnTarget', 'evidenceDate'
    )
    assert rows.rows() == [
        ('curated', 'ENSG1', 'EFO_0000001', 'protect', 'LoF', '2020-05-01'),
        ('mapped', 'ENSG1', 'EFO_0000002', 'risk', 'LoF', '2020-05-01'),
        ('pLoF', 'ENSG1', 'EFO_0000003', 'risk', 'LoF', None),
    ]
    assert valid.sort('statisticalMethod')['score'].to_list() == [
        0.3867931555708239,
        0.3867931555708239,
        0.3774227503252014,
    ]
    assert valid['id'].n_unique() == 3
    assert failed['statisticalMethod'].to_list() == ['unmapped']


def test_unmapped_lands_in_failed_evidence(tmp_path: Path) -> None:
    valid, failed = _run(tmp_path, [curated_line(diseaseFromSource='Nothing at all', diseaseFromSourceMappedId='')])
    assert valid.height == 0
    assert failed.select('diseaseFromSource', 'diseaseFromSourceMappedId', 'targetId').rows() == [
        ('Nothing at all', None, 'ENSG1')
    ]


def test_outputs_have_normalised_mantissas(tmp_path: Path) -> None:
    valid, _ = _run(tmp_path, [curated_line(resourceScore='1e-8', pValueMantissa='10.0', pValueExponent='-9')])
    assert valid.select('pValueMantissa', 'pValueExponent').rows() == [(1.0, -8)]


def test_empty_sub_source_writes_outputs(tmp_path: Path) -> None:
    empty_valid, empty_failed = _run(tmp_path / 'empty', [])
    valid, failed = _run(
        tmp_path / 'full',
        [curated_line(), curated_line(diseaseFromSource='Nothing at all', diseaseFromSourceMappedId='')],
    )
    assert (empty_valid.height, empty_failed.height, valid.height, failed.height) == (0, 0, 1, 1)
    assert empty_valid.schema == valid.schema
    assert empty_failed.schema == failed.schema
    for column in ('pValueExponent', 'studySampleSize', 'urls', 'directionOnTrait', 'directionOnTarget', 'score'):
        assert empty_valid.schema[column] == evidence_schema[column]
    assert 'curatedDiseaseFromSourceMappedId' not in empty_valid.columns
