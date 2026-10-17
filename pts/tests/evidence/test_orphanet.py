"""Tests for orphanet evidence."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from pts.pyspark.orphanet import process_orphanet as spark_process_orphanet
from pts.transformers.evidence.orphanet import (
    DIRECTION_ON_TARGET,
    DIRECTION_ON_TRAIT,
    ORPHANET,
    orphanet,
    parse_orphanet,
    process_orphanet,
)
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol, normalise_term
from tests.evidence.fixtures import write_disease, write_literature, write_ontoma_lut, write_target

GERMLINE_LOF = 'Disease-causing germline mutation(s) (loss of function) in'
GERMLINE_GOF = 'Disease-causing germline mutation(s) (gain of function) in'
GERMLINE = 'Disease-causing germline mutation(s) in'


def _association(
    gene: str = 'CFTR',
    ensg: str | None = 'ENSG00000001',
    kind: str | None = GERMLINE,
    status: str = 'Assessed',
    validation: str | None = '11111[PMID]_22222[PMID]',
) -> str:
    refs = ''.join(
        f'<ExternalReference><Source>x</Source><Reference>{r}</Reference></ExternalReference>'
        for r in ['OMIM1', ensg]
        if r
    )
    return (
        '<DisorderGeneAssociation>'
        + (f'<SourceOfValidation>{validation}</SourceOfValidation>' if validation is not None else '')
        + (f'<DisorderGeneAssociationType><Name>{kind}</Name></DisorderGeneAssociationType>' if kind else '')
        + f'<DisorderGeneAssociationStatus><Name>{status}</Name></DisorderGeneAssociationStatus>'
        + f'<Gene><Name>{gene}</Name><ExternalReferenceList>{refs}</ExternalReferenceList></Gene>'
        + '</DisorderGeneAssociation>'
    )


def _disorder(code: str | None, name: str, *associations: str) -> str:
    return (
        '<Disorder>'
        + (f'<OrphaCode>{code}</OrphaCode>' if code is not None else '')
        + f'<Name>{name}</Name><DisorderType><Name>Disease</Name></DisorderType>'
        + f'<DisorderGeneAssociationList count="{len(associations)}">'
        + ''.join(associations)
        + '</DisorderGeneAssociationList>'
        + '</Disorder>'
    )


def _write(path: Path, *disorders: str) -> str:
    path.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?><JDBOR><DisorderList count="{len(disorders)}">'
        + ''.join(disorders)
        + '</DisorderList></JDBOR>'
    )
    return str(path)


def _parse(tmp_path: Path, *disorders: str) -> pl.DataFrame:
    return parse_orphanet(_write(tmp_path / 'orphanet.xml', *disorders)).collect()


def test_disorders_without_orphacode_are_skipped(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path, _disorder(None, 'nameless', _association()), _disorder('558', 'cystic fibrosis', _association())
    )
    assert raw['diseaseFromSourceId'].to_list() == ['Orphanet_558']
    assert raw['diseaseFromSource'].to_list() == ['cystic fibrosis']


def test_first_ensg_reference_is_the_target(tmp_path: Path) -> None:
    association = (
        '<DisorderGeneAssociation><Gene><Name>G</Name><ExternalReferenceList>'
        '<ExternalReference><Reference>ENSGR0000001</Reference></ExternalReference>'
        '<ExternalReference><Reference>ENSG00000002</Reference></ExternalReference>'
        '</ExternalReferenceList></Gene></DisorderGeneAssociation>'
    )
    raw = _parse(tmp_path, _disorder('1', 'd', association))
    assert raw['targetFromSourceId'].to_list() == ['ENSGR0000001']
    assert raw['targetFromSource'].to_list() == ['G']


def test_literature_pmids_distinct_first_seen(tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        _disorder(
            '1',
            'd',
            _association(validation='3[PMID]_1[PMID]_3[PMID]_ignored_2[PMID]'),
            _association(validation='no pmids here'),
            _association(validation=None),
        ),
    )
    assert process_orphanet(raw.lazy()).collect()['literature'].to_list() == [['3', '1', '2'], [], None]


def test_pmid_is_right_stripped_of_python_whitespace(tmp_path: Path) -> None:
    raw = _parse(tmp_path, _disorder('1', 'd', _association(validation='12\N{IDEOGRAPHIC SPACE}[PMID]_34 [PMID]')))
    assert raw['literature'].to_list() == [['12', '34']]


def test_rows_without_ensg_reference_are_dropped(tmp_path: Path) -> None:
    raw = _parse(tmp_path, _disorder('1', 'd', _association(ensg=None), _association(gene='KEPT')))
    assert raw['targetFromSourceId'].to_list() == [None, 'ENSG00000001']
    assert process_orphanet(raw.lazy()).collect()['targetFromSource'].to_list() == ['KEPT']


def _sources(root: Path, *disorders: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    return {
        'evidence': _write(root / 'orphanet.xml', *disorders),
        'ontoma_disease_label_lut': write_ontoma_lut(
            root / 'label_lut', 'label', normalise_term('cystic fibrosis'), 'MONDO_0009061'
        ),
        'ontoma_disease_id_lut': write_ontoma_lut(
            root / 'id_lut', 'id', normalise_symbol('MONDO:0007254'), 'MONDO_0007254'
        ),
        'disease': write_disease(root / 'disease', ['MONDO_0009061', 'MONDO_0007254']),
        'target': write_target(root / 'target', ['ENSG00000001', 'ENSG00000002'], {'ENSG00000001': 'CFTR'}),
        'literature': write_literature(root / 'literature', {'11111': '2020-05-01'}),
    }


def _destination(root: Path) -> dict[str, str]:
    return {'evidence': str(root / 'valid'), 'failed_evidence': str(root / 'failed')}


def _run(tmp_path: Path, *disorders: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    destination = _destination(tmp_path / 'out')
    orphanet(_sources(tmp_path / 'src', *disorders), destination, {}, None)
    return pl.read_parquet(destination['evidence']), pl.read_parquet(destination['failed_evidence'])


@pytest.mark.parametrize(
    'kind',
    [
        'Major susceptibility factor in',
        'Part of a fusion gene in',
        'Candidate gene tested in',
        'Role in the phenotype of',
        'Biomarker tested in',
        'Disease-causing somatic mutation(s) in',
    ],
)
def test_excluded_association_types_dropped(tmp_path: Path, kind: str) -> None:
    valid, failed = _run(tmp_path, _disorder('558', 'cystic fibrosis', _association(kind=kind), _association(gene='K')))
    assert pl.concat([valid, failed], how='diagonal')['targetFromSource'].to_list() == ['K']


def test_null_association_type_dropped(tmp_path: Path) -> None:
    valid, failed = _run(tmp_path, _disorder('558', 'cystic fibrosis', _association(kind=None), _association(gene='K')))
    assert pl.concat([valid, failed], how='diagonal')['targetFromSource'].to_list() == ['K']


def test_allele_origins_are_germline_and_somatic(tmp_path: Path) -> None:
    valid, _ = _run(tmp_path, _disorder('558', 'cystic fibrosis', _association()))
    assert valid['alleleOrigins'].to_list() == [['germline', 'somatic']]
    assert valid['datasourceId'].to_list() == ['orphanet']
    assert valid['datatypeId'].to_list() == ['genetic_association']


@pytest.mark.parametrize(
    ('confidence', 'score'),
    [('Assessed', 1.0), ('Not yet assessed', 0.5), ('other', None), (None, None)],
)
def test_score_follows_the_confidence(confidence: str | None, score: float | None) -> None:
    df = pl.DataFrame({'confidence': [confidence]}, schema={'confidence': pl.String})
    assert df.select(ORPHANET.score).item() == score


def test_score_and_direction() -> None:
    df = pl.DataFrame(
        {
            'diseaseId': ['EFO_1', 'EFO_1', None, 'EFO_1'],
            'variantFunctionalConsequenceId': ['SO_0002053', 'SO_0002054', 'SO_0002054', None],
        },
        schema={'diseaseId': pl.String, 'variantFunctionalConsequenceId': pl.String},
    )
    out = df.select(trait=DIRECTION_ON_TRAIT, target=DIRECTION_ON_TARGET)
    assert out['trait'].to_list() == ['risk', 'risk', None, 'risk']
    assert out['target'].to_list() == ['GoF', 'LoF', 'LoF', None]


def test_unique_fields_follow_the_spark_config() -> None:
    assert ORPHANET.datasource_id == 'orphanet'
    assert ORPHANET.unique_fields == [
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'diseaseFromSourceId',
    ]


def test_maps_scores_and_directions(tmp_path: Path) -> None:
    valid, _ = _run(
        tmp_path,
        _disorder(
            '558',
            'cystic fibrosis',
            _association(kind=GERMLINE_LOF, validation='11111[PMID]'),
            _association(gene='G2', ensg='ENSG00000002', kind=GERMLINE_GOF, status='Not yet assessed', validation=None),
        ),
    )
    assert valid.sort('targetId').select(
        'targetId', 'diseaseId', 'score', 'directionOnTrait', 'directionOnTarget', 'evidenceDate'
    ).rows() == [
        ('ENSG00000001', 'MONDO_0009061', 1.0, 'risk', 'LoF', '2020-05-01'),
        ('ENSG00000002', 'MONDO_0009061', 0.5, 'risk', 'GoF', None),
    ]


def test_unmapped_disease_lands_in_failed_evidence(tmp_path: Path) -> None:
    valid, failed = _run(tmp_path, _disorder('1', 'nothing at all', _association()))
    assert failed['diseaseFromSource'].to_list() == ['nothing at all']
    assert failed['diseaseFromSourceMappedId'].to_list() == [None]
    assert valid.height == 0


def test_empty_input_writes_empty_outputs(tmp_path: Path) -> None:
    valid, failed = _run(tmp_path)
    assert valid.height == 0
    assert failed.height == 0


def test_process_matches_spark(spark: SparkSession, tmp_path: Path) -> None:
    raw = _parse(
        tmp_path,
        _disorder(
            '1',
            'd1',
            _association(kind=GERMLINE_LOF),
            _association(kind=GERMLINE_GOF, validation='5[PMID]_5[PMID]_4[PMID]'),
            _association(kind='Candidate gene tested in'),
            _association(kind=None),
            _association(kind='Modifying germline mutation in', ensg=None),
            _association(kind='unknown type', validation=None, status='Not yet assessed'),
        ),
        _disorder('2', 'd2', _association(kind=GERMLINE, validation='')),
    )
    schema = StructType([
        StructField(name, ArrayType(StringType()) if name == 'literature' else StringType()) for name in raw.columns
    ])
    expected = spark_process_orphanet(spark.createDataFrame(raw.rows(), schema))
    got = process_orphanet(raw.lazy()).collect()
    assert got.columns == expected.columns
    assert got.rows() == [tuple(r) for r in expected.collect()]
