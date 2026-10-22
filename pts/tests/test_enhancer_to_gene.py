"""Tests for enhancer-to-gene ingestion and annotation."""

import gzip
import hashlib
from typing import Any

import polars as pl
import pytest

from pts.schemas.enhancer_to_gene import enhancer_to_gene_schema
from pts.transformers.enhancer_to_gene import parse_enhancer_to_gene, read_enhancer_to_gene


def raw_frame(**changes):
    """Provide one unified raw evidence record."""
    row = {
        'chromosome': 'chr1',
        'start': '100',
        'end': '200',
        'geneId': 'ENSG1',
        'biosampleName': 'cell',
        'score': '0.75',
        'intervalType': ' Enhancer ',
        'dnase': '1.5',
        'hicContacts': '2.25',
        'filePath': '/data/study.bed.gz',
    }
    row.update(changes)
    return pl.DataFrame([row], schema=dict.fromkeys(row, pl.String)).lazy()


def targets(tss=150, duplicate=False):
    """Create targets with the full lookup input schema."""
    return pl.DataFrame(
        [{'id': 'ENSG1', 'tss': tss, 'canonicalTranscript': None, 'genomicLocation': None}] * (2 if duplicate else 1),
        schema={
            'id': pl.String,
            'tss': pl.Int64,
            'canonicalTranscript': pl.Struct({'strand': pl.String, 'start': pl.Int64, 'end': pl.Int64}),
            'genomicLocation': pl.Struct({'strand': pl.Int64, 'start': pl.Int64, 'end': pl.Int64}),
        },
    ).lazy()


def mapping(duplicate=False):
    """Create a biosample mapping."""
    return pl.DataFrame({
        'biosampleName': ['cell'] * (2 if duplicate else 1),
        'biosampleId': ['CL_1'] * (2 if duplicate else 1),
    }).lazy()


def test_parse_output_contract():
    """Parsing produces normalized values and the exact nested output schema."""
    result = parse_enhancer_to_gene(raw_frame(), mapping(), targets())
    assert isinstance(result, pl.LazyFrame)
    assert result.collect_schema() == enhancer_to_gene_schema
    row = result.collect().row(0, named=True)
    assert row == {
        'chromosome': '1',
        'start': 100,
        'end': 200,
        'geneId': 'ENSG1',
        'score': 0.75,
        'distanceToTss': 0,
        'resourceScore': [{'name': 'DNase', 'value': 1.5}, {'name': 'HiC_contacts', 'value': 2.25}],
        'datasourceId': 'E2G',
        'intervalType': 'enhancer',
        'pmid': '38014075',
        'biosampleName': 'cell',
        'biosampleFromSourceId': None,
        'biosampleId': 'CL_1',
        'studyId': 'study',
        'intervalId': hashlib.md5(b'1|100|200|ENSG1|study|enhancer').hexdigest(),
        'qualityControls': [],
    }


@pytest.mark.parametrize(
    ('start', 'score', 'expected_start', 'expected_score'),
    [(' 100 ', ' 0.75 ', 100, 0.75), ('100.9', '0.75', 100, 0.75), ('abc', 'x', None, None)],
    ids=['padded', 'fraction', 'invalid'],
)
def test_parse_casts_numbers_like_spark(start, score, expected_start, expected_score):
    """Padding is trimmed and fractions truncated, as Gentropy's Spark casts did; bad text is null."""
    row = parse_enhancer_to_gene(raw_frame(start=start, score=score), mapping(), targets()).collect().row(0, named=True)
    assert (row['start'], row['score']) == (expected_start, expected_score)


@pytest.mark.parametrize(
    ('tss', 'interval_type', 'distance'),
    [
        (100, 'enhancer', 0),
        (200, 'enhancer', 0),
        (90, 'enhancer', 10),
        (250, 'enhancer', 50),
        (None, 'enhancer', None),
        (None, 'promoter', 0),
        (900, 'promoter', 0),
    ],
)
def test_distance_to_tss(tss, interval_type, distance):
    """Interval bounds are inclusive and promoter distances always zero."""
    row = (
        parse_enhancer_to_gene(raw_frame(intervalType=interval_type), mapping(), targets(tss))
        .collect()
        .row(0, named=True)
    )
    assert row['distanceToTss'] == distance


def test_interval_ids_do_not_collide_across_column_boundaries():
    """chr1:12-345 and chr11:2-345 concatenate alike without a separator but must differ."""
    ids = [
        parse_enhancer_to_gene(raw_frame(chromosome=chromosome, start=start, end='345'), mapping(), targets())
        .collect()
        .row(0, named=True)['intervalId']
        for chromosome, start in [('chr1', '12'), ('chr11', '2')]
    ]
    assert ids[0] != ids[1]


def test_missing_lookups_and_null_identifier_component():
    """Unknown identifiers retain evidence with null annotations."""
    row = (
        parse_enhancer_to_gene(raw_frame(geneId=None, biosampleName='unknown'), mapping(), targets())
        .collect()
        .row(0, named=True)
    )
    assert row['distanceToTss'] is None
    assert row['biosampleId'] is None
    assert row['intervalId'] == hashlib.md5(b'1|100|200|None|study|enhancer').hexdigest()


def test_duplicate_lookup_rows_preserve_join_multiplicity():
    """Repeated mapping records must not silently deduplicate evidence."""
    result = parse_enhancer_to_gene(raw_frame(), mapping(duplicate=True), targets(duplicate=True)).collect()
    assert result.height == 4


def test_reader_unifies_mixed_headers_from_gzip_glob(tmp_path):
    """Both upstream chromosome headers become the same string columns."""
    for index, chr_header in enumerate(['#chr', 'chr']):
        with gzip.open(tmp_path / f'study{index}.bed.gz', 'wt') as handle:
            handle.write(
                f'{chr_header}\tstart\tend\tTargetGeneEnsemblID\tCellType\tScore\tclass\tnormalizedDNase_prom.Feature\t3DContact.Feature\textra\n'
            )
            handle.write('chr1\t100\t200\tENSG1\tcell\t0.75\tenhancer\t1.5\t2.25\tdiscard\n')
    result = read_enhancer_to_gene(str(tmp_path / '*.bed.gz'))
    assert isinstance(result, pl.LazyFrame)
    frame = result.collect()
    assert frame.height == 2
    assert set(frame.columns) == set(raw_frame().collect_schema().names())
    assert all(dtype == pl.String for dtype in frame.schema.values())
    assert frame['chromosome'].to_list() == ['chr1', 'chr1']
    assert {path.rsplit('/', 1)[-1] for path in frame['filePath']} == {'study0.bed.gz', 'study1.bed.gz'}


def test_reader_rejects_empty_glob(tmp_path):
    """No matching evidence files must fail explicitly."""
    with pytest.raises((ValueError, FileNotFoundError)):
        read_enhancer_to_gene(str(tmp_path / '*.bed.gz'))


def evidence_rows(*changes):
    """Create normalized evidence, with independent identifiers by default."""
    base = parse_enhancer_to_gene(raw_frame(), mapping(), targets()).collect().row(0, named=True)
    return pl.DataFrame(
        [{**base, 'intervalId': f'interval{index}', **change} for index, change in enumerate(changes)],
        schema=enhancer_to_gene_schema,
    ).lazy()


def qc_target(*chromosomes):
    """Create genomic chromosome lookups."""
    return pl.DataFrame(
        {
            'id': ['ENSG1'] * len(chromosomes),
            'genomicLocation': [{'chromosome': chromosome} for chromosome in chromosomes],
        },
        schema={'id': pl.String, 'genomicLocation': pl.Struct({'chromosome': pl.String})},
    ).lazy()


def qc_contig():
    """Include canonical, noncanonical and missing-bound chromosomes."""
    return pl.DataFrame(
        {'id': ['1', 'X', 'Y', 'MT', 'XY', '2'], 'start': [0, 0, 0, 0, 0, None], 'end': [300] * 6},
        schema={'id': pl.String, 'start': pl.Int64, 'end': pl.Int64},
    ).lazy()


@pytest.mark.parametrize(
    ('method', 'changes', 'expected'),
    [
        (
            'validate_score',
            [{'score': value} for value in [0.6, 1.0, 0.59, 1.01, None]],
            [[], [], ['SCORE_OUTSIDE_BOUNDS'], ['SCORE_OUTSIDE_BOUNDS'], ['SCORE_OUTSIDE_BOUNDS']],
        ),
        (
            'validate_interval_range',
            [{'chromosome': value} for value in ['1', 'X', 'Y', 'MT', 'XY', '2', None]]
            + [{'start': -1}, {'end': 301}, {'start': 250, 'end': 100}, {'start': None}],
            [
                [],
                [],
                [],
                [],
                ['INVALID_CHROMOSOME'],
                ['INVALID_CHROMOSOME'],
                ['INVALID_CHROMOSOME'],
                ['INVALID_RANGE'],
                ['INVALID_RANGE'],
                [],
                [],
            ],
        ),
        (
            'validate_target',
            [{'geneId': 'unknown'}, {'geneId': None}, {'chromosome': '2'}, {'chromosome': None}, {}],
            [['UNRESOLVED_TARGET'], ['UNRESOLVED_TARGET'], ['TRANS_CHROMOSOMAL_EFFECT'], [], []],
        ),
        (
            'validate_biosample',
            [{'biosampleId': value} for value in ['CL_1', 'unknown', None]],
            [[], ['UNKNOWN_BIOSAMPLE'], ['UNKNOWN_BIOSAMPLE']],
        ),
    ],
)
def test_qc_flags_and_idempotence(method, changes, expected):
    """Each validator remains lazy, preserves rows and appends its flag only once."""
    # Import the module explicitly because the package may expose its entry point.
    from importlib import import_module

    module = import_module('pts.transformers.enhancer_to_gene')
    function = getattr(module, method)
    args = {
        'validate_score': (0.6, 1.0),
        'validate_interval_range': (qc_contig(),),
        'validate_target': (qc_target('1'),),
        'validate_biosample': (pl.DataFrame({'biosampleId': ['CL_1']}).lazy(),),
    }.get(method, ())
    result = function(evidence_rows(*changes), *args)
    assert isinstance(result, pl.LazyFrame)
    assert result.collect_schema() == enhancer_to_gene_schema
    frame = result.collect().sort('intervalId')
    assert frame.height == len(changes)
    # Sort numerically for the larger range fixture.
    actual = {row['intervalId']: row['qualityControls'] for row in frame.to_dicts()}
    assert actual == {
        f'interval{index}': [module.QualityFlag[name].value for name in flags] for index, flags in enumerate(expected)
    }
    repeated = function(result, *args).collect()
    assert {row['intervalId']: row['qualityControls'] for row in repeated.to_dicts()} == actual


def test_qc_creates_missing_quality_controls():
    """A raw normalized frame may not contain the QC array yet."""
    from pts.transformers.enhancer_to_gene import QualityFlag, validate_score

    result = validate_score(evidence_rows({'score': 2.0}).drop('qualityControls'), 0.6, 1.0).collect()
    assert result['qualityControls'].to_list() == [[QualityFlag.SCORE_OUTSIDE_BOUNDS.value]]


def test_interval_type_ambiguity_ignores_null_and_groups_null_keys():
    """Only distinct nonnull types count, including groups with missing genes."""
    from pts.transformers.enhancer_to_gene import QualityFlag, validate_interval_type

    changes: list[dict[str, Any]] = [
        {'geneId': None, 'intervalType': value} for value in ['promoter', 'enhancer', None]
    ]
    changes += [
        {'start': 300 + index, 'intervalType': value}
        for index, value in enumerate(['intragenic', 'intergenic', 'genic', 'invalid', '', None])
    ]
    result = validate_interval_type(evidence_rows(*changes)).collect().sort('intervalId')
    flags = result['qualityControls'].to_list()
    assert flags[:3] == [[QualityFlag.AMBIGUOUS_INTERVAL_TYPE.value]] * 3
    assert flags[3:] == [
        [],
        [],
        [],
        [QualityFlag.UNKNOWN_INTERVAL_TYPE.value],
        [QualityFlag.UNKNOWN_INTERVAL_TYPE.value],
        [],
    ]


def test_score_ambiguity_ignores_null_and_separates_biosamples():
    """Distinct scores share the full source identity even when keys are null."""
    from pts.transformers.enhancer_to_gene import QualityFlag, validate_id_has_unique_score

    changes: list[dict[str, Any]] = [{'geneId': None, 'score': value} for value in [0.7, 0.9, None]]
    changes += [{'biosampleId': 'other', 'score': value} for value in [0.7, 0.7, None]]
    result = validate_id_has_unique_score(evidence_rows(*changes)).collect().sort('intervalId')
    assert result['qualityControls'].to_list() == [[QualityFlag.AMBIGUOUS_SCORE.value]] * 3 + [[], [], []]


def test_target_null_chromosome_and_duplicate_lookups():
    """Unknown target chromosomes do not produce trans flags, duplicates are retained."""
    from pts.transformers.enhancer_to_gene import validate_biosample, validate_interval_range, validate_target

    assert validate_target(evidence_rows({}), qc_target(None)).collect()['qualityControls'].to_list() == [[]]
    assert validate_target(evidence_rows({}), qc_target('1', '1')).collect().height == 2
    biosample = pl.DataFrame({'biosampleId': ['CL_1', 'CL_1']}).lazy()
    assert validate_biosample(evidence_rows({}), biosample).collect().height == 2
    contig = pl.DataFrame({'id': ['1', '1'], 'start': [0, 0], 'end': [300, 300]}).lazy()
    assert validate_interval_range(evidence_rows({}), contig).collect().height == 2


def test_composed_qc_pipeline_and_valid_rows():
    """QC composition separates configured failures while retaining other flags."""
    from pts.transformers.enhancer_to_gene import (
        QualityFlag,
        validate_biosample,
        validate_id_has_unique_score,
        validate_interval_range,
        validate_interval_type,
        validate_score,
        validate_target,
    )
    from pts.transformers.utils.quality_flags import valid_rows

    result = (
        evidence_rows({}, {'score': None, 'start': 201}, {'geneId': 'unknown', 'start': 202})
        .pipe(validate_interval_range, qc_contig())
        .pipe(validate_target, qc_target('1'))
        .pipe(validate_biosample, pl.DataFrame({'biosampleId': ['CL_1']}).lazy())
        .pipe(validate_interval_type)
        .pipe(validate_score, 0.6, 1.0)
        .pipe(validate_id_has_unique_score)
        .pipe(valid_rows, [QualityFlag.SCORE_OUTSIDE_BOUNDS])
    )
    assert set(result.valid.collect()['intervalId']) == {'interval0', 'interval2'}
    assert result.invalid.collect()['intervalId'].to_list() == ['interval1']




@pytest.mark.parametrize(('chromosome', 'expected'), [('1', False), ('2', True)])
def test_target_qc_uses_genomic_chromosome_when_transcript_disagrees(chromosome, expected):
    from pts.transformers.enhancer_to_gene import QualityFlag, validate_target

    target = qc_target(chromosome).with_columns(
        pl.struct(pl.lit('2' if chromosome == '1' else '1').alias('chromosome')).alias('canonicalTranscript')
    )
    result = validate_target(evidence_rows({}), target).collect()
    assert (QualityFlag.TRANS_CHROMOSOMAL_EFFECT in result['qualityControls'][0]) == expected
