"""Tests for lazy credible set quality control and annotations."""

import polars as pl
import pytest

from pts.schemas.credible_set import credible_set_schema
from pts.transformers import credible_set as cs


def credible_sets(rows):
    """Normalize partial records, including nested locus fields."""
    return pl.DataFrame(
        [{name: row.get(name) for name in credible_set_schema} for row in rows],
        schema=credible_set_schema,
        strict=False,
    ).lazy()


def flags(result):
    """Collect only QC flags, also checking the lazy API."""
    assert isinstance(result, pl.LazyFrame)
    return result.collect()['qualityControls'].to_list()


@pytest.mark.parametrize(
    ('chromosome', 'position', 'expected'),
    [
        ('6', 25726063, True),
        ('6', 33400556, True),
        ('6', 25726062, False),
        ('6', 33400557, False),
        ('1', 30000000, False),
        (None, 30000000, False),
        ('6', None, False),
    ],
)
def test_mhc_inclusive_boundaries(chromosome, position, expected):
    result = flags(cs.qc_mhc_region(credible_sets([{'chromosome': chromosome, 'position': position}])))
    assert (cs.QualityFlag.IN_MHC in result[0]) is expected


@pytest.mark.parametrize(
    ('chromosome', 'expected'),
    [
        ('1', False),
        ('22', False),
        ('X', False),
        ('Y', False),
        ('XY', False),
        ('MT', False),
        ('23', True),
        ('chr1', True),
        ('x', True),
        ('', True),
        (None, False),
    ],
)
def test_chromosome_validation(chromosome, expected):
    result = flags(cs.validate_chromosome_label(credible_sets([{'chromosome': chromosome}])))
    assert (cs.QualityFlag.INVALID_CHROMOSOME in result[0]) is expected


def test_qc_flags_preserve_existing_values_and_do_not_duplicate():
    frame = credible_sets([
        {'chromosome': '6', 'position': 30000000, 'qualityControls': [cs.QualityFlag.TOP_HIT, cs.QualityFlag.IN_MHC]}
    ])
    assert flags(cs.qc_mhc_region(frame)) == [[cs.QualityFlag.IN_MHC, cs.QualityFlag.TOP_HIT]]


@pytest.mark.parametrize(
    ('study_qc', 'expected'),
    [
        ([], [cs.QualityFlag.TOP_HIT, cs.QualityFlag.TOP_HIT_AND_SUMMARY_STATS]),
        (
            ['Harmonized summary statistics are not available or empty'],
            [cs.QualityFlag.TOP_HIT, cs.QualityFlag.FLAGGED_STUDY],
        ),
        (None, [cs.QualityFlag.TOP_HIT]),
    ],
)
def test_study_qc_and_summary_statistics(study_qc, expected):
    study = pl.DataFrame(
        {'studyId': ['s'], 'qualityControls': [study_qc]},
        schema={'studyId': pl.String, 'qualityControls': pl.List(pl.String)},
    ).lazy()
    frame = credible_sets([{'studyId': 's', 'qualityControls': [cs.QualityFlag.TOP_HIT]}])
    assert flags(cs.validate_study(frame, study)) == [sorted(expected)]


def test_missing_study_is_flagged():
    study = pl.LazyFrame(schema={'studyId': pl.String})
    assert flags(cs.validate_study(credible_sets([{'studyId': 'missing'}]), study)) == [[cs.QualityFlag.MISSING_STUDY]]


def test_study_type_replaces_stale_annotation_and_preserves_missing_rows():
    frame = credible_sets([{'studyId': 's', 'studyType': 'gwas'}, {'studyId': 'missing', 'studyType': 'gwas'}])
    study = pl.DataFrame({'studyId': ['s'], 'studyType': ['eqtl']}).lazy()
    result = cs.annotate_study_type(frame, study).collect()
    assert dict(zip(result['studyId'], result['studyType'], strict=True)) == {'s': 'eqtl', 'missing': None}


@pytest.mark.parametrize(
    ('summary_method', 'summary_qc', 'expected'),
    [
        ('PICS', [], True),
        ('SuSie', [], False),
        ('PICS', [cs.QualityFlag.TOP_HIT], False),
        ('PICS', None, False),
    ],
)
def test_pics_top_hit_requires_same_study_summary_statistics(summary_method, summary_qc, expected):
    frame = credible_sets([
        {
            'studyLocusId': 'top',
            'studyId': 's',
            'finemappingMethod': 'PICS',
            'qualityControls': [cs.QualityFlag.TOP_HIT],
        },
        {'studyLocusId': 'summary', 'studyId': 's', 'finemappingMethod': summary_method, 'qualityControls': summary_qc},
        {
            'studyLocusId': 'other',
            'studyId': 'other',
            'finemappingMethod': 'PICS',
            'qualityControls': [cs.QualityFlag.TOP_HIT],
        },
    ])
    result = cs.qc_redundant_top_hits_from_pics(frame).collect()
    by_id = dict(zip(result['studyLocusId'], result['qualityControls'].to_list(), strict=True))
    assert (cs.QualityFlag.REDUNDANT_PICS_TOP_HIT in by_id['top']) is expected
    assert cs.QualityFlag.REDUNDANT_PICS_TOP_HIT not in by_id['other']


@pytest.mark.parametrize(
    ('position', 'study_id', 'chromosome', 'expected'),
    [
        (100, 's', '1', True),
        (200, 's', '1', True),
        (99, 's', '1', False),
        (201, 's', '1', False),
        (150, 'other', '1', False),
        (150, 's', '2', False),
    ],
)
@pytest.mark.parametrize('susie_method', ['SuSie', 'SuSiE-inf'])
def test_susie_explanation_requires_tag_in_same_study_region(position, study_id, chromosome, expected, susie_method):
    frame = credible_sets([
        {
            'studyLocusId': 'susie',
            'studyId': 's',
            'chromosome': '1',
            'finemappingMethod': susie_method,
            'locusStart': 100,
            'locusEnd': 200,
        },
        {
            'studyLocusId': 'pics',
            'studyId': study_id,
            'chromosome': chromosome,
            'finemappingMethod': 'PICS',
            'locus': [{'variantId': f'{chromosome}_{position}_A_C'}],
        },
    ])
    result = cs.qc_explained_by_susie(frame).collect()
    by_id = dict(zip(result['studyLocusId'], result['qualityControls'].to_list(), strict=True))
    assert (cs.QualityFlag.EXPLAINED_BY_SUSIE in by_id['pics']) is expected
    assert cs.QualityFlag.EXPLAINED_BY_SUSIE not in by_id['susie']


@pytest.mark.parametrize(
    ('interval', 'expected'), [('IS95', ['high', 'crossing']), ('IS99', ['high', 'crossing', 'last'])]
)
def test_filter_credible_set_sorts_and_includes_crossing_variant(interval, expected):
    frame = credible_sets([
        {
            'locus': [
                {'variantId': 'last', 'posteriorProbability': 0.04},
                {'variantId': 'crossing', 'posteriorProbability': 0.06},
                {'variantId': 'high', 'posteriorProbability': 0.9},
            ]
        }
    ])
    result = cs.filter_credible_set(frame, getattr(cs.CredibleInterval, interval)).collect()['locus'][0].to_list()
    assert [tag['variantId'] for tag in result] == expected
    assert all(tag['is95CredibleSet'] for tag in result[:2])


@pytest.mark.parametrize('locus', [None, []])
def test_filter_credible_set_empty_locus_becomes_null(locus):
    assert (
        cs.filter_credible_set(credible_sets([{'locus': locus}]), cs.CredibleInterval.IS95).collect()['locus'][0]
        is None
    )


@pytest.mark.parametrize(
    ('pips', 'expected'),
    [
        ([0.95], False),
        ([1.0001], False),
        ([0.949], True),
        ([1.001], True),
        ([], True),
        (None, False),
        ([0.1, None], False),
    ],
)
def test_abnormal_pips_thresholds_and_null_semantics(pips, expected):
    locus = None if pips is None else [{'posteriorProbability': pip} for pip in pips]
    result = flags(cs.qc_abnormal_pips(credible_sets([{'locus': locus}]), 0.95, 1.0001))
    assert (cs.QualityFlag.ABNORMAL_PIPS in result[0]) is expected


@pytest.mark.parametrize(
    ('method', 'qc', 'expected'),
    [
        ('SuSie', [], 'FINEMAPPED_IN_SAMPLE_LD'),
        ('SuSiE-inf', [], 'FINEMAPPED_IN_SAMPLE_LD'),
        ('SuSie', [cs.QualityFlag.OUT_OF_SAMPLE_LD], 'FINEMAPPED_OUT_OF_SAMPLE_LD'),
        ('PICS', [], 'PICSED_SUMMARY_STATS'),
        ('PICS', [cs.QualityFlag.TOP_HIT], 'PICSED_TOP_HIT'),
        ('PICS', None, 'UNKNOWN'),
        ('other', [], 'UNKNOWN'),
        (None, [], 'UNKNOWN'),
    ],
)
def test_confidence_classes(method, qc, expected):
    frame = credible_sets([{'finemappingMethod': method, 'qualityControls': qc}])
    assert (
        cs.assign_confidence(frame).collect()['confidence'][0]
        == getattr(cs.CredibleSetConfidenceClasses, expected).value
    )


@pytest.mark.parametrize(
    ('study_type', 'gene_id', 'variants', 'expected'),
    [
        ('eqtl', 'gene', ['1_110_A_C'], False),
        ('eqtl', 'gene', ['1_111_A_C'], True),
        ('eqtl', 'gene', ['2_100_A_C'], True),
        ('eqtl', 'gene', ['2_100_A_C', '1_100_A_C'], False),
        ('eqtl', 'missing', ['2_100_A_C'], False),
        ('gwas', 'gene', ['2_100_A_C'], None),
        ('eqtl', 'gene', [], None),
        ('eqtl', 'gene', None, None),
    ],
)
def test_trans_qtl_requires_all_tags_trans(study_type, gene_id, variants, expected):
    study = pl.DataFrame({'studyId': ['s'], 'studyType': [study_type], 'geneId': [gene_id], 'projectId': ['p']}).lazy()
    target = pl.DataFrame({'id': ['gene'], 'tss': [100], 'genomicLocation': [{'chromosome': '1'}]}).lazy()
    locus = None if variants is None else [{'variantId': variant} for variant in variants]
    frame = credible_sets([{'studyLocusId': 'cs', 'studyId': 's', 'locus': locus, 'isTransQtl': True}])
    assert cs.flag_trans_qtls(frame, study, target, 10).collect()['isTransQtl'][0] is expected


def test_trans_qtl_without_gene_column_leaves_existing_annotation():
    frame = credible_sets([{'studyLocusId': 'cs', 'isTransQtl': True}])
    study = pl.LazyFrame(schema={'studyId': pl.String, 'studyType': pl.String})
    target = pl.LazyFrame(schema={'id': pl.String})
    assert cs.flag_trans_qtls(frame, study, target).collect()['isTransQtl'][0] is True


def test_filter_credible_set_null_pip_poisons_subsequent_cumulative_sum():
    frame = credible_sets([
        {
            'locus': [
                {'variantId': 'last-null', 'posteriorProbability': None},
                {'variantId': 'first-null', 'posteriorProbability': None},
                {'variantId': 'high', 'posteriorProbability': 0.5},
            ]
        }
    ])
    result = cs.filter_credible_set(frame, cs.CredibleInterval.IS95).collect()['locus'][0].to_list()
    assert len(result) == 2
    assert result[0]['variantId'] == 'high'
    assert result[1]['posteriorProbability'] is None


def test_qc_pipeline_preserves_empty_frame_schema():
    frame = pl.LazyFrame(schema=credible_set_schema)
    study = pl.LazyFrame(
        schema={
            'studyId': pl.String,
            'studyType': pl.String,
            'geneId': pl.String,
            'projectId': pl.String,
            'qualityControls': pl.List(pl.String),
        }
    )
    target = pl.LazyFrame(
        schema={
            'id': pl.String,
            'tss': pl.Int64,
            'genomicLocation': pl.Struct({'chromosome': pl.String}),
        }
    )
    result = (
        frame
        .pipe(cs.qc_mhc_region)
        .pipe(cs.validate_chromosome_label)
        .pipe(cs.validate_study, study)
        .pipe(cs.annotate_study_type, study)
        .pipe(cs.qc_redundant_top_hits_from_pics)
        .pipe(cs.qc_explained_by_susie)
        .pipe(cs.filter_credible_set, cs.CredibleInterval.IS95)
        .pipe(cs.qc_abnormal_pips, 0.95, 1.0001)
        .pipe(cs.assign_confidence)
        .pipe(cs.flag_trans_qtls, study, target)
        .select(credible_set_schema.names())
        .collect()
    )
    assert result.is_empty()
    assert result.schema == credible_set_schema


def test_trans_qtl_null_tag_prevents_all_tags_being_trans():
    study = pl.DataFrame({'studyId': ['s'], 'studyType': ['eqtl'], 'geneId': ['gene'], 'projectId': ['p']}).lazy()
    target = pl.DataFrame({'id': ['gene'], 'tss': [100], 'genomicLocation': [{'chromosome': '1'}]}).lazy()
    frame = credible_sets([
        {
            'studyLocusId': 'cs',
            'studyId': 's',
            'locus': [None, {'variantId': '2_100_A_C'}],
        }
    ])
    assert cs.flag_trans_qtls(frame, study, target, 10).collect()['isTransQtl'][0] is False


@pytest.mark.parametrize(
    ('qc', 'expected'),
    [
        ([None], 'UNKNOWN'),
        ([None, cs.QualityFlag.TOP_HIT], 'PICSED_TOP_HIT'),
    ],
)
def test_confidence_respects_null_elements_in_quality_controls(qc, expected):
    frame = credible_sets([{'finemappingMethod': 'PICS', 'qualityControls': qc}])
    result = cs.assign_confidence(frame).collect()['confidence'][0]
    assert result == getattr(cs.CredibleSetConfidenceClasses, expected).value


def diseases(measurements=(), other=()):
    """Build a disease index filing `measurements` under the measurement therapeutic area."""
    rows = [{'id': term, 'therapeuticAreas': [cs.MEASUREMENT_THERAPEUTIC_AREA]} for term in measurements]
    rows += [{'id': term, 'therapeuticAreas': ['EFO_0000651']} for term in other]
    return pl.DataFrame(rows, schema={'id': pl.String, 'therapeuticAreas': pl.List(pl.String)}).lazy()


def studies(rows):
    """Build typed study annotations used for replication."""
    from pts.schemas.study import study_schema

    return pl.DataFrame(
        [{name: row.get(name) for name in study_schema} for row in rows],
        schema=study_schema,
        strict=False,
    ).lazy()


@pytest.mark.parametrize(('sizes', 'survivor'), [([3, 1, 2], 1), ([None, 0, 1], 0)])
def test_unique_locus_keeps_smallest_set(sizes, survivor):
    frame = credible_sets([
        {
            'studyLocusId': 'duplicate',
            'beta': float(index),
            'locus': None if size is None else [{'variantId': f'1_{tag}_A_C'} for tag in range(size)],
        }
        for index, size in enumerate(sizes)
    ])
    result = cs.validate_unique_study_locus_id(frame)
    assert isinstance(result, pl.LazyFrame)
    rows = result.collect().sort('beta').to_dicts()
    assert [cs.QualityFlag.DUPLICATED_STUDYLOCUS_ID in row['qualityControls'] for row in rows] == [
        index != survivor for index in range(len(sizes))
    ]


def test_unique_locus_ignores_ineligible_rows():
    """Rows already excluded are neither flagged nor counted as duplicates."""
    frame = credible_sets([
        {
            'studyLocusId': 'duplicate',
            'beta': 0.0,
            'locus': [{'variantId': '1_1_A_C'}],
            'qualityControls': ['excluded'],
        },
        {'studyLocusId': 'duplicate', 'beta': 1.0, 'locus': [{'variantId': f'1_{i}_A_C'} for i in range(3)]},
    ])
    eligible = ~pl.col('qualityControls').fill_null([]).list.contains('excluded')
    rows = cs.validate_unique_study_locus_id(frame, eligible).collect().sort('beta')
    assert rows['qualityControls'].to_list() == [['excluded'], []]


def test_unique_locus_does_not_flag_distinct_identifiers():
    frame = credible_sets([{'studyLocusId': 'a'}, {'studyLocusId': 'b'}])
    assert flags(cs.validate_unique_study_locus_id(frame)) == [[], []]


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [
        (['D1', 'D2'], ['D2', 'D1', 'D2'], True),
        (['D1', 'D2'], ['D1'], False),
        ([], [], False),
        (None, None, False),
    ],
)
def test_gwas_replication_matches_entire_disease_set(left, right, expected):
    study = studies([
        {'studyId': 's1', 'studyType': 'gwas', 'diseaseIds': left, 'pubmedId': 'p1'},
        {'studyId': 's2', 'studyType': 'gwas', 'diseaseIds': right, 'pubmedId': 'p2'},
    ])
    frame = credible_sets([
        {'studyLocusId': 'cs1', 'studyId': 's1', 'variantId': '1_100_A_C'},
        {'studyLocusId': 'cs2', 'studyId': 's2', 'variantId': '1_100_A_C'},
    ])
    assert [cs.QualityFlag.REPLICATED in qc for qc in flags(cs.qc_replication(frame, study, diseases()))] == [
        expected,
        expected,
    ]


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [
        (['M1', 'M2'], ['M2'], True),
        (['M1'], ['M2'], False),
        (['M1', 'D1'], ['M2', 'D1'], True),
        (['M1', 'D1', 'D2'], ['D1'], False),
        (['M1', 'D1'], ['M1'], False),
        (['D1', 'D2'], ['D2', 'D1'], True),
        (['UNINDEXED', 'D1'], ['D1'], False),
    ],
    ids=[
        'measurements-share-one',
        'measurements-disjoint',
        'measurements-dropped-diseases-match',
        'measurements-dropped-diseases-differ',
        'diseases-vs-measurement-only',
        'diseases-in-any-order',
        'unindexed-term-is-a-disease',
    ],
)
def test_gwas_replication_matches_on_the_kind_of_terms(left, right, expected):
    """Measurements replicate one at a time; diseases replicate as a whole list."""
    study = studies([
        {'studyId': 's1', 'studyType': 'gwas', 'diseaseIds': left, 'pubmedId': 'p1'},
        {'studyId': 's2', 'studyType': 'gwas', 'diseaseIds': right, 'pubmedId': 'p2'},
    ])
    frame = credible_sets([
        {'studyLocusId': 'cs1', 'studyId': 's1', 'variantId': '1_100_A_C'},
        {'studyLocusId': 'cs2', 'studyId': 's2', 'variantId': '1_100_A_C'},
    ])
    disease = diseases(measurements=['M1', 'M2'], other=['D1', 'D2'])
    result = flags(cs.qc_replication(frame, study, disease))
    assert [cs.QualityFlag.REPLICATED in qc for qc in result] == [expected, expected]


@pytest.mark.parametrize(
    ('difference', 'expected'),
    [
        ({}, False),
        ({'cohorts': ['different']}, True),
        ({'pubmedId': 'different'}, True),
        ({'ldPopulationStructure': [{'ldPopulation': 'afr', 'relativeSampleSize': 1.0}]}, True),
    ],
)
def test_gwas_replication_collapses_identical_evidence(difference, expected):
    annotation = {
        'studyType': 'gwas',
        'diseaseIds': ['D'],
        'cohorts': ['cohort'],
        'pubmedId': 'pubmed',
        'ldPopulationStructure': [{'ldPopulation': 'eur', 'relativeSampleSize': 1.0}],
    }
    study = studies([
        {'studyId': 's1', **annotation},
        {'studyId': 's2', **annotation, **difference},
    ])
    frame = credible_sets([
        {'studyLocusId': 'cs1', 'studyId': 's1', 'variantId': '1_100_A_C'},
        {'studyLocusId': 'cs2', 'studyId': 's2', 'variantId': '1_100_A_C'},
    ])
    assert [cs.QualityFlag.REPLICATED in qc for qc in flags(cs.qc_replication(frame, study, diseases()))] == [
        expected,
        expected,
    ]


@pytest.mark.parametrize(
    ('gene_left', 'gene_right', 'expected'),
    [
        ('gene', 'gene', True),
        ('gene', 'other', False),
        (None, None, False),
    ],
)
def test_molqtl_replication_matches_gene_across_studies(gene_left, gene_right, expected):
    study = studies([
        {'studyId': 's1', 'studyType': 'eqtl', 'geneId': gene_left},
        {'studyId': 's2', 'studyType': 'pqtl', 'geneId': gene_right},
    ])
    frame = credible_sets([
        {'studyLocusId': 'cs1', 'studyId': 's1', 'variantId': '1_100_A_C'},
        {'studyLocusId': 'cs2', 'studyId': 's2', 'variantId': '1_100_A_C'},
    ])
    assert [cs.QualityFlag.REPLICATED in qc for qc in flags(cs.qc_replication(frame, study, diseases()))] == [
        expected,
        expected,
    ]


def test_replication_null_variants_never_match():
    study = studies([
        {'studyId': 's1', 'studyType': 'eqtl', 'geneId': 'gene'},
        {'studyId': 's2', 'studyType': 'eqtl', 'geneId': 'gene'},
    ])
    frame = credible_sets([{'studyLocusId': 'cs1', 'studyId': 's1'}, {'studyLocusId': 'cs2', 'studyId': 's2'}])
    assert flags(cs.qc_replication(frame, study, diseases())) == [[], []]


def test_credible_set_step_filters_before_deduplication_and_replication(tmp_path):
    """Real scans and writes exercise invalid top-hit removal and survivor-only replication."""
    from unittest.mock import Mock

    from otter.config.model import Config

    study_rows = [
        {'studyId': sid, 'studyType': 'gwas', 'diseaseIds': ['D'], 'pubmedId': sid, 'qualityControls': []}
        for sid in ['s1', 's2', 's3']
    ]
    top_hit = {
        'studyLocusId': 'same',
        'studyId': 's1',
        'variantId': '1_100_A_C',
        'chromosome': '1',
        'position': 100,
        'finemappingMethod': 'PICS',
        'qualityControls': [cs.QualityFlag.TOP_HIT],
        'locus': [{'variantId': '1_100_A_C', 'posteriorProbability': 1.0}],
    }
    valid = {
        **top_hit,
        'qualityControls': [],
        'beta': 1.0,
        'locus': [
            {'variantId': '1_100_A_C', 'posteriorProbability': 0.6},
            {'variantId': '1_101_A_C', 'posteriorProbability': 0.4},
        ],
    }
    duplicate = {**valid, 'beta': 2.0}
    other_study = {**valid, 'studyLocusId': 'other', 'studyId': 's2'}
    singleton = {**valid, 'studyLocusId': 'alone', 'studyId': 's3', 'variantId': '1_200_A_C'}
    invalid_singleton_partner = {
        **top_hit,
        'studyLocusId': 'invalid-partner',
        'studyId': 's2',
        'variantId': '1_200_A_C',
    }
    frames = {
        'study': studies(study_rows).collect(),
        'target': pl.DataFrame({'id': ['gene'], 'tss': [100], 'genomicLocation': [{'chromosome': '1'}]}),
        'disease': diseases().collect(),
        'credible_set': credible_sets([
            top_hit,
            valid,
            duplicate,
            other_study,
            singleton,
            invalid_singleton_partner,
        ]).collect(),
    }
    paths = {}
    for name, frame in frames.items():
        path = tmp_path / f'{name}.parquet'
        frame.write_parquet(path)
        paths[name] = [str(path)] if name == 'credible_set' else str(path)
    output = tmp_path / 'output'
    excluded = tmp_path / 'excluded'
    cs.credible_set(
        source=paths,
        destination={'output': str(output), 'excluded': str(excluded)},
        settings={
            'trans_qtl_threshold': 5000000,
            'invalid_qc_reasons': ['TOP_HIT_AND_SUMMARY_STATS', 'DUPLICATED_STUDYLOCUS_ID'],
        },
        config=Mock(spec=Config),
    )
    accepted = pl.read_parquet(list(output.glob('*.parquet')))
    rejected = pl.read_parquet(list(excluded.glob('*.parquet')))
    assert accepted.schema == rejected.schema == credible_set_schema
    assert accepted.height == rejected.height == 3
    by_id = {row['studyLocusId']: row for row in accepted.to_dicts()}
    assert set(by_id) == {'same', 'other', 'alone'}
    assert by_id['same']['beta'] in {1.0, 2.0}
    same_rows = pl.concat([accepted, rejected]).filter(
        (pl.col('studyLocusId') == 'same') & ~pl.col('qualityControls').list.contains(cs.QualityFlag.TOP_HIT.value)
    )
    assert sorted(same_rows['beta'].to_list()) == [1.0, 2.0]
    assert cs.QualityFlag.REPLICATED in by_id['same']['qualityControls']
    assert cs.QualityFlag.REPLICATED in by_id['other']['qualityControls']
    assert cs.QualityFlag.REPLICATED not in by_id['alone']['qualityControls']
    assert sum(cs.QualityFlag.DUPLICATED_STUDYLOCUS_ID in qc for qc in rejected['qualityControls'].to_list()) == 1
    assert all(cs.QualityFlag.REPLICATED not in qc for qc in rejected['qualityControls'].to_list())


def test_unique_locus_equal_size_tie_keeps_same_content_when_input_reversed():
    rows = [{'studyLocusId': 'same', 'beta': beta, 'locus': [{'variantId': '1_100_A_C'}]} for beta in [1.0, 2.0, 3.0]]
    survivors = []
    for inputs in [rows, list(reversed(rows))]:
        result = cs.validate_unique_study_locus_id(credible_sets(inputs)).collect()
        kept = result.filter(~pl.col('qualityControls').list.contains(cs.QualityFlag.DUPLICATED_STUDYLOCUS_ID.value))
        assert kept.height == 1
        assert result.height == 3
        survivors.append(kept['beta'][0])
    assert survivors[0] == survivors[1]
