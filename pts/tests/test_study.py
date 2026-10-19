"""Tests for study transformations."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from pts.schemas.study import study_schema
from pts.transformers.study import (
    QualityFlag,
    StudySource,
    collect_heritability,
    deconvolute_studies,
    disease_lut,
    study,
    validate_analysis_flags,
    validate_biosample,
    validate_ccs,
    validate_disease,
    validate_project_id,
    validate_study_type,
    validate_target,
)


class TestDiseaseLut:
    """Canonical and obsolete identifiers retain their lookup semantics."""

    @pytest.mark.parametrize(
        ('obsolete_terms', 'expected_efo'),
        [
            (None, ['EFO_001']),
            ([], ['EFO_001']),
            (['OLD_001', 'OLD_002'], ['EFO_001', 'OLD_001', 'OLD_002']),
            (['OLD_001', 'EFO_001', 'OLD_001'], ['EFO_001', 'OLD_001']),
            ([None, None, 'OLD_001'], ['EFO_001', 'EFO_001', 'OLD_001']),
        ],
        ids=['null-list', 'empty-list', 'obsolete-terms', 'duplicate-terms', 'null-terms'],
    )
    def test_identifier_mapping(self, obsolete_terms, expected_efo):
        """Null terms fall back after deduplication, preserving duplicate output rows."""
        disease = pl.DataFrame(
            {'id': ['EFO_001'], 'obsoleteTerms': [obsolete_terms], 'name': ['Disease']},
            schema={'id': pl.String, 'obsoleteTerms': pl.List(pl.String), 'name': pl.String},
        ).lazy()

        result = disease_lut(disease)

        assert isinstance(result, pl.LazyFrame)
        expected = pl.DataFrame({'diseaseId': ['EFO_001'] * len(expected_efo), 'efo': expected_efo})
        assert_frame_equal(result.collect(), expected, check_row_order=False)

    def test_shared_obsolete_term_preserves_both_mappings(self):
        """An obsolete identifier shared by diseases must retain each mapping."""
        disease = pl.DataFrame({
            'id': ['EFO_001', 'EFO_002'],
            'obsoleteTerms': [['OLD_001'], ['OLD_001']],
        }).lazy()
        expected = pl.DataFrame({
            'diseaseId': ['EFO_001', 'EFO_001', 'EFO_002', 'EFO_002'],
            'efo': ['EFO_001', 'OLD_001', 'EFO_002', 'OLD_001'],
        })

        assert_frame_equal(disease_lut(disease).collect(), expected, check_row_order=False)

    def test_empty_dataset_preserves_output_schema(self):
        """No input records produce a typed empty lookup."""
        disease = pl.LazyFrame(schema={'id': pl.String, 'obsoleteTerms': pl.List(pl.String)})
        expected = pl.DataFrame(schema={'diseaseId': pl.String, 'efo': pl.String})

        assert_frame_equal(disease_lut(disease).collect(), expected)


def studies_frame(rows):
    """Build typed study inputs with absent optional fields filled with nulls."""
    rows = [
        {
            name: [str(item) if isinstance(item, QualityFlag) else item for item in value]
            if isinstance(value, list)
            else value
            for name, value in row.items()
        }
        for row in rows
    ]
    fields = {name for row in rows for name in row}
    return (
        pl
        .DataFrame(rows, schema={name: dtype for name, dtype in study_schema.items() if name in fields})
        .lazy()
        .match_to_schema(study_schema, missing_columns='insert')
    )


@pytest.mark.parametrize(
    ('study_type', 'flagged'),
    [('gwas', False), ('eqtl', False), ('customqtl', False), ('GWAS', True), ('other', True), (None, True)],
)
def test_validate_study_type(study_type, flagged):
    """Only gwas and types ending in qtl are supported, including null rejection."""
    result = validate_study_type(studies_frame([{'studyId': 's', 'studyType': study_type}]))
    assert isinstance(result, pl.LazyFrame)
    assert result.collect()['qualityControls'].to_list() == [[QualityFlag.UNKNOWN_STUDY_TYPE] if flagged else []]


@pytest.mark.parametrize(
    ('project', 'deprecated', 'flagged'),
    [('old', ['old'], True), ('new', ['old'], False), (None, ['old'], False), ('old', [], False)],
)
def test_validate_project_id(project, deprecated, flagged):
    """Deprecated projects add their flag without losing existing flags."""
    result = validate_project_id(
        studies_frame([{'projectId': project, 'qualityControls': ['existing']}]), deprecated
    ).collect()
    assert set(result['qualityControls'][0]) == (
        {'existing', QualityFlag.DEPRECATED_PROJECT} if flagged else {'existing'}
    )


@pytest.mark.parametrize(
    ('study_type', 'gene', 'flagged'),
    [
        ('eqtl', 'known', False),
        ('eqtl', 'unknown', True),
        ('eqtl', None, True),
        ('gwas', None, False),
        (None, None, False),
        ('other', 'unknown', True),
    ],
)
def test_validate_target(study_type, gene, flagged):
    """Unknown targets flag non-GWAS studies while Spark null predicates stay unflagged."""
    result = validate_target(
        studies_frame([{'studyType': study_type, 'geneId': gene}]), pl.DataFrame({'id': ['known']}).lazy()
    ).collect()
    assert result['qualityControls'].to_list() == [[QualityFlag.UNRESOLVED_TARGET] if flagged else []]
    assert result.schema == study_schema


@pytest.mark.parametrize(
    ('study_type', 'biosample', 'flagged'),
    [
        ('eqtl', 'known', False),
        ('eqtl', 'unknown', True),
        ('eqtl', None, True),
        ('gwas', None, False),
        (None, None, False),
    ],
)
def test_validate_biosample(study_type, biosample, flagged):
    """Validated biosample IDs replace previous annotations and flag unresolved QTLs."""
    result = validate_biosample(
        studies_frame([{'studyType': study_type, 'biosampleFromSourceId': biosample, 'biosampleId': 'stale'}]),
        pl.DataFrame({'biosampleId': ['known']}).lazy(),
    ).collect()
    assert result['biosampleId'].to_list() == ['known' if biosample == 'known' else None]
    assert result['qualityControls'].to_list() == [[QualityFlag.UNKNOWN_BIOSAMPLE] if flagged else []]


@pytest.mark.parametrize(
    ('flags', 'flagged'),
    [(None, False), ([], False), (['Case-case study'], True), (['Case-case study', 'other'], True), (['other'], False)],
)
def test_validate_analysis_flags(flags, flagged):
    """Case-case design is flagged once while existing QC flags are preserved."""
    result = validate_analysis_flags(
        studies_frame([{'analysisFlags': flags, 'qualityControls': ['existing']}])
    ).collect()
    assert set(result['qualityControls'][0]) == (
        {'existing', QualityFlag.CASE_CASE_STUDY_DESIGN} if flagged else {'existing'}
    )


@pytest.mark.parametrize(
    ('cases', 'controls', 'samples', 'flag'),
    [
        (10, 20, 30, QualityFlag.CASE_CONTROL_STUDY_DESIGN),
        (10, 20, 40, QualityFlag.CASE_CONTROL_SUM_NEQ_SAMPLE_SIZE),
        (None, None, 30, QualityFlag.MEASUREMENT_STUDY_DESIGN),
        (0, 0, 30, QualityFlag.MEASUREMENT_STUDY_DESIGN),
        (10, None, 30, QualityFlag.ONE_ONLY_CASE_OR_CONTROL),
        (0, 20, 30, QualityFlag.ONE_ONLY_CASE_OR_CONTROL),
        (10, 20, None, QualityFlag.INVALID_SAMPLE_SIZE),
        (None, None, 0, QualityFlag.INVALID_SAMPLE_SIZE),
        (-1, 20, 30, None),
        (10, 20, -1, None),
    ],
)
def test_validate_ccs(cases, controls, samples, flag):
    """Sample-count design branches retain the exact null, zero and negative semantics."""
    result = validate_ccs(studies_frame([{'nCases': cases, 'nControls': controls, 'nSamples': samples}])).collect()
    assert result['qualityControls'].to_list() == [[flag] if flag else []]


def test_validate_disease_normalizes_foreground_and_background_and_replaces_old_annotations():
    """Obsolete IDs map to all matching canonical diseases and unknown IDs disappear."""
    studies = studies_frame([
        {
            'studyId': 's',
            'studyType': 'gwas',
            'traitFromSourceMappedIds': ['old', 'old', 'unknown'],
            'backgroundTraitFromSourceMappedIds': ['background'],
            'diseaseIds': ['stale'],
            'backgroundDiseaseIds': ['stale'],
        }
    ])
    lookup = pl.DataFrame({'efo': ['old', 'old', 'background'], 'diseaseId': ['a', 'b', 'c']}).lazy()
    result = validate_disease(studies, lookup).collect()
    assert set(result['diseaseIds'][0]) == {'a', 'b'}
    assert result['backgroundDiseaseIds'].to_list() == [['c']]
    assert result['qualityControls'].to_list() == [[]]


@pytest.mark.parametrize(
    ('study_type', 'mapped', 'flagged'),
    [('gwas', None, True), ('gwas', [], True), ('gwas', ['unknown'], True), ('eqtl', [], False), (None, [], False)],
)
def test_validate_disease_flags_only_gwas_without_resolved_diseases(study_type, mapped, flagged):
    """Absent or unresolved mappings become typed empty disease lists."""
    result = validate_disease(
        studies_frame([{'studyId': 's', 'studyType': study_type, 'traitFromSourceMappedIds': mapped}]),
        pl.LazyFrame(schema={'efo': pl.String, 'diseaseId': pl.String}),
    ).collect()
    assert result['diseaseIds'].to_list() == [[]]
    assert result['backgroundDiseaseIds'].to_list() == [[]]
    assert result['qualityControls'].to_list() == [[QualityFlag.UNRESOLVED_DISEASE] if flagged else []]


def test_deconvolute_studies_merges_flags_sumstats_and_granular_disease():
    """Duplicate sources propagate QC and select the most granular disease assignment."""
    studies = studies_frame([
        {
            'studyId': 's',
            'studyType': 'gwas',
            'hasSumstats': False,
            'qualityControls': [QualityFlag.NO_OT_CURATION, QualityFlag.SUMSTATS_NOT_AVAILABLE, 'top'],
            'traitFromSourceMappedIds': ['a', 'b'],
            'traitFromSource': 'broad',
            'analysisFlags': ['one'],
        },
        {
            'studyId': 's',
            'studyType': 'gwas',
            'hasSumstats': True,
            'qualityControls': [QualityFlag.NO_OT_CURATION, 'summary'],
            'traitFromSourceMappedIds': ['a'],
            'traitFromSource': 'specific',
            'analysisFlags': ['two'],
            'sumstatQCValues': [{'QCCheckName': 'test', 'QCCheckValue': 1.0}],
        },
    ])
    result = deconvolute_studies(studies)
    assert isinstance(result, pl.LazyFrame)
    row = result.collect().row(0, named=True)
    assert set(row['qualityControls']) == {QualityFlag.NO_OT_CURATION, 'top', 'summary'}
    assert row['hasSumstats'] is True
    assert row['traitFromSourceMappedIds'] == ['a']
    assert row['traitFromSource'] == 'specific'
    assert sorted(row['analysisFlags']) == ['one', 'two']
    assert row['sumstatQCValues'] == [{'QCCheckName': 'test', 'QCCheckValue': 1.0}]


@pytest.mark.parametrize(
    ('sumstats', 'expected'), [([None, None], None), ([False, None], False), ([True, False], True)]
)
def test_deconvolute_studies_propagates_tristate_sumstats(sumstats, expected):
    """Summary statistics availability prefers true, then false, then null."""
    result = deconvolute_studies(
        studies_frame([{'studyId': 's', 'studyType': 'gwas', 'hasSumstats': value} for value in sumstats])
    ).collect()
    assert result.height == 1
    assert result['hasSumstats'].to_list() == [expected]
    assert result['qualityControls'].to_list() == [[]]
    assert result['analysisFlags'].to_list() == [[]]


def test_deconvolute_studies_without_sumstats_removes_curation_flag():
    """Top-hit studies retain unavailable-sumstats flags but drop absent curation."""
    result = deconvolute_studies(
        studies_frame([
            {
                'studyId': 's',
                'studyType': 'gwas',
                'hasSumstats': False,
                'qualityControls': [QualityFlag.NO_OT_CURATION, QualityFlag.SUMSTATS_NOT_AVAILABLE],
            }
        ])
    ).collect()
    assert result['qualityControls'].to_list() == [[QualityFlag.SUMSTATS_NOT_AVAILABLE]]


def test_deconvolute_studies_combines_ambiguous_types_and_keeps_distinct_studies():
    """Different study types remain visible to validation and separate IDs survive."""
    result = deconvolute_studies(
        studies_frame([
            {'studyId': 's', 'studyType': 'gwas'},
            {'studyId': 's', 'studyType': 'eqtl'},
            {'studyId': 'other', 'studyType': 'gwas'},
        ])
    ).collect()
    assert result.height == 2
    assert set(result.filter(pl.col('studyId') == 's')['studyType'][0].split(',')) == {'gwas', 'eqtl'}


def test_quality_flags_are_sorted_and_not_duplicated_when_validation_is_repeated():
    """An already applied flag is preserved once and triggered flags sort the QC list."""
    studies = studies_frame([
        {'studyType': 'other', 'qualityControls': ['z', QualityFlag.UNKNOWN_STUDY_TYPE, 'a', 'a']}
    ])
    result = studies.pipe(validate_study_type).pipe(validate_study_type).collect()
    assert result['qualityControls'].to_list() == [sorted(['a', 'z', str(QualityFlag.UNKNOWN_STUDY_TYPE)])]


def test_deconvolute_studies_null_mapping_takes_precedence_with_legacy_spark_size():
    """Spark's legacy null array size is -1, so a null mapping outranks a populated one."""
    result = deconvolute_studies(
        studies_frame([
            {'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': ['a'], 'traitFromSource': 'populated'},
            {'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': None, 'traitFromSource': 'null mapping'},
        ])
    ).collect()
    assert result['traitFromSourceMappedIds'].to_list() == [None]
    assert result['traitFromSource'].to_list() == ['null mapping']


def heritability_frame(rows):
    """Build LDSC source rows with typed estimates, including null results."""
    return pl.DataFrame(
        rows,
        schema={
            'studyId': pl.String,
            'runStatus': pl.String,
            **dict.fromkeys(('h2', 'h2_se', 'intercept', 'intercept_se', 'mean_chisq', 'lambda_gc'), pl.Float64),
        },
    ).lazy()


@pytest.mark.parametrize('existing', [None, [], [{'QCCheckName': 'existing', 'QCCheckValue': 2.0}]])
def test_collect_heritability_appends_six_ordered_float32_estimates(existing):
    """Successful LDSC results append all six named estimates to the existing QC list."""
    estimates = {'h2': 0.5, 'h2_se': 0.25, 'intercept': 1.0, 'intercept_se': 0.125, 'mean_chisq': 2.0, 'lambda_gc': 1.5}
    result = collect_heritability(
        studies_frame([{'studyId': 's', 'sumstatQCValues': existing}]),
        heritability_frame([{'studyId': 's', 'runStatus': 'success', **estimates}]),
    )
    assert isinstance(result, pl.LazyFrame)
    output = result.collect()
    assert output.schema == study_schema
    expected = (existing or []) + [
        {'QCCheckName': label, 'QCCheckValue': value}
        for label, value in zip(
            [
                'LDSC_SNP-h2',
                'LDSC_SNP-h2_se',
                'LDSC_intercept',
                'LDSC_intercept_se',
                'LDSC_mean_chisq',
                'LDSC_gc_lambda',
            ],
            estimates.values(),
            strict=True,
        )
    ]
    assert output['sumstatQCValues'].to_list() == [expected]


@pytest.mark.parametrize(('status', 'source_id'), [('failed', 's'), (None, 's'), ('success', 'unmatched')])
@pytest.mark.parametrize('existing', [None, [], [{'QCCheckName': 'existing', 'QCCheckValue': 2.0}]])
def test_collect_heritability_preserves_existing_values_without_successful_match(status, source_id, existing):
    """Failed, null-status and unmatched results leave existing QC values unchanged."""
    studies = studies_frame([{'studyId': 's', 'sumstatQCValues': existing}])
    result = collect_heritability(studies, heritability_frame([{'studyId': source_id, 'runStatus': status}]))
    assert_frame_equal(result.collect(), studies.collect())


def test_collect_heritability_preserves_null_estimates_as_named_entries():
    """A successful run with null estimates still appends six named null values."""
    result = collect_heritability(
        studies_frame([{'studyId': 's'}]),
        heritability_frame([{'studyId': 's', 'runStatus': 'success'}]),
    ).collect()
    entries = result['sumstatQCValues'][0].to_list()
    assert len(entries) == 6
    assert all(entry['QCCheckValue'] is None for entry in entries)
    assert entries[0]['QCCheckName'] == 'LDSC_SNP-h2'


def test_collect_heritability_preserves_duplicate_successful_join_rows():
    """Two successful source rows produce two study rows instead of silently deduplicating."""
    result = collect_heritability(
        studies_frame([{'studyId': 's'}]),
        heritability_frame([
            {'studyId': 's', 'runStatus': 'success', 'h2': 0.5},
            {'studyId': 's', 'runStatus': 'success', 'h2': 1.0},
        ]),
    ).collect()
    assert result.height == 2
    assert sorted(row[0]['QCCheckValue'] for row in result['sumstatQCValues'].to_list()) == [0.5, 1.0]


def test_study_source_requires_heritability_path():
    """The configured heritability source must be present before study execution."""
    with pytest.raises(ValidationError, match='heritability'):
        StudySource.model_validate({
            'study': ['study'],
            'target': 'target',
            'disease': 'disease',
            'biosample': 'biosample',
        })


def test_study_writes_one_parquet_file_per_accepted_and_excluded_dataset(tmp_path):
    """Real input scans run the complete pipeline and write correctly separated datasets."""
    input_frames = {
        'study': studies_frame([
            {'studyId': 'valid', 'studyType': 'gwas', 'traitFromSourceMappedIds': ['old'], 'nSamples': 100},
            {'studyId': 'invalid', 'studyType': 'unsupported', 'nSamples': 100},
        ]).collect(),
        'target': pl.DataFrame({'id': ['gene']}),
        'biosample': pl.DataFrame({'biosampleId': ['tissue']}),
        'disease': pl.DataFrame({'id': ['disease'], 'obsoleteTerms': [['old']]}),
        'heritability': heritability_frame([{'studyId': 'valid', 'runStatus': 'success', 'h2': 0.5}]).collect(),
    }
    paths = {}
    for name, frame in input_frames.items():
        path = tmp_path / f'{name}.parquet'
        frame.write_parquet(path)
        paths[name] = str(path)
    paths['study'] = [paths['study']]
    output_path = tmp_path / 'output'
    excluded_path = tmp_path / 'excluded'

    study(
        source=paths,
        destination={'output': str(output_path), 'excluded': str(excluded_path)},
        settings={'invalid_qc_reasons': ['UNKNOWN_STUDY_TYPE']},
        config=Mock(spec=Config),
    )

    output_files = list(output_path.glob('*.parquet'))
    excluded_files = list(excluded_path.glob('*.parquet'))
    assert len(output_files) == len(excluded_files) == 1
    valid = pl.read_parquet(output_files[0])
    invalid = pl.read_parquet(excluded_files[0])
    assert valid['studyId'].to_list() == ['valid']
    assert invalid['studyId'].to_list() == ['invalid']
    assert valid.schema == invalid.schema == study_schema
    assert valid['diseaseIds'].to_list() == [['disease']]
    assert valid['sumstatQCValues'][0][0]['QCCheckValue'] == 0.5
    assert str(QualityFlag.UNKNOWN_STUDY_TYPE) in invalid['qualityControls'][0]


@pytest.mark.parametrize('seed', [1, 7, 42])
def test_deconvolution_is_independent_of_row_order(seed):
    """Explicit source ranks and canonical content break otherwise conflicting ties."""
    rows = (
        studies_frame([
            {
                'studyId': 's',
                'projectId': 'z',
                'traitFromSource': 'z',
                'traitFromSourceMappedIds': ['b'],
                'studyType': 'gwas',
                'qualityControls': ['z', 'a'],
                'analysisFlags': ['z', 'a', 'a'],
            },
            {
                'studyId': 's',
                'projectId': 'a',
                'traitFromSource': 'a',
                'traitFromSourceMappedIds': ['a'],
                'studyType': 'no_licence',
                'qualityControls': ['a', 'z'],
                'analysisFlags': ['a'],
            },
            {
                'studyId': 's',
                'projectId': 'preferred',
                'traitFromSource': 'last source',
                'traitFromSourceMappedIds': ['c'],
                'studyType': 'gwas',
            },
        ])
        .with_columns(pl.Series('_source_rank', [0, 0, 1]))
        .collect()
    )
    expected = deconvolute_studies(rows.lazy()).collect()
    actual = deconvolute_studies(rows.sample(fraction=1, shuffle=True, seed=seed).lazy()).collect()
    assert_frame_equal(actual, expected)
    row = actual.row(0, named=True)
    assert row['projectId'] == 'z'
    assert row['traitFromSource'] == 'last source'
    assert row['traitFromSourceMappedIds'] == ['c']
    assert row['studyType'] == 'gwas,no_licence'
    assert row['qualityControls'] == ['a', 'z']
    assert row['analysisFlags'] == ['a', 'a', 'a', 'z']
    assert '_source_rank' not in actual.columns


def test_collection_ordering_preserves_nulls_and_sample_multiplicity():
    """Canonical ordering changes order only, including nested sample arrays."""
    from pts.transformers.study import sort_study_collections

    samples = [
        {'sampleSize': 20, 'ancestry': 'B'},
        {'sampleSize': 10, 'ancestry': 'A'},
        {'sampleSize': 10, 'ancestry': 'A'},
    ]
    frame = studies_frame([
        {'studyId': 's', 'discoverySamples': samples, 'replicationSamples': None, 'diseaseIds': ['z', 'a']}
    ])
    result = frame.pipe(sort_study_collections).collect().row(0, named=True)
    assert result['discoverySamples'] == [samples[1], samples[2], samples[0]]
    assert result['replicationSamples'] is None
    assert result['diseaseIds'] == ['a', 'z']
