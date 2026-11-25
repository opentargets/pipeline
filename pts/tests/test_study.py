"""Tests for study transformations."""

from unittest.mock import Mock

import polars as pl
import pytest
from loguru import logger
from ot_datamodels.outputs.study import Study
from otter.config.model import Config
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from pts.schemas.dataset.conform import conformed_schema
from pts.schemas.study import study_schema
from pts.transformers.evidence.utils.luts import build_disease_lut
from pts.transformers.study import (
    QualityFlag,
    StudySource,
    collect_heritability,
    read_superseded_studies,
    study,
    validate_analysis_flags,
    validate_biosample,
    validate_ccs,
    validate_disease,
    validate_duplicated_studies,
    validate_project_id,
    validate_study_type,
    validate_superseded_studies,
    validate_target,
)
from tests.study_fixtures import source_schema, studies_frame


def disease_lookup(tmp_path, rows):
    """The study step's lookup: evidence's disease LUT, keyed on `efo`."""
    path = tmp_path / 'disease'
    path.mkdir()
    pl.DataFrame(rows, schema={'id': pl.String, 'obsoleteTerms': pl.List(pl.String)}).write_parquet(
        path / 'part-0.parquet'
    )
    return build_disease_lut(str(path)).lazy().rename({'diseaseFromSourceMappedId': 'efo'})


@pytest.mark.parametrize(
    ('obsolete_terms', 'mapped', 'expected'),
    [
        (None, ['EFO_001'], ['EFO_001']),
        ([], ['EFO_001'], ['EFO_001']),
        (['OLD_001', 'OLD_002'], ['OLD_002'], ['EFO_001']),
        (['OLD_001', 'EFO_001', 'OLD_001'], ['EFO_001', 'OLD_001'], ['EFO_001']),
        ([None, None, 'OLD_001'], ['OLD_001'], ['EFO_001']),
        (['OLD_001'], ['UNKNOWN'], []),
    ],
    ids=['null-list', 'empty-list', 'obsolete-term', 'repeated-own-id', 'null-terms', 'unknown'],
)
def test_validate_disease_resolves_current_and_obsolete_ids(tmp_path, obsolete_terms, mapped, expected):
    """Current and obsolete identifiers resolve to one current disease, without duplicates."""
    lookup = disease_lookup(tmp_path, {'id': ['EFO_001'], 'obsoleteTerms': [obsolete_terms]})
    studies = studies_frame([{'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': mapped}])

    assert validate_disease(studies, lookup).collect()['diseaseIds'].to_list() == [expected]


def test_validate_disease_keeps_every_disease_sharing_an_obsolete_id(tmp_path):
    """An obsolete identifier shared by two diseases maps to both."""
    lookup = disease_lookup(tmp_path, {'id': ['EFO_001', 'EFO_002'], 'obsoleteTerms': [['OLD_001'], ['OLD_001']]})
    studies = studies_frame([{'studyId': 's', 'studyType': 'gwas', 'traitFromSourceMappedIds': ['OLD_001']}])

    assert validate_disease(studies, lookup).collect()['diseaseIds'].to_list() == [['EFO_001', 'EFO_002']]


@pytest.mark.parametrize(
    ('study_type', 'flagged'),
    [
        ('gwas', False),
        ('eqtl', False),
        ('sctuqtl', False),
        ('customqtl', True),
        ('ccqtl', True),
        ('pQTL', True),
        ('GWAS', True),
        ('other', True),
        (None, True),
    ],
)
def test_validate_study_type(study_type, flagged):
    """Only the StudyType values, as in gentropy, are supported, including null rejection."""
    result = validate_study_type(studies_frame([{'studyId': 's', 'studyType': study_type}]))
    assert isinstance(result, pl.LazyFrame)
    assert result.collect()['qualityControls'].to_list() == [[QualityFlag.UNKNOWN_STUDY_TYPE] if flagged else []]


@pytest.mark.parametrize('order', [[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]])
def test_validate_duplicated_studies_keeps_the_first_record_by_content(order):
    """Of the records sharing a study ID, the first by content is kept and the rest flagged, in any row order."""
    rows = [
        {'studyId': 's', 'geneId': 'ENSG2'},
        {'studyId': 's', 'geneId': 'ENSG1'},
        {'studyId': 's', 'geneId': 'ENSG3'},
        {'studyId': 'other', 'geneId': 'ENSG9'},
    ]
    result = validate_duplicated_studies(studies_frame([rows[i] for i in order])).collect()

    assert sorted(result.select('studyId', 'geneId', 'qualityControls').rows()) == [
        ('other', 'ENSG9', []),
        ('s', 'ENSG1', []),
        ('s', 'ENSG2', [QualityFlag.DUPLICATED_STUDY]),
        ('s', 'ENSG3', [QualityFlag.DUPLICATED_STUDY]),
    ]


def test_validate_duplicated_studies_keeps_one_of_identical_records():
    """Identical records are still reduced to one kept record."""
    result = validate_duplicated_studies(studies_frame([{'studyId': 's'}, {'studyId': 's'}])).collect()

    assert sorted(result['qualityControls'].to_list()) == [[], [QualityFlag.DUPLICATED_STUDY]]


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


def supersede(rows, superseded_studies):
    """Run the supersession check over `rows`, returning each study's flags and the warnings it logged."""
    warnings = []
    handler = logger.add(lambda message: warnings.append(message.record['message']), level='WARNING')
    try:
        result = validate_superseded_studies(
            studies_frame(rows).collect(),
            superseded_studies,
            [QualityFlag.UNRESOLVED_DISEASE, QualityFlag.SUPERSEDED_STUDY],
        )
    finally:
        logger.remove(handler)
    return dict(zip(result['studyId'], result['qualityControls'].to_list(), strict=True)), warnings


@pytest.mark.parametrize(
    ('replacement', 'superseded'),
    [
        ({'studyId': 'pgc'}, True),
        ({'studyId': 'pgc', 'qualityControls': [QualityFlag.CASE_CONTROL_STUDY_DESIGN]}, True),
        ({'studyId': 'pgc', 'qualityControls': [QualityFlag.UNRESOLVED_DISEASE]}, False),
        ({'studyId': 'other'}, False),
    ],
    ids=['valid', 'valid-with-a-non-invalidating-flag', 'invalid', 'absent'],
)
def test_validate_superseded_studies_supersedes_a_study_only_when_its_replacement_is_valid(replacement, superseded):
    """A study is superseded only when its replacement is present and carries no invalidating flag."""
    flags, warnings = supersede([{'studyId': 'gcst'}, replacement], {'gcst': 'pgc'})

    assert (QualityFlag.SUPERSEDED_STUDY in flags['gcst']) is superseded
    assert QualityFlag.SUPERSEDED_STUDY not in flags[replacement['studyId']]
    assert bool(warnings) is not superseded


def test_validate_superseded_studies_names_kept_studies_and_the_flags_of_their_replacements():
    """One warning names every kept study, its replacement and why that replacement does not count."""
    _, warnings = supersede(
        [
            {'studyId': 'missing'},
            {'studyId': 'failed'},
            {'studyId': 'pgc_failed', 'qualityControls': [QualityFlag.UNRESOLVED_DISEASE]},
        ],
        {'missing': 'pgc_missing', 'failed': 'pgc_failed'},
    )

    [warning] = warnings
    assert 'missing -> pgc_missing (absent)' in warning
    assert 'failed -> pgc_failed (UNRESOLVED_DISEASE)' in warning


def test_validate_superseded_studies_ignores_studies_absent_from_the_run():
    """A listed study the run does not hold is skipped without a warning, whatever its replacement."""
    flags, warnings = supersede([{'studyId': 'pgc'}], {'gcst': 'pgc', 'other': 'pgc_missing'})

    assert flags == {'pgc': []}
    assert warnings == []


def test_validate_superseded_studies_without_a_mapping_leaves_studies_unchanged():
    """No mapping, no change."""
    studies = studies_frame([{'studyId': 'gcst', 'qualityControls': ['existing']}]).collect()

    assert_frame_equal(validate_superseded_studies(studies, {}, [QualityFlag.SUPERSEDED_STUDY]), studies)


def superseded_studies_file(tmp_path, rows):
    """Write the curation file of superseded studies: a header, then one tab-separated pair per row."""
    path = tmp_path / 'superseded_studies.tsv'
    path.write_text(
        ''.join(f'{study_id}\t{superseded_by}\n' for study_id, superseded_by in [('studyId', 'supersededBy'), *rows])
    )
    return str(path)


def test_read_superseded_studies_maps_each_study_to_the_one_superseding_it(tmp_path):
    """Each row maps a superseded study to the study that supersedes it."""
    path = superseded_studies_file(tmp_path, [('GCST1', 'pgc_a'), ('GCST2', 'pgc_b')])

    assert read_superseded_studies(path) == {'GCST1': 'pgc_a', 'GCST2': 'pgc_b'}


@pytest.mark.parametrize(
    'rows',
    [[('GCST1', '')], [('', 'pgc_a')], [('GCST1', 'pgc_a'), ('GCST1', 'pgc_b')]],
    ids=['missing-replacement', 'missing-study', 'repeated-study'],
)
def test_read_superseded_studies_rejects_incomplete_or_repeated_rows(tmp_path, rows):
    """Every row names both studies, and a study is superseded by one study only."""
    with pytest.raises(ValueError, match='superseded studies'):
        read_superseded_studies(superseded_studies_file(tmp_path, rows))


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
    assert result.schema == source_schema


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
    """Source biosample IDs resolve against the biosample index and unresolved QTLs are flagged."""
    result = validate_biosample(
        studies_frame([{'studyType': study_type, 'biosampleFromSourceId': biosample}]),
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


def test_validate_analysis_flags_turns_missing_flags_into_an_empty_list():
    """A source without analysis flags publishes an empty list, as deconvoluted studies do."""
    result = validate_analysis_flags(studies_frame([{'analysisFlags': None}, {'analysisFlags': ['one']}])).collect()

    assert result['analysisFlags'].to_list() == [[], ['one']]


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


def test_quality_flags_are_sorted_and_not_duplicated_when_validation_is_repeated():
    """An already applied flag is preserved once and triggered flags sort the QC list."""
    studies = studies_frame([
        {'studyType': 'other', 'qualityControls': ['z', QualityFlag.UNKNOWN_STUDY_TYPE, 'a', 'a']}
    ])
    result = studies.pipe(validate_study_type).pipe(validate_study_type).collect()
    assert result['qualityControls'].to_list() == [sorted(['a', 'z', str(QualityFlag.UNKNOWN_STUDY_TYPE)])]


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
        studies_frame([{'studyId': 's', 'sumstatQCValues': existing}], study_schema),
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


@pytest.mark.parametrize('order', [[0.5, 1.0], [1.0, 0.5]], ids=['ascending', 'descending'])
def test_collect_heritability_keeps_one_estimate_per_study(order):
    """Several successful LDSC runs for a study never duplicate it; the choice ignores input order."""
    result = collect_heritability(
        studies_frame([{'studyId': 's'}]),
        heritability_frame([{'studyId': 's', 'runStatus': 'success', 'h2': value} for value in order]),
    ).collect()
    assert result.height == 1
    assert result['sumstatQCValues'].to_list()[0][0]['QCCheckValue'] == 0.5


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
            {'studyId': 'superseded', 'studyType': 'gwas', 'traitFromSourceMappedIds': ['old'], 'nSamples': 100},
            {'studyId': 'repeated', 'studyType': 'gwas', 'traitFromSourceMappedIds': ['old'], 'nSamples': 100},
        ]).collect(),
        'other_study': studies_frame([
            {'studyId': 'repeated', 'studyType': 'gwas', 'traitFromSourceMappedIds': ['old'], 'nSamples': 100},
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
    paths['study'] = [paths['study'], paths.pop('other_study')]
    paths['superseded_studies'] = superseded_studies_file(tmp_path, [('superseded', 'valid')])
    output_path = tmp_path / 'output'
    excluded_path = tmp_path / 'excluded'

    study(
        source=paths,
        destination={'output': str(output_path), 'excluded': str(excluded_path)},
        settings={'invalid_qc_reasons': ['UNKNOWN_STUDY_TYPE', 'DUPLICATED_STUDY', 'SUPERSEDED_STUDY']},
        config=Mock(spec=Config),
    )

    output_files = list(output_path.glob('*.parquet'))
    excluded_files = list(excluded_path.glob('*.parquet'))
    assert len(output_files) == len(excluded_files) == 1
    valid = pl.read_parquet(output_files[0])
    invalid = pl.read_parquet(excluded_files[0])
    assert sorted(valid['studyId']) == ['repeated', 'valid']
    assert sorted(invalid['studyId']) == ['invalid', 'repeated', 'superseded']
    assert invalid.schema == study_schema
    assert valid.schema == conformed_schema(study_schema, Study)
    accepted = valid.filter(pl.col('studyId') == 'valid')
    assert accepted['diseaseIds'].to_list() == [['disease']]
    assert accepted['sumstatQCValues'][0][0]['QCCheckValue'] == 0.5
    flags = {(row['studyId'], flag) for row in invalid.iter_rows(named=True) for flag in row['qualityControls']}
    assert {
        ('invalid', QualityFlag.UNKNOWN_STUDY_TYPE),
        ('superseded', QualityFlag.SUPERSEDED_STUDY),
        ('repeated', QualityFlag.DUPLICATED_STUDY),
    } <= flags


def test_collection_ordering_preserves_nulls_and_sample_multiplicity():
    """Canonical ordering changes order only, including nested sample arrays."""
    from pts.transformers.study import sort_study_collections

    samples = [
        {'sampleSize': 20, 'ancestry': 'B'},
        {'sampleSize': 10, 'ancestry': 'A'},
        {'sampleSize': 10, 'ancestry': 'A'},
    ]
    frame = studies_frame(
        [{'studyId': 's', 'discoverySamples': samples, 'replicationSamples': None, 'diseaseIds': ['z', 'a']}],
        study_schema,
    )
    result = frame.pipe(sort_study_collections).collect().row(0, named=True)
    assert result['discoverySamples'] == [samples[1], samples[2], samples[0]]
    assert result['replicationSamples'] is None
    assert result['diseaseIds'] == ['a', 'z']


def test_validators_accept_inputs_without_derived_columns():
    """Source studies carry no diseaseIds, backgroundDiseaseIds or biosampleId; the validators derive them."""
    studies = studies_frame([
        {
            'studyId': 's',
            'studyType': 'gwas',
            'traitFromSourceMappedIds': ['old'],
            'backgroundTraitFromSourceMappedIds': ['background'],
            'biosampleFromSourceId': 'known',
        }
    ])
    lookup = pl.DataFrame({'efo': ['old', 'background'], 'diseaseId': ['a', 'c']}).lazy()

    result = validate_biosample(validate_disease(studies, lookup), pl.DataFrame({'biosampleId': ['known']}).lazy())

    row = result.collect().row(0, named=True)
    assert (row['diseaseIds'], row['backgroundDiseaseIds'], row['biosampleId']) == (['a'], ['c'], 'known')


def test_validators_replace_derived_columns_carried_by_the_source():
    """A v4 study index carries diseaseIds, backgroundDiseaseIds and biosampleId; the validators replace them."""
    studies = pl.concat(
        [
            studies_frame([
                {
                    'studyId': 'gwas',
                    'studyType': 'gwas',
                    'traitFromSourceMappedIds': ['old'],
                    'backgroundTraitFromSourceMappedIds': ['background'],
                }
            ]),
            pl.LazyFrame(
                {
                    'studyId': ['qtl'],
                    'studyType': ['eqtl'],
                    'biosampleFromSourceId': ['known'],
                    'diseaseIds': [['stale']],
                    'backgroundDiseaseIds': [['stale']],
                    'biosampleId': ['stale'],
                },
            ),
        ],
        how='diagonal_relaxed',
    )
    lookup = pl.DataFrame({'efo': ['old', 'background'], 'diseaseId': ['a', 'c']}).lazy()

    result = validate_biosample(validate_disease(studies, lookup), pl.DataFrame({'biosampleId': ['known']}).lazy())

    rows = {row['studyId']: row for row in result.collect().iter_rows(named=True)}
    assert not [name for name in result.collect_schema().names() if name.endswith('_right')]
    assert (rows['gwas']['diseaseIds'], rows['gwas']['backgroundDiseaseIds'], rows['gwas']['qualityControls']) == (
        ['a'],
        ['c'],
        [],
    )
    assert (rows['qtl']['diseaseIds'], rows['qtl']['biosampleId'], rows['qtl']['qualityControls']) == ([], 'known', [])
