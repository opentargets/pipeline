"""Credible set QC runs per source; validation, with the cross-set QC, and annotation run over every source."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from otter.config.yaml import parse_yaml

from pts.schemas.credible_set import FinemappingMethod
from pts.tasks.transform import Transform
from pts.transformers import credible_set
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.set_qc import SetQcSettings
from pts.transformers.credible_set.validation import ValidationSettings
from pts.transformers.credible_set.variant_qc import VariantQcSettings

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

SOURCES = {
    'gcca': 'gwas',
    'gcss_pics': 'gwas',
    'gcss_susie': 'gwas',
    'finngen_gwas': 'gwas',
    'eqtl_catalogue': 'molqtl',
    'ukb_ppp_eur': 'molqtl',
    'finngen_multiome': 'molqtl',
}
"""The platform credible set sources, named after their PIS folders, with their kind."""


def _substitute(value: Any, each: str) -> Any:
    if isinstance(value, str):
        return value.replace('${each}', each)
    if isinstance(value, list):
        return [_substitute(item, each) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, each) for key, item in value.items()}
    return value


def _steps() -> list[dict[str, Any]]:
    return parse_yaml(CONFIG_PATH)['steps']['credible_set']


def _tasks() -> dict[str, dict[str, Any]]:
    """The tasks of the credible set step, with every explode expanded as otter expands it."""
    tasks = {}
    for task in _steps():
        tasks[task['name']] = task
        for each in task.get('foreach', []):
            for spec in task['do']:
                expanded = _substitute(spec, each)
                tasks[expanded['name']] = expanded
    return tasks


def _explode() -> list[str]:
    """The source folders below `input/` (`<kind>/<source>`) that the QC explode iterates over."""
    [task] = [task for task in _steps() if task['name'] == 'explode qc credible sets']
    return task['foreach']


def test_the_qc_explode_covers_every_platform_source_once() -> None:
    assert sorted(_explode()) == sorted(f'{kind}/{source}' for source, kind in SOURCES.items())
    assert len(_explode()) == len(SOURCES)


def test_each_source_is_checked_against_its_own_dataset_and_mirrors_it() -> None:
    """QC reads one dataset of one source; the checks that pair sets with variants run in validation."""
    tasks = _tasks()
    for source, kind in SOURCES.items():
        folder = f'{kind}/{source}'
        set_qc = tasks[f'transform qc credible sets {folder}']
        variant_qc = tasks[f'transform qc credible set variants {folder}']
        assert set_qc['transformer'] == 'credible_set.set_qc'
        assert set_qc['source'] == {
            'credible_set': f'input/{folder}/credible_set',
            'study': 'output/study',
        }
        assert set_qc['destination'] == f'intermediate/qc/{folder}/credible_set'
        assert variant_qc['transformer'] == 'credible_set.variant_qc'
        assert variant_qc['source'] == {
            'credible_set_variant': f'input/{folder}/credible_set_variant',
        }
        assert variant_qc['destination'] == f'intermediate/qc/{folder}/credible_set_variant'


def test_validation_reads_every_source_in_explode_order_and_writes_single_datasets() -> None:
    validation = _tasks()['transform validate credible sets']
    assert validation['requires'] == ['explode qc credible sets']
    assert validation['source'] == {
        'credible_set': [f'intermediate/qc/{folder}/credible_set' for folder in _explode()],
        'credible_set_variant': [f'intermediate/qc/{folder}/credible_set_variant' for folder in _explode()],
    }
    assert validation['destination'] == {
        'valid_set': 'intermediate/validated/credible_set',
        'valid_variant': 'intermediate/validated/credible_set_variant',
        'excluded_set': 'excluded/credible_set',
        'excluded_variant': 'excluded/credible_set_variant',
    }


def test_annotation_runs_in_its_own_step_after_the_variant_index() -> None:
    """Annotation reads `output/variant`, whose index reads the validated variants, so it is a separate step."""
    steps = parse_yaml(CONFIG_PATH)['steps']
    assert 'transform annotate credible sets' not in _tasks()
    [annotation] = steps['credible_set_annotation']
    assert annotation['name'] == 'transform annotate credible sets'
    assert annotation['transformer'] == 'credible_set.annotation'
    assert 'requires' not in annotation
    assert annotation['source'] == {
        'credible_set': 'intermediate/validated/credible_set',
        'credible_set_variant': 'intermediate/validated/credible_set_variant',
        'study': 'output/study',
        'variant': 'output/variant',
        'target': 'output/target',
        'disease': 'output/disease',
    }
    assert annotation['settings'] == {'trans_qtl_threshold': 5_000_000}
    assert annotation['destination'] == {
        'credible_set': 'output/credible_set',
        'credible_set_variant': 'output/credible_set_variant',
    }


def _validation_settings() -> ValidationSettings:
    return ValidationSettings.model_validate(_tasks()['transform validate credible sets']['settings'])


def test_validation_excludes_the_agreed_flags() -> None:
    """Every set-excluding flag of the old step and the PRD, except those gentropy removed.

    ABNORMAL_PIP stays metadata: the old step flagged abnormal PIP sums without excluding them.
    """
    settings = _validation_settings()
    assert set(settings.cs_invalid_qc_reasons) == {
        CredibleSetFlag.INVALID_CS_ID,
        CredibleSetFlag.IN_MHC,
        CredibleSetFlag.INVALID_CHROMOSOME,
        CredibleSetFlag.LEAD_CHROMOSOME_MISMATCH,
        CredibleSetFlag.TOP_HIT_AND_SUMMARY_STATS,
        CredibleSetFlag.MISSING_STUDY,
        CredibleSetFlag.REDUNDANT_PICS_TOP_HIT,
        CredibleSetFlag.DUPLICATED_CREDIBLE_SET_ID_ACROSS_SOURCES,
        CredibleSetFlag.INVALID_FINEMAPPING_METHOD,
        CredibleSetFlag.EXPLAINED_BY_SUSIE,
        CredibleSetFlag.COMPOSITE_FLAG,
        CredibleSetFlag.INCONSISTENCY_FLAG,
        CredibleSetFlag.SUBSIGNIFICANT_FLAG,
        CredibleSetFlag.LD_CLUMPED,
        CredibleSetFlag.WINDOW_CLUMPED,
        CredibleSetFlag.STUDY_TYPE_MISMATCH,
        CredibleSetFlag.LEAD_VARIANT_MISSING,
    }
    assert set(settings.csv_set_invalid_qc_reasons) == {
        CredibleSetVariantFlag.INVALID_CS_ID,
        CredibleSetVariantFlag.INVALID_VARIANT_ID,
        CredibleSetVariantFlag.DUPLICATED_VARIANT,
    }
    assert set(settings.csv_row_invalid_qc_reasons) == {
        CredibleSetVariantFlag.EXCEEDS_COVERAGE_95,
        CredibleSetVariantFlag.ORPHAN_VARIANT,
    }


def test_configured_validation_excludes_window_clumped_and_keeps_abnormal_pip(validate_source) -> None:
    settings = _validation_settings()
    sets = pl.LazyFrame(
        {
            'id': ['clumped', 'abnormal'],
            'studyId': ['s', 's'],
            'chromosome': ['1', '1'],
            'locusStart': [100, 200],
            'locusEnd': [100, 200],
            'leadVariantId': ['1_100_A_T', '1_200_A_T'],
            'fineMappingMethod': ['PICS', 'PICS'],
            'qualityControls': [[CredibleSetFlag.WINDOW_CLUMPED.value], []],
        },
        schema={
            'id': pl.String,
            'studyId': pl.String,
            'chromosome': pl.String,
            'locusStart': pl.Int32,
            'locusEnd': pl.Int32,
            'leadVariantId': pl.String,
            'fineMappingMethod': pl.String,
            'qualityControls': pl.List(pl.String),
        },
    )
    variants = pl.LazyFrame(
        {
            'credibleSetId': ['clumped', 'abnormal'],
            'variantId': ['1_100_A_T', '1_200_A_T'],
            'qualityControls': [[], [CredibleSetVariantFlag.ABNORMAL_PIP.value]],
        },
        schema={'credibleSetId': pl.String, 'variantId': pl.String, 'qualityControls': pl.List(pl.String)},
    )
    valid_set, valid_variant, excluded_set, excluded_variant = validate_source(sets, variants, settings)
    assert valid_set.collect()['id'].to_list() == ['abnormal']
    assert excluded_set.collect()['id'].to_list() == ['clumped']
    assert valid_variant.collect()['credibleSetId'].to_list() == ['abnormal']
    assert excluded_variant.collect()['credibleSetId'].to_list() == ['clumped']


def test_variant_qc_of_every_source_cuts_at_95_percent_coverage() -> None:
    tasks = _tasks()
    for source, kind in SOURCES.items():
        settings = VariantQcSettings.model_validate(
            tasks[f'transform qc credible set variants {kind}/{source}']['settings']
        )
        assert (settings.coverage, settings.pip_tolerance) == (0.95, 1e-4), source


def test_set_qc_of_every_source_allows_every_finemapping_method() -> None:
    tasks = _tasks()
    for source, kind in SOURCES.items():
        settings = SetQcSettings.model_validate(tasks[f'transform qc credible sets {kind}/{source}']['settings'])
        assert settings.allowed_fm_methods == list(FinemappingMethod), source


def test_the_variant_index_reads_the_validated_variants() -> None:
    [index] = [task for task in parse_yaml(CONFIG_PATH)['steps']['variant'] if task['transformer'] == 'variant']
    assert index['source']['credible_set_variant'] == 'intermediate/validated/credible_set_variant'
    assert 'credible_set' not in index['source']


@pytest.mark.parametrize(
    ('step', 'task'), [('vep_view', 'transform vep view'), ('coding_variant', 'transform coding_variant view')]
)
def test_the_variant_views_read_the_published_credible_sets_and_their_variants(step: str, task: str) -> None:
    [view] = [spec for spec in parse_yaml(CONFIG_PATH)['steps'][step] if spec['name'] == task]
    assert view['source']['credible_set'] == 'output/credible_set'
    assert view['source']['credible_set_variant'] == 'output/credible_set_variant'


CREDIBLE_SET_DATASETS = {
    'intermediate/validated/credible_set',
    'intermediate/validated/credible_set_variant',
    'output/credible_set',
    'output/credible_set_variant',
}
"""The credible sets and their variants as validation and annotation write them."""

CONSUMER_TEST = {
    'variant': 'test_variant_rows.py',
    'credible_set.annotation': 'test_credible_set_stages.py',
    'coding_variant': 'test_coding_variant.py',
    'l2g_train': 'test_l2g_train.py',
    'l2g_predict': 'test_l2g_predict.py',
    'evidence.gwas_credible_sets': 'evidence/test_gwas_credible_sets.py',
    'vep_view': 'test_vep_view.py',
    'search': 'test_search_entrypoint.py',
}
"""The tests of every task that reads `CREDIBLE_SET_DATASETS`, which build their inputs as the producers write them."""


def _paths(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return set().union(*(_paths(item) for item in value))
    return set()


def _consumers() -> set[str]:
    """The transformers of every task in the config that reads `CREDIBLE_SET_DATASETS`."""
    return {
        spec['transformer']
        for tasks in parse_yaml(CONFIG_PATH)['steps'].values()
        for task in tasks
        for spec in [task, *task.get('do', [])]
        if 'transformer' in spec and _paths(spec.get('source')) & CREDIBLE_SET_DATASETS
    }


def test_every_task_reading_the_credible_sets_has_a_contract_test() -> None:
    """A new reader of the credible sets fails here until its tests are listed in `CONSUMER_TEST`."""
    assert _consumers() == set(CONSUMER_TEST)


@pytest.mark.parametrize('consumer', sorted(CONSUMER_TEST))
def test_the_tests_of_a_credible_set_reader_build_its_input_as_the_producers_write_it(consumer: str) -> None:
    test_module = Path(__file__).parent / CONSUMER_TEST[consumer]
    assert 'tests.credible_set_fixtures import' in test_module.read_text(), consumer


def test_every_credible_set_task_runs_an_entry_point_the_package_exports() -> None:
    names = {
        spec['transformer']
        for tasks in parse_yaml(CONFIG_PATH)['steps'].values()
        for task in tasks
        for spec in [task, *task.get('do', [])]
        if spec.get('transformer', '').startswith('credible_set.')
    }
    assert names == {f'credible_set.{name}' for name in credible_set.__all__} - {'credible_set.validation_check'}
    for name in names:
        assert Transform.load_transformer(name) is getattr(credible_set, name.removeprefix('credible_set.'))
