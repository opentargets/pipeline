"""Credible set QC runs per source; validation, with the cross-set QC, and annotation run over every source."""

from pathlib import Path
from typing import Any

import polars as pl
from otter.config.yaml import parse_yaml

from pts.transformers.credible_set.enums import FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.set_qc import SetQcSettings
from pts.transformers.credible_set.validation import ValidationSettings, cross_set_flags_schema, validate
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


def test_each_source_is_checked_against_its_own_input_and_mirrors_it() -> None:
    tasks = _tasks()
    for source, kind in SOURCES.items():
        folder = f'{kind}/{source}'
        set_qc = tasks[f'transform qc credible sets {folder}']
        variant_qc = tasks[f'transform qc credible set variants {folder}']
        assert set_qc['transformer'] == 'credible_set.set_qc'
        assert set_qc['source'] == {'credible_set': f'input/{folder}/credible_set', 'study': 'output/study'}
        assert set_qc['destination'] == f'intermediate/qc/{folder}/credible_set'
        assert variant_qc['transformer'] == 'credible_set.variant_qc'
        assert variant_qc['source'] == {'credible_set_variant': f'input/{folder}/credible_set_variant'}
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


def test_annotation_publishes_the_validated_datasets() -> None:
    annotation = _tasks()['transform annotate credible sets']
    assert annotation['requires'] == ['transform validate credible sets']
    assert annotation['source'] == {
        'credible_set': 'intermediate/validated/credible_set',
        'credible_set_variant': 'intermediate/validated/credible_set_variant',
    }
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
        CredibleSetFlag.DUPLICATED_STUDYLOCUS_ID,
        CredibleSetFlag.INVALID_FINEMAPPING_METHOD,
        CredibleSetFlag.EXPLAINED_BY_SUSIE,
        CredibleSetFlag.COMPOSITE_FLAG,
        CredibleSetFlag.INCONSISTENCY_FLAG,
        CredibleSetFlag.SUBSIGNIFICANT_FLAG,
        CredibleSetFlag.LD_CLUMPED,
        CredibleSetFlag.WINDOW_CLUMPED,
        CredibleSetFlag.STUDY_TYPE_MISMATCH,
    }
    assert set(settings.csv_set_invalid_qc_reasons) == {
        CredibleSetVariantFlag.INVALID_CS_ID,
        CredibleSetVariantFlag.INVALID_VARIANT_ID,
    }


def test_configured_validation_excludes_window_clumped_and_keeps_abnormal_pip() -> None:
    settings = _validation_settings()
    sets = pl.LazyFrame(
        {'id': ['clumped', 'abnormal'], 'qualityControls': [[CredibleSetFlag.WINDOW_CLUMPED.value], []]},
        schema={'id': pl.String, 'qualityControls': pl.List(pl.String)},
    )
    variants = pl.LazyFrame(
        {
            'credibleSetId': ['clumped', 'abnormal'],
            'variantId': ['1_100_A_T', '1_200_A_T'],
            'qualityControls': [[], [CredibleSetVariantFlag.ABNORMAL_PIP.value]],
        },
        schema={'credibleSetId': pl.String, 'variantId': pl.String, 'qualityControls': pl.List(pl.String)},
    )
    no_cross_set_flags = pl.LazyFrame(schema=cross_set_flags_schema)
    valid_set, valid_variant, excluded_set, excluded_variant = validate(sets, variants, no_cross_set_flags, settings)
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
