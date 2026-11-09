"""Two credible sets of one source under one id: variant QC, then validation, with the shipped settings.

The id hashes the study, chromosome, locus, lead and method, so two sets under one id share their
lead. Their variants share the id too, so they cannot be told apart, and every copy leaves.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import polars as pl
from otter.config.model import Config
from otter.config.yaml import parse_yaml

from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.validation import validation
from pts.transformers.credible_set.variant_qc import VariantQcSettings, flag_variants

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

SET_ID = 'e' * 32
LEAD = '1_100_A_T'


def _settings(transformer: str) -> dict[str, Any]:
    """The settings of the credible set task that runs `transformer`, as shipped."""
    for task in parse_yaml(CONFIG_PATH)['steps']['credible_set']:
        for spec in [task, *task.get('do', [])]:
            if spec.get('transformer') == transformer:
                return spec['settings']
    raise LookupError(transformer)


def _set(flags: list[str] | None = None) -> dict[str, Any]:
    return {
        'id': SET_ID,
        'chromosome': '1',
        'locusStart': 1,
        'locusEnd': 1_000_000,
        'studyId': 'GCST1',
        'studyType': 'gwas',
        'leadVariantId': LEAD,
        'fineMappingMethod': 'SuSiE',
        'qualityControls': flags or [],
    }


def _variant(variant_id: str, pip: float) -> dict[str, Any]:
    return {
        'credibleSetId': SET_ID,
        'variantId': variant_id,
        'posteriorProbability': pip,
        'pValueMantissa': 1.0,
        'pValueExponent': -10,
        'sampleSize': 1000,
        'qualityControls': [],
    }


def _frame(schema: pl.Schema, rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame([{name: row.get(name) for name in schema} for row in rows], schema=schema)


def _write(frame: pl.DataFrame, path: Path) -> str:
    path.mkdir(parents=True)
    frame.write_parquet(path / 'part-0.parquet')
    return str(path)


def _qc_and_validate(tmp_path: Path, sets: list[dict], variants: list[dict]) -> dict[str, pl.DataFrame]:
    """Variant QC of one source's variants, then validation of the source; return the four written datasets."""
    qc = VariantQcSettings.model_validate(_settings('credible_set.variant_qc'))
    flagged_variant = (
        _frame(credible_set_variant_schema, variants)
        .lazy()
        .pipe(flag_variants, qc.coverage, qc.pip_tolerance)
        .collect()
    )
    out = tmp_path / 'out'
    names = ('valid_set', 'valid_variant', 'excluded_set', 'excluded_variant')
    validation(
        source={
            'credible_set': [_write(_frame(credible_set_schema, sets), tmp_path / 'qc/gwas/s/credible_set')],
            'credible_set_variant': [_write(flagged_variant, tmp_path / 'qc/gwas/s/credible_set_variant')],
        },
        destination={name: str(out / name) for name in names},
        settings=_settings('credible_set.validation'),
        config=Mock(spec=Config, work_path=str(tmp_path), release_uri=None),
    )
    return {name: pl.read_parquet(out / name / '*.parquet') for name in names}


def _flag_name(frame: pl.DataFrame, enum: type[CredibleSetFlag] | type[CredibleSetVariantFlag]) -> list[list[str]]:
    """The flags of each row by name, rows sorted by variant id where there is one."""
    if 'variantId' in frame.columns:
        frame = frame.sort('variantId', 'posteriorProbability')
    return [sorted(enum(flag).name for flag in row) for row in frame['qualityControls']]


def test_two_sets_of_only_their_lead_are_both_excluded_as_duplicated(tmp_path: Path) -> None:
    written = _qc_and_validate(tmp_path, [_set(), _set()], [_variant(LEAD, 1.0), _variant(LEAD, 1.0)])

    assert written['valid_set'].height == 0
    assert written['valid_variant'].height == 0
    assert (
        _flag_name(written['excluded_set'], CredibleSetFlag)
        == [['DUPLICATED_CREDIBLE_SET_ID', 'INVALID_VARIANT_IN_SET']] * 2
    )
    # the PIPs of both sets add up under the one id, so one lead row is past the 95% coverage; the two
    # rows are otherwise identical, so they are compared in any order
    assert sorted(_flag_name(written['excluded_variant'], CredibleSetVariantFlag)) == [
        ['CREDIBLE_SET_EXCLUDED', 'DUPLICATED_VARIANT'],
        ['DUPLICATED_VARIANT', 'EXCEEDS_COVERAGE_95'],
    ]


def test_two_sets_of_different_sizes_sharing_their_lead_are_both_excluded(tmp_path: Path) -> None:
    larger = [_variant(LEAD, 0.9), _variant('1_200_C_G', 0.06), _variant('1_300_G_A', 0.04)]
    smaller = [_variant(LEAD, 0.9), _variant('1_400_T_C', 0.1)]
    written = _qc_and_validate(tmp_path, [_set(), _set()], [*larger, *smaller])

    assert written['valid_set'].height == 0
    assert written['valid_variant'].height == 0
    assert (
        _flag_name(written['excluded_set'], CredibleSetFlag)
        == [['DUPLICATED_CREDIBLE_SET_ID', 'INVALID_VARIANT_IN_SET']] * 2
    )
    # under the one id the PIPs sum to 2: every row is ABNORMAL_PIP, and the two leads cover 95%
    assert written['excluded_variant'].sort('variantId', 'posteriorProbability')['variantId'].to_list() == [
        LEAD,
        LEAD,
        '1_200_C_G',
        '1_300_G_A',
        '1_400_T_C',
    ]
    assert _flag_name(written['excluded_variant'], CredibleSetVariantFlag) == [
        ['ABNORMAL_PIP', 'CREDIBLE_SET_EXCLUDED', 'DUPLICATED_VARIANT'],
        ['ABNORMAL_PIP', 'CREDIBLE_SET_EXCLUDED', 'DUPLICATED_VARIANT'],
        ['ABNORMAL_PIP', 'EXCEEDS_COVERAGE_95'],
        ['ABNORMAL_PIP', 'EXCEEDS_COVERAGE_95'],
        ['ABNORMAL_PIP', 'EXCEEDS_COVERAGE_95'],
    ]


def test_two_sets_listing_their_variants_once_are_both_excluded_whatever_their_own_flags(tmp_path: Path) -> None:
    """The review's case: one copy carries an excluding flag of its own; neither copy can keep the variants."""
    excluded = _set([CredibleSetFlag.INCONSISTENCY_FLAG.value])
    written = _qc_and_validate(tmp_path, [_set(), excluded], [_variant(LEAD, 0.9), _variant('1_200_C_G', 0.1)])

    assert written['valid_set'].height == 0
    assert written['valid_variant'].height == 0
    assert sorted(_flag_name(written['excluded_set'], CredibleSetFlag)) == [
        ['DUPLICATED_CREDIBLE_SET_ID'],
        ['DUPLICATED_CREDIBLE_SET_ID', 'INCONSISTENCY_FLAG'],
    ]
    assert _flag_name(written['excluded_variant'], CredibleSetVariantFlag) == [['CREDIBLE_SET_EXCLUDED']] * 2
