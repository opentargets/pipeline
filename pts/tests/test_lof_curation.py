"""LoF annotation transformation and isolated CSV/Parquet I/O."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from polars.testing import assert_frame_equal

from pts.schemas.variant_annotation import variant_annotation_schema
from pts.transformers.lof_curation import _transform_lof_curation, lof_curation

_RAW_SCHEMA = {
    'Variant ID GRCh37': pl.String,
    'Variant ID GRCh38': pl.String,
    'Verdict': pl.String,
}


@pytest.mark.parametrize(
    ('input_row', 'output_row'),
    [
        pytest.param(
            {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'lof'},
            {
                'variantId': '1_20_A_G',
                'chromosome': '1',
                'position': 20,
                'referenceAllele': 'A',
                'alternateAllele': 'G',
                'variantEffect': [
                    {
                        'method': 'LossOfFunctionCuration',
                        'assessment': 'lof',
                        'score': None,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 1.0,
                    }
                ],
                'variantDescription': 'Assessed to cause LoF by OTAR2075 variant curation effort.',
            },
            id='lof',
        ),
        pytest.param(
            {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'likely_lof'},
            {
                'variantId': '1_20_A_G',
                'chromosome': '1',
                'position': 20,
                'referenceAllele': 'A',
                'alternateAllele': 'G',
                'variantEffect': [
                    {
                        'method': 'LossOfFunctionCuration',
                        'assessment': 'likely_lof',
                        'score': None,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 0.5,
                    }
                ],
                'variantDescription': 'Suspected to cause LoF by OTAR2075 variant curation effort.',
            },
            id='likely-lof',
        ),
        pytest.param(
            {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'uncertain'},
            {
                'variantId': '1_20_A_G',
                'chromosome': '1',
                'position': 20,
                'referenceAllele': 'A',
                'alternateAllele': 'G',
                'variantEffect': [
                    {
                        'method': 'LossOfFunctionCuration',
                        'assessment': 'uncertain',
                        'score': None,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 0.0,
                    }
                ],
                'variantDescription': 'Uncertain LoF assessment by OTAR2075 variant curation effort.',
            },
            id='uncertain',
        ),
        pytest.param(
            {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'likely_not_lof'},
            {
                'variantId': '1_20_A_G',
                'chromosome': '1',
                'position': 20,
                'referenceAllele': 'A',
                'alternateAllele': 'G',
                'variantEffect': [
                    {
                        'method': 'LossOfFunctionCuration',
                        'assessment': 'likely_not_lof',
                        'score': None,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': -0.5,
                    }
                ],
                'variantDescription': 'Suspected not to cause LoF by OTAR2075 variant curation effort.',
            },
            id='likely-not-lof',
        ),
        pytest.param(
            {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'not_lof'},
            {
                'variantId': '1_20_A_G',
                'chromosome': '1',
                'position': 20,
                'referenceAllele': 'A',
                'alternateAllele': 'G',
                'variantEffect': [
                    {
                        'method': 'LossOfFunctionCuration',
                        'assessment': 'not_lof',
                        'score': None,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': -1.0,
                    }
                ],
                'variantDescription': 'Assessed not to cause LoF by OTAR2075 variant curation effort.',
            },
            id='not-lof',
        ),
        pytest.param(
            {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '2-200-C-T', 'Verdict': 'lof'},
            {
                'variantId': '2_200_A_G',
                'chromosome': '2',
                'position': 200,
                'referenceAllele': 'A',
                'alternateAllele': 'G',
                'variantEffect': [
                    {
                        'method': 'LossOfFunctionCuration',
                        'assessment': 'lof',
                        'score': None,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 1.0,
                    }
                ],
                'variantDescription': 'Assessed to cause LoF by OTAR2075 variant curation effort.',
            },
            id='reference-allele-from-grch37',
        ),
        pytest.param(
            {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': None},
            {
                'variantId': '1_20_A_G',
                'chromosome': '1',
                'position': 20,
                'referenceAllele': 'A',
                'alternateAllele': 'G',
                'variantEffect': [
                    {
                        'method': 'LossOfFunctionCuration',
                        'assessment': None,
                        'score': None,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': None,
                    }
                ],
                'variantDescription': None,
            },
            id='empty-lof-class',
        ),
        # The inspected input has one missing GRCh38 ID, leaving both coordinates
        # null. No isolated missing chromosome/position or missing REF/ALT cases
        # occur in that input, so we do not add synthetic cases for them here.
        pytest.param(
            {
                'Variant ID GRCh37': '18-72228107-GTCTTCCAATACCTCCCA-G',
                'Variant ID GRCh38': None,
                'Verdict': 'lof',
            },
            {
                'variantId': 'OTVAR_f61f8e0a0e77bdd4a448620d69674736',
                'chromosome': None,
                'position': None,
                'referenceAllele': 'GTCTTCCAATACCTCCCA',
                'alternateAllele': 'G',
                'variantEffect': [
                    {
                        'method': 'LossOfFunctionCuration',
                        'assessment': 'lof',
                        'score': None,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 1.0,
                    }
                ],
                'variantDescription': 'Assessed to cause LoF by OTAR2075 variant curation effort.',
            },
            id='missing-chromosome-and-position-uses-otvar',
        ),
    ],
)
def test_transform_lof_curation(input_row, output_row):
    source = pl.LazyFrame([input_row], schema=_RAW_SCHEMA)
    expected = pl.DataFrame([output_row], schema=variant_annotation_schema)

    result = _transform_lof_curation(source).collect()

    assert_frame_equal(result, expected)


def test_transform_lof_curation_rejects_empty_input():
    with pytest.raises(ValueError, match='empty'):
        _transform_lof_curation(pl.LazyFrame(schema=_RAW_SCHEMA)).collect()


@pytest.mark.parametrize(
    'second_verdict',
    [
        pytest.param('lof', id='identical-annotations'),
        pytest.param('likely_lof', id='different-annotations-same-variant'),
    ],
)
def test_transform_lof_curation_rejects_duplicate_variants(second_verdict):
    row = {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'lof'}
    source = pl.LazyFrame([row, {**row, 'Verdict': second_verdict}], schema=_RAW_SCHEMA)

    with pytest.raises(ValueError, match='1_20_A_G'):
        _transform_lof_curation(source).collect()


def test_step_reads_multiline_csv_and_writes_parquet(tmp_path, monkeypatch):
    source_frame = pl.DataFrame([
        {
            'Variant ID GRCh37': '1-10-A-G',
            'Variant ID GRCh38': '2-20-C-T',
            'Verdict': 'likely_lof',
            'Notes': 'Comment, with "quotes"\nand a newline',
        },
        {
            'Variant ID GRCh37': '1-11-G-T',
            'Variant ID GRCh38': '2-21-G-T',
            'Verdict': '',
            'Notes': 'Other note',
        },
    ])
    source = tmp_path / 'curation.csv'
    source_frame.write_csv(source)
    dummy = pl.DataFrame({'dummy': ['transformed']})
    transform = Mock(return_value=dummy.lazy())
    monkeypatch.setattr('pts.transformers.lof_curation._transform_lof_curation', transform)
    destination = tmp_path / 'annotations'

    lof_curation(str(source), str(destination), {}, Mock(spec=Config))

    transform.assert_called_once()
    assert_frame_equal(
        transform.call_args.args[0].collect(),
        source_frame.with_columns(pl.col('Verdict').replace('', None)),
    )
    assert_frame_equal(pl.read_parquet(destination / '*.parquet'), dummy)
