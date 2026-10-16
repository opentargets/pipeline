"""Variant index task configuration and real annotation enrichment."""

from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
import yaml
from otter.config.model import Config
from pydantic import ValidationError

from pts.schemas.variant import variant_schema
from pts.tasks.transform import TransformSpec
from pts.transformers.variant import (
    VariantDestination,
    VariantSettings,
    VariantSource,
    read_variant_annotations,
    variant,
)


@pytest.fixture
def doctest_namespace():
    """These Polars tests do not need the Spark doctest session."""
    return {}


def test_variant_configuration_matches_models():
    config = yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text())
    step = config['steps']['variant'][0]
    TransformSpec.model_validate(step)
    VariantSource.model_validate(step['source'])
    VariantDestination.model_validate(step['destination'])
    assert VariantSettings.model_validate(step['settings']).hash_threshold == 300


@pytest.mark.parametrize('threshold', [0, -1])
def test_variant_hash_threshold_must_be_positive(threshold):
    with pytest.raises(ValidationError):
        VariantSettings(hash_threshold=threshold)


def test_recursive_annotations_insert_missing_fields(tmp_path):
    directory = tmp_path / 'annotations' / 'chromosome=1'
    directory.mkdir(parents=True)
    pl.DataFrame({
        'variantId': ['1_10_A_T'],
        'chromosome': ['1'],
        'position': pl.Series([10], dtype=pl.Int32),
    }).write_parquet(directory / 'part.parquet')

    frame = read_variant_annotations(str(directory.parent))
    assert isinstance(frame, pl.LazyFrame)
    result = frame.collect()
    assert result.schema == variant_schema
    assert result['variantId'].to_list() == ['1_10_A_T']
    assert result['variantEffect'].to_list() == [[]]
    assert result['variantDescription'].to_list() == [None]


def test_recursive_annotations_require_parquet_parts(tmp_path):
    with pytest.raises(ValueError, match='No Parquet'):
        read_variant_annotations(str(tmp_path))


def test_variant_step_merges_annotations_and_amino_acid_predictions(tmp_path):
    supplemental = pl.DataFrame(
        {
            'variantId': ['20_10645397_C_T'],
            'chromosome': ['20'],
            'position': [10645397],
            'alleleFrequencies': [[{'populationName': 'test_population', 'alleleFrequency': 0.125}]],
            'dbXrefs': [[{'id': 'extra_id', 'source': 'test_source'}]],
            'variantDescription': ['Additional annotation.'],
        },
        schema={
            name: variant_schema[name]
            for name in ['variantId', 'chromosome', 'position', 'alleleFrequencies', 'dbXrefs', 'variantDescription']
        },
    )
    annotation_path = tmp_path / 'annotation.parquet'
    supplemental.write_parquet(annotation_path)
    amino = pl.DataFrame(
        {
            'uniprotAccession': ['P78504'],
            'aminoAcidChange': ['C691Y'],
            'variantEffect': [
                [
                    {
                        'method': 'FoldX',
                        'assessment': None,
                        'score': 7.0,
                        'assessmentFlag': None,
                        'targetId': None,
                        'normalisedScore': 1.0,
                    }
                ]
            ],
        },
        schema={
            'uniprotAccession': pl.String,
            'aminoAcidChange': pl.String,
            'variantEffect': variant_schema['variantEffect'],
        },
    )
    amino_path = tmp_path / 'amino.parquet'
    amino.write_parquet(amino_path)
    output = tmp_path / 'variant'

    variant(
        {
            'vep': str(Path(__file__).parent / 'test_data' / 'vep_sample.jsonl'),
            'variant_annotations': [str(annotation_path)],
            'amino_acid_annotations': [str(amino_path)],
        },
        {'output': str(output)},
        {},
        Mock(spec=Config),
    )

    result = pl.read_parquet(list(output.glob('*.parquet')))
    assert result.schema == variant_schema
    assert result['variantId'].to_list() == ['17_29510931_T_C', '20_10645397_C_T', '20_10649087_G_A', '9_82445881_C_T']
    row = result.filter(pl.col('variantId') == '20_10645397_C_T').to_dicts()[0]
    assert {'id': 'extra_id', 'source': 'test_source'} in row['dbXrefs']
    assert {'populationName': 'test_population', 'alleleFrequency': 0.125} in row['alleleFrequencies']
    assert row['variantDescription'].endswith('Additional annotation.')
    assert any(effect['method'] == 'FoldX' and effect['score'] == 7.0 for effect in row['variantEffect'])
