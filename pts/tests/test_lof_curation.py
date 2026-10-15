"""LoF curation transformation and complete CSV-to-Parquet behaviour."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers.lof_curation import transform_lof_curation


@pytest.fixture
def doctest_namespace():
    """These Polars tests do not need the shared Spark doctest session."""
    return {}


def raw_annotations(rows):
    """Use the three string columns consumed from the multiline curation CSV."""
    return pl.DataFrame(
        rows,
        schema={
            'Variant ID GRCh37': pl.String,
            'Variant ID GRCh38': pl.String,
            'Verdict': pl.String,
        },
    ).lazy()


def test_coordinates_and_alleles_come_from_their_respective_genome_builds():
    """GRCh38 coordinates retain GRCh37 alleles when liftover changes them."""
    result = (
        transform_lof_curation(
            raw_annotations([
                {'Variant ID GRCh37': '1-100-A-T', 'Variant ID GRCh38': '2-200-C-G', 'Verdict': 'lof'},
            ])
        )
        .collect()
        .row(0, named=True)
    )
    assert result == {
        'variantId': '2_200_A_T',
        'chromosome': '2',
        'position': 200,
        'referenceAllele': 'A',
        'alternateAllele': 'T',
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
    }


@pytest.mark.parametrize(
    ('verdict', 'score', 'description'),
    [
        ('lof', 1.0, 'Assessed to cause LoF'),
        ('likely_lof', 0.5, 'Suspected to cause LoF'),
        ('uncertain', 0.0, 'Uncertain LoF assessment'),
        ('likely_not_lof', -0.5, 'Suspected not to cause LoF'),
        ('not_lof', -1.0, 'Assessed not to cause LoF'),
        ('LOF', None, None),
        ('unknown', None, None),
        (None, None, None),
    ],
)
def test_verdict_scores_and_descriptions(verdict, score, description):
    """All five verdicts retain their ordinal mapping; unknowns are not inferred."""
    row = (
        transform_lof_curation(
            raw_annotations([
                {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': verdict},
            ])
        )
        .collect()
        .row(0, named=True)
    )
    assert row['variantEffect'][0]['assessment'] == verdict
    assert row['variantEffect'][0]['normalisedScore'] == score
    assert row['variantDescription'] == (description + ' by OTAR2075 variant curation effort.' if description else None)


@pytest.mark.parametrize(
    ('h37', 'h38', 'variant_id', 'position'),
    [
        (None, None, '', None),
        ('1-10-A', '2-20-C-G', '2_20_A', 20),
        ('1-10--G', '2-20-C-G', '2_20_G', 20),
        ('1-10-A-T-extra', '2-20-C-G', '2_20_A_T', 20),
        ('1-10-A-G', '2-1.9-C-T', '2_1.9_A_G', 1),
        ('1-10-A-G', '2-2147483648-C-T', '2_2147483648_A_G', None),
        ('1-10-A-G', '2-1e2-C-T', '2_1e2_A_G', None),
    ],
)
def test_partial_identifiers_and_integer_casting(h37, h38, variant_id, position):
    """Identifier concatenation skips nulls while invalid Int32 coordinates remain null."""
    row = (
        transform_lof_curation(
            raw_annotations([
                {'Variant ID GRCh37': h37, 'Variant ID GRCh38': h38, 'Verdict': 'lof'},
            ])
        )
        .collect()
        .row(0, named=True)
    )
    import hashlib

    expected_id = 'OTVAR_' + hashlib.md5(variant_id.encode()).hexdigest() if position is None else variant_id
    assert row['variantId'] == expected_id
    assert row['position'] == position


def test_duplicate_annotations_are_preserved():
    """Curation ingestion preserves source records instead of choosing a verdict."""
    row = {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'lof'}
    assert transform_lof_curation(raw_annotations([row, row])).collect().height == 2


def test_empty_input_has_the_annotation_schema():
    """An empty raw export still yields typed annotation columns."""
    from pts.schemas.variant_annotation import variant_annotation_schema

    result = transform_lof_curation(raw_annotations([])).collect()
    assert result.is_empty()
    assert result.schema == variant_annotation_schema


def test_step_reads_multiline_csv_and_writes_typed_parquet(tmp_path):
    """Quoted commas and newlines in unused curation columns do not split records."""
    import csv

    from pts.schemas.variant_annotation import variant_annotation_schema
    from pts.transformers.lof_curation import lof_curation

    source = tmp_path / 'curation.csv'
    with source.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant ID GRCh37', 'Variant ID GRCh38', 'Verdict', 'Notes'])
        writer.writerow(['1-10-A-G', '2-20-C-T', 'likely_lof', 'Comment, with "quotes"\nand a newline'])
        writer.writerow(['1-11-G-T', '2-21-G-T', '', 'Other note'])
    destination = tmp_path / 'annotations'
    lof_curation(str(source), str(destination), {}, Mock(spec=Config))
    result = pl.read_parquet(destination / '*.parquet').sort('position')
    assert result.height == 2
    assert result.schema == variant_annotation_schema
    assert result['variantId'].to_list() == ['2_20_A_G', '2_21_G_T']
    assert result['variantEffect'][0][0]['normalisedScore'] == 0.5
    assert result['variantEffect'][1][0]['assessment'] is None


def test_settings_reject_unrecognized_options():
    """Unsupported settings fail before trying to read the input."""
    from pydantic import ValidationError

    from pts.transformers.lof_curation import lof_curation

    with pytest.raises(ValidationError):
        lof_curation('unused.csv', 'unused-output', {'unknown': True}, Mock(spec=Config))


def test_missing_coordinate_uses_the_stored_annotation_identifier():
    """Real curation without GRCh38 coordinates receives VariantIndex's stable hash."""
    row = (
        transform_lof_curation(
            raw_annotations([
                {'Variant ID GRCh37': '1-10-GTCTTCCAATACCTCCCA-G', 'Variant ID GRCh38': None, 'Verdict': 'lof'},
            ])
        )
        .collect()
        .row(0, named=True)
    )
    assert row['variantId'] == 'OTVAR_f61f8e0a0e77bdd4a448620d69674736'


def test_long_alleles_use_hashed_id_with_coordinates():
    """Long alleles preserve genomic context in the compact identifier."""
    import hashlib

    allele = 'A' * 300
    identifier = '2_200_' + allele + '_T'
    row = (
        transform_lof_curation(
            raw_annotations([
                {'Variant ID GRCh37': '1-10-' + allele + '-T', 'Variant ID GRCh38': '2-200-A-G', 'Verdict': 'lof'},
            ])
        )
        .collect()
        .row(0, named=True)
    )
    assert row['variantId'] == 'OTVAR_2_200_' + hashlib.md5(identifier.encode()).hexdigest()
