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


def reference_bases(sequence='A', chromosome='1', position=20):
    return pl.LazyFrame(
        {
            'chr': [chromosome] * len(sequence),
            'pos': list(range(position, position + len(sequence))),
            'ref': list(sequence),
        },
        schema={'chr': pl.String, 'pos': pl.Int64, 'ref': pl.String},
    )


def test_step_validates_full_grch38_ref_against_reference_parquet(tmp_path):
    """Only complete GRCh38 REF matches survive, including across Parquet parts."""
    import csv

    from pts.transformers.lof_curation import lof_curation
    from pts.transformers.reference import reference

    fasta = tmp_path / 'genome.fasta'
    fasta.write_text('>chr1\nACGTAC\n>chr19_KI270938v1_alt\nGC\n')
    reference_path = tmp_path / 'reference'
    reference(str(fasta), str(reference_path), {'batch_size': 2}, Mock(spec=Config))
    source = tmp_path / 'curation.csv'
    with source.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Variant ID GRCh37', 'Variant ID GRCh38', 'Verdict'])
        writer.writerows([
            ['1-20-T-G', '1-2-C-T', 'lof'],
            ['1-20-T-G', '1-2-C-T', 'uncertain'],
            ['1-20-T-G', '1-2-CGT-C', 'likely_lof'],
            ['1-20-T-G', '1-2-CGA-C', 'lof'],  # anchor matches, deletion suffix does not
            ['1-20-T-G', '1-6-CA-C', 'lof'],  # reference span runs beyond contig
            ['1-20-T-G', '1-1-T-A', 'lof'],  # wrong SNV REF
            ['1-20-T-G', 'missing-1-A-T', 'lof'],
            ['1-20-T-G', '', 'lof'],
            ['', '19_KI270938v1_alt-1-G-A', 'lof'],  # GRCh37 ID is not required
        ])
    destination = tmp_path / 'annotations'
    lof_curation(
        {'curation': str(source), 'reference': str(reference_path)},
        str(destination),
        {},
        Mock(spec=Config),
    )
    result = pl.read_parquet(destination / '*.parquet')
    assert result['variantId'].to_list() == ['1_2_C_T', '1_2_C_T', '1_2_CGT_C', '19_KI270938v1_alt_1_G_A']
    assert [entry[0]['assessment'] for entry in result['variantEffect']] == ['lof', 'uncertain', 'likely_lof', 'lof']


def test_all_identifier_components_come_from_grch38():
    """GRCh37 alleles never replace the validated GRCh38 alleles."""
    result = (
        transform_lof_curation(
            raw_annotations([
                {'Variant ID GRCh37': '1-100-A-T', 'Variant ID GRCh38': '2-200-C-G', 'Verdict': 'lof'},
            ]),
            reference_bases('C', '2', 200),
        )
        .collect()
        .row(0, named=True)
    )
    assert result == {
        'variantId': '2_200_C_G',
        'chromosome': '2',
        'position': 200,
        'referenceAllele': 'C',
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
            ]),
            reference_bases(),
        )
        .collect()
        .row(0, named=True)
    )
    assert row['variantEffect'][0]['assessment'] == verdict
    assert row['variantEffect'][0]['normalisedScore'] == score
    assert row['variantDescription'] == (description + ' by OTAR2075 variant curation effort.' if description else None)


@pytest.mark.parametrize(
    'h38',
    [
        None,
        '',
        '1-20-A',
        '1-20--G',
        '1-20-A-G-extra',
        '1-20.9-A-T',
        '1-2147483648-A-T',
        '1-2e1-A-T',
        '1-0-A-T',
        '1--20-A-T',
        '1-20-A-A',
        '1-20-A-<DEL>',
        '1-20-a-g',
    ],
)
def test_invalid_or_missing_grch38_identifiers_are_dropped(h38):
    """Malformed IDs are not truncated, repaired, or converted to fallback hashes."""
    result = transform_lof_curation(
        raw_annotations([
            {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': h38, 'Verdict': 'lof'},
        ]),
        reference_bases(),
    ).collect()
    assert result.is_empty()


def test_duplicate_annotations_are_preserved():
    """Curation ingestion preserves source records instead of choosing a verdict."""
    row = {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'lof'}
    assert transform_lof_curation(raw_annotations([row, row]), reference_bases()).collect().height == 2


def test_empty_input_has_the_annotation_schema():
    """An empty raw export still yields typed annotation columns."""
    from pts.schemas.variant_annotation import variant_annotation_schema

    result = transform_lof_curation(raw_annotations([]), reference_bases()).collect()
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
    reference_path = tmp_path / 'reference.parquet'
    reference_bases('CG', '2', 20).collect().write_parquet(reference_path)
    lof_curation({'curation': str(source), 'reference': str(reference_path)}, str(destination), {}, Mock(spec=Config))
    result = pl.read_parquet(destination / '*.parquet').sort('position')
    assert result.height == 2
    assert result.schema == variant_annotation_schema
    assert result['variantId'].to_list() == ['2_20_C_T', '2_21_G_T']
    assert result['variantEffect'][0][0]['normalisedScore'] == 0.5
    assert result['variantEffect'][1][0]['assessment'] is None


def test_settings_reject_unrecognized_options():
    """Unsupported settings fail before trying to read the input."""
    from pydantic import ValidationError

    from pts.transformers.lof_curation import lof_curation

    with pytest.raises(ValidationError):
        lof_curation(
            {'curation': 'unused.csv', 'reference': 'unused'}, 'unused-output', {'unknown': True}, Mock(spec=Config)
        )


def test_missing_reference_base_inside_deletion_is_dropped():
    """A gap within the reference span cannot validate a partial REF sequence."""
    result = transform_lof_curation(
        raw_annotations([
            {'Variant ID GRCh38': '1-20-ACG-A', 'Verdict': 'lof'},
        ]),
        reference_bases('ACG').filter(pl.col('pos') != 21),
    ).collect()
    assert result.is_empty()


def test_long_alleles_use_hashed_id_with_coordinates():
    """Long alleles preserve genomic context in the compact identifier."""
    import hashlib

    allele = 'A' * 300
    identifier = '2_200_' + allele + '_T'
    row = (
        transform_lof_curation(
            raw_annotations([
                {'Variant ID GRCh37': '1-10-A-G', 'Variant ID GRCh38': '2-200-' + allele + '-T', 'Verdict': 'lof'},
            ]),
            reference_bases(allele, '2', 200),
        )
        .collect()
        .row(0, named=True)
    )
    assert row['variantId'] == 'OTVAR_2_200_' + hashlib.md5(identifier.encode()).hexdigest()


def test_duplicate_reference_coordinates_fail_instead_of_multiplying_annotations():
    rows = raw_annotations([{'Variant ID GRCh38': '1-20-A-G', 'Verdict': 'lof'}])
    with pytest.raises(pl.exceptions.ComputeError, match='m:1'):
        transform_lof_curation(rows, pl.concat([reference_bases(), reference_bases()]))


def test_chromosome_aliases_and_insertions_are_validated():
    rows = raw_annotations([{'Variant ID GRCh38': 'chrM-1-A-ACG', 'Verdict': 'lof'}])
    result = transform_lof_curation(rows, reference_bases('A', 'MT', 1)).collect()
    assert result['variantId'].to_list() == ['MT_1_A_ACG']


def test_step_writes_a_readable_empty_dataset_when_every_ref_fails(tmp_path):
    from pts.schemas.variant_annotation import variant_annotation_schema
    from pts.transformers.lof_curation import lof_curation

    source = tmp_path / 'curation.csv'
    source.write_text('Variant ID GRCh38,Verdict\n1-20-C-T,lof\n')
    reference_path = tmp_path / 'reference.parquet'
    reference_bases('A').collect().write_parquet(reference_path)
    destination = tmp_path / 'annotations'

    lof_curation(
        {'curation': str(source), 'reference': str(reference_path)},
        str(destination),
        {},
        Mock(spec=Config),
    )

    result = pl.read_parquet(destination / '*.parquet')
    assert result.is_empty()
    assert result.schema == variant_annotation_schema
