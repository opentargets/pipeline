"""The record sets of the credible sets follow the CredibleSet and CredibleSetVariant datasets."""

import json
from pathlib import Path

import pytest

RECORDSET_DIR = Path(__file__).parent.parent / 'src/ot_croissant/assets/recordset'

CREDIBLE_SET_FIELDS = {
    'id', 'chromosome', 'locusStart', 'locusEnd', 'studyId', 'studyType', 'leadVariantId', 'fineMappingMethod',
    'logBayesFactor', 'confidence', 'purityMeanR2', 'purityMinR2', 'qualityControls',
}  # fmt: skip
CREDIBLE_SET_VARIANT_FIELDS = {
    'credibleSetId', 'variantId', 'pValueMantissa', 'pValueExponent', 'effectAlleleFrequency', 'sampleSize',
    'reportedBeta', 'reportedStandardError', 'posteriorProbability', 'logBayesFactor', 'r2Overall',
    'conditionalBeta', 'conditionalStandardError', 'rescaledAbsBeta', 'rescaledStandardError', 'rescaledDirection',
    'qualityControls',
}  # fmt: skip


def _fields(dataset: str) -> dict[str, dict]:
    return {entry['id']: entry for entry in json.loads((RECORDSET_DIR / f'{dataset}.json').read_text())}


@pytest.mark.parametrize(
    ('dataset', 'fields', 'primary_key'),
    [
        ('credible_set', CREDIBLE_SET_FIELDS, {'id'}),
        ('credible_set_variant', CREDIBLE_SET_VARIANT_FIELDS, {'credibleSetId', 'variantId'}),
    ],
)
def test_the_credible_set_record_sets_describe_every_field_and_their_key(
    dataset: str, fields: set[str], primary_key: set[str]
) -> None:
    curated = _fields(dataset)
    assert set(curated) == fields
    assert {name for name, entry in curated.items() if entry.get('isPrimaryKey')} == primary_key


@pytest.mark.parametrize(
    ('dataset', 'field'),
    [
        ('credible_set_variant', 'credibleSetId'),
        ('colocalisation', 'leftLocusId'),
        ('colocalisation', 'rightLocusId'),
        ('l2g_prediction', 'credibleSetId'),
        ('evidence_gwas_credible_sets', 'studyLocusId'),
    ],
)
def test_references_to_a_credible_set_point_at_its_id(dataset: str, field: str) -> None:
    assert _fields(dataset)[field]['foreign_key'] == 'credible_set/id'
