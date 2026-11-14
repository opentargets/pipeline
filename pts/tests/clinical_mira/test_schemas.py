import hashlib
import json

from pts.transformers.clinical_mira.schemas import (
    AssociatedDisease,
    ClinicalProvider,
    ClinicalReportExtractionSchema,
    ClinicalSource,
)

MIRA_1_0_5_EXTRACTION_SCHEMA_DIGEST = '82af315af6273f063cac40e605abdd11e33bbfa128db92d3f8d2078139626061'


def test_associated_disease_fields_are_optional() -> None:
    disease = AssociatedDisease()

    assert disease.diseaseFromSource is None
    assert disease.diseaseId is None


def test_extraction_schema_digest_matches_mira() -> None:
    """The extraction schema digest equals the one mira 1.0.5 produced."""
    schema = ClinicalReportExtractionSchema.model_json_schema(by_alias=True)
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode('utf-8')).hexdigest()

    assert digest == MIRA_1_0_5_EXTRACTION_SCHEMA_DIGEST


def test_fda_owns_drugs_at_fda() -> None:
    assert ClinicalProvider.FDA.value == 'FDA'
    assert ClinicalSource.DRUGS_AT_FDA.value == 'Drugs@FDA'
    assert ClinicalProvider.owns_source('FDA', 'Drugs@FDA')
    assert not ClinicalProvider.owns_source('ChEMBL', 'Drugs@FDA')
