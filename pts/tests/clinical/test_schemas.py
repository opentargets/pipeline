import hashlib
import json

from pts.transformers.clinical import ClinicalReportExtractionSchema
from pts.transformers.clinical.schemas import AssociatedDisease

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
