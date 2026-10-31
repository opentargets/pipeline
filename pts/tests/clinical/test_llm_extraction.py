import pytest
from pydantic import ValidationError

from pts.transformers.clinical.schemas import (
    ClinicalReportExtractionSchema as ClinicalReportExtraction,
)
from pts.transformers.clinical.schemas import (
    ExtractedDisease,
    ExtractedDrug,
)


def test_investigated_drug_required_fields():
    drug = ExtractedDrug(drug='ibuprofen', evidence_quote='patients received ibuprofen')
    assert drug.drug == 'ibuprofen'
    assert drug.evidence_quote == 'patients received ibuprofen'


def test_investigated_drug_missing_required_raises():
    with pytest.raises(ValidationError):
        ExtractedDrug()


def test_investigated_drug_dosages_is_list():
    drug = ExtractedDrug(
        drug='metformin',
        evidence_quote='metformin 500 mg twice daily',
        dosages=['500 mg twice daily', '1000 mg once daily'],
    )
    assert isinstance(drug.dosages, list)
    assert len(drug.dosages) == 2


def test_extracted_drug_modifier_fields_default_to_none():
    drug = ExtractedDrug(drug='metformin', evidence_quote='metformin was administered')
    assert drug.route is None
    assert drug.formulation is None


def test_extracted_drug_route_captures_inhaled():
    """Inhaled budesonide → drug='budesonide', route='inhaled'."""
    drug = ExtractedDrug(
        drug='budesonide',
        route='inhaled',
        evidence_quote='inhaled budesonide 400 mcg twice daily',
    )
    assert drug.route == 'inhaled'
    assert drug.formulation is None


def test_extracted_drug_route_and_formulation_independent():
    """'oral metformin tablet' → route='oral', formulation='tablet'."""
    drug = ExtractedDrug(
        drug='metformin',
        route='oral',
        formulation='tablet',
        evidence_quote='oral metformin tablet 500 mg',
    )
    assert drug.drug == 'metformin'
    assert drug.route == 'oral'
    assert drug.formulation == 'tablet'


def test_extracted_drug_modifiers_round_trip_json():
    """route/formulation survive JSON round-trip."""
    import json

    drug = ExtractedDrug(
        drug='metformin',
        route='oral',
        formulation='tablet',
        evidence_quote='oral metformin tablet',
    )
    data = json.loads(drug.model_dump_json())
    assert data['route'] == 'oral'
    assert data['formulation'] == 'tablet'
    restored = ExtractedDrug.model_validate(data)
    assert restored.route == 'oral'
    assert restored.formulation == 'tablet'


def test_extracted_disease_required_fields():
    disease = ExtractedDisease(name='type 2 diabetes', evidence_quote='patients with type 2 diabetes')
    assert disease.name == 'type 2 diabetes'
    assert disease.evidence_quote == 'patients with type 2 diabetes'


def test_extracted_disease_evidence_quote_optional():
    """background_conditions may omit evidence_quote when no standalone span exists."""
    disease = ExtractedDisease(name='colorectal cancer')
    assert disease.name == 'colorectal cancer'
    assert disease.evidence_quote is None


def test_extracted_disease_modifiers_are_separate_fields():
    disease = ExtractedDisease(
        name='peripheral neurotoxicity',
        evidence_quote='severe chronic oxaliplatin-induced peripheral neurotoxicity',
        severity='severe',
        stage='stage III',
        onset='chronic',
        etiology='oxaliplatin-induced',
    )
    assert disease.severity == 'severe'
    assert disease.stage == 'stage III'
    assert disease.onset == 'chronic'
    assert disease.etiology == 'oxaliplatin-induced'


def test_extracted_disease_etiology_defaults_to_none():
    disease = ExtractedDisease(name='lung cancer', evidence_quote='lung cancer')
    assert disease.etiology is None


def test_clinical_report_extraction_full():
    extraction = ClinicalReportExtraction(
        id='NCT04012606',
        drug_intent='therapeutic',
        drug_intent_confidence=0.95,
        primary_indications=[
            ExtractedDisease(
                name='type 2 diabetes',
                evidence_quote='patients with type 2 diabetes were enrolled',
            )
        ],
        investigated_drugs=[
            ExtractedDrug(drug='metformin', evidence_quote='metformin 500 mg was administered'),
        ],
    )
    assert extraction.id == 'NCT04012606'
    assert extraction.drug_intent == 'therapeutic'
    assert extraction.primary_indications[0].name == 'type 2 diabetes'
    assert len(extraction.investigated_drugs) == 1


def test_clinical_report_extraction_supports_multiple_primary_indications():
    """A trial studying NHL, ALL, and CLL in parallel should list all three."""
    extraction = ClinicalReportExtraction(
        id='NCT07166419',
        drug_intent='therapeutic',
        drug_intent_confidence=0.95,
        primary_indications=[
            ExtractedDisease(name='non-Hodgkin lymphoma', evidence_quote='Non-Hodgkin Lymphoma'),
            ExtractedDisease(
                name='acute lymphoblastic leukemia',
                evidence_quote='Acute Lymphoblastic Leukemia',
            ),
            ExtractedDisease(
                name='chronic lymphocytic leukemia',
                evidence_quote='Chronic Lymphocytic Leukemia',
            ),
        ],
        investigated_drugs=[
            ExtractedDrug(drug='TriCAR19.20.22 T cells', evidence_quote='TriCAR19.20.22 T cells'),
        ],
    )
    assert len(extraction.primary_indications) == 3


def test_clinical_report_extraction_diagnostic_drug_intent():
    """Diagnostic/imaging trials use drug_intent='diagnostic' to flag the relationship as detection."""
    extraction = ClinicalReportExtraction(
        id='NCT07218224',
        drug_intent='diagnostic',
        drug_intent_confidence=0.9,
        primary_indications=[
            ExtractedDisease(
                name='parathyroid adenomas',
                evidence_quote='localization of parathyroid adenomas',
            )
        ],
        investigated_drugs=[
            ExtractedDrug(drug='18F-fluorocholine', evidence_quote='18F-fluorocholine (FCH)'),
        ],
    )
    assert extraction.drug_intent == 'diagnostic'
    assert extraction.primary_indications[0].name == 'parathyroid adenomas'


def test_clinical_report_extraction_prevention_drug_intent():
    """Prevention trials: primary_indications is the prevented event, not the chronic condition."""
    extraction = ClinicalReportExtraction(
        id='NCT00000620',
        drug_intent='prevention',
        drug_intent_confidence=0.85,
        primary_indications=[
            ExtractedDisease(
                name='cardiovascular events',
                evidence_quote='prevent major cardiovascular events',
            )
        ],
        background_conditions=[
            ExtractedDisease(
                name='type 2 diabetes',
                evidence_quote='adults with type 2 diabetes',
            )
        ],
        investigated_drugs=[
            ExtractedDrug(drug='simvastatin', evidence_quote='simvastatin'),
        ],
    )
    assert extraction.drug_intent == 'prevention'
    assert extraction.background_conditions is not None
    assert extraction.background_conditions[0].name == 'type 2 diabetes'


def test_clinical_report_extraction_optional_fields_default_none():
    extraction = ClinicalReportExtraction(
        id='NCT00000001',
        drug_intent='therapeutic',
        drug_intent_confidence=0.95,
        primary_indications=[ExtractedDisease(name='asthma', evidence_quote='asthma patients')],
        investigated_drugs=[
            ExtractedDrug(drug='budesonide', evidence_quote='budesonide inhaler'),
        ],
    )
    assert extraction.background_conditions is None
    assert extraction.comparator_drugs is None
    assert extraction.supportive_drugs is None
    assert extraction.conclusion is None


def test_clinical_report_extraction_from_json():
    json_str = """{
        "id": "NCT00000001",
        "drug_intent": "therapeutic",
    "drug_intent_confidence": 0.95,
        "primary_indications": [{
            "name": "headache",
            "evidence_quote": "patients with chronic headache"
        }],
        "investigated_drugs": [
            {"drug": "aspirin", "evidence_quote": "aspirin 100mg was given"}
        ]
    }"""
    extraction = ClinicalReportExtraction.model_validate_json(json_str)
    assert extraction.id == 'NCT00000001'
    assert extraction.drug_intent == 'therapeutic'
    assert extraction.primary_indications[0].name == 'headache'
    assert extraction.investigated_drugs[0].drug == 'aspirin'


TRIAL_FIELDS = {
    'trialOfficialTitle': 'Official Title',
    'trialDescription': 'Description',
    'trialDetailedDescription': 'Detailed Description',
    'trialPhase': 'Phase',
    'trialOverallStatus': 'Overall Status',
    'trialPrimaryPurpose': 'Primary Purpose',
    'trialStudyType': 'Study Type',
    'trialNumberOfArms': 'Number of Arms',
    'trialWhyStopped': 'Why Stopped',
    'trialStartDate': 'Start Date',
}


def test_build_prompt_contains_id_and_trial_fields():
    from pts.transformers.clinical.providers.aact_llm import build_prompt

    row = {
        'id': 'NCT04012606',
        'trialOfficialTitle': 'A Phase 2 Study',
        'trialDescription': 'Tests aspirin',
        'trialDetailedDescription': 'Detailed protocol.',
        'trialPhase': 'PHASE_2',
        'trialOverallStatus': 'COMPLETED',
        'trialPrimaryPurpose': 'TREATMENT',
        'trialStudyType': 'INTERVENTIONAL',
        'trialNumberOfArms': 2,
        'trialWhyStopped': None,
        'trialLiterature': None,
        'trialStartDate': '2020-01-01',
    }
    prompt = build_prompt(row, trial_fields=TRIAL_FIELDS)
    assert 'NCT04012606' in prompt
    assert 'A Phase 2 Study' in prompt
    assert 'Detailed protocol.' in prompt
    assert 'PHASE_2' in prompt
    assert 'null' in prompt  # None values rendered as null


def test_build_prompt_handles_missing_trial_fields():
    from pts.transformers.clinical.providers.aact_llm import build_prompt

    row = {'id': 'NCT00000001'}  # no trial_* fields
    prompt = build_prompt(row, trial_fields=TRIAL_FIELDS)
    assert 'NCT00000001' in prompt


def test_build_prompt_includes_interventions():
    from pts.transformers.clinical.providers.aact_llm import build_prompt

    row = {
        'id': 'NCT00000001',
        'drugs': [
            {'drugFromSource': 'aspirin', 'drugId': None},
            {'drugFromSource': 'ibuprofen', 'drugId': None},
        ],
    }
    prompt = build_prompt(row, trial_fields=TRIAL_FIELDS)
    assert 'Interventions' in prompt
    assert 'aspirin' in prompt
    assert 'ibuprofen' in prompt


def test_build_prompt_omits_interventions_when_empty():
    from pts.transformers.clinical.providers.aact_llm import build_prompt

    row = {'id': 'NCT00000001', 'drugs': []}
    assert 'Interventions' not in build_prompt(row, trial_fields=TRIAL_FIELDS)
    row2 = {'id': 'NCT00000001'}
    assert 'Interventions' not in build_prompt(row2, trial_fields=TRIAL_FIELDS)
