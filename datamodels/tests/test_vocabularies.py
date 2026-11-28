"""The closed vocabularies and the fields that store them as Enums."""

from typing import Any, get_args

import polars as pl
import pytest

from ot_datamodels import vocabularies as v
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import Node, dataset_spec, walk

DATASETS = discover_datasets()
NODES = {f'{name}.{path}': node for name, model in DATASETS.items() for path, node in walk(dataset_spec(model))}

EVIDENCE = [
    'cancer_biomarkers',
    'cancer_gene_census',
    'clingen',
    'clinical_precedence',
    'crispr',
    'crispr_screen',
    'encore',
    'europepmc',
    'eva',
    'eva_somatic',
    'expression_atlas',
    'gene2phenotype',
    'gene_burden',
    'genomics_england',
    'gwas_credible_sets',
    'impc',
    'intogen',
    'orphanet',
    'ot_crispr',
    'ot_crispr_validation',
    'reactome',
    'uniprot_literature',
    'uniprot_variants',
]
ASSOCIATIONS = [
    f'association_{kind}_{scope}'
    for kind in ('overall', 'by_datasource', 'by_datatype')
    for scope in ('direct', 'indirect')
]

#: every field stored as a vocabulary's Enum
FIELDS: dict[str, Any] = {
    **{f'{a}.aggregationType': v.AggregationType for a in ASSOCIATIONS},
    **{f'association_overall_{s}.aggregationValue': v.OverallAggregationValue for s in ('direct', 'indirect')},
    **{f'association_by_datasource_{s}.aggregationValue': v.Datasource for s in ('direct', 'indirect')},
    **{f'association_by_datatype_{s}.aggregationValue': v.Datatype for s in ('direct', 'indirect')},
    **dict.fromkeys(
        (
            'colocalisation.chromosome',
            'credible_set.chromosome',
            'enhancer_to_gene.chromosome',
            'target.genomicLocation/chromosome',
            'transcript.exons[]/chromosome',
            'transcript.chromosome',
            'variant.chromosome',
        ),
        v.Chromosome,
    ),
    'clinical_report.clinicalStage': v.ClinicalStage,
    'evidence_clinical_precedence.clinicalStage': v.ClinicalStage,
    'clinical_indication.maxClinicalStage': v.ClinicalStage,
    'clinical_target.maxClinicalStage': v.ClinicalStage,
    'drug_molecule.maximumClinicalStage': v.ClinicalStage,
    'colocalisation.colocalisationMethod': v.ColocalisationMethod,
    'target.constraint[]/constraintType': v.ConstraintType,
    'drug_mechanism_of_action.curation': v.MechanismCuration,
    **{f'evidence_{e}.datasourceId': v.Datasource for e in EVIDENCE},
    'baseline_expression.datasourceId': v.ExpressionDatasource,
    'baseline_expression.datatypeId': v.ExpressionDatatype,
    'enhancer_to_gene.datasourceId': v.IntervalDatasource,
    'pharmacogenomics.datasourceId': v.PharmacogenomicsDatasource,
    'pharmacogenomics.datatypeId': v.PharmacogenomicsDatatype,
    **{f'evidence_{e}.datatypeId': v.Datatype for e in EVIDENCE},
    **{
        f'evidence_{e}.directionOnTarget': v.DirectionOnTarget
        for e in (
            'cancer_gene_census',
            'clinical_precedence',
            'eva',
            'eva_somatic',
            'gene2phenotype',
            'gene_burden',
            'impc',
            'intogen',
            'orphanet',
        )
    },
    **{
        f'evidence_{e}.directionOnTrait': v.DirectionOnTrait
        for e in (
            'cancer_gene_census',
            'clinical_precedence',
            'eva',
            'eva_somatic',
            'gene2phenotype',
            'gene_burden',
            'impc',
            'intogen',
            'orphanet',
        )
    },
    'biosample_enrichment.enrichmentApproach': v.EnrichmentApproach,
    'credible_set.fineMappingMethod': v.FineMappingMethod,
    'enhancer_to_gene.intervalType': v.IntervalType,
    'literature_entity_lut.keywordType': v.LiteratureKeywordType,
    'variant.transcriptConsequences[]/lofteePrediction': v.LofteePrediction,
    'variant.variantEffect[]/method': v.VariantEffectMethod,
    'interaction.speciesA/mnemonic': v.SpeciesMnemonic,
    'interaction.speciesA/scientificName': v.SpeciesScientificName,
    'clinical_report.origin': v.Origin,
    'clinical_report.provider': v.Provider,
    'disease_phenotype.evidence[]/resource': v.HpoResource,
    'colocalisation.rightStudyType': v.StudyType,
    'credible_set.studyType': v.StudyType,
    'study.studyType': v.StudyType,
    'evidence_europepmc.textMiningSentences[]/section': v.EuropePmcSection,
    'interaction.sourceDatabase': v.InteractionSource,
    'interaction_evidence.interactionResources/sourceDatabase': v.InteractionSource,
    'clinical_report.trialStudyType': v.TrialStudyType,
    'clinical_report.trialLiterature[]/type': v.TrialReferenceType,
    'clinical_report.type': v.ClinicalReportType,
    'target_safety_event.studies[]/type': v.SafetyStudyType,
    'drug_mechanism_of_action.actionType': v.ChemblActionType,
    'chembl_bioactivity.assayActionType': v.ChemblActionType,
    'chembl_bioactivity.activityOutcome': v.ChemblActivityOutcome,
    'clinical_report.trialSponsor/agencyClass': v.TrialAgencyClass,
    'variant.transcriptConsequences[]/appris': v.Appris,
    'disease_phenotype.evidence[]/aspect': v.HpoAspect,
    'target.go[]/aspect': v.GoAspect,
    'chembl_bioactivity.assayType': v.ChemblAssayType,
    'chembl_bioactivity.docType': v.ChemblDocumentType,
    'drug_molecule.drugType': v.ChemblMoleculeType,
    'pharmacogenomics.variantAnnotation[]/effectType': v.PgxEffectType,
    'target.go[]/evidence': v.GoEvidenceCode,
    'pharmacogenomics.evidenceLevel': v.PgxEvidenceLevel,
    'disease_phenotype.evidence[]/evidenceType': v.HpoEvidenceCode,
    'homology.homologyType': v.HomologyType,
    'study.ldPopulationStructure[]/ldPopulation': v.LdPopulation,
    'chembl_bioactivity.measurementClass': v.ChemblMeasurementClass,
    'target_tractability.modality': v.TractabilityModality,
    'go.namespace': v.GoNamespace,
    'pharmacogenomics.pgxCategory': v.PgxCategory,
    'variant.alleleFrequencies[]/populationName': v.GnomadPopulation,
    'disease_phenotype.evidence[]/sex': v.HpoSex,
    'chembl_bioactivity.standardMeasurementType': v.ChemblStandardType,
    'drug_mechanism_of_action.targetType': v.ChemblTargetType,
    'clinical_report.trialOverallStatus': v.TrialOverallStatus,
    'clinical_report.trialPhase': v.TrialPhase,
    'clinical_report.trialPrimaryPurpose': v.TrialPrimaryPurpose,
    'transcript.flags[]/value': v.TranscriptFlagValue,
    'drug_warning.warningType': v.ChemblWarningType,
    'clinical_report.trialStopReasonCategories[]': v.TrialStopReasonCategory,
    'evidence_clinical_precedence.trialStopReasonCategories[]': v.TrialStopReasonCategory,
    'chembl_bioactivity.actionTypes[]': v.ChemblActionType,
    'evidence_gene_burden.sex[]': v.GeneBurdenSex,
    'transcript.flags[]/label': v.TranscriptFlagLabel,
    'transcript.biotype': v.TranscriptBiotype,
    'credible_set.qualityControls[]': v.CredibleSetQualityControl,
    'credible_set_variant.qualityControls[]': v.VariantQualityControl,
    'clinical_report.source': v.ClinicalSource,
}

#: API `Datasource`, in its order (platform-api-nextgen schema.graphql)
API_DATASOURCES = [
    'clinical_precedence',
    'gwas_credible_sets',
    'gene_burden',
    'eva',
    'genomics_england',
    'gene2phenotype',
    'uniprot_literature',
    'uniprot_variants',
    'orphanet',
    'clingen',
    'cancer_gene_census',
    'intogen',
    'eva_somatic',
    'cancer_biomarkers',
    'crispr_screen',
    'crispr',
    'reactome',
    'europepmc',
    'expression_atlas',
    'impc',
]


def _values(vocabulary: Any) -> list[str]:
    return list(get_args(vocabulary))


def test_today_s_clickhouse_enums_keep_their_order() -> None:
    assert _values(v.Chromosome) == [str(n) for n in range(1, 23)] + ['X', 'Y', 'MT']
    assert _values(v.StudyType) == ['tuqtl', 'pqtl', 'eqtl', 'sqtl', 'sctuqtl', 'scpqtl', 'sceqtl', 'scsqtl', 'gwas']
    assert _values(v.InteractionSource) == ['intact', 'reactome', 'signor', 'string']
    assert _values(v.FineMappingMethod) == ['PICS', 'SuSiE', 'SuSiE-inf']


def test_api_backed_vocabularies_follow_the_api_order() -> None:
    assert _values(v.ClinicalStage) == [
        'UNKNOWN',
        'PRECLINICAL',
        'IND',
        'EARLY_PHASE_1',
        'PHASE_1',
        'PHASE_1_2',
        'PHASE_2',
        'PHASE_2_3',
        'PHASE_3',
        'PREAPPROVAL',
        'APPROVAL',
        'PHASE_4',
        'WITHDRAWAL',
    ]
    assert _values(v.Provider) == ['ChEMBL', 'TTD', 'AACT', 'EMA', 'PMDA', 'FDA']
    assert _values(v.Origin) == ['CLINICAL_TRIAL', 'DRUG_LABEL', 'REGULATORY_AGENCY', 'CURATED_RESOURCE']
    assert _values(v.TrialStudyType) == ['INTERVENTIONAL', 'OBSERVATIONAL', 'EXPANDED_ACCESS']
    assert _values(v.AggregationType) == ['overall', 'datasourceId', 'datatypeId']
    assert _values(v.Datasource)[: len(API_DATASOURCES)] == API_DATASOURCES


def test_datasource_is_exactly_the_evidence_datasources() -> None:
    assert sorted(_values(v.Datasource)) == sorted(EVIDENCE)
    assert set(_values(v.AssociationDatatype)) <= set(_values(v.Datatype))


@pytest.mark.parametrize('name', [n for n in v.__all__ if n != 'enum_of'])
def test_a_vocabulary_has_no_repeated_value_and_fits_enum8(name: str) -> None:
    values = _values(getattr(v, name))

    assert len(values) == len(set(values))
    assert 0 < len(values) <= 127


def test_every_vocabulary_field_is_modelled() -> None:
    assert sorted(set(FIELDS) - set(NODES)) == []


@pytest.mark.parametrize('path', sorted(FIELDS))
def test_a_vocabulary_field_stores_its_enum(path: str) -> None:
    node: Node = NODES[path]
    values = _values(FIELDS[path])

    assert node.dtype == pl.Enum(values)
    assert node.constraints.allowed is not None
    assert set(node.constraints.allowed) <= set(values)


def test_an_evidence_model_allows_only_its_own_datasource_and_datatype() -> None:
    for e in EVIDENCE:
        assert NODES[f'evidence_{e}.datasourceId'].constraints.allowed == (e,)
        allowed = NODES[f'evidence_{e}.datatypeId'].constraints.allowed
        assert allowed is not None
        assert len(allowed) == 1, e


def test_homology_high_confidence_is_a_bool() -> None:
    assert NODES['homology.isHighConfidence'].dtype == pl.Boolean()
