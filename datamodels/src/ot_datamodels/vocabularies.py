"""The closed vocabularies of the release, each a `Literal` in its stored order.

A field uses one as `Annotated[Vocabulary, PolarsDtype(pl.Enum)]`: the categories, and the generated
ClickHouse `Enum8` numbering, follow the order here. A field restricted to part of a vocabulary keeps its
own `Literal` and stores the whole vocabulary's Enum: `Annotated[Literal['eva'], enum_of(Datasource)]`.
Where order carries no meaning, a new value is appended, so no existing value is renumbered.

Order: the ClickHouse Enum order where one existed, else the API's GraphQL enum, else the standard's
published order, else sorted.
"""

from typing import Any, Literal, get_args

import polars as pl

from ot_datamodels.base import PolarsDtype

__all__ = [
    'AggregationType',
    'Appris',
    'AssociationDatatype',
    'ChemblActionType',
    'ChemblActivityOutcome',
    'ChemblAssayType',
    'ChemblDocumentType',
    'ChemblMeasurementClass',
    'ChemblMoleculeType',
    'ChemblStandardType',
    'ChemblTargetType',
    'ChemblWarningType',
    'Chromosome',
    'ClinicalReportType',
    'ClinicalSource',
    'ClinicalStage',
    'CodingVariantDatasource',
    'ColocalisationMethod',
    'ConstraintType',
    'CredibleSetQualityControl',
    'Datasource',
    'Datatype',
    'DirectionOnTarget',
    'DirectionOnTrait',
    'EnrichmentApproach',
    'EuropePmcSection',
    'ExpressionDatasource',
    'ExpressionDatatype',
    'FineMappingMethod',
    'GeneBurdenSex',
    'GnomadPopulation',
    'GoAspect',
    'GoEvidenceCode',
    'GoNamespace',
    'HomologyType',
    'HpoAspect',
    'HpoEvidenceCode',
    'HpoResource',
    'HpoSex',
    'InteractionSource',
    'IntervalDatasource',
    'IntervalType',
    'LdPopulation',
    'LiteratureKeywordType',
    'LofteePrediction',
    'MechanismCuration',
    'Origin',
    'OverallAggregationValue',
    'PgxCategory',
    'PgxEffectType',
    'PgxEvidenceLevel',
    'PharmacogenomicsDatasource',
    'PharmacogenomicsDatatype',
    'Provider',
    'SafetyStudyType',
    'SpeciesMnemonic',
    'SpeciesScientificName',
    'StudyType',
    'TractabilityModality',
    'TranscriptBiotype',
    'TranscriptFlagLabel',
    'TranscriptFlagValue',
    'TrialAgencyClass',
    'TrialOverallStatus',
    'TrialPhase',
    'TrialPrimaryPurpose',
    'TrialReferenceType',
    'TrialStopReasonCategory',
    'TrialStudyType',
    'VariantEffectMethod',
    'VariantQualityControl',
    'enum_of',
]


def enum_of(vocabulary: Any) -> PolarsDtype:
    """The Enum of a whole vocabulary, for a field whose `Literal` allows only part of it."""
    return PolarsDtype(pl.Enum(list(get_args(vocabulary))))


# A: the hand-written ClickHouse Enums before phase 5

Chromosome = Literal[
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', 'X', 'Y', 'MT',
]  # fmt: skip
StudyType = Literal['tuqtl', 'pqtl', 'eqtl', 'sqtl', 'sctuqtl', 'scpqtl', 'sceqtl', 'scsqtl', 'gwas']
InteractionSource = Literal['intact', 'reactome', 'signor', 'string']
FineMappingMethod = Literal['PICS', 'SuSiE', 'SuSiE-inf']

# B: Open Targets' own vocabularies

# Evidence datasources: API `Datasource` order, then those it lacks (PPP evidence, from pts `association`
# datasource_weights); append new ones
Datasource = Literal[
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
    'encore',
    'ot_crispr',
    'ot_crispr_validation',
]
# Evidence datatypes, sorted: the evidence `datatypeId`s and pts `association` datasource_weights; append new ones
Datatype = Literal[
    'affected_pathway',
    'animal_model',
    'clinical',
    'genetic_association',
    'genetic_literature',
    'known_drug',
    'literature',
    'ot_partner',
    'ot_validation_lab',
    'rna_expression',
    'somatic_mutation',
]
# pts `association` datasource_weights
AssociationDatatype = Literal[
    'affected_pathway',
    'animal_model',
    'genetic_association',
    'genetic_literature',
    'known_drug',
    'literature',
    'ot_partner',
    'ot_validation_lab',
    'rna_expression',
    'somatic_mutation',
]
# baseline expression's sources and assay types
ExpressionDatasource = Literal['DICE', 'PRIDE', 'gtex', 'tabula_sapiens']
ExpressionDatatype = Literal['bulk rna-seq', 'mass-spectrometry proteomics', 'scrna-seq']
# enhancer-to-gene interval source
IntervalDatasource = Literal['E2G']
# pharmacogenomics provider and data type
PharmacogenomicsDatasource = Literal['clinpgx']
PharmacogenomicsDatatype = Literal['clinical_annotation']
# pts `coding_variant` datasources: its variant evidence, GWAS credible sets, `mol_qtl` and pharmacogenomics;
# sorted, as pts sorts `datasources[]` by it
CodingVariantDatasource = Literal['clinpgx', 'eva', 'eva_somatic', 'gwas_credible_sets', 'mol_qtl', 'uniprot_variants']
# API `AggregationType`, then datatypeId
AggregationType = Literal['overall', 'datasourceId', 'datatypeId']
OverallAggregationValue = Literal['None']
# API `ClinicalStage`
ClinicalStage = Literal[
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
# API `Provider` (which spells ChEMBL `CHEMBL`), then FDA, which the API lacks
Provider = Literal['ChEMBL', 'TTD', 'AACT', 'EMA', 'PMDA', 'FDA']
# API `Origin`
Origin = Literal['CLINICAL_TRIAL', 'DRUG_LABEL', 'REGULATORY_AGENCY', 'CURATED_RESOURCE']
ClinicalReportType = Literal['INDICATION', 'SAFETY']
# pts `clinical_report` sources: clinical_mira `ClinicalSource` and the ChEMBL indication and warning ref_types;
# the 2612-3 values, sorted
ClinicalSource = Literal[
    'ATC',
    'ClinicalTrials.gov',
    'DOI',
    'DailyMed',
    'Drugs@FDA',
    'EMA',
    'EMA Human Drugs',
    'FDA',
    'HIS',
    'HPRA',
    'Health Canada',
    'INN',
    'MEDSAFE',
    'MHRA',
    'NICE',
    'NTP',
    'Other',
    'PMDA',
    'PubMed',
    'TGA',
    'TTD',
    'USAN',
    'USGPO',
    'WHO',
]
DirectionOnTarget = Literal['GoF', 'LoF']
DirectionOnTrait = Literal['protect', 'risk']
ColocalisationMethod = Literal['COLOC_PIP_ECAVIAR']
ConstraintType = Literal['lof', 'mis', 'syn']
MechanismCuration = Literal['electronic', 'manual']
EnrichmentApproach = Literal['l2g_regression']
# sorted, then the types pts accepts that 2612-3 lacks; append new ones
IntervalType = Literal['genic', 'intergenic', 'promoter', 'enhancer', 'intragenic']
LiteratureKeywordType = Literal['CD', 'DS', 'GP']
LofteePrediction = Literal['HC', 'LC']
VariantEffectMethod = Literal[
    'AlphaMissense', 'FoldX', 'GERP', 'GPN-Star', 'LOFTEE', 'LossOfFunctionCuration', 'SIFT', 'VEP'
]
SpeciesMnemonic = Literal['human']
SpeciesScientificName = Literal['Homo sapiens']
HpoResource = Literal['HPO']
EuropePmcSection = Literal[
    'abstract', 'appendix', 'case', 'concl', 'discuss', 'figure', 'intro', 'other', 'results', 'table', 'title'
]
SafetyStudyType = Literal['biochemical', 'cell-based', 'clinical', 'preclinical']
# labels of the opentargets/clinical_trial_stop_reasons classifier (rev 5594f9e5, config.json id2label), then
# pts's `Uncategorised` (clinical_mira/stop_reasons.py)
TrialStopReasonCategory = Literal[
    'Another_Study',
    'Business_Administrative',
    'Covid19',
    'Endpoint_Met',
    'Ethical_Reason',
    'Insufficient_Data',
    'Insufficient_Enrollment',
    'Interim_Analysis',
    'Invalid_Reason',
    'Logistics_Resources',
    'Negative',
    'No_Context',
    'Regulatory',
    'Safety_Sideeffects',
    'Study_Design',
    'Study_Staff_Moved',
    'Success',
    'Uncategorised',
]
# pts `transcript` flag labels (_STATIC_FLAGS, then appris)
TranscriptFlagLabel = Literal['appris', 'ensembl_canonical', 'gencode_primary', 'mane']
# pts `credible_set` flags (credible_set/flags.py CredibleSetFlag), in its order
CredibleSetQualityControl = Literal[
    'Explained by a more significant variant in the same window',
    'Explained by a more significant variant in high LD',
    'Study does not have population annotation to resolve LD',
    'Variant not found in LD reference',
    'Study locus finemapped without in-sample LD reference',
    'Subsignificant p-value',
    'Composite association',
    'Variant inconsistency',
    'Palindrome alleles - cannot harmonize',
    'Study locus from curated top hit',
    'Trans QTL',
    'Lead variant is replicated in an independent study',
    'Study not found in the study index',
    'Curated top hit is flagged because summary statistics are available for study',
    'MHC region',
    'Chromosome not in 1:22, X, Y or MT',
    'PICS results from summary statistics available for this same study',
    'Study locus in region explained by a SuSiE credible set',
    'Credible set identifier repeated within its source',
    'Credible set identifier also in another source, whose copy is kept',
    'Credible set has a variant with an excluding quality flag',
    'Credible set identifier is not a 32-character hexadecimal md5',
    'Chromosome of the lead variant differs from the credible set chromosome',
    'Fine-mapping method is not supported',
    'Study type of the source is not a supported study type',
    'Study type of the credible set differs from the study index',
    'Lead variant of the credible set has no row among its variants',
]
# pts `credible_set_variant` flags (credible_set/flags.py CredibleSetVariantFlag), in its order
VariantQualityControl = Literal[
    'Allele frequency reported by the study',
    'Allele frequency taken from gnomAD',
    'Allele frequency is missing from the study and from gnomAD',
    'Allele frequency reported by the study is not in [0, 1]',
    'Rescaled effect derived from the conditional effect',
    'Rescaled effect derived from the marginal effect',
    'Variant in LD with the lead has no statistics in the locus',
    'Variant identifier is not a valid chromosome_position_ref_alt',
    'Variant appears more than once in its credible set',
    'Credible set of the variant is excluded',
    'Credible set identifier is not a 32-character hexadecimal md5',
    'Variant is outside the 95% coverage of the credible set',
    'Credible set PIPs do not sum to the expected coverage, or a PIP is not in (0, 1]',
    'Reported and conditional effects have opposite signs',
    'Only one of the reported beta and standard error is present',
    'Only one of the conditional beta and standard error is present',
    'P-value mantissa or exponent is missing',
    'Credible set of the variant is not among the credible sets of its source',
    'Chromosome of the variant differs from the credible set chromosome',
]
# gene burden sample sex, as the gene burden sources write it
GeneBurdenSex = Literal['men', 'women']
# API `TrialStudyType`; ClinicalTrials.gov StudyType
TrialStudyType = Literal['INTERVENTIONAL', 'OBSERVATIONAL', 'EXPANDED_ACCESS']
# ClinicalTrials.gov ReferenceType, https://clinicaltrials.gov/api/v2/studies/enums (2026-11-22)
TrialReferenceType = Literal['BACKGROUND', 'RESULT', 'DERIVED']

# C: external standards

# ClinicalTrials.gov API v2 enums, https://clinicaltrials.gov/api/v2/studies/enums (2026-11-22)
TrialOverallStatus = Literal[
    'ACTIVE_NOT_RECRUITING',
    'COMPLETED',
    'ENROLLING_BY_INVITATION',
    'NOT_YET_RECRUITING',
    'RECRUITING',
    'SUSPENDED',
    'TERMINATED',
    'WITHDRAWN',
    'AVAILABLE',
    'NO_LONGER_AVAILABLE',
    'TEMPORARILY_NOT_AVAILABLE',
    'APPROVED_FOR_MARKETING',
    'WITHHELD',
    'UNKNOWN',
]
# Phase, with AACT's combined phases; pts nulls NA
TrialPhase = Literal['EARLY_PHASE1', 'PHASE1', 'PHASE1/PHASE2', 'PHASE2', 'PHASE2/PHASE3', 'PHASE3', 'PHASE4']
TrialPrimaryPurpose = Literal[
    'TREATMENT',
    'PREVENTION',
    'DIAGNOSTIC',
    'ECT',
    'SUPPORTIVE_CARE',
    'SCREENING',
    'HEALTH_SERVICES_RESEARCH',
    'BASIC_SCIENCE',
    'DEVICE_FEASIBILITY',
    'OTHER',
]
TrialAgencyClass = Literal['NIH', 'FED', 'OTHER_GOV', 'INDIV', 'INDUSTRY', 'NETWORK', 'AMBIG', 'OTHER', 'UNKNOWN']

# ChEMBL 37 (2026-05), terms of the chembl_mechanism and chembl_activity action_type fields of
# https://www.ebi.ac.uk/chembl/elk/es; sorted, OTHER last as in pts's ActionType
ChemblActionType = Literal[
    'ACTIVATOR',
    'AGONIST',
    'ALLOSTERIC ANTAGONIST',
    'ANTAGONIST',
    'ANTISENSE INHIBITOR',
    'BINDING AGENT',
    'BLOCKER',
    'CHELATING AGENT',
    'CROSS-LINKING AGENT',
    'DEGRADER',
    'DISRUPTING AGENT',
    'EXOGENOUS GENE',
    'EXOGENOUS PROTEIN',
    'GENE EDITING NEGATIVE MODULATOR',
    'HYDROLYTIC ENZYME',
    'INHIBITOR',
    'INVERSE AGONIST',
    'METHYLATING AGENT',
    'MODULATOR',
    'NEGATIVE ALLOSTERIC MODULATOR',
    'NEGATIVE MODULATOR',
    'OPENER',
    'OXIDATIVE ENZYME',
    'PARTIAL AGONIST',
    'POSITIVE ALLOSTERIC MODULATOR',
    'POSITIVE MODULATOR',
    'PROTEOLYTIC ENZYME',
    'REDUCING AGENT',
    'RELEASING AGENT',
    'RNAI INHIBITOR',
    'SEQUESTERING AGENT',
    'STABILISER',
    'SUBSTRATE',
    'VACCINE ANTIGEN',
    'OTHER',
]
# ChEMBL 37 target_type terms of chembl_target, lowercased as pts writes them; sorted, then pts's `other`
ChemblTargetType = Literal[
    '3d cell culture',
    'admet',
    'cell-line',
    'chimeric protein',
    'lipid',
    'macromolecule',
    'metal',
    'no target',
    'non-molecular',
    'nucleic-acid',
    'oligosaccharide',
    'organism',
    'phenotype',
    'protein complex',
    'protein complex group',
    'protein family',
    'protein nucleic-acid complex',
    'protein-protein interaction',
    'selectivity group',
    'single protein',
    'small molecule',
    'subcellular',
    'tissue',
    'unchecked',
    'unknown',
    'other',
]
# ChEMBL 37 assay_type: binding, functional, ADMET, toxicity, physicochemical, unclassified
ChemblAssayType = Literal['B', 'F', 'A', 'T', 'P', 'U']
# pts `chembl_bioactivity`: ACTIVITY_OUTCOMES, MEASUREMENT_CLASSES and its keys (the pChEMBL standard types)
ChemblActivityOutcome = Literal['active', 'inactive', 'inconclusive']
ChemblMeasurementClass = Literal['affinity', 'functional_potency', 'inhibition']
ChemblStandardType = Literal['AC50', 'EC50', 'ED50', 'IC50', 'Kd', 'Ki', 'Potency', 'XC50']
# ChEMBL 37 doc_type terms of chembl_document, sorted
ChemblDocumentType = Literal['BOOK', 'DATASET', 'PATENT', 'PUBLICATION']
# ChEMBL 37 molecule_type terms of chembl_molecule, sorted
ChemblMoleculeType = Literal[
    'Antibody',
    'Antibody drug conjugate',
    'Cell',
    'Enzyme',
    'Gene',
    'Oligonucleotide',
    'Oligosaccharide',
    'Protein',
    'Small molecule',
    'Unknown',
    'Vaccine component',
]
# ChEMBL 37 warning_type terms of chembl_drug_warning
ChemblWarningType = Literal['Black Box Warning', 'Withdrawn']

# GO evidence codes, https://geneontology.org/docs/guide-go-evidence-codes/ (2026-11-22)
GoEvidenceCode = Literal[
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP',
    'IBA', 'IBD', 'IKR', 'IRD',
    'ISS', 'ISO', 'ISA', 'ISM', 'IGC', 'RCA',
    'TAS', 'NAS',
    'IC', 'ND',
    'IEA',
]  # fmt: skip
# GAF 2.2 column 9, https://geneontology.org/docs/go-annotation-file-gaf-format-2.2/
GoAspect = Literal['P', 'F', 'C']
GoNamespace = Literal['biological_process', 'molecular_function', 'cellular_component']
# phenotype.hpoa, https://obophenotype.github.io/human-phenotype-ontology/annotations/phenotype_hpoa/ (2026-11-22)
HpoAspect = Literal['P', 'C', 'I', 'M', 'H']
HpoEvidenceCode = Literal['IEA', 'TAS', 'PCS']
HpoSex = Literal['MALE', 'FEMALE']

# Ensembl Compara 116 homology.description,
# https://github.com/Ensembl/ensembl-compara/blob/release/116/sql/table.sql
HomologyType = Literal[
    'ortholog_one2one',
    'ortholog_one2many',
    'ortholog_many2many',
    'within_species_paralog',
    'other_paralog',
    'gene_split',
    'between_species_paralog',
    'alt_allele',
    'homoeolog_one2one',
    'homoeolog_one2many',
    'homoeolog_many2many',
]
# Ensembl appris transcript attributes (principal1-5, alternative1-2, and the M forms Ensembl 116 also
# writes), as VEP abbreviates them
Appris = Literal['P1', 'P2', 'P3', 'P4', 'P5', 'PM', 'A1', 'A2', 'AM']
# Ensembl 116 transcript biotypes of the human GTF (Homo_sapiens.GRCh38.116.gtf.gz), sorted
TranscriptBiotype = Literal[
    'IG_C_gene',
    'IG_C_pseudogene',
    'IG_D_gene',
    'IG_J_gene',
    'IG_J_pseudogene',
    'IG_V_gene',
    'IG_V_pseudogene',
    'IG_pseudogene',
    'Mt_rRNA',
    'Mt_tRNA',
    'TEC',
    'TR_C_gene',
    'TR_D_gene',
    'TR_J_gene',
    'TR_J_pseudogene',
    'TR_V_gene',
    'TR_V_pseudogene',
    'artifact',
    'lncRNA',
    'miRNA',
    'misc_RNA',
    'non_stop_decay',
    'nonsense_mediated_decay',
    'processed_pseudogene',
    'processed_transcript',
    'protein_coding',
    'protein_coding_CDS_not_defined',
    'protein_coding_LoF',
    'pseudogene',
    'rRNA',
    'rRNA_pseudogene',
    'retained_intron',
    'ribozyme',
    'sRNA',
    'scaRNA',
    'snRNA',
    'snoRNA',
    'transcribed_processed_pseudogene',
    'transcribed_unitary_pseudogene',
    'transcribed_unprocessed_pseudogene',
    'translated_processed_pseudogene',
    'unitary_pseudogene',
    'unprocessed_pseudogene',
    'vault_RNA',
]
# pts `transcript` flags: MANE (select, plus_clinical), gencode_primary and ensembl_canonical (true), and
# the appris attributes spelled as GENCODE tags them
TranscriptFlagValue = Literal[
    'select',
    'plus_clinical',
    'true',
    'principal_1',
    'principal_2',
    'principal_3',
    'principal_4',
    'principal_5',
    'principal_M',
    'alternative_1',
    'alternative_2',
    'alternative_M',
]

# gnomAD populations, gentropy v3.4.0 `common/types.py` (LDPopulation; VariantPopulation with `_adj`)
LdPopulation = Literal['afr', 'amr', 'asj', 'eas', 'nfe', 'est', 'fin', 'nwe', 'seu', 'csa']
GnomadPopulation = Literal[
    'afr_adj',
    'amr_adj',
    'ami_adj',
    'asj_adj',
    'eas_adj',
    'fin_adj',
    'nfe_adj',
    'mid_adj',
    'sas_adj',
    'remaining_adj',
]

# ClinPGx levels of evidence, https://pmc.ncbi.nlm.nih.gov/articles/PMC8457105/ (table 2)
PgxEvidenceLevel = Literal['1A', '1B', '2A', '2B', '3', '4']
# ClinPGx phenotype categories and variant annotation types, lowercased by the evidence parser; no
# fetchable published list, so the 2612-3 values
PgxCategory = Literal['dosage', 'efficacy', 'metabolism/pk', 'other', 'pd', 'toxicity']
PgxEffectType = Literal['drug', 'functional', 'phenotype']

# Open Targets tractability: small molecule, antibody, PROTAC, other clinical modalities
TractabilityModality = Literal['SM', 'AB', 'PR', 'OC']
