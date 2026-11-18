"""ClinVar somatic evidence: the `evidence_eva_somatic` output.

Generated from the 26.12 release output (8,875 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import Bioregistry, DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, RS_ID, SHA1_HEX


class EvidenceEvaSomatic(DatasetModel):
    """ClinVar somatic evidence."""

    dataset_name: ClassVar[str] = 'evidence_eva_somatic'
    dataset_title: ClassVar[str] = 'ClinVar somatic evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on somatic variation from ClinVar, provided by the European Variation Archive'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000103197',
        'id': 'f3e9df1a0d6faa12b3de493492da69c643a997be',
        'targetFromSourceId': 'ENSG00000103197',
        'diseaseFromSourceMappedId': 'MONDO_0013199',
        'alleleOrigins': ['somatic'],
        'datasourceId': 'eva_somatic',
        'datatypeId': 'somatic_mutation',
        'allelicRequirements': ['Somatic mutation'],
        'clinicalSignificances': ['pathogenic'],
        'confidence': 'criteria provided, single submitter',
        'studyId': 'RCV002266919',
        'releaseDate': '2022-07-23',
        'variantFunctionalConsequenceId': 'SO_0001574',
        'variantId': '16_2071509_G_A',
        'variantRsId': 'rs397514953',
        'variantFromSourceId': 'VCV000064936',
        'cohortPhenotypes': ['Tuberous sclerosis 2'],
        'diseaseFromSource': 'Tuberous sclerosis 2',
        'diseaseFromSourceId': 'C1860707',
        'variantHgvsId': 'NC_000016.10:g.2071509G>A',
        'literature': ['22658544'],
        'qualityControls': [],
        'diseaseId': 'MONDO_0013199',
        'publicationDate': '2012-05-31',
        'evidenceDate': '2022-07-23',
        'score': 0.92,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the gene the variant was assigned to by EVA'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    alleleOrigins: list[Literal['somatic']] = Field(
        description=("Origin(s) of the variant allele as reported in ClinVar (e.g. 'somatic')")
    )
    datasourceId: str = Field(pattern=ONTOLOGY_ID, description='Identifer of the evidence source')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    allelicRequirements: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='Mode(s) of inheritance reported in the ClinVar submissions'
    )
    clinicalSignificances: list[Annotated[str, Field(min_length=1)]] = Field(
        description=(
            'ClinVar clinical significance(s) of the variant for the condition '
            "(e.g. 'pathogenic', 'uncertain significance')"
        )
    )
    confidence: str = Field(
        min_length=1,
        description=(
            "ClinVar review status of the record (e.g. 'criteria provided, single "
            "submitter', 'reviewed by expert panel')"
        ),
    )
    studyId: str = Field(min_length=1, description='ClinVar reference (RCV) accession of the variant-condition record')
    releaseDate: str = Field(pattern=ISO_DATE, description="Date of the release of the data in a 'YYYY-MM-DD' format")
    variantFunctionalConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] = Field(
        pattern=ONTOLOGY_ID,
        description=('Sequence ontology (SO) identifier of the functional consequence of the variant'),
    )
    variantId: Annotated[str, Reference('variant', 'variantId')] | None = Field(
        min_length=1, description=('Identifier in CHROM_POS_REF_ALT notation of the disease-causing variant')
    )
    variantRsId: str | None = Field(pattern=RS_ID, description='Variant reference SNP cluster ID (Rsid)')
    variantFromSourceId: str = Field(min_length=1, description='ClinVar variation (VCV) accession of the variant')
    cohortPhenotypes: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Condition names submitted to ClinVar for the record'
    )
    diseaseFromSource: str = Field(min_length=1, description='Condition name as reported in ClinVar')
    diseaseFromSourceId: str | None = Field(
        min_length=1, description='MedGen identifier of the condition as reported in ClinVar'
    )
    variantHgvsId: str = Field(min_length=1, description='Genomic HGVS notation of the variant as reported in ClinVar')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='PubMed identifiers cited in the ClinVar submissions'
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship, derived from the ClinVar clinical significance and '
            'review status'
        ),
    )
    directionOnTrait: Literal['risk', 'protect'] | None = Field(
        description='Predicted direction of effect on the trait'
    )
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
