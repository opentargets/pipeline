"""ClinVar somatic evidence: the `evidence_eva_somatic` output.

Generated from the 26.09 release output (8,875 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, RS_ID, SHA1_HEX


class EvidenceEvaSomatic(DatasetModel):
    """ClinVar somatic evidence."""

    dataset_name: ClassVar[str] = 'evidence_eva_somatic'
    dataset_title: ClassVar[str] = 'ClinVar somatic evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on somatic variation from ClinVar, provided by the European Variation Archive'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000112038',
        'id': '095fd1e3faf52f440389b85a677b49eb0d9758e9',
        'targetFromSourceId': 'ENSG00000112038',
        'diseaseFromSourceMappedId': 'EFO_0022041',
        'alleleOrigins': ['somatic'],
        'allelicRequirements': ['Somatic mutation'],
        'clinicalSignificances': ['drug response'],
        'cohortPhenotypes': ['Tramadol response'],
        'confidence': 'no assertion criteria provided',
        'datasourceId': 'eva_somatic',
        'datatypeId': 'somatic_mutation',
        'diseaseFromSource': 'Tramadol response',
        'diseaseFromSourceId': 'CN078023',
        'literature': ['12843323'],
        'releaseDate': '2020-03-29',
        'studyId': 'RCV001029145',
        'variantFromSourceId': 'VCV000829225',
        'variantFunctionalConsequenceId': 'SO_0001627',
        'variantHgvsId': 'NC_000006.12:g.154087849del',
        'variantId': '6_154087846_GT_G',
        'variantRsId': 'rs1583463133',
        'qualityControls': [],
        'diseaseId': 'EFO_0022041',
        'publicationDate': '2003-07-01',
        'evidenceDate': '2020-03-29',
        'score': 0.9,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    alleleOrigins: list[Literal['somatic']] = Field(min_length=1, description='Origin of the variant allele')
    allelicRequirements: (
        list[
            Literal[
                'Autosomal dominant inheritance',
                'Autosomal recessive inheritance',
                'Mitochondrial inheritance',
                'Semidominant inheritance',
                'Somatic mutation',
                'Sporadic',
                'Unknown mechanism',
                'X-linked dominant inheritance',
                'X-linked inheritance',
                'X-linked recessive inheritance',
            ]
        ]
        | None
    ) = Field(min_length=1, description='Inheritance patterns')
    clinicalSignificances: list[
        Literal[
            'affects',
            'association',
            'association not found',
            'benign',
            'conflicting classifications of pathogenicity',
            'drug response',
            'established risk allele',
            'likely benign',
            'likely pathogenic',
            'likely risk allele',
            'not provided',
            'other',
            'pathogenic',
            'risk factor',
            'uncertain risk allele',
            'uncertain significance',
        ]
    ] = Field(min_length=1, description='Standard terms to define clinical significance')
    cohortPhenotypes: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, description='Clinical features/phenotypes observed in studied individuals'
    )
    confidence: Literal[
        'criteria provided, conflicting classifications',
        'criteria provided, multiple submitters, no conflicts',
        'criteria provided, single submitter',
        'no assertion criteria provided',
        'no classification provided',
        'reviewed by expert panel',
    ] = Field(description='Confidence qualifier on the reported evidence')
    datasourceId: Literal['eva_somatic'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['somatic_mutation'] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    diseaseFromSourceId: str | None = Field(min_length=1, description='Disease identifier from the original source')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    releaseDate: str = Field(pattern=ISO_DATE, description="Date of the release of the data in a 'YYYY-MM-DD' format")
    studyId: str = Field(min_length=1, description='Identifier of the study generating the data')
    variantFromSourceId: str = Field(min_length=1, description='Identifier of the disease-causing variant at source')
    variantFunctionalConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] = Field(
        pattern=ONTOLOGY_ID,
        description=('Sequence ontology (SO) identifier of the functional consequence of the variant'),
    )
    variantHgvsId: str = Field(min_length=1, description='Identifier in HGVS notation of the disease-causing variant')
    variantId: Annotated[str, Reference('variant', 'variantId')] | None = Field(
        min_length=1, description=('Identifier in CHROM_POS_REF_ALT notation of the disease-causing variant')
    )
    variantRsId: str | None = Field(pattern=RS_ID, description='Variant reference SNP cluster ID (Rsid)')
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
        le=0.97,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
    directionOnTrait: Literal['risk'] | None = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['LoF'] | None = Field(description='Effect of direction on target')
