"""ClinVar germline evidence: the `evidence_eva` output.

Generated from the 26.09 release output (3,667,606 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, RS_ID, SHA1_HEX


class EvidenceEva(DatasetModel):
    """ClinVar germline evidence."""

    dataset_name: ClassVar[str] = 'evidence_eva'
    dataset_title: ClassVar[str] = 'ClinVar germline evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on germline variation from ClinVar, provided by the European Variation Archive'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000075043',
        'id': '00390b0868a26b434bd38c144b01b8d713b319ee',
        'targetFromSourceId': 'ENSG00000075043',
        'diseaseFromSourceMappedId': 'MONDO_0800491',
        'alleleOrigins': ['germline'],
        'allelicRequirements': ['X-linked inheritance'],
        'clinicalSignificances': ['pathogenic'],
        'cohortPhenotypes': ['Early infantile epileptic encephalopathy'],
        'confidence': 'criteria provided, single submitter',
        'datasourceId': 'eva',
        'datatypeId': 'genetic_association',
        'diseaseFromSource': 'Early-infantile DEE',
        'diseaseFromSourceId': 'C0393706',
        'literature': ['23623386'],
        'releaseDate': '2026-02-23',
        'studyId': 'RCV006561402',
        'variantFromSourceId': 'VCV002572483',
        'variantFunctionalConsequenceId': 'SO_0001583',
        'variantHgvsId': 'NC_000020.11:g.63439602G>A',
        'variantId': '20_63439602_G_A',
        'variantRsId': 'rs2516418427',
        'qualityControls': [],
        'diseaseId': 'EFO_1000643',
        'publicationDate': '2013-04-25',
        'evidenceDate': '2026-02-23',
        'score': 0.92,
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
    alleleOrigins: (
        list[Literal['biparental', 'de novo', 'germline', 'inherited', 'maternal', 'paternal', 'uniparental']] | None
    ) = Field(min_length=1, description='Origin of the variant allele')
    allelicRequirements: (
        list[
            Literal[
                'Autosomal dominant inheritance',
                'Autosomal dominant inheritance with maternal imprinting',
                'Autosomal dominant inheritance with paternal imprinting',
                'Autosomal recessive inheritance',
                'Autosomal unknown',
                'Codominant',
                'Genetic anticipation',
                'Mitochondrial inheritance',
                'Oligogenic inheritance',
                'Other',
                'Semidominant inheritance',
                'Sex-limited autosomal dominant',
                'Somatic mutation',
                'Sporadic',
                'Unknown mechanism',
                'X-linked dominant inheritance',
                'X-linked inheritance',
                'X-linked recessive inheritance',
                'Y-linked inheritance',
                'autosomal unknown',
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
            'confers sensitivity',
            'conflicting classifications of pathogenicity',
            'drug response',
            'established risk allele',
            'evidence only',
            'likely benign',
            'likely pathogenic',
            'likely risk allele',
            'low penetrance',
            'not provided',
            'other',
            'pathogenic',
            'protective',
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
        'practice guideline',
        'reviewed by expert panel',
    ] = Field(description='Confidence qualifier on the reported evidence')
    datasourceId: Literal['eva'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['genetic_association'] = Field(description='Type of the evidence')
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
    variantHgvsId: str | None = Field(
        min_length=1, description='Identifier in HGVS notation of the disease-causing variant'
    )
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
        ge=0.0, le=1.0, description=('Score of the evidence reflecting the strength of the disease/target relationship')
    )
    directionOnTrait: Literal['protect', 'risk'] | None = Field(
        description='Predicted direction of effect on the trait'
    )
    directionOnTarget: Literal['LoF'] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
